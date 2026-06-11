"""
TrustChain-MedAI: Model Poisoning Detection Engine.

Detects malicious gradient/weight submissions using multi-layer analysis:
  - Gradient Norm Analysis (L2 z-score)
  - Cosine Similarity Analysis (directional outliers)
  - Mahalanobis Distance Analysis (statistical outliers via PCA+SVD)
  - Ensemble detector with consensus voting

Design principle: This module ONLY outputs verdicts (CLEAN/SUSPICIOUS/MALICIOUS)
and risk scores. It does NOT aggregate. The byzantine aggregators in
byzantine.py consume these verdicts to adjust their weighting.
"""

import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple


VERDICT_CLEAN = "CLEAN"
VERDICT_SUSPICIOUS = "SUSPICIOUS"
VERDICT_MALICIOUS = "MALICIOUS"


def _flatten(params: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of parameter arrays into a single 1-D vector."""
    return np.concatenate([p.ravel() for p in params])


class GradientNormAnalyzer:
    """
    Detects poisoning via L2 norm outlier analysis.

    Computes the L2 norm of each client's flattened parameter vector and
    flags statistical outliers using z-scores:
      - z > 3.0  → MALICIOUS
      - z > 2.0  → SUSPICIOUS
      - else     → CLEAN
    """

    def __init__(self, malicious_threshold: float = 3.0, suspicious_threshold: float = 2.0):
        self.malicious_threshold = malicious_threshold
        self.suspicious_threshold = suspicious_threshold

    def analyze(self, client_params: List[List[np.ndarray]]) -> List[Dict]:
        """
        Analyze client updates by L2 norm.

        Uses two detection methods:
          1. Z-score outlier detection (standard)
          2. Ratio to median (robust to outlier-inflated std)
        A client is flagged if EITHER method triggers.

        Returns list of {client_idx, norm, z_score, ratio_to_median, verdict}.
        """
        norms = np.array([np.linalg.norm(_flatten(p)) for p in client_params])
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        median_norm = float(np.median(norms))

        results = []
        for i, norm_val in enumerate(norms):
            # Method 1: Z-score
            if std_norm > 1e-10:
                z = abs(float(norm_val) - float(mean_norm)) / float(std_norm)
            else:
                z = 0.0

            # Method 2: Ratio to median (robust to single large outlier
            # inflating std and pulling mean towards it)
            ratio = float(norm_val) / max(1e-10, median_norm)

            # Combined verdict: either method can trigger
            if z > self.malicious_threshold or ratio > 5.0:
                verdict = VERDICT_MALICIOUS
            elif z > self.suspicious_threshold or ratio > 3.0:
                verdict = VERDICT_SUSPICIOUS
            else:
                verdict = VERDICT_CLEAN

            results.append({
                "client_idx": i,
                "norm": round(float(norm_val), 6),
                "mean_norm": round(float(mean_norm), 6),
                "median_norm": round(median_norm, 6),
                "z_score": round(z, 4),
                "ratio_to_median": round(ratio, 4),
                "verdict": verdict,
            })

        return results


class CosineSimilarityAnalyzer:
    """
    Detects poisoning via cosine similarity with a reference direction.

    If no reference is provided, uses the mean of all client updates.
    Flags clients with:
      - negative cosine similarity → MALICIOUS (adversarial direction)
      - cosine < 0.3              → SUSPICIOUS (weak alignment)
      - else                      → CLEAN
    """

    def __init__(self, malicious_threshold: float = 0.0, suspicious_threshold: float = 0.3):
        self.malicious_threshold = malicious_threshold
        self.suspicious_threshold = suspicious_threshold

    def analyze(
        self,
        client_params: List[List[np.ndarray]],
        reference: Optional[List[np.ndarray]] = None,
    ) -> List[Dict]:
        """
        Analyze client updates by cosine similarity to a reference.

        Returns list of {client_idx, cosine_sim, verdict}.
        """
        flat_params = [_flatten(p) for p in client_params]

        if reference is not None:
            ref_flat = _flatten(reference)
        else:
            # Use mean of all clients as reference
            ref_flat = np.mean(flat_params, axis=0)

        ref_norm = np.linalg.norm(ref_flat)
        if ref_norm < 1e-10:
            return [
                {"client_idx": i, "cosine_sim": 0.0, "verdict": VERDICT_SUSPICIOUS}
                for i in range(len(client_params))
            ]

        results = []
        for i, fp in enumerate(flat_params):
            client_norm = np.linalg.norm(fp)
            if client_norm < 1e-10:
                cosine = 0.0
            else:
                cosine = float(np.dot(fp, ref_flat) / (client_norm * ref_norm))

            if cosine < self.malicious_threshold:
                verdict = VERDICT_MALICIOUS
            elif cosine < self.suspicious_threshold:
                verdict = VERDICT_SUSPICIOUS
            else:
                verdict = VERDICT_CLEAN

            results.append({
                "client_idx": i,
                "cosine_sim": round(cosine, 6),
                "verdict": verdict,
            })

        return results


class MahalanobisAnalyzer:
    """
    Detects poisoning via Mahalanobis distance in PCA-reduced space.

    Reduces dimensionality via SVD (manual PCA, no sklearn dependency),
    then computes Mahalanobis distance for each client from the empirical
    distribution. Uses chi-squared thresholds for verdicts:
      - p < 0.01  → MALICIOUS
      - p < 0.05  → SUSPICIOUS
      - else      → CLEAN
    """

    def __init__(self, max_components: int = 10):
        self.max_components = max_components

    def analyze(self, client_params: List[List[np.ndarray]]) -> List[Dict]:
        """
        Analyze client updates via Mahalanobis distance.

        Returns list of {client_idx, mahalanobis_dist, p_value_approx, verdict}.
        """
        n = len(client_params)
        if n < 3:
            return [
                {"client_idx": i, "mahalanobis_dist": 0.0,
                 "p_value_approx": 1.0, "verdict": VERDICT_CLEAN}
                for i in range(n)
            ]

        flat_params = np.array([_flatten(p) for p in client_params])

        # PCA via SVD (manual, no sklearn)
        k = min(self.max_components, n - 1, flat_params.shape[1])
        mean_vec = np.mean(flat_params, axis=0)
        centered = flat_params - mean_vec

        # Truncated SVD on the centered data
        # For n << d, compute SVD on the Gram matrix (n x n) for efficiency
        if n < flat_params.shape[1]:
            gram = centered @ centered.T
            eigvals, eigvecs = np.linalg.eigh(gram)
            # Sort descending
            idx = np.argsort(eigvals)[::-1][:k]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            # Project to PCA space
            # Principal components in original space: V = X^T U Lambda^{-1/2}
            valid = eigvals > 1e-10
            projected = eigvecs[:, valid] * np.sqrt(np.maximum(eigvals[valid], 1e-10))
        else:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            projected = U[:, :k] * S[:k]

        # Compute Mahalanobis distance in PCA space
        pca_mean = np.mean(projected, axis=0)
        pca_centered = projected - pca_mean

        # Covariance matrix in PCA space
        if projected.shape[0] > 1:
            cov = np.cov(projected, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            # Regularize for numerical stability
            cov += np.eye(cov.shape[0]) * 1e-6
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.eye(cov.shape[0])
        else:
            cov_inv = np.eye(projected.shape[1])

        # Chi-squared critical values for degrees of freedom = k
        # Approximation: for large k, chi2(k) ≈ k * (1 - 2/(9k) + z*sqrt(2/(9k)))^3
        # For p=0.01 (z=2.326) and p=0.05 (z=1.645)
        df = projected.shape[1]
        chi2_01 = self._chi2_ppf(0.99, df)
        chi2_05 = self._chi2_ppf(0.95, df)

        results = []
        for i in range(n):
            diff = pca_centered[i]
            mahal_sq = float(diff @ cov_inv @ diff)
            mahal = float(np.sqrt(max(0, mahal_sq)))

            # Approximate p-value from chi-squared CDF
            p_value = self._chi2_sf(mahal_sq, df)

            if mahal_sq > chi2_01:
                verdict = VERDICT_MALICIOUS
            elif mahal_sq > chi2_05:
                verdict = VERDICT_SUSPICIOUS
            else:
                verdict = VERDICT_CLEAN

            results.append({
                "client_idx": i,
                "mahalanobis_dist": round(mahal, 4),
                "mahalanobis_sq": round(mahal_sq, 4),
                "p_value_approx": round(p_value, 6),
                "verdict": verdict,
            })

        return results

    @staticmethod
    def _chi2_ppf(p: float, df: int) -> float:
        """Approximate chi-squared inverse CDF using Wilson-Hilferty transformation."""
        if df <= 0:
            return 0.0
        # Normal quantile approximation
        z_map = {0.99: 2.326, 0.95: 1.645, 0.90: 1.282}
        z = z_map.get(p, 1.645)
        # Wilson-Hilferty approximation
        term = 1.0 - 2.0 / (9.0 * df) + z * np.sqrt(2.0 / (9.0 * df))
        return df * max(0, term ** 3)

    @staticmethod
    def _chi2_sf(x: float, df: int) -> float:
        """Approximate chi-squared survival function (1 - CDF).
        Uses the normal approximation for the chi-squared distribution."""
        if df <= 0 or x <= 0:
            return 1.0
        # Normalize to standard normal
        z = (((x / df) ** (1.0 / 3.0)) - (1.0 - 2.0 / (9.0 * df))) / np.sqrt(2.0 / (9.0 * df))
        # Standard normal CDF approximation (Abramowitz and Stegun)
        if z < -8:
            return 1.0
        if z > 8:
            return 0.0
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
        cdf = 1.0 - pdf * poly if z > 0 else pdf * poly
        return 1.0 - cdf


class EnsemblePoisoningDetector:
    """
    Ensemble poisoning detector combining multiple analysis strategies.

    Consensus logic:
      - If ANY analyzer says MALICIOUS → final verdict = MALICIOUS
      - If 2+ analyzers say SUSPICIOUS → final verdict = SUSPICIOUS
      - Else → CLEAN

    Risk score ∈ [0, 1] computed as weighted average of sub-analyzer signals.
    """

    def __init__(self, analyzers: Optional[List] = None):
        if analyzers is None:
            self.analyzers = {
                "gradient_norm": GradientNormAnalyzer(),
                "cosine_similarity": CosineSimilarityAnalyzer(),
                "mahalanobis": MahalanobisAnalyzer(),
            }
        else:
            self.analyzers = {type(a).__name__: a for a in analyzers}

    def detect(
        self,
        client_params: List[List[np.ndarray]],
        reference: Optional[List[np.ndarray]] = None,
    ) -> Dict[int, Dict]:
        """
        Run all analyzers and produce ensemble verdicts.

        Args:
            client_params: List of client parameter lists.
            reference: Optional reference params for cosine similarity.

        Returns:
            Dict mapping client_idx to {verdict, risk_score, details}.
        """
        n = len(client_params)
        all_results = {}

        # Run each analyzer
        for name, analyzer in self.analyzers.items():
            if name == "cosine_similarity" or isinstance(analyzer, CosineSimilarityAnalyzer):
                results = analyzer.analyze(client_params, reference=reference)
            else:
                results = analyzer.analyze(client_params)
            all_results[name] = {r["client_idx"]: r for r in results}

        # Build ensemble verdicts
        ensemble = {}
        for i in range(n):
            sub_verdicts = {}
            malicious_count = 0
            suspicious_count = 0

            for name in self.analyzers:
                if i in all_results[name]:
                    v = all_results[name][i]["verdict"]
                    sub_verdicts[name] = all_results[name][i]
                    if v == VERDICT_MALICIOUS:
                        malicious_count += 1
                    elif v == VERDICT_SUSPICIOUS:
                        suspicious_count += 1

            # Consensus
            if malicious_count > 0:
                final_verdict = VERDICT_MALICIOUS
            elif suspicious_count >= 2:
                final_verdict = VERDICT_SUSPICIOUS
            else:
                final_verdict = VERDICT_CLEAN

            # Risk score: weighted combination
            risk = 0.0
            num_analyzers = len(self.analyzers)
            for name in self.analyzers:
                if i in all_results[name]:
                    v = all_results[name][i]["verdict"]
                    if v == VERDICT_MALICIOUS:
                        risk += 1.0 / num_analyzers
                    elif v == VERDICT_SUSPICIOUS:
                        risk += 0.5 / num_analyzers

            ensemble[i] = {
                "client_idx": i,
                "verdict": final_verdict,
                "risk_score": round(risk, 4),
                "malicious_votes": malicious_count,
                "suspicious_votes": suspicious_count,
                "details": sub_verdicts,
            }

        return ensemble

    def get_verdicts_dict(
        self,
        client_params: List[List[np.ndarray]],
        reference: Optional[List[np.ndarray]] = None,
    ) -> Dict[int, str]:
        """
        Convenience method returning just {client_idx: verdict_string}.
        This is the format consumed by byzantine.py aggregators.
        """
        full = self.detect(client_params, reference=reference)
        return {idx: info["verdict"] for idx, info in full.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Model Poisoning Detection — Self-Test")
    print("=" * 60)

    np.random.seed(42)

    # Create 5 clients: 3 honest, 1 scaling attack (50x), 1 sign-flip attack
    base = [np.random.randn(20, 10).astype(np.float32),
            np.random.randn(10).astype(np.float32)]

    honest = []
    for _ in range(3):
        noise = [np.random.randn(*p.shape).astype(np.float32) * 0.05 for p in base]
        honest.append([b + n for b, n in zip(base, noise)])

    # Attack 1: 50x scaling
    scaling_attack = [p * 50.0 for p in base]

    # Attack 2: sign-flip (directional reversal)
    sign_flip = [(-p + np.random.randn(*p.shape).astype(np.float32) * 0.01) for p in base]

    all_clients = honest + [scaling_attack, sign_flip]
    print(f"\n  Clients: 5 (3 honest, 1 scaling@50x, 1 sign-flip)")
    print(f"  Client indices: 0,1,2=honest, 3=scaling, 4=sign-flip")

    # Test 1: Gradient Norm Analyzer
    print("\n  [1] Gradient Norm Analyzer:")
    gna = GradientNormAnalyzer()
    for r in gna.analyze(all_clients):
        flag = " ⚠" if r["verdict"] != VERDICT_CLEAN else ""
        print(f"      Client {r['client_idx']}: norm={r['norm']:.2f}, "
              f"z={r['z_score']:.2f} → {r['verdict']}{flag}")

    # Test 2: Cosine Similarity Analyzer
    print("\n  [2] Cosine Similarity Analyzer:")
    csa = CosineSimilarityAnalyzer()
    for r in csa.analyze(all_clients):
        flag = " ⚠" if r["verdict"] != VERDICT_CLEAN else ""
        print(f"      Client {r['client_idx']}: cosine={r['cosine_sim']:.4f} → "
              f"{r['verdict']}{flag}")

    # Test 3: Mahalanobis Analyzer
    print("\n  [3] Mahalanobis Analyzer:")
    ma = MahalanobisAnalyzer()
    for r in ma.analyze(all_clients):
        flag = " ⚠" if r["verdict"] != VERDICT_CLEAN else ""
        print(f"      Client {r['client_idx']}: mahal={r['mahalanobis_dist']:.2f}, "
              f"p≈{r['p_value_approx']:.4f} → {r['verdict']}{flag}")

    # Test 4: Ensemble Detector
    print("\n  [4] Ensemble Detector (consensus):")
    detector = EnsemblePoisoningDetector()
    ensemble = detector.detect(all_clients)
    for idx, info in ensemble.items():
        flag = " ⚠" if info["verdict"] != VERDICT_CLEAN else " ✓"
        print(f"      Client {idx}: risk={info['risk_score']:.2f}, "
              f"mal_votes={info['malicious_votes']}, "
              f"sus_votes={info['suspicious_votes']} → "
              f"{info['verdict']}{flag}")

    # Verify correctness
    assert ensemble[3]["verdict"] == VERDICT_MALICIOUS, "Scaling attack should be MALICIOUS"
    assert ensemble[4]["verdict"] in (VERDICT_MALICIOUS, VERDICT_SUSPICIOUS), \
        "Sign-flip should be at least SUSPICIOUS"
    for i in range(3):
        assert ensemble[i]["verdict"] == VERDICT_CLEAN, f"Honest client {i} should be CLEAN"
    print("\n  ✓ All honest clients correctly marked CLEAN")
    print("  ✓ Scaling attack correctly marked MALICIOUS")
    print(f"  ✓ Sign-flip attack marked {ensemble[4]['verdict']}")

    # Test 5: Verdicts dict (format for byzantine.py)
    print("\n  [5] Verdicts dict for byzantine.py:")
    verdicts = detector.get_verdicts_dict(all_clients)
    print(f"      {verdicts}")

    print("\n" + "=" * 60)
    print("  Poisoning detection tests completed successfully!")
    print("=" * 60)
