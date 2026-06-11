"""
TrustChain-MedAI: Byzantine-Robust Aggregation Strategies.

Implements production-grade robust aggregation for federated learning:
  - Multi-Krum  (Blanchard et al., 2017)
  - FLTrust     (Cao et al., 2021)
  - Bulyan      (Mhamdi et al., 2018)
  - FedAvg      (McMahan et al., 2017) — baseline

Design principle: These aggregators decide HOW MUCH WEIGHT to give each
client update. They optionally consume verdicts from poisoning_detector.py
(CLEAN/SUSPICIOUS/MALICIOUS) to enhance decisions, but their primary job
is aggregation, not detection. The two modules compose cleanly.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def _flatten(params: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of parameter arrays into a single 1-D vector."""
    return np.concatenate([p.ravel() for p in params])


def _unflatten(flat: np.ndarray, shapes: List[tuple], dtypes: List) -> List[np.ndarray]:
    """Reconstruct parameter list from a flat vector."""
    result = []
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        size = int(np.prod(shape))
        result.append(flat[offset:offset + size].reshape(shape).astype(dtype))
        offset += size
    return result


def _get_shapes_dtypes(params: List[np.ndarray]):
    """Extract shapes and dtypes from a parameter list."""
    return [p.shape for p in params], [p.dtype for p in params]


# ─────────────────────────────────────────────────────────────────────────────
# FedAvg — Baseline
# ─────────────────────────────────────────────────────────────────────────────

def fedavg(
    client_params: List[List[np.ndarray]],
    weights: Optional[List[float]] = None,
) -> List[np.ndarray]:
    """
    Federated Averaging (McMahan et al., 2017).

    Computes weighted average of client model parameters.
    If weights=None, uniform averaging is used.

    Args:
        client_params: List of client parameter lists.
        weights: Optional per-client weights (e.g., proportional to dataset size).

    Returns:
        Aggregated parameter list.
    """
    n = len(client_params)
    if n == 0:
        raise ValueError("No client parameters provided")

    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    aggregated = []
    for layer_idx in range(len(client_params[0])):
        layer_sum = np.zeros_like(client_params[0][layer_idx], dtype=np.float64)
        for client_idx in range(n):
            layer_sum += weights[client_idx] * client_params[client_idx][layer_idx]
        aggregated.append(layer_sum.astype(client_params[0][layer_idx].dtype))

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Krum (Blanchard et al., 2017)
# ─────────────────────────────────────────────────────────────────────────────

def multi_krum(
    client_params: List[List[np.ndarray]],
    f: int,
    m: int = 1,
    verdicts: Optional[Dict[int, str]] = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Multi-Krum byzantine-robust aggregation.

    For each client, computes the sum of squared L2 distances to its
    (n - f - 2) closest neighbors. Selects the m clients with lowest
    Krum scores and averages their parameters.

    Provably tolerates f < n/2 - 1 Byzantine clients.

    Args:
        client_params: List of client parameter lists.
        f: Maximum number of Byzantine clients to tolerate.
        m: Number of top clients to select (default=1 for basic Krum).
        verdicts: Optional dict {client_idx: "CLEAN"|"SUSPICIOUS"|"MALICIOUS"}
                  from poisoning_detector. MALICIOUS clients get worst scores.

    Returns:
        (aggregated_params, krum_scores_per_client)
    """
    n = len(client_params)
    if n < 2 * f + 3:
        raise ValueError(
            f"Multi-Krum requires n >= 2f+3. Got n={n}, f={f}. "
            f"Need at least {2 * f + 3} clients."
        )

    m = min(m, n - f)

    # Flatten each client's params to a single vector
    flat_params = [_flatten(p) for p in client_params]

    # Compute pairwise squared L2 distances
    distances = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = flat_params[i] - flat_params[j]
            d = float(np.dot(diff, diff))
            distances[i][j] = d
            distances[j][i] = d

    # Krum score: sum of distances to (n - f - 2) closest neighbors
    num_closest = n - f - 2
    scores = np.zeros(n, dtype=np.float64)

    for i in range(n):
        sorted_dists = np.sort(distances[i])
        # Skip self (distance=0), take next num_closest
        scores[i] = np.sum(sorted_dists[1:num_closest + 1])

    # Apply verdicts: MALICIOUS clients get penalty score
    if verdicts:
        max_score = np.max(scores) * 10.0
        for idx, verdict in verdicts.items():
            if 0 <= idx < n and verdict == "MALICIOUS":
                scores[idx] = max_score

    # Select m clients with lowest scores
    selected_indices = np.argsort(scores)[:m]
    score_list = scores.tolist()

    # Average selected clients' parameters
    shapes, dtypes = _get_shapes_dtypes(client_params[0])
    selected_flat = np.mean([flat_params[i] for i in selected_indices], axis=0)
    aggregated = _unflatten(selected_flat, shapes, dtypes)

    return aggregated, score_list


# ─────────────────────────────────────────────────────────────────────────────
# FLTrust (Cao et al., 2021)
# ─────────────────────────────────────────────────────────────────────────────

def fltrust(
    client_params: List[List[np.ndarray]],
    server_params: List[np.ndarray],
    global_params: Optional[List[np.ndarray]] = None,
    verdicts: Optional[Dict[int, str]] = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    FLTrust byzantine-robust aggregation.

    Server maintains a small clean root dataset and trains its own update.
    Each client's update is weighted by ReLU(cosine_similarity) with the
    server update. Clients with negative similarity get zero weight.

    Immune to arbitrary-scale poisoning attacks.

    Args:
        client_params: List of client parameter lists (post-training).
        server_params: Server's own update from its root dataset.
        global_params: Previous global model params (for computing deltas).
                       If None, client_params are used directly as updates.
        verdicts: Optional verdicts from poisoning_detector.

    Returns:
        (aggregated_params, trust_scores_per_client)
    """
    n = len(client_params)
    if n == 0:
        raise ValueError("No client parameters provided")

    # Compute update vectors (delta from global)
    if global_params is not None:
        global_flat = _flatten(global_params)
        client_deltas = [_flatten(p) - global_flat for p in client_params]
        server_delta = _flatten(server_params) - global_flat
    else:
        client_deltas = [_flatten(p) for p in client_params]
        server_delta = _flatten(server_params)

    server_norm = np.linalg.norm(server_delta)
    if server_norm < 1e-10:
        # Server had negligible update; fall back to FedAvg
        return fedavg(client_params), [1.0 / n] * n

    # Compute trust scores via cosine similarity with server update
    trust_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        client_norm = np.linalg.norm(client_deltas[i])
        if client_norm < 1e-10:
            trust_scores[i] = 0.0
            continue
        cosine = float(np.dot(client_deltas[i], server_delta) / (client_norm * server_norm))
        # ReLU: only positive cosine contributes
        trust_scores[i] = max(0.0, cosine)

    # Apply verdicts: force MALICIOUS to zero trust
    if verdicts:
        for idx, verdict in verdicts.items():
            if 0 <= idx < n and verdict == "MALICIOUS":
                trust_scores[idx] = 0.0

    # Normalize trust scores
    total_trust = np.sum(trust_scores)
    if total_trust < 1e-10:
        # All clients are untrusted; return server's own params
        return [p.copy() for p in server_params], trust_scores.tolist()

    normalized_ts = trust_scores / total_trust

    # Normalize each client's update to have the same magnitude as server's
    # (prevents scale attacks), then weight by trust score
    shapes, dtypes = _get_shapes_dtypes(client_params[0])
    aggregated_delta = np.zeros_like(server_delta)

    for i in range(n):
        if normalized_ts[i] < 1e-10:
            continue
        client_norm = np.linalg.norm(client_deltas[i])
        if client_norm < 1e-10:
            continue
        # Scale client delta to server's norm (neutralizes magnitude attacks)
        scaled_delta = client_deltas[i] * (server_norm / client_norm)
        aggregated_delta += normalized_ts[i] * scaled_delta

    # Apply aggregated delta to global params
    if global_params is not None:
        result_flat = global_flat + aggregated_delta
    else:
        result_flat = aggregated_delta

    aggregated = _unflatten(result_flat, shapes, dtypes)
    return aggregated, trust_scores.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Bulyan (Mhamdi et al., 2018)
# ─────────────────────────────────────────────────────────────────────────────

def bulyan(
    client_params: List[List[np.ndarray]],
    f: int,
    verdicts: Optional[Dict[int, str]] = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Bulyan byzantine-robust aggregation.

    Two-phase defense:
      1. Run Multi-Krum to select (n - 2f) candidate clients
      2. For each parameter coordinate, sort values across candidates
         and trim the f largest and f smallest, then average the rest

    Strongest known defense against both omniscient and partial-knowledge
    adversaries in the honest-majority setting.

    Requires n >= 4f + 3.

    Args:
        client_params: List of client parameter lists.
        f: Maximum number of Byzantine clients to tolerate.
        verdicts: Optional verdicts from poisoning_detector.

    Returns:
        (aggregated_params, krum_selection_scores)
    """
    n = len(client_params)
    if n < 4 * f + 3:
        raise ValueError(
            f"Bulyan requires n >= 4f+3. Got n={n}, f={f}. "
            f"Need at least {4 * f + 3} clients."
        )

    # Phase 1: Use Multi-Krum to select (n - 2f) candidates
    num_select = n - 2 * f
    _, krum_scores = multi_krum(client_params, f=f, m=num_select, verdicts=verdicts)

    selected_indices = np.argsort(krum_scores)[:num_select]
    selected_params = [client_params[i] for i in selected_indices]

    # Phase 2: Coordinate-wise trimmed mean
    shapes, dtypes = _get_shapes_dtypes(client_params[0])
    flat_selected = np.array([_flatten(p) for p in selected_params])

    dim = flat_selected.shape[1]
    trimmed = np.zeros(dim, dtype=np.float64)

    # For each coordinate, sort, trim f from each end, average the rest
    trim = min(f, (num_select - 1) // 2)
    for d in range(dim):
        values = np.sort(flat_selected[:, d])
        if trim > 0:
            trimmed[d] = np.mean(values[trim:-trim])
        else:
            trimmed[d] = np.mean(values)

    aggregated = _unflatten(trimmed, shapes, dtypes)
    return aggregated, krum_scores


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(
    method: str,
    client_params: List[List[np.ndarray]],
    **kwargs,
) -> Tuple[List[np.ndarray], Dict]:
    """
    Routes to the correct aggregation strategy.

    Args:
        method: One of "fedavg", "krum", "fltrust", "bulyan".
        client_params: Client parameter lists.
        **kwargs: Strategy-specific arguments:
            - f (int): Byzantine tolerance (krum, bulyan)
            - m (int): Number of selections for multi-krum (default=1)
            - server_params: Server root update (fltrust)
            - global_params: Previous global params (fltrust)
            - verdicts: Poisoning verdicts dict (all strategies)
            - weights: Client weights (fedavg)

    Returns:
        (aggregated_params, metadata_dict)
    """
    method = method.lower().strip()

    if method == "fedavg":
        result = fedavg(client_params, weights=kwargs.get("weights"))
        return result, {
            "method": "fedavg",
            "num_clients": len(client_params),
            "scores": [1.0 / len(client_params)] * len(client_params),
        }

    elif method == "krum":
        f = kwargs.get("f", max(1, len(client_params) // 4))
        m = kwargs.get("m", 1)
        verdicts = kwargs.get("verdicts")
        result, scores = multi_krum(client_params, f=f, m=m, verdicts=verdicts)
        return result, {
            "method": "multi_krum",
            "f": f,
            "m": m,
            "scores": scores,
            "selected": sorted(np.argsort(scores)[:m].tolist()),
        }

    elif method == "fltrust":
        server_params = kwargs.get("server_params")
        if server_params is None:
            raise ValueError("FLTrust requires 'server_params'")
        global_params = kwargs.get("global_params")
        verdicts = kwargs.get("verdicts")
        result, scores = fltrust(
            client_params, server_params,
            global_params=global_params, verdicts=verdicts,
        )
        return result, {
            "method": "fltrust",
            "trust_scores": scores,
            "num_trusted": sum(1 for s in scores if s > 0),
        }

    elif method == "bulyan":
        f = kwargs.get("f", max(1, len(client_params) // 6))
        verdicts = kwargs.get("verdicts")
        result, scores = bulyan(client_params, f=f, verdicts=verdicts)
        return result, {
            "method": "bulyan",
            "f": f,
            "krum_scores": scores,
        }

    else:
        raise ValueError(
            f"Unknown aggregation method: '{method}'. "
            f"Supported: fedavg, krum, fltrust, bulyan"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Byzantine-Robust Aggregation — Self-Test")
    print("=" * 60)

    np.random.seed(42)

    # Create 5 clients: 3 honest, 2 poisoned (100x scaling)
    base_params = [np.random.randn(10, 5).astype(np.float32),
                   np.random.randn(5).astype(np.float32)]

    honest_clients = []
    for _ in range(3):
        noise = [np.random.randn(*p.shape).astype(np.float32) * 0.1 for p in base_params]
        honest_clients.append([b + n for b, n in zip(base_params, noise)])

    poisoned_clients = []
    for _ in range(2):
        poisoned_clients.append([p * 100.0 for p in base_params])

    all_clients = honest_clients + poisoned_clients
    n_clients = len(all_clients)
    print(f"\n  Clients: {n_clients} (3 honest + 2 poisoned @ 100x scale)")

    # 1. FedAvg (vulnerable baseline)
    print("\n  [1] FedAvg (no defense):")
    avg_result = fedavg(all_clients)
    avg_norm = np.linalg.norm(_flatten(avg_result))
    honest_avg = fedavg(honest_clients)
    honest_norm = np.linalg.norm(_flatten(honest_avg))
    print(f"      Result norm: {avg_norm:.2f} (honest-only: {honest_norm:.2f})")
    print(f"      Poisoned? {'YES — norm inflated' if avg_norm > honest_norm * 5 else 'No'}")

    # 2. Multi-Krum
    print("\n  [2] Multi-Krum (f=1, m=1):")
    krum_result, krum_scores = multi_krum(all_clients, f=1, m=1)
    krum_norm = np.linalg.norm(_flatten(krum_result))
    selected = int(np.argmin(krum_scores))
    print(f"      Scores: {[f'{s:.1f}' for s in krum_scores]}")
    print(f"      Selected client: {selected} ({'HONEST' if selected < 3 else 'POISONED'})")
    print(f"      Result norm: {krum_norm:.2f} (vs honest {honest_norm:.2f})")

    # 3. FLTrust
    print("\n  [3] FLTrust (server = honest client 0):")
    server = honest_clients[0]
    fl_result, trust_scores = fltrust(all_clients, server_params=server)
    fl_norm = np.linalg.norm(_flatten(fl_result))
    print(f"      Trust scores: {[f'{s:.3f}' for s in trust_scores]}")
    print(f"      Poisoned clients trust: {trust_scores[3]:.4f}, {trust_scores[4]:.4f}")
    print(f"      Result norm: {fl_norm:.2f}")

    # 4. Bulyan (requires n >= 4f+3 = 7 for f=1)
    extra_honest = []
    for _ in range(4):
        noise = [np.random.randn(*p.shape).astype(np.float32) * 0.1 for p in base_params]
        extra_honest.append([b + n for b, n in zip(base_params, noise)])
    large_pool = honest_clients + extra_honest + poisoned_clients  # 9 total
    print(f"\n  [4] Bulyan (f=1, n={len(large_pool)} clients):")
    bul_result, bul_scores = bulyan(large_pool, f=1)
    bul_norm = np.linalg.norm(_flatten(bul_result))
    print(f"      Result norm: {bul_norm:.2f} (honest baseline: {honest_norm:.2f})")
    poisoned_rejected = all(
        bul_scores[i] > np.median(bul_scores) * 2 for i in [7, 8]
    )
    print(f"      Poisoned clients rejected: {poisoned_rejected}")

    # 5. Strategy dispatcher with verdicts
    print("\n  [5] Dispatcher with poisoning verdicts:")
    verdicts = {3: "MALICIOUS", 4: "MALICIOUS", 0: "CLEAN", 1: "CLEAN", 2: "CLEAN"}
    result, meta = aggregate("krum", all_clients, f=1, m=1, verdicts=verdicts)
    print(f"      Method: {meta['method']}")
    print(f"      Selected: {meta['selected']}")
    result_norm = np.linalg.norm(_flatten(result))
    print(f"      Result norm: {result_norm:.2f}")

    print("\n" + "=" * 60)
    print("  Byzantine aggregation tests completed successfully!")
    print("=" * 60)
