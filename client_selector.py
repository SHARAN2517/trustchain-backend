"""
TrustChain-MedAI: Federated Client Selection & Orchestration.

Implements:
  - ClientProfile: Tracks hospital client state and reputation
  - ReputationScorer: Composite reputation from accuracy, reliability, data volume
  - StragglerDetector: Timeout-based straggler detection
  - ClientSelector: Selection strategies (random, reputation-weighted, resource-aware)
  - AsyncFederationOrchestrator: Non-blocking FL rounds with quorum-based aggregation
"""

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple


@dataclass
class ClientProfile:
    """Tracks a hospital client's state for federated learning."""
    client_id: str
    hospital_name: str = ""
    specialty: str = ""
    dataset_size: int = 0
    compute_tier: str = "standard"  # "standard", "gpu", "tpu"

    # Performance history
    rounds_participated: int = 0
    rounds_accepted: int = 0
    rounds_rejected: int = 0
    avg_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    last_contribution_score: float = 0.0
    byzantine_rejections: int = 0

    # Current state
    is_available: bool = True
    last_seen: float = field(default_factory=time.time)
    reputation_score: float = 50.0  # 0-100

    def to_dict(self) -> Dict:
        return {
            "client_id": self.client_id,
            "hospital_name": self.hospital_name,
            "specialty": self.specialty,
            "dataset_size": self.dataset_size,
            "compute_tier": self.compute_tier,
            "rounds_participated": self.rounds_participated,
            "rounds_accepted": self.rounds_accepted,
            "rounds_rejected": self.rounds_rejected,
            "avg_accuracy": round(self.avg_accuracy, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "last_contribution_score": round(self.last_contribution_score, 2),
            "byzantine_rejections": self.byzantine_rejections,
            "is_available": self.is_available,
            "reputation_score": round(self.reputation_score, 1),
        }


class ReputationScorer:
    """
    Computes composite reputation scores for FL clients.

    Score is a weighted combination of:
      - Historical accuracy contribution (40%)
      - Byzantine filter pass rate (30%)
      - Reliability / availability (20%)
      - Data volume contribution (10%)
    """

    def __init__(
        self,
        accuracy_weight: float = 0.4,
        byzantine_weight: float = 0.3,
        reliability_weight: float = 0.2,
        data_weight: float = 0.1,
    ):
        self.accuracy_weight = accuracy_weight
        self.byzantine_weight = byzantine_weight
        self.reliability_weight = reliability_weight
        self.data_weight = data_weight

    def compute(self, profile: ClientProfile, max_dataset_size: int = 50000) -> float:
        """
        Compute reputation score (0-100) for a client.

        Returns updated score.
        """
        # Accuracy component (0-100)
        accuracy_score = min(100.0, profile.avg_accuracy * 100)

        # Byzantine pass rate (0-100)
        total = profile.rounds_accepted + profile.rounds_rejected
        if total > 0:
            byzantine_score = (profile.rounds_accepted / total) * 100
        else:
            byzantine_score = 50.0  # neutral for new clients

        # Reliability: fraction of rounds participated vs expected
        # New clients get neutral score
        if profile.rounds_participated > 0:
            reliability_score = min(100.0, (profile.rounds_participated / 50) * 100)
        else:
            reliability_score = 30.0

        # Data volume contribution (0-100)
        data_score = min(100.0, (profile.dataset_size / max(1, max_dataset_size)) * 100)

        # Weighted combination
        score = (
            self.accuracy_weight * accuracy_score +
            self.byzantine_weight * byzantine_score +
            self.reliability_weight * reliability_score +
            self.data_weight * data_score
        )

        return min(100.0, max(0.0, score))

    def update_profile(self, profile: ClientProfile, **kwargs) -> ClientProfile:
        """
        Update a client's profile after a round and recompute reputation.

        kwargs can include: accuracy, latency_ms, contribution_score,
        accepted (bool), byzantine_status.
        """
        profile.rounds_participated += 1

        if kwargs.get("accepted", True):
            profile.rounds_accepted += 1
        else:
            profile.rounds_rejected += 1

        if kwargs.get("byzantine_status") == "MALICIOUS":
            profile.byzantine_rejections += 1

        # Running average for accuracy
        acc = kwargs.get("accuracy")
        if acc is not None:
            n = profile.rounds_participated
            profile.avg_accuracy = (
                profile.avg_accuracy * (n - 1) + acc
            ) / n

        # Running average for latency
        lat = kwargs.get("latency_ms")
        if lat is not None:
            n = profile.rounds_participated
            profile.avg_latency_ms = (
                profile.avg_latency_ms * (n - 1) + lat
            ) / n

        contrib = kwargs.get("contribution_score")
        if contrib is not None:
            profile.last_contribution_score = contrib

        profile.last_seen = time.time()
        profile.reputation_score = self.compute(profile)

        return profile


class StragglerDetector:
    """
    Detects straggler clients that exceed round timeout deadlines.

    Clients exceeding the timeout are marked as stragglers.
    The round proceeds with available updates using partial aggregation
    once a quorum is reached.
    """

    def __init__(self, timeout_seconds: float = 120.0):
        self.timeout_seconds = timeout_seconds

    def check_stragglers(
        self,
        start_time: float,
        completed_clients: Set[str],
        all_clients: Set[str],
    ) -> Dict:
        """
        Check for stragglers based on elapsed time.

        Returns:
            {
                timed_out: bool,
                elapsed: float,
                completed: list,
                stragglers: list,
                completion_rate: float,
            }
        """
        elapsed = time.time() - start_time
        timed_out = elapsed > self.timeout_seconds
        stragglers = all_clients - completed_clients

        return {
            "timed_out": timed_out,
            "elapsed_seconds": round(elapsed, 2),
            "timeout_seconds": self.timeout_seconds,
            "completed": sorted(completed_clients),
            "stragglers": sorted(stragglers),
            "completion_rate": len(completed_clients) / max(1, len(all_clients)),
        }


class ClientSelector:
    """
    Selects clients for each federated learning round.

    Strategies:
      - random: Uniform random selection from available pool
      - reputation: Probability proportional to reputation score
      - resource_aware: Selects clients with sufficient compute and data
    """

    def __init__(self, profiles: Optional[Dict[str, ClientProfile]] = None):
        self.profiles: Dict[str, ClientProfile] = profiles or {}

    def register_client(self, profile: ClientProfile):
        """Register a new client."""
        self.profiles[profile.client_id] = profile

    def remove_client(self, client_id: str):
        """Remove a client from the pool."""
        self.profiles.pop(client_id, None)

    def get_available(self) -> List[ClientProfile]:
        """Returns list of currently available clients."""
        return [p for p in self.profiles.values() if p.is_available]

    def select(
        self,
        strategy: str = "reputation",
        n: int = 4,
        min_dataset_size: int = 0,
        min_reputation: float = 0.0,
        exclude: Optional[Set[str]] = None,
    ) -> List[ClientProfile]:
        """
        Select n clients for a federation round.

        Args:
            strategy: "random", "reputation", or "resource_aware".
            n: Number of clients to select.
            min_dataset_size: Minimum dataset size filter.
            min_reputation: Minimum reputation score filter.
            exclude: Set of client_ids to exclude.

        Returns:
            List of selected ClientProfile objects.
        """
        exclude = exclude or set()
        candidates = [
            p for p in self.get_available()
            if p.client_id not in exclude
            and p.dataset_size >= min_dataset_size
            and p.reputation_score >= min_reputation
        ]

        if not candidates:
            return []

        n = min(n, len(candidates))

        if strategy == "random":
            return self._select_random(candidates, n)
        elif strategy == "reputation":
            return self._select_reputation(candidates, n)
        elif strategy == "resource_aware":
            return self._select_resource_aware(candidates, n)
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'. Use random/reputation/resource_aware")

    def _select_random(self, candidates: List[ClientProfile], n: int) -> List[ClientProfile]:
        """Uniform random selection."""
        return random.sample(candidates, n)

    def _select_reputation(self, candidates: List[ClientProfile], n: int) -> List[ClientProfile]:
        """Weighted selection proportional to reputation score."""
        # Softmax-style weighting to avoid zero weights
        scores = [max(1.0, p.reputation_score) for p in candidates]
        total = sum(scores)
        weights = [s / total for s in scores]

        # Weighted sampling without replacement
        selected = []
        remaining = list(zip(candidates, weights))

        for _ in range(n):
            if not remaining:
                break
            cands, wts = zip(*remaining)
            total_w = sum(wts)
            normalized = [w / total_w for w in wts]

            r = random.random()
            cumulative = 0.0
            chosen_idx = 0
            for idx, w in enumerate(normalized):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = idx
                    break

            selected.append(cands[chosen_idx])
            remaining = [(c, w) for i, (c, w) in enumerate(remaining) if i != chosen_idx]

        return selected

    def _select_resource_aware(self, candidates: List[ClientProfile], n: int) -> List[ClientProfile]:
        """Select clients with best compute resources and data volume."""
        # Score: data_size * compute_multiplier / latency
        compute_multipliers = {"tpu": 3.0, "gpu": 2.0, "standard": 1.0}

        scored = []
        for p in candidates:
            mult = compute_multipliers.get(p.compute_tier, 1.0)
            latency_factor = 1.0 / max(1.0, p.avg_latency_ms / 100.0)
            score = p.dataset_size * mult * latency_factor
            scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:n]]


class AsyncFederationOrchestrator:
    """
    Non-blocking federation round orchestrator.

    Clients can join rounds asynchronously. Server aggregates once a
    configurable quorum (default 60%) is reached.
    """

    def __init__(
        self,
        selector: ClientSelector,
        quorum_fraction: float = 0.6,
        timeout_seconds: float = 120.0,
    ):
        self.selector = selector
        self.quorum_fraction = quorum_fraction
        self.straggler_detector = StragglerDetector(timeout_seconds)
        self._round_counter = 0

    def prepare_round(
        self,
        strategy: str = "reputation",
        n_clients: int = 4,
        **select_kwargs,
    ) -> Dict:
        """
        Prepare a new federation round by selecting participants.

        Returns round configuration dict.
        """
        self._round_counter += 1
        selected = self.selector.select(strategy=strategy, n=n_clients, **select_kwargs)

        quorum_needed = max(1, int(len(selected) * self.quorum_fraction))

        return {
            "round_id": self._round_counter,
            "selected_clients": [p.client_id for p in selected],
            "num_selected": len(selected),
            "quorum_needed": quorum_needed,
            "quorum_fraction": self.quorum_fraction,
            "strategy": strategy,
            "timeout_seconds": self.straggler_detector.timeout_seconds,
            "profiles": {p.client_id: p.to_dict() for p in selected},
        }

    def check_quorum(
        self,
        round_config: Dict,
        received_clients: Set[str],
        start_time: float,
    ) -> Dict:
        """
        Check if quorum has been reached or timeout exceeded.

        Returns:
            {ready: bool, reason: str, straggler_info: dict}
        """
        all_clients = set(round_config["selected_clients"])
        quorum_needed = round_config["quorum_needed"]

        straggler_info = self.straggler_detector.check_stragglers(
            start_time, received_clients, all_clients,
        )

        if len(received_clients) >= quorum_needed:
            return {
                "ready": True,
                "reason": f"Quorum reached: {len(received_clients)}/{quorum_needed}",
                "straggler_info": straggler_info,
            }

        if straggler_info["timed_out"]:
            if len(received_clients) >= 2:
                return {
                    "ready": True,
                    "reason": f"Timeout — proceeding with {len(received_clients)} clients "
                              f"(quorum was {quorum_needed})",
                    "straggler_info": straggler_info,
                }
            else:
                return {
                    "ready": False,
                    "reason": f"Timeout — insufficient clients ({len(received_clients)} < 2 minimum)",
                    "straggler_info": straggler_info,
                }

        return {
            "ready": False,
            "reason": f"Waiting: {len(received_clients)}/{quorum_needed} clients",
            "straggler_info": straggler_info,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Client Selection & Orchestration — Self-Test")
    print("=" * 60)

    random.seed(42)

    # Create test hospital profiles
    hospitals = [
        ClientProfile("HOSP-RAD-01", "Aster Radiology", "Radiology", 12450, "gpu",
                       rounds_participated=20, rounds_accepted=19, avg_accuracy=0.912),
        ClientProfile("HOSP-ONC-02", "Nexus Oncology", "Oncology", 8230, "gpu",
                       rounds_participated=18, rounds_accepted=17, avg_accuracy=0.897),
        ClientProfile("HOSP-PED-03", "Kaveri Pediatrics", "Pediatrics", 6890, "standard",
                       rounds_participated=15, rounds_accepted=14, avg_accuracy=0.885),
        ClientProfile("HOSP-CAR-04", "Pulse Cardiology", "Cardiology", 9120, "gpu",
                       rounds_participated=12, rounds_accepted=10, avg_accuracy=0.873,
                       byzantine_rejections=2),
        ClientProfile("HOSP-NEW-05", "New Hospital", "Radiology", 2000, "standard",
                       rounds_participated=2, rounds_accepted=2, avg_accuracy=0.780),
    ]

    # Test Reputation Scorer
    print("\n  [1] Reputation Scoring:")
    scorer = ReputationScorer()
    for h in hospitals:
        h.reputation_score = scorer.compute(h)
        print(f"      {h.client_id}: rep={h.reputation_score:.1f} "
              f"(acc={h.avg_accuracy:.3f}, accepted={h.rounds_accepted}/{h.rounds_participated})")

    # Test Client Selector
    print("\n  [2] Client Selection Strategies:")
    selector = ClientSelector()
    for h in hospitals:
        selector.register_client(h)

    # Random
    selected = selector.select("random", n=3)
    print(f"      Random (n=3): {[p.client_id for p in selected]}")

    # Reputation-weighted
    selected = selector.select("reputation", n=3)
    print(f"      Reputation (n=3): {[p.client_id for p in selected]}")
    print(f"        Scores: {[f'{p.reputation_score:.1f}' for p in selected]}")

    # Resource-aware
    selected = selector.select("resource_aware", n=3)
    print(f"      Resource-aware (n=3): {[p.client_id for p in selected]}")

    # With filters
    selected = selector.select("reputation", n=3, min_dataset_size=5000)
    print(f"      Filtered (data≥5k): {[p.client_id for p in selected]}")

    # Test Straggler Detection
    print("\n  [3] Straggler Detection:")
    detector = StragglerDetector(timeout_seconds=2.0)
    start = time.time()
    all_c = {"HOSP-RAD-01", "HOSP-ONC-02", "HOSP-PED-03"}
    completed = {"HOSP-RAD-01"}
    result = detector.check_stragglers(start, completed, all_c)
    print(f"      Stragglers: {result['stragglers']}")
    print(f"      Completion: {result['completion_rate']:.1%}")

    # Test Orchestrator
    print("\n  [4] Async Federation Orchestrator:")
    orchestrator = AsyncFederationOrchestrator(
        selector=selector, quorum_fraction=0.6, timeout_seconds=5.0,
    )

    round_config = orchestrator.prepare_round(strategy="reputation", n_clients=4)
    print(f"      Round {round_config['round_id']}: {round_config['num_selected']} clients selected")
    print(f"      Quorum needed: {round_config['quorum_needed']}")
    print(f"      Selected: {round_config['selected_clients']}")

    # Simulate quorum check
    start = time.time()
    received = set()
    for cid in round_config["selected_clients"][:3]:  # 3 of 4 respond
        received.add(cid)

    quorum = orchestrator.check_quorum(round_config, received, start)
    print(f"      Quorum ready: {quorum['ready']}")
    print(f"      Reason: {quorum['reason']}")

    # Test profile update
    print("\n  [5] Profile Update After Round:")
    h = hospitals[0]
    scorer.update_profile(h, accuracy=0.925, latency_ms=145, contribution_score=94.5, accepted=True)
    print(f"      {h.client_id}: rep={h.reputation_score:.1f}, "
          f"acc={h.avg_accuracy:.3f}, rounds={h.rounds_participated}")

    print("\n" + "=" * 60)
    print("  Client selection tests completed successfully!")
    print("=" * 60)
