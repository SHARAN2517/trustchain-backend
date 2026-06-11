"""
TrustChain-MedAI: Per-Hospital Metrics & Federation Analytics Engine.

Tracks per-hospital performance, per-round convergence data, privacy budget
consumption, and provides time-series analytics for the dashboard.
"""

import json
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class HospitalMetricsStore:
    """
    Tracks per-hospital performance metrics across federated learning rounds.
    Stores accuracy, loss, contribution scores, Byzantine status, latency,
    and privacy budget consumption.
    """

    def __init__(self, db_path: str = "trustchain.db"):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        with _get_db(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hospital_metrics (
                    id INTEGER PRIMARY KEY,
                    hospital_id TEXT NOT NULL,
                    round_id INTEGER NOT NULL,
                    accuracy REAL,
                    loss REAL,
                    f1_score REAL,
                    contribution_score REAL,
                    byzantine_status TEXT DEFAULT 'CLEAN',
                    latency_ms REAL,
                    epsilon_spent REAL DEFAULT 0.0,
                    samples_used INTEGER DEFAULT 0,
                    aggregation_weight REAL DEFAULT 1.0,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_metrics (
                    id INTEGER PRIMARY KEY,
                    hospital_id TEXT NOT NULL,
                    specialty TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    tier TEXT NOT NULL,
                    is_correct INTEGER,
                    latency_ms REAL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hm_hospital
                ON hospital_metrics(hospital_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hm_round
                ON hospital_metrics(round_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pm_hospital
                ON prediction_metrics(hospital_id)
            """)

    # ── Recording Methods ──

    def record_round_metrics(
        self,
        hospital_id: str,
        round_id: int,
        accuracy: float = None,
        loss: float = None,
        f1_score: float = None,
        contribution_score: float = None,
        byzantine_status: str = "CLEAN",
        latency_ms: float = None,
        epsilon_spent: float = 0.0,
        samples_used: int = 0,
        aggregation_weight: float = 1.0,
        metadata: dict = None,
    ) -> int:
        """Records metrics for a hospital's participation in an FL round."""
        with _get_db(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO hospital_metrics (
                    hospital_id, round_id, accuracy, loss, f1_score,
                    contribution_score, byzantine_status, latency_ms,
                    epsilon_spent, samples_used, aggregation_weight,
                    metadata, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hospital_id, round_id, accuracy, loss, f1_score,
                    contribution_score, byzantine_status, latency_ms,
                    epsilon_spent, samples_used, aggregation_weight,
                    json.dumps(metadata) if metadata else None,
                    _utc_now(),
                ),
            )
            return cursor.lastrowid

    def record_prediction(
        self,
        hospital_id: str,
        specialty: str,
        confidence: float,
        tier: str,
        is_correct: bool = None,
        latency_ms: float = None,
    ) -> int:
        """Records a single prediction event for analytics."""
        with _get_db(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO prediction_metrics (
                    hospital_id, specialty, confidence, tier,
                    is_correct, latency_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hospital_id, specialty, confidence, tier,
                    int(is_correct) if is_correct is not None else None,
                    latency_ms, _utc_now(),
                ),
            )
            return cursor.lastrowid

    # ── Query Methods ──

    def get_hospital_summary(self, hospital_id: str) -> Dict:
        """Returns a full performance summary for a hospital."""
        with _get_db(self.db_path) as conn:
            # Round participation
            round_stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_rounds,
                    AVG(accuracy) as avg_accuracy,
                    AVG(loss) as avg_loss,
                    AVG(f1_score) as avg_f1,
                    AVG(contribution_score) as avg_contribution,
                    SUM(epsilon_spent) as total_epsilon,
                    SUM(samples_used) as total_samples,
                    AVG(latency_ms) as avg_latency,
                    SUM(CASE WHEN byzantine_status = 'CLEAN' THEN 1 ELSE 0 END) as clean_rounds,
                    SUM(CASE WHEN byzantine_status = 'MALICIOUS' THEN 1 ELSE 0 END) as malicious_rounds,
                    SUM(CASE WHEN byzantine_status = 'SUSPICIOUS' THEN 1 ELSE 0 END) as suspicious_rounds
                FROM hospital_metrics
                WHERE hospital_id = ?
                """,
                (hospital_id,),
            ).fetchone()

            # Prediction stats
            pred_stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total_predictions,
                    AVG(confidence) as avg_confidence,
                    AVG(latency_ms) as avg_pred_latency,
                    SUM(CASE WHEN tier = 'HIGH' THEN 1 ELSE 0 END) as high_tier,
                    SUM(CASE WHEN tier = 'MEDIUM' THEN 1 ELSE 0 END) as medium_tier,
                    SUM(CASE WHEN tier = 'LOW' THEN 1 ELSE 0 END) as low_tier,
                    SUM(CASE WHEN tier = 'ESCALATE' THEN 1 ELSE 0 END) as escalations
                FROM prediction_metrics
                WHERE hospital_id = ?
                """,
                (hospital_id,),
            ).fetchone()

            # Specialty breakdown
            specialties = conn.execute(
                """
                SELECT specialty, COUNT(*) as count, AVG(confidence) as avg_conf
                FROM prediction_metrics
                WHERE hospital_id = ?
                GROUP BY specialty
                """,
                (hospital_id,),
            ).fetchall()

            # Recent rounds (last 10)
            recent = conn.execute(
                """
                SELECT round_id, accuracy, loss, contribution_score,
                       byzantine_status, epsilon_spent, timestamp
                FROM hospital_metrics
                WHERE hospital_id = ?
                ORDER BY round_id DESC
                LIMIT 10
                """,
                (hospital_id,),
            ).fetchall()

        # Compute reputation score
        total_rounds = round_stats["total_rounds"] or 0
        clean = round_stats["clean_rounds"] or 0
        byzantine_pass_rate = (clean / total_rounds * 100) if total_rounds > 0 else 100.0

        reputation = min(100.0, (
            (round_stats["avg_contribution"] or 0) * 0.4 +
            byzantine_pass_rate * 0.3 +
            min((total_rounds / 50) * 100, 100) * 0.2 +
            min(((round_stats["total_samples"] or 0) / 50000) * 100, 100) * 0.1
        ))

        return {
            "hospital_id": hospital_id,
            "reputation_score": round(reputation, 1),
            "federation": {
                "total_rounds": total_rounds,
                "avg_accuracy": round(round_stats["avg_accuracy"] or 0, 4),
                "avg_loss": round(round_stats["avg_loss"] or 0, 4),
                "avg_f1": round(round_stats["avg_f1"] or 0, 4),
                "avg_contribution": round(round_stats["avg_contribution"] or 0, 2),
                "total_epsilon_spent": round(round_stats["total_epsilon"] or 0, 4),
                "total_samples": round_stats["total_samples"] or 0,
                "avg_latency_ms": round(round_stats["avg_latency"] or 0, 1),
                "byzantine_pass_rate": round(byzantine_pass_rate, 1),
                "clean_rounds": clean,
                "suspicious_rounds": round_stats["suspicious_rounds"] or 0,
                "malicious_rounds": round_stats["malicious_rounds"] or 0,
            },
            "predictions": {
                "total": pred_stats["total_predictions"] or 0,
                "avg_confidence": round(pred_stats["avg_confidence"] or 0, 2),
                "avg_latency_ms": round(pred_stats["avg_pred_latency"] or 0, 1),
                "tier_distribution": {
                    "HIGH": pred_stats["high_tier"] or 0,
                    "MEDIUM": pred_stats["medium_tier"] or 0,
                    "LOW": pred_stats["low_tier"] or 0,
                    "ESCALATE": pred_stats["escalations"] or 0,
                },
                "specialties": [
                    {"specialty": s["specialty"], "count": s["count"],
                     "avg_confidence": round(s["avg_conf"], 2)}
                    for s in specialties
                ],
            },
            "recent_rounds": [dict(r) for r in recent],
        }

    def get_federation_overview(self) -> Dict:
        """Returns aggregated federation-wide health metrics."""
        with _get_db(self.db_path) as conn:
            # Global stats
            global_stats = conn.execute("""
                SELECT
                    COUNT(DISTINCT hospital_id) as active_hospitals,
                    MAX(round_id) as latest_round,
                    AVG(accuracy) as global_avg_accuracy,
                    AVG(loss) as global_avg_loss,
                    SUM(epsilon_spent) as total_epsilon_spent,
                    SUM(samples_used) as total_samples,
                    SUM(CASE WHEN byzantine_status != 'CLEAN' THEN 1 ELSE 0 END) as total_anomalies
                FROM hospital_metrics
            """).fetchone()

            # Per-round convergence
            convergence = conn.execute("""
                SELECT round_id,
                       AVG(accuracy) as avg_accuracy,
                       AVG(loss) as avg_loss,
                       COUNT(DISTINCT hospital_id) as participants,
                       SUM(CASE WHEN byzantine_status = 'MALICIOUS' THEN 1 ELSE 0 END) as rejected
                FROM hospital_metrics
                GROUP BY round_id
                ORDER BY round_id
            """).fetchall()

            # Hospital leaderboard
            leaderboard = conn.execute("""
                SELECT hospital_id,
                       COUNT(*) as rounds,
                       AVG(accuracy) as avg_acc,
                       AVG(contribution_score) as avg_contrib,
                       SUM(epsilon_spent) as epsilon
                FROM hospital_metrics
                WHERE byzantine_status = 'CLEAN'
                GROUP BY hospital_id
                ORDER BY avg_contrib DESC
                LIMIT 20
            """).fetchall()

        return {
            "active_hospitals": global_stats["active_hospitals"] or 0,
            "latest_round": global_stats["latest_round"] or 0,
            "global_accuracy": round(global_stats["global_avg_accuracy"] or 0, 4),
            "global_loss": round(global_stats["global_avg_loss"] or 0, 4),
            "total_epsilon": round(global_stats["total_epsilon_spent"] or 0, 4),
            "total_samples": global_stats["total_samples"] or 0,
            "total_anomalies": global_stats["total_anomalies"] or 0,
            "convergence": [
                {
                    "round": r["round_id"],
                    "accuracy": round(r["avg_accuracy"] or 0, 4),
                    "loss": round(r["avg_loss"] or 0, 4),
                    "participants": r["participants"],
                    "rejected": r["rejected"],
                }
                for r in convergence
            ],
            "leaderboard": [
                {
                    "hospital_id": h["hospital_id"],
                    "rounds": h["rounds"],
                    "avg_accuracy": round(h["avg_acc"] or 0, 4),
                    "avg_contribution": round(h["avg_contrib"] or 0, 2),
                    "epsilon_spent": round(h["epsilon"] or 0, 4),
                }
                for h in leaderboard
            ],
        }

    def get_round_details(self, round_id: int) -> Dict:
        """Returns detailed metrics for a specific federation round."""
        with _get_db(self.db_path) as conn:
            participants = conn.execute(
                """
                SELECT hospital_id, accuracy, loss, contribution_score,
                       byzantine_status, latency_ms, epsilon_spent,
                       samples_used, aggregation_weight, timestamp
                FROM hospital_metrics
                WHERE round_id = ?
                ORDER BY contribution_score DESC
                """,
                (round_id,),
            ).fetchall()

        if not participants:
            return {"round_id": round_id, "found": False}

        accuracies = [p["accuracy"] for p in participants if p["accuracy"] is not None]
        return {
            "round_id": round_id,
            "found": True,
            "num_participants": len(participants),
            "avg_accuracy": round(sum(accuracies) / len(accuracies), 4) if accuracies else 0,
            "participants": [dict(p) for p in participants],
        }

    def get_privacy_budget_report(self) -> Dict:
        """Returns privacy budget consumption per hospital."""
        with _get_db(self.db_path) as conn:
            budgets = conn.execute("""
                SELECT hospital_id,
                       SUM(epsilon_spent) as total_epsilon,
                       COUNT(*) as rounds,
                       AVG(epsilon_spent) as avg_epsilon_per_round,
                       MAX(epsilon_spent) as max_epsilon_single_round
                FROM hospital_metrics
                GROUP BY hospital_id
                ORDER BY total_epsilon DESC
            """).fetchall()

        max_budget = 10.0  # configurable target epsilon
        return {
            "target_epsilon": max_budget,
            "hospitals": [
                {
                    "hospital_id": b["hospital_id"],
                    "total_epsilon": round(b["total_epsilon"] or 0, 4),
                    "rounds": b["rounds"],
                    "avg_per_round": round(b["avg_epsilon_per_round"] or 0, 4),
                    "max_single_round": round(b["max_epsilon_single_round"] or 0, 4),
                    "budget_remaining": round(max(0, max_budget - (b["total_epsilon"] or 0)), 4),
                    "utilization_pct": round(min(100, ((b["total_epsilon"] or 0) / max_budget) * 100), 1),
                }
                for b in budgets
            ],
        }

    def get_latency_percentiles(self, hospital_id: str = None) -> Dict:
        """Computes latency percentiles (p50, p90, p95, p99) for predictions."""
        with _get_db(self.db_path) as conn:
            where = "WHERE hospital_id = ?" if hospital_id else ""
            params = (hospital_id,) if hospital_id else ()
            rows = conn.execute(
                f"""
                SELECT latency_ms FROM prediction_metrics
                {where}
                ORDER BY latency_ms
                """,
                params,
            ).fetchall()

        latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        if not latencies:
            return {"count": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0}

        latencies.sort()
        n = len(latencies)

        def percentile(p):
            k = (n - 1) * p / 100.0
            f = int(k)
            c = f + 1 if f + 1 < n else f
            d = k - f
            return latencies[f] + d * (latencies[c] - latencies[f])

        return {
            "count": n,
            "p50": round(percentile(50), 2),
            "p90": round(percentile(90), 2),
            "p95": round(percentile(95), 2),
            "p99": round(percentile(99), 2),
            "mean": round(sum(latencies) / n, 2),
        }


if __name__ == "__main__":
    import os
    import tempfile

    print("Testing HospitalMetricsStore...")

    db_path = os.path.join(tempfile.gettempdir(), "test_metrics.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    store = HospitalMetricsStore(db_path=db_path)

    # Record round metrics for 3 hospitals across 5 rounds
    hospitals = ["HOSP-MUM-001", "HOSP-DEL-002", "HOSP-BLR-003"]
    for r in range(1, 6):
        for i, h in enumerate(hospitals):
            acc = 0.75 + r * 0.03 + i * 0.01
            store.record_round_metrics(
                hospital_id=h,
                round_id=r,
                accuracy=acc,
                loss=1.0 - acc + 0.05,
                f1_score=acc - 0.02,
                contribution_score=85 + r * 2 - i * 3,
                byzantine_status="CLEAN" if r != 3 or i != 2 else "SUSPICIOUS",
                latency_ms=150 + i * 30 + r * 5,
                epsilon_spent=0.5 + r * 0.1,
                samples_used=5000 + i * 1000,
            )

    # Record predictions
    for h in hospitals:
        for _ in range(20):
            store.record_prediction(
                hospital_id=h,
                specialty="Radiology",
                confidence=0.6 + 0.35 * (hash(h + str(_)) % 100) / 100.0,
                tier="HIGH" if _ % 3 == 0 else "MEDIUM",
                latency_ms=80 + (_ * 7) % 200,
            )

    # Test queries
    summary = store.get_hospital_summary("HOSP-MUM-001")
    print(f"  Hospital HOSP-MUM-001 reputation: {summary['reputation_score']}")
    print(f"  Rounds participated: {summary['federation']['total_rounds']}")
    print(f"  Avg accuracy: {summary['federation']['avg_accuracy']}")
    print(f"  Byzantine pass rate: {summary['federation']['byzantine_pass_rate']}%")

    overview = store.get_federation_overview()
    print(f"\n  Federation: {overview['active_hospitals']} hospitals, "
          f"{overview['latest_round']} rounds, "
          f"accuracy {overview['global_accuracy']}")

    privacy = store.get_privacy_budget_report()
    print(f"\n  Privacy budget report: {len(privacy['hospitals'])} hospitals tracked")
    for h in privacy["hospitals"]:
        print(f"    {h['hospital_id']}: ε={h['total_epsilon']}, "
              f"{h['utilization_pct']}% used")

    latency = store.get_latency_percentiles()
    print(f"\n  Latency p50={latency['p50']}ms, p95={latency['p95']}ms, p99={latency['p99']}ms")

    round_detail = store.get_round_details(3)
    print(f"\n  Round 3: {round_detail['num_participants']} participants, "
          f"avg accuracy {round_detail['avg_accuracy']}")

    # Cleanup
    os.remove(db_path)
    print("\nMetrics store test completed successfully!")
