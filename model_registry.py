"""
TrustChain-MedAI: Model Versioning & Registry.

Production-grade model registry with versioning, rollback, promotion pipeline,
and metrics tracking. Stores versioned .pt files and tracks lineage through
federated learning rounds.
"""

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class ModelVersion:
    """Represents a single model version in the registry."""

    __slots__ = (
        "id", "version", "round_id", "weight_hash", "accuracy", "f1_score",
        "auroc", "ece", "aggregation_method", "status", "file_path",
        "created_at", "deployed_at", "metadata",
    )

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> Dict:
        return {s: getattr(self, s, None) for s in self.__slots__}


class ModelRegistry:
    """
    SQLite-backed model registry for TrustChain-MedAI.

    Lifecycle: TESTING → DEPLOYED → ARCHIVED
    Rollback:  Any ARCHIVED version can be re-promoted to DEPLOYED.
    Invariant: Exactly one version is DEPLOYED at any time (or zero if none promoted).
    """

    STATUSES = ("TESTING", "DEPLOYED", "ARCHIVED", "ROLLED_BACK")

    def __init__(self, db_path: str = "trustchain.db", registry_dir: str = "models/registry"):
        self.db_path = db_path
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self._init_table()

    def _init_table(self):
        with _get_db(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY,
                    version TEXT UNIQUE NOT NULL,
                    round_id INTEGER,
                    weight_hash TEXT NOT NULL,
                    accuracy REAL,
                    f1_score REAL,
                    auroc REAL,
                    ece REAL,
                    aggregation_method TEXT,
                    status TEXT NOT NULL DEFAULT 'TESTING',
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    deployed_at TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_transitions (
                    id INTEGER PRIMARY KEY,
                    version TEXT NOT NULL,
                    from_status TEXT NOT NULL,
                    to_status TEXT NOT NULL,
                    actor_id TEXT,
                    reason TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

    # ── Core Operations ──

    def register_model(
        self,
        model: torch.nn.Module,
        version: str,
        round_id: int = None,
        accuracy: float = None,
        f1_score: float = None,
        auroc: float = None,
        ece: float = None,
        aggregation_method: str = None,
        metadata: dict = None,
    ) -> Dict:
        """
        Registers a new model version in TESTING status.
        Saves the model weights to disk and computes a SHA-256 hash.
        """
        # Save model weights
        file_path = os.path.join(self.registry_dir, f"{version}.pt")
        torch.save(model.state_dict(), file_path)

        # Compute weight hash
        weight_hash = self._compute_file_hash(file_path)

        with _get_db(self.db_path) as conn:
            # Check for duplicate version
            existing = conn.execute(
                "SELECT id FROM model_versions WHERE version = ?", (version,)
            ).fetchone()
            if existing:
                raise ValueError(f"Version '{version}' already exists in registry")

            conn.execute(
                """
                INSERT INTO model_versions (
                    version, round_id, weight_hash, accuracy, f1_score,
                    auroc, ece, aggregation_method, status, file_path,
                    created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'TESTING', ?, ?, ?)
                """,
                (
                    version, round_id, weight_hash, accuracy, f1_score,
                    auroc, ece, aggregation_method, file_path,
                    _utc_now(), json.dumps(metadata) if metadata else None,
                ),
            )
            self._log_transition(conn, version, "NEW", "TESTING")

        return {
            "version": version,
            "status": "TESTING",
            "weight_hash": weight_hash,
            "file_path": file_path,
        }

    def promote(self, version: str, actor_id: str = "system") -> Dict:
        """
        Promotes a TESTING version to DEPLOYED.
        Archives the currently deployed version (if any).
        """
        with _get_db(self.db_path) as conn:
            # Validate the target version exists and is TESTING
            target = conn.execute(
                "SELECT * FROM model_versions WHERE version = ?", (version,)
            ).fetchone()
            if not target:
                raise ValueError(f"Version '{version}' not found")
            if target["status"] != "TESTING":
                raise ValueError(
                    f"Cannot promote version with status '{target['status']}'. "
                    f"Only TESTING versions can be promoted."
                )

            # Archive current deployed version
            current = conn.execute(
                "SELECT version FROM model_versions WHERE status = 'DEPLOYED'"
            ).fetchone()
            if current:
                conn.execute(
                    "UPDATE model_versions SET status = 'ARCHIVED' WHERE version = ?",
                    (current["version"],),
                )
                self._log_transition(
                    conn, current["version"], "DEPLOYED", "ARCHIVED",
                    actor_id=actor_id, reason=f"Replaced by {version}",
                )

            # Deploy target version
            conn.execute(
                "UPDATE model_versions SET status = 'DEPLOYED', deployed_at = ? WHERE version = ?",
                (_utc_now(), version),
            )
            self._log_transition(
                conn, version, "TESTING", "DEPLOYED", actor_id=actor_id,
            )

        return {"version": version, "status": "DEPLOYED", "previous": current["version"] if current else None}

    def rollback(self, version: str, actor_id: str = "system", reason: str = None) -> Dict:
        """
        Rolls back to a previously ARCHIVED version.
        The current DEPLOYED version gets status ROLLED_BACK.
        """
        with _get_db(self.db_path) as conn:
            # Validate target
            target = conn.execute(
                "SELECT * FROM model_versions WHERE version = ?", (version,)
            ).fetchone()
            if not target:
                raise ValueError(f"Version '{version}' not found")
            if target["status"] not in ("ARCHIVED", "ROLLED_BACK"):
                raise ValueError(
                    f"Can only rollback to ARCHIVED versions. "
                    f"'{version}' has status '{target['status']}'."
                )

            # Verify the weight file still exists
            if not os.path.exists(target["file_path"]):
                raise FileNotFoundError(
                    f"Weight file for version '{version}' not found at {target['file_path']}"
                )

            # Mark current deployed as ROLLED_BACK
            current = conn.execute(
                "SELECT version FROM model_versions WHERE status = 'DEPLOYED'"
            ).fetchone()
            if current:
                conn.execute(
                    "UPDATE model_versions SET status = 'ROLLED_BACK' WHERE version = ?",
                    (current["version"],),
                )
                self._log_transition(
                    conn, current["version"], "DEPLOYED", "ROLLED_BACK",
                    actor_id=actor_id, reason=reason or f"Rolled back to {version}",
                )

            # Re-deploy target
            conn.execute(
                "UPDATE model_versions SET status = 'DEPLOYED', deployed_at = ? WHERE version = ?",
                (_utc_now(), version),
            )
            self._log_transition(
                conn, version, target["status"], "DEPLOYED",
                actor_id=actor_id, reason=reason or "Rollback",
            )

        return {
            "version": version,
            "status": "DEPLOYED",
            "rolled_back_from": current["version"] if current else None,
        }

    def load_deployed_model(self, model: torch.nn.Module) -> Optional[str]:
        """
        Loads the currently DEPLOYED model weights into the given model.
        Returns the version string, or None if no model is deployed.
        """
        with _get_db(self.db_path) as conn:
            deployed = conn.execute(
                "SELECT version, file_path, weight_hash FROM model_versions WHERE status = 'DEPLOYED'"
            ).fetchone()

        if not deployed:
            return None

        file_path = deployed["file_path"]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Weight file missing: {file_path}")

        # Verify integrity
        actual_hash = self._compute_file_hash(file_path)
        if actual_hash != deployed["weight_hash"]:
            raise RuntimeError(
                f"Weight file integrity check failed for {deployed['version']}! "
                f"Expected hash {deployed['weight_hash'][:16]}..., "
                f"got {actual_hash[:16]}..."
            )

        model.load_state_dict(torch.load(file_path, map_location="cpu"))
        return deployed["version"]

    # ── Query Operations ──

    def list_versions(self, status: str = None, limit: int = 50) -> List[Dict]:
        """Lists model versions, optionally filtered by status."""
        with _get_db(self.db_path) as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM model_versions WHERE status = ? ORDER BY id DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM model_versions ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    def get_current(self) -> Optional[Dict]:
        """Returns the currently DEPLOYED version, or None."""
        with _get_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE status = 'DEPLOYED'"
            ).fetchone()
        return dict(row) if row else None

    def get_version(self, version: str) -> Optional[Dict]:
        """Returns details for a specific version."""
        with _get_db(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM model_versions WHERE version = ?", (version,)
            ).fetchone()
        return dict(row) if row else None

    def compare_versions(self, v1: str, v2: str) -> Dict:
        """Compares metrics between two versions."""
        ver1 = self.get_version(v1)
        ver2 = self.get_version(v2)
        if not ver1 or not ver2:
            raise ValueError(f"One or both versions not found: {v1}, {v2}")

        metrics = ["accuracy", "f1_score", "auroc", "ece"]
        diff = {}
        for m in metrics:
            val1 = ver1.get(m)
            val2 = ver2.get(m)
            if val1 is not None and val2 is not None:
                diff[m] = {
                    v1: round(val1, 4),
                    v2: round(val2, 4),
                    "delta": round(val2 - val1, 4),
                    "improved": (val2 > val1) if m != "ece" else (val2 < val1),
                }

        return {
            "v1": v1, "v2": v2,
            "v1_status": ver1["status"],
            "v2_status": ver2["status"],
            "metrics_comparison": diff,
        }

    def get_transition_history(self, version: str = None, limit: int = 50) -> List[Dict]:
        """Returns model transition history."""
        with _get_db(self.db_path) as conn:
            if version:
                rows = conn.execute(
                    "SELECT * FROM model_transitions WHERE version = ? ORDER BY id DESC LIMIT ?",
                    (version, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM model_transitions ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Internal Helpers ──

    def _compute_file_hash(self, file_path: str) -> str:
        """Computes SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _log_transition(
        self, conn, version: str, from_status: str, to_status: str,
        actor_id: str = "system", reason: str = None,
    ):
        conn.execute(
            """
            INSERT INTO model_transitions (version, from_status, to_status, actor_id, reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (version, from_status, to_status, actor_id, reason, _utc_now()),
        )


if __name__ == "__main__":
    import tempfile

    print("Testing ModelRegistry...")

    db_path = os.path.join(tempfile.gettempdir(), "test_registry.db")
    reg_dir = os.path.join(tempfile.gettempdir(), "test_model_registry")
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(reg_dir):
        shutil.rmtree(reg_dir)

    registry = ModelRegistry(db_path=db_path, registry_dir=reg_dir)

    # Create a simple test model
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    # Register 3 versions with improving metrics
    for i, (ver, acc, f1, auroc, ece) in enumerate([
        ("v0.1.0", 0.821, 0.752, 0.840, 0.051),
        ("v0.2.0", 0.891, 0.824, 0.892, 0.024),
        ("v0.3.0", 0.912, 0.850, 0.940, 0.015),
    ]):
        model = TestModel()
        result = registry.register_model(
            model=model, version=ver, round_id=i + 1,
            accuracy=acc, f1_score=f1, auroc=auroc, ece=ece,
            aggregation_method="krum",
        )
        print(f"  Registered {ver}: hash={result['weight_hash'][:16]}...")

    # Test promotion pipeline
    print("\n  Promoting v0.1.0 to DEPLOYED...")
    registry.promote("v0.1.0", actor_id="admin")
    current = registry.get_current()
    assert current["version"] == "v0.1.0", "Expected v0.1.0 to be deployed"
    print(f"  Current deployed: {current['version']}")

    print("  Promoting v0.2.0 (should archive v0.1.0)...")
    result = registry.promote("v0.2.0", actor_id="admin")
    assert result["previous"] == "v0.1.0", "Expected v0.1.0 to be archived"
    current = registry.get_current()
    assert current["version"] == "v0.2.0"
    print(f"  Current deployed: {current['version']}, archived: {result['previous']}")

    # Test rollback
    print("\n  Rolling back to v0.1.0...")
    rb = registry.rollback("v0.1.0", actor_id="admin", reason="Regression detected")
    assert rb["version"] == "v0.1.0"
    assert rb["rolled_back_from"] == "v0.2.0"
    current = registry.get_current()
    assert current["version"] == "v0.1.0"
    print(f"  Current deployed: {current['version']}, rolled back from: {rb['rolled_back_from']}")

    # Test version comparison
    print("\n  Comparing v0.1.0 vs v0.3.0...")
    comp = registry.compare_versions("v0.1.0", "v0.3.0")
    for metric, data in comp["metrics_comparison"].items():
        print(f"    {metric}: {data['v0.1.0']} → {data['v0.3.0']} "
              f"(Δ={data['delta']:+.4f}, {'✓' if data['improved'] else '✗'})")

    # Test model loading with integrity check
    print("\n  Loading deployed model with integrity verification...")
    load_model = TestModel()
    loaded_ver = registry.load_deployed_model(load_model)
    assert loaded_ver == "v0.1.0"
    print(f"  Loaded version: {loaded_ver} — integrity verified ✓")

    # Test transition history
    history = registry.get_transition_history()
    print(f"\n  Transition history: {len(history)} events")
    for h in history[:5]:
        print(f"    {h['version']}: {h['from_status']} → {h['to_status']} ({h['actor_id']})")

    # List all versions
    versions = registry.list_versions()
    print(f"\n  Registry contains {len(versions)} versions:")
    for v in versions:
        print(f"    {v['version']}: status={v['status']}, accuracy={v['accuracy']}")

    # Cleanup
    os.remove(db_path)
    shutil.rmtree(reg_dir)
    print("\nModel registry test completed successfully!")
