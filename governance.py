"""
TrustChain-MedAI: HIPAA/GDPR/DPDP Governance Engine.

Implements:
  - AuditTrailLogger: Immutable append-only audit chain with SHA-256 linking
    and verify_chain() for integrity verification on read.
  - ConsentManager: Per-patient consent records with expiry and revocation.
  - DataMinimizer: PHI stripping and data retention policies.
  - RBACPolicy: Role-based access control policy checker.
"""

import copy
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Audit Trail Logger
# ─────────────────────────────────────────────────────────────────────────────

class AuditTrailLogger:
    """
    Immutable append-only audit log with tamper-evident SHA-256 chain linking.

    Each entry's chain_hash = SHA-256(prev_hash || timestamp || actor_id ||
    action || resource || patient_id || details_json).

    verify_chain() walks the chain from oldest to newest, recomputing each
    chain_hash to detect any tampering.
    """

    GENESIS_HASH = "GENESIS_0" * 8  # 64 chars

    def __init__(self, db_path: str = "trustchain.db"):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        with _get_db(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY,
                    entry_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    patient_id TEXT,
                    ip_address TEXT,
                    details TEXT,
                    chain_hash TEXT NOT NULL,
                    prev_hash TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_actor
                ON audit_log(actor_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_patient
                ON audit_log(patient_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_action
                ON audit_log(action)
            """)

    def _compute_chain_hash(
        self, prev_hash: str, timestamp: str, actor_id: str,
        action: str, resource: str, patient_id: str, details: str,
    ) -> str:
        """Compute SHA-256 chain hash for tamper evidence."""
        payload = f"{prev_hash}|{timestamp}|{actor_id}|{action}|{resource}|{patient_id or ''}|{details or ''}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def log(
        self,
        actor_id: str,
        action: str,
        resource: str,
        patient_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> str:
        """
        Append an audit entry to the immutable chain.

        Args:
            actor_id: Who performed the action (user/system ID).
            action: What was done (e.g., 'PREDICT', 'ACCESS_RECORD', 'MODEL_UPDATE').
            resource: What resource was accessed (endpoint, model, record).
            patient_id: Optional patient identifier.
            ip_address: Optional requester IP.
            details: Optional dict with extra context.

        Returns:
            entry_id of the new audit entry.
        """
        entry_id = str(uuid.uuid4())[:12]
        timestamp = _utc_now()
        details_json = json.dumps(details) if details else None

        with _get_db(self.db_path) as conn:
            # Get previous hash
            last = conn.execute(
                "SELECT chain_hash FROM audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev_hash = last["chain_hash"] if last else self.GENESIS_HASH

            # Compute chain hash
            chain_hash = self._compute_chain_hash(
                prev_hash, timestamp, actor_id, action, resource, patient_id, details_json,
            )

            conn.execute(
                """
                INSERT INTO audit_log (
                    entry_id, timestamp, actor_id, action, resource,
                    patient_id, ip_address, details, chain_hash, prev_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id, timestamp, actor_id, action, resource,
                    patient_id, ip_address, details_json, chain_hash, prev_hash,
                ),
            )

        return entry_id

    def verify_chain(self, limit: int = 1000) -> Dict:
        """
        Walk the audit chain and verify integrity by recomputing hashes.

        Returns:
            {valid: bool, entries_checked: int, first_invalid_id: Optional[int], message: str}
        """
        with _get_db(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id ASC LIMIT ?", (limit,)
            ).fetchall()

        if not rows:
            return {
                "valid": True,
                "entries_checked": 0,
                "first_invalid_id": None,
                "message": "Audit chain is empty.",
            }

        expected_prev = self.GENESIS_HASH

        for i, row in enumerate(rows):
            # Verify prev_hash linkage
            if row["prev_hash"] != expected_prev:
                return {
                    "valid": False,
                    "entries_checked": i + 1,
                    "first_invalid_id": row["id"],
                    "message": f"Chain broken at entry {row['id']}: "
                               f"prev_hash mismatch (expected {expected_prev[:16]}..., "
                               f"got {row['prev_hash'][:16]}...)",
                }

            # Recompute chain hash
            recomputed = self._compute_chain_hash(
                row["prev_hash"], row["timestamp"], row["actor_id"],
                row["action"], row["resource"], row["patient_id"], row["details"],
            )

            if recomputed != row["chain_hash"]:
                return {
                    "valid": False,
                    "entries_checked": i + 1,
                    "first_invalid_id": row["id"],
                    "message": f"Tampered entry detected at id={row['id']}: "
                               f"chain_hash mismatch (recomputed {recomputed[:16]}..., "
                               f"stored {row['chain_hash'][:16]}...)",
                }

            expected_prev = row["chain_hash"]

        return {
            "valid": True,
            "entries_checked": len(rows),
            "first_invalid_id": None,
            "message": f"Audit chain verified — {len(rows)} entries intact.",
        }

    def query(
        self,
        actor_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Query audit entries with optional filters."""
        conditions = []
        params = []

        if actor_id:
            conditions.append("actor_id = ?")
            params.append(actor_id)
        if patient_id:
            conditions.append("patient_id = ?")
            params.append(patient_id)
        if action:
            conditions.append("action = ?")
            params.append(action)
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        with _get_db(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM audit_log WHERE {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()

        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Consent Manager
# ─────────────────────────────────────────────────────────────────────────────

class ConsentManager:
    """
    Manages per-patient consent records with expiry and revocation.

    Consent types: 'treatment', 'research', 'federation', 'data_sharing'.
    """

    VALID_TYPES = ("treatment", "research", "federation", "data_sharing")

    def __init__(self, db_path: str = "trustchain.db"):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        with _get_db(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consent_records (
                    id INTEGER PRIMARY KEY,
                    consent_id TEXT UNIQUE NOT NULL,
                    patient_id TEXT NOT NULL,
                    hospital_id TEXT NOT NULL,
                    consent_type TEXT NOT NULL,
                    granted_at TEXT NOT NULL,
                    expires_at TEXT,
                    revoked_at TEXT,
                    status TEXT NOT NULL DEFAULT 'ACTIVE'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_consent_patient
                ON consent_records(patient_id)
            """)

    def grant_consent(
        self,
        patient_id: str,
        hospital_id: str,
        consent_type: str,
        expires_in_days: int = 365,
    ) -> str:
        """
        Grant consent for a specific type of data use.

        Returns consent_id.
        """
        if consent_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid consent type: '{consent_type}'. "
                f"Valid: {self.VALID_TYPES}"
            )

        consent_id = f"CST-{uuid.uuid4().hex[:8].upper()}"
        granted_at = _utc_now()
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        ).isoformat()

        with _get_db(self.db_path) as conn:
            # Revoke any existing active consent of the same type
            conn.execute(
                """
                UPDATE consent_records
                SET status = 'SUPERSEDED', revoked_at = ?
                WHERE patient_id = ? AND consent_type = ? AND status = 'ACTIVE'
                """,
                (granted_at, patient_id, consent_type),
            )
            conn.execute(
                """
                INSERT INTO consent_records (
                    consent_id, patient_id, hospital_id, consent_type,
                    granted_at, expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE')
                """,
                (consent_id, patient_id, hospital_id, consent_type, granted_at, expires_at),
            )

        return consent_id

    def revoke_consent(self, patient_id: str, consent_type: str) -> bool:
        """
        Revoke active consent for a patient.

        Returns True if a consent was revoked.
        """
        with _get_db(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE consent_records
                SET status = 'REVOKED', revoked_at = ?
                WHERE patient_id = ? AND consent_type = ? AND status = 'ACTIVE'
                """,
                (_utc_now(), patient_id, consent_type),
            )
            return cursor.rowcount > 0

    def check_consent(self, patient_id: str, consent_type: str) -> bool:
        """
        Check if a patient has active, non-expired consent.

        Returns True if valid consent exists.
        """
        now = _utc_now()
        with _get_db(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id FROM consent_records
                WHERE patient_id = ? AND consent_type = ? AND status = 'ACTIVE'
                  AND (expires_at IS NULL OR expires_at > ?)
                LIMIT 1
                """,
                (patient_id, consent_type, now),
            ).fetchone()
        return row is not None

    def get_patient_consents(self, patient_id: str) -> List[Dict]:
        """Get all consent records for a patient."""
        with _get_db(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM consent_records
                WHERE patient_id = ?
                ORDER BY id DESC
                """,
                (patient_id,),
            ).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Data Minimizer
# ─────────────────────────────────────────────────────────────────────────────

class DataMinimizer:
    """
    Strips Protected Health Information (PHI) from data payloads
    and enforces retention policies.
    """

    PHI_FIELDS = frozenset([
        "patient_name", "name", "ssn", "social_security",
        "dob", "date_of_birth", "birth_date",
        "address", "street", "city", "zip", "zip_code", "postal_code",
        "phone", "phone_number", "mobile", "telephone",
        "email", "email_address",
        "mrn", "medical_record_number",
        "insurance_id", "policy_number",
    ])

    REDACTED = "[REDACTED]"

    def strip_phi(self, data: dict) -> dict:
        """
        Deep-copies data and redacts all PHI fields to '[REDACTED]'.

        Recursively processes nested dicts and lists.
        """
        return self._redact_recursive(copy.deepcopy(data))

    def _redact_recursive(self, obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key.lower() in self.PHI_FIELDS:
                    result[key] = self.REDACTED
                else:
                    result[key] = self._redact_recursive(value)
            return result
        elif isinstance(obj, list):
            return [self._redact_recursive(item) for item in obj]
        else:
            return obj

    def apply_retention_policy(self, db_path: str, max_age_days: int = 365):
        """
        Mark prediction records older than max_age_days for purging.

        Creates/updates a data_retention tracking table.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=max_age_days)
        ).isoformat()

        with _get_db(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_retention (
                    id INTEGER PRIMARY KEY,
                    record_type TEXT NOT NULL,
                    record_id INTEGER NOT NULL,
                    patient_id TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    purged INTEGER DEFAULT 0,
                    purged_at TEXT
                )
            """)

            # Find expired prediction records
            expired = conn.execute(
                """
                SELECT id, patient_id, created_at
                FROM predictions
                WHERE created_at < ?
                """,
                (cutoff,),
            ).fetchall()

            count = 0
            for row in expired:
                existing = conn.execute(
                    "SELECT id FROM data_retention WHERE record_type = 'prediction' AND record_id = ?",
                    (row["id"],),
                ).fetchone()
                if not existing:
                    conn.execute(
                        """
                        INSERT INTO data_retention (record_type, record_id, patient_id, created_at, expires_at)
                        VALUES ('prediction', ?, ?, ?, ?)
                        """,
                        (row["id"], row["patient_id"], row["created_at"], cutoff),
                    )
                    count += 1

            return {"expired_records": count, "cutoff_date": cutoff}


# ─────────────────────────────────────────────────────────────────────────────
# RBAC Policy
# ─────────────────────────────────────────────────────────────────────────────

class RBACPolicy:
    """
    Role-Based Access Control policy checker.

    Roles: DOCTOR, ADMIN, AUDITOR, SYSTEM
    """

    PERMISSIONS = {
        "DOCTOR": [
            "/predict", "/gradcam", "/explain", "/feedback",
            "/history", "/models/current",
        ],
        "ADMIN": [
            "/predict", "/gradcam", "/explain", "/feedback", "/history",
            "/federation", "/models", "/metrics", "/governance",
            "/auth/register", "/tasks",
        ],
        "AUDITOR": [
            "/blockchain", "/audit", "/metrics", "/models/registry",
            "/models/compare", "/governance/audit",
        ],
        "SYSTEM": ["*"],
    }

    def check_access(self, role: str, endpoint: str) -> bool:
        """
        Check if a role is allowed to access an endpoint.

        Matches by prefix: '/models' allows '/models/registry', '/models/current', etc.
        SYSTEM role has wildcard access.
        """
        allowed = self.PERMISSIONS.get(role, [])
        if "*" in allowed:
            return True

        endpoint_clean = endpoint.rstrip("/").lower()
        for perm in allowed:
            perm_clean = perm.rstrip("/").lower()
            if endpoint_clean.startswith(perm_clean):
                return True

        return False

    def get_role_permissions(self, role: str) -> List[str]:
        """Returns the list of allowed endpoint prefixes for a role."""
        return self.PERMISSIONS.get(role, [])


# ─────────────────────────────────────────────────────────────────────────────
# Self-Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import tempfile

    print("=" * 60)
    print("  HIPAA Governance Engine — Self-Test")
    print("=" * 60)

    db_path = os.path.join(tempfile.gettempdir(), "test_governance.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    # ── Audit Trail ──
    print("\n  [1] Audit Trail Logger:")
    audit = AuditTrailLogger(db_path=db_path)

    # Log 10 entries
    for i in range(10):
        audit.log(
            actor_id=f"DR-{i % 3 + 1:03d}",
            action="PREDICT" if i % 2 == 0 else "ACCESS_RECORD",
            resource=f"/predict" if i % 2 == 0 else f"/history/PAT-{i:03d}",
            patient_id=f"PAT-{i:03d}",
            ip_address="10.0.1.50",
            details={"model": "v0.6.2", "specialty": "Radiology"},
        )
    print("      Logged 10 audit entries")

    # Verify chain integrity
    result = audit.verify_chain()
    print(f"      Chain valid: {result['valid']}")
    print(f"      Entries checked: {result['entries_checked']}")
    print(f"      Message: {result['message']}")
    assert result["valid"], "Audit chain should be valid"

    # Tamper with an entry
    with _get_db(db_path) as conn:
        conn.execute("UPDATE audit_log SET action = 'TAMPERED' WHERE id = 5")
    tamper_result = audit.verify_chain()
    print(f"      After tamper — valid: {tamper_result['valid']}")
    print(f"      First invalid at: id={tamper_result['first_invalid_id']}")
    assert not tamper_result["valid"], "Tampered chain should be invalid"
    assert tamper_result["first_invalid_id"] == 5, "Should detect tamper at entry 5"

    # Restore for next tests
    os.remove(db_path)
    audit2 = AuditTrailLogger(db_path=db_path)

    # Query test
    audit2.log("DR-001", "PREDICT", "/predict", patient_id="PAT-100")
    audit2.log("ADMIN-001", "MODEL_UPDATE", "/models", patient_id=None)
    results = audit2.query(actor_id="DR-001")
    print(f"      Query by actor: {len(results)} results")

    # ── Consent Manager ──
    print("\n  [2] Consent Manager:")
    consent = ConsentManager(db_path=db_path)

    cid = consent.grant_consent("PAT-001", "HOSP-MUM", "treatment")
    print(f"      Granted consent: {cid}")

    has_consent = consent.check_consent("PAT-001", "treatment")
    print(f"      Has treatment consent: {has_consent}")
    assert has_consent

    no_consent = consent.check_consent("PAT-001", "research")
    print(f"      Has research consent: {no_consent}")
    assert not no_consent

    consent.grant_consent("PAT-001", "HOSP-MUM", "research", expires_in_days=30)
    consent.revoke_consent("PAT-001", "research")
    revoked = consent.check_consent("PAT-001", "research")
    print(f"      After revocation: {revoked}")
    assert not revoked

    all_consents = consent.get_patient_consents("PAT-001")
    print(f"      Total consent records: {len(all_consents)}")

    # ── Data Minimizer ──
    print("\n  [3] Data Minimizer:")
    minimizer = DataMinimizer()

    test_data = {
        "prediction": "Pneumonia",
        "confidence": 0.92,
        "patient_name": "John Doe",
        "ssn": "123-45-6789",
        "email": "john@hospital.com",
        "nested": {
            "phone": "+1-555-0100",
            "address": "123 Medical Ave",
            "clinical_note": "Patient presents with cough",
        },
    }

    stripped = minimizer.strip_phi(test_data)
    print(f"      Original patient_name: {test_data['patient_name']}")
    print(f"      Stripped patient_name: {stripped['patient_name']}")
    print(f"      Stripped SSN: {stripped['ssn']}")
    print(f"      Stripped nested phone: {stripped['nested']['phone']}")
    print(f"      Preserved clinical_note: {stripped['nested']['clinical_note'][:30]}...")
    assert stripped["patient_name"] == "[REDACTED]"
    assert stripped["ssn"] == "[REDACTED]"
    assert stripped["nested"]["phone"] == "[REDACTED]"
    assert stripped["nested"]["clinical_note"] == test_data["nested"]["clinical_note"]

    # ── RBAC ──
    print("\n  [4] RBAC Policy:")
    rbac = RBACPolicy()

    assert rbac.check_access("DOCTOR", "/predict")
    assert rbac.check_access("DOCTOR", "/history/PAT-001")
    assert not rbac.check_access("DOCTOR", "/federation/start")
    assert rbac.check_access("ADMIN", "/federation/start")
    assert rbac.check_access("AUDITOR", "/blockchain/proofs")
    assert not rbac.check_access("AUDITOR", "/predict")
    assert rbac.check_access("SYSTEM", "/anything")
    print("      DOCTOR → /predict: ✓")
    print("      DOCTOR → /federation: ✗")
    print("      ADMIN → /federation: ✓")
    print("      AUDITOR → /blockchain: ✓")
    print("      AUDITOR → /predict: ✗")
    print("      SYSTEM → wildcard: ✓")

    # Cleanup
    os.remove(db_path)

    print("\n" + "=" * 60)
    print("  Governance tests completed successfully!")
    print("=" * 60)
