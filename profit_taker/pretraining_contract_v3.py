from __future__ import annotations

import hashlib
import sqlite3
from typing import Any

from . import pretraining_contract_v2 as _base

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


def _dedupe_economic_collapse(db: str) -> int:
    with sqlite3.connect(db) as conn:
        migrate(conn)
        conn.execute(
            f"""DELETE FROM {TARGET_TABLE}
                WHERE target_kind='economic_collapse'
                  AND rowid NOT IN (
                    SELECT MAX(rowid) FROM {TARGET_TABLE}
                    WHERE target_kind='economic_collapse'
                    GROUP BY token_key,decision_at,target_kind,horizon_minutes
                  )"""
        )
        conn.commit()
        return int(conn.execute(
            f"SELECT COUNT(*) FROM {TARGET_TABLE} WHERE target_kind='economic_collapse'"
        ).fetchone()[0])


def refresh_pretraining_targets(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    out = _base.refresh_pretraining_targets(db, cfg)
    out["economic_collapse_rows"] = _dedupe_economic_collapse(db)
    return out


def training_readiness(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    out = _base.training_readiness(db, cfg)
    # V1 helpers called inside the retained readiness implementation also refresh
    # targets. Compact after the complete operation, not only before it.
    _dedupe_economic_collapse(db)
    return out


def assert_training_ready(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    report = training_readiness(db, cfg)
    if not report["ready"]:
        failed = [
            k for k, v in report["gates"].items()
            if not bool(v.get("pass")) and k != "operational_death_tokens"
        ]
        raise RuntimeError(
            "V24 production bootstrap refused by pretraining readiness gates: " + ", ".join(failed)
        )
    return report


def evaluate_baselines(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    out = _base.evaluate_baselines(db, cfg)
    _dedupe_economic_collapse(db)
    return out


def enrich_counterfactual_friction(conn: sqlite3.Connection, cfg: PretrainingConfig | None = None) -> dict[str, int]:
    cfg = cfg or PretrainingConfig()
    table = "axiom_v24_counterfactual_policy_targets"
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
        return {"updated": 0}
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if "pre_friction_source_fingerprint" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN pre_friction_source_fingerprint TEXT")
        conn.execute(
            f"UPDATE {table} SET pre_friction_source_fingerprint=source_fingerprint "
            "WHERE pre_friction_source_fingerprint IS NULL"
        )
        conn.commit()
    else:
        conn.execute(
            f"UPDATE {table} SET pre_friction_source_fingerprint=source_fingerprint "
            "WHERE pre_friction_source_fingerprint IS NULL"
        )
        conn.commit()

    out = _base.enrich_counterfactual_friction(conn, cfg)
    rows = conn.execute(
        f"""SELECT rowid,COALESCE(pre_friction_source_fingerprint,''),
                   COALESCE(friction_definition_hash,'') FROM {table}"""
    ).fetchall()
    for rowid, raw_fp, friction_hash in rows:
        stable = hashlib.sha256(
            f"{raw_fp}|friction:{friction_hash}".encode("utf-8")
        ).hexdigest()
        conn.execute(f"UPDATE {table} SET source_fingerprint=? WHERE rowid=?", (stable, rowid))
    conn.commit()
    out["stable_friction_provenance_rows"] = len(rows)
    return out
