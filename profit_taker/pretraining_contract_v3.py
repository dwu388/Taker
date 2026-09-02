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
    """Materialize friction exactly once from immutable gross economics.

    The retained policy learner consumes compatibility aliases such as
    ``entry_execution_return``.  Those aliases become net values after enrichment,
    so a repeated refresh must never use them as the next gross input.  Explicit
    ``*_gross`` columns are therefore the canonical economic source after the first
    pass, while ``pre_friction_source_fingerprint`` permanently records provenance
    before friction was attached.
    """
    cfg = cfg or PretrainingConfig()
    table = "axiom_v24_counterfactual_policy_targets"
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
        return {"updated": 0, "policy_aliases_updated_to_net": 0, "stable_friction_provenance_rows": 0}

    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    additions = {
        "entry_return_gross": "REAL", "entry_return_net": "REAL",
        "exit_now_return_gross": "REAL", "exit_now_return_net": "REAL",
        "hold_terminal_return_gross": "REAL", "hold_terminal_return_net": "REAL",
        "hold_advantage_gross": "REAL", "hold_advantage_net": "REAL",
        "friction_bps": "REAL", "friction_definition_hash": "TEXT",
        "pre_friction_source_fingerprint": "TEXT",
    }
    for name, typ in additions.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")
    conn.execute(
        f"UPDATE {table} SET pre_friction_source_fingerprint=source_fingerprint "
        "WHERE pre_friction_source_fingerprint IS NULL"
    )

    rows = conn.execute(
        f"""SELECT rowid,
                   entry_execution_return,exit_now_return,hold_terminal_return,hold_advantage_return,
                   entry_return_gross,exit_now_return_gross,hold_terminal_return_gross,hold_advantage_gross,
                   COALESCE(pre_friction_source_fingerprint,'')
            FROM {table}"""
    ).fetchall()
    one_way = cfg.default_round_trip_bps / 20000.0
    round_trip = cfg.default_round_trip_bps / 10000.0
    fhash = _hash(target_contract_payload(cfg)["counterfactual_friction"])

    for row in rows:
        (rowid, entry_alias, exit_alias, hold_alias, adv_alias,
         entry_g_saved, exit_g_saved, hold_g_saved, adv_g_saved, raw_fp) = row

        # First enrichment reads retained gross aliases; every later enrichment
        # reads the immutable explicit gross columns instead of already-net aliases.
        entry_g = float(entry_g_saved) if entry_g_saved is not None else (float(entry_alias) if entry_alias is not None else None)
        exit_g = float(exit_g_saved) if exit_g_saved is not None else (float(exit_alias) if exit_alias is not None else None)
        hold_g = float(hold_g_saved) if hold_g_saved is not None else (float(hold_alias) if hold_alias is not None else None)
        adv_g = float(adv_g_saved) if adv_g_saved is not None else (float(adv_alias) if adv_alias is not None else None)

        entry_n = entry_g - round_trip if entry_g is not None else None
        exit_n = exit_g - one_way if exit_g is not None else None
        hold_n = hold_g - one_way if hold_g is not None else None
        # HOLD-vs-exit compares future exit costs only; equal fixed exit friction
        # cancels, so the advantage itself is unchanged.
        adv_n = adv_g
        stable_fp = hashlib.sha256(f"{raw_fp}|friction:{fhash}".encode("utf-8")).hexdigest()

        conn.execute(
            f"""UPDATE {table} SET
                entry_return_gross=?,entry_return_net=?,
                exit_now_return_gross=?,exit_now_return_net=?,
                hold_terminal_return_gross=?,hold_terminal_return_net=?,
                hold_advantage_gross=?,hold_advantage_net=?,
                entry_execution_return=?,exit_now_return=?,hold_terminal_return=?,hold_advantage_return=?,
                friction_bps=?,friction_definition_hash=?,source_fingerprint=?
                WHERE rowid=?""",
            (entry_g, entry_n, exit_g, exit_n, hold_g, hold_n, adv_g, adv_n,
             entry_n, exit_n, hold_n, adv_n,
             cfg.default_round_trip_bps, fhash, stable_fp, rowid),
        )
    conn.commit()
    n = len(rows)
    return {"updated": n, "policy_aliases_updated_to_net": n, "stable_friction_provenance_rows": n}
