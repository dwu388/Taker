from __future__ import annotations

"""Compatibility-tightened pretraining contract.

V1 owns the definitions.  This layer makes repeated materialization idempotent,
feeds friction-adjusted values through the retained policy compatibility aliases,
and strengthens readiness/audit reporting with mature development-block and
collector-context evidence.
"""

import argparse
import hashlib
import json
import sqlite3
from typing import Any, Sequence

import pandas as pd

from . import pretraining_contract as _base
from .db import RAW_DB_DEFAULT

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)


def refresh_pretraining_targets(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    out = _base.refresh_pretraining_targets(db, cfg)
    # SQLite UNIQUE constraints treat NULLs as distinct. Economic-collapse rows
    # intentionally have no up/down barrier, so compact each decision to the newest
    # materialization after every refresh.
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
        out["economic_collapse_rows"] = int(conn.execute(
            f"SELECT COUNT(*) FROM {TARGET_TABLE} WHERE target_kind='economic_collapse'"
        ).fetchone()[0])
    return out


def enrich_counterfactual_friction(conn: sqlite3.Connection, cfg: PretrainingConfig | None = None) -> dict[str, int]:
    cfg = cfg or PretrainingConfig()
    out = _base.enrich_counterfactual_friction(conn, cfg)
    table = "axiom_v24_counterfactual_policy_targets"
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
        return out
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if not {"entry_return_net", "hold_advantage_net", "friction_definition_hash"}.issubset(cols):
        return out
    # The retained V24 policy learner reads entry_execution_return and
    # hold_advantage_return. Preserve gross values in explicit *_gross columns and
    # move the compatibility aliases to net economics.
    conn.execute(
        f"""UPDATE {table} SET
            entry_execution_return=COALESCE(entry_return_net,entry_execution_return),
            exit_now_return=COALESCE(exit_now_return_net,exit_now_return),
            hold_terminal_return=COALESCE(hold_terminal_return_net,hold_terminal_return),
            hold_advantage_return=COALESCE(hold_advantage_net,hold_advantage_return)"""
    )
    # Friction is part of target provenance. Do not leave an old fingerprint
    # looking equivalent after the economic target changes.
    rows = conn.execute(
        f"SELECT rowid,COALESCE(source_fingerprint,''),COALESCE(friction_definition_hash,'') FROM {table}"
    ).fetchall()
    for rowid, old, fhash in rows:
        fp = hashlib.sha256(f"{old}|friction:{fhash}".encode("utf-8")).hexdigest()
        conn.execute(f"UPDATE {table} SET source_fingerprint=? WHERE rowid=?", (fp, rowid))
    conn.commit()
    out["policy_aliases_updated_to_net"] = int(len(rows))
    return out


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _mature_development_blocks(conn: sqlite3.Connection, latest: pd.Timestamp | None) -> int:
    table = "axiom_v24_calendar_cohorts"
    if latest is None or not _table_exists(conn, table):
        return 0
    cutoff = latest - pd.Timedelta(hours=24)
    try:
        return int(conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE role='train' AND end_at<=?",
            (cutoff.isoformat(),),
        ).fetchone()[0])
    except sqlite3.DatabaseError:
        return 0


def _development_token_count(conn: sqlite3.Connection, fallback: int) -> int:
    table = "axiom_v24_token_assignment"
    if not _table_exists(conn, table):
        return fallback
    try:
        # Before first bootstrap, only ordinary train-role tokens are development
        # data. One-use promotion/audit tokens remain reserved.
        return int(conn.execute(
            f"SELECT COUNT(DISTINCT token_key) FROM {table} WHERE forecast_role='train'"
        ).fetchone()[0])
    except sqlite3.DatabaseError:
        return fallback


def training_readiness(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    # Materialize current truth first, then start from the V1 token-balanced report.
    refresh_pretraining_targets(db, cfg)
    report = _base.training_readiness(db, cfg)
    with sqlite3.connect(db) as conn:
        migrate(conn)
        obs, _ = peak.load_observations(conn)
        latest = None if obs.empty else pd.to_datetime(obs.snapshot_at, utc=True).max()
        fallback = int(obs.token_key.nunique()) if not obs.empty else 0
        dev_tokens = _development_token_count(conn, fallback)
        blocks = _mature_development_blocks(conn, latest)
        report["gates"]["train_tokens"] = {
            "value": dev_tokens,
            "minimum": cfg.min_train_tokens,
            "pass": dev_tokens >= cfg.min_train_tokens,
        }
        report["gates"]["mature_development_blocks"] = {
            "value": blocks,
            "minimum": cfg.min_cpcv_blocks,
            "pass": blocks >= cfg.min_cpcv_blocks,
        }
        core = (
            report["gates"]["collection_days"]["pass"]
            and report["gates"]["train_tokens"]["pass"]
            and report["gates"]["confirmed_peak_tokens"]["pass"]
            and report["gates"]["retained_barrier_cells"]["pass"]
            and report["gates"]["mature_development_blocks"]["pass"]
        )
        report["ready"] = bool(core)
        report["head_enablement"]["peak_and_barrier"] = bool(core)
        report["head_enablement"]["ts2vec"] = bool(dev_tokens >= cfg.ts2vec_min_tokens)
        report["note"] = (
            "Production readiness uses independent development-role tokens and mature calendar blocks. "
            "Promotion/audit tokens are not counted as first-model training support. Sparse death remains modular."
        )
        conn.execute(
            f"INSERT INTO {READINESS_TABLE}(target_definition_hash,ready,report_json) VALUES(?,?,?)",
            (target_contract_hash(cfg), int(report["ready"]), json.dumps(report, sort_keys=True, default=str)),
        )
        conn.commit()
    return report


def assert_training_ready(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    report = training_readiness(db, cfg)
    if not report["ready"]:
        modular = {"operational_death_tokens"}
        failed = [k for k,v in report["gates"].items() if not bool(v.get("pass")) and k not in modular]
        raise RuntimeError("V24 production bootstrap refused by pretraining readiness gates: " + ", ".join(failed))
    return report


def collection_audit(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    report = _base.collection_audit(db, cfg)
    with sqlite3.connect(db) as conn:
        sessions = {}
        if _table_exists(conn, "collection_sessions"):
            for schema, n in conn.execute("SELECT collector_schema,COUNT(*) FROM collection_sessions GROUP BY collector_schema"):
                sessions[str(schema)] = int(n)
        attempts = {}
        if _table_exists(conn, "capture_attempts"):
            for success, n in conn.execute("SELECT success,COUNT(*) FROM capture_attempts GROUP BY success"):
                attempts["successful" if int(success) else "failed"] = int(n)
        context = {}
        if _table_exists(conn, "axiom_v24_capture_context"):
            row = conn.execute(
                """SELECT COUNT(*),AVG(visible_tokens),AVG(new_tokens_this_cycle),
                          AVG(capture_control_latency_ms),AVG(post_capture_processing_latency_ms)
                   FROM axiom_v24_capture_context"""
            ).fetchone()
            context = {
                "cycles": int(row[0] or 0),
                "mean_visible_tokens": float(row[1]) if row[1] is not None else None,
                "mean_new_tokens": float(row[2]) if row[2] is not None else None,
                "mean_capture_control_latency_ms": float(row[3]) if row[3] is not None else None,
                "mean_post_capture_processing_latency_ms": float(row[4]) if row[4] is not None else None,
            }
        report["collector_schema_counts"] = sessions
        report["capture_attempt_counts"] = attempts
        report["capture_context"] = context
        # Persist the augmented form as the newest authoritative audit snapshot.
        migrate(conn)
        conn.execute(
            f"INSERT INTO {AUDIT_TABLE}(target_definition_hash,report_json) VALUES(?,?)",
            (target_contract_hash(cfg), json.dumps(report, sort_keys=True, default=str)),
        )
        conn.commit()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="V24 pre-first-training hardening contracts")
    sp = p.add_subparsers(dest="cmd", required=True)
    for name in ("refresh-targets", "audit", "readiness", "baselines", "profile", "enrich-friction"):
        x = sp.add_parser(name); x.add_argument("--db", default=RAW_DB_DEFAULT)
    args = p.parse_args(argv); cfg = PretrainingConfig()
    if args.cmd == "refresh-targets": out = refresh_pretraining_targets(args.db, cfg)
    elif args.cmd == "audit": out = collection_audit(args.db, cfg)
    elif args.cmd == "readiness": out = training_readiness(args.db, cfg)
    elif args.cmd == "baselines": out = evaluate_baselines(args.db, cfg)
    elif args.cmd == "profile": out = first_model_profile(cfg)
    elif args.cmd == "enrich-friction":
        with sqlite3.connect(args.db) as conn: out = enrich_counterfactual_friction(conn, cfg)
    else: raise RuntimeError(args.cmd)
    print(json.dumps(out, indent=2, default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
