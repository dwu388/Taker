from __future__ import annotations

"""Cheap, read-only pretraining readiness snapshot for V24 status.

The production target materializer is intentionally not invoked here.  This module
summarizes raw collection breadth plus whatever current-contract targets have
already been materialized, and reports target freshness explicitly so a quick
status check can never masquerade as a full target refresh.
"""

import sqlite3
from typing import Any

import pandas as pd

from . import axiom_peak_structure as peak
from . import pretraining_contract_v4 as contract


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone())


def _utc(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def status_snapshot(db: str, cfg: contract.PretrainingConfig | None = None) -> dict[str, Any]:
    """Return readiness evidence without rebuilding historical targets.

    Barrier support is considered current only when current-definition target rows
    reach the latest observation timestamp.  A stale/missing materialization is
    reported as such rather than treated as a failed market-data gate.
    """
    cfg = cfg or contract.PretrainingConfig()
    thash = contract.target_contract_hash(cfg)
    with sqlite3.connect(db) as conn:
        contract.migrate(conn)
        if not _table_exists(conn, "axiom_observations"):
            return {
                "available": False,
                "reason": "axiom_observations table missing",
                "targets_refreshed": False,
                "target_definition_hash": thash,
            }

        row = conn.execute(
            """SELECT COUNT(*),COUNT(DISTINCT token_key),MIN(snapshot_at),MAX(snapshot_at)
               FROM axiom_observations
               WHERE token_key IS NOT NULL AND market_cap_usd IS NOT NULL AND market_cap_usd > 0"""
        ).fetchone()
        observations = int(row[0] or 0)
        tokens = int(row[1] or 0)
        first_obs = _utc(row[2])
        latest_obs = _utc(row[3])
        span_days = (
            float((latest_obs - first_obs).total_seconds() / 86400.0)
            if first_obs is not None and latest_obs is not None else 0.0
        )

        event_table = getattr(peak, "PEAK_EVENT_TABLE", "axiom_peak_events_v21")
        peak_tokens = 0
        if _table_exists(conn, event_table):
            peak_tokens = int(conn.execute(
                f"SELECT COUNT(DISTINCT token_key) FROM {event_table}"
            ).fetchone()[0] or 0)

        target_rows = 0
        latest_target = None
        cells: dict[tuple[int, float, float], dict[str, int]] = {}
        if _table_exists(conn, contract.TARGET_TABLE):
            target_rows, latest_raw = conn.execute(
                f"""SELECT COUNT(*),MAX(decision_at) FROM {contract.TARGET_TABLE}
                    WHERE target_definition_hash=?""",
                (thash,),
            ).fetchone()
            target_rows = int(target_rows or 0)
            latest_target = _utc(latest_raw)
            rows = conn.execute(
                f"""SELECT horizon_minutes,up_barrier,down_barrier,outcome,COUNT(DISTINCT token_key)
                    FROM {contract.TARGET_TABLE}
                    WHERE target_kind='triple_barrier' AND target_definition_hash=?
                    GROUP BY horizon_minutes,up_barrier,down_barrier,outcome""",
                (thash,),
            ).fetchall()
            for h, up, down, outcome, n in rows:
                if up is None or down is None:
                    continue
                cells.setdefault((int(h), float(up), float(down)), {})[str(outcome)] = int(n)

        retained: list[dict[str, Any]] = []
        for (h, up, down), counts in sorted(cells.items()):
            floor = min(
                int(counts.get("up_first", 0)),
                int(counts.get("down_first", 0)),
                int(counts.get("neither", 0)),
            )
            if floor >= int(cfg.min_multiclass_tokens_per_class):
                retained.append({
                    "horizon_minutes": h,
                    "up": up,
                    "down": down,
                    **counts,
                    "minimum_resolved_class_tokens": floor,
                })

        targets_current = bool(
            latest_obs is not None
            and latest_target is not None
            and latest_target >= latest_obs
        )
        gates = {
            "collection_days": {
                "value": span_days,
                "minimum": float(cfg.min_collection_days),
                "pass": span_days >= float(cfg.min_collection_days),
            },
            "raw_distinct_tokens": {
                "value": tokens,
                "minimum": int(cfg.min_train_tokens),
                "pass": tokens >= int(cfg.min_train_tokens),
                "note": "Raw distinct-token count; production bootstrap still enforces development-role eligibility.",
            },
            "confirmed_peak_tokens": {
                "value": peak_tokens,
                "minimum": int(cfg.min_confirmed_peak_tokens),
                "pass": peak_tokens >= int(cfg.min_confirmed_peak_tokens),
            },
            "retained_barrier_cells": {
                "value": len(retained),
                "minimum": 1,
                "pass": bool(targets_current and retained),
                "fresh": targets_current,
                "note": "Unknown/current=false until current-contract targets are materialized through the latest observation.",
            },
        }
        return {
            "available": True,
            "targets_refreshed": False,
            "observations": observations,
            "raw_distinct_tokens": tokens,
            "first_observation_at": first_obs,
            "latest_observation_at": latest_obs,
            "collection_span_days": span_days,
            "confirmed_peak_tokens": peak_tokens,
            "target_materialization": {
                "current_definition_rows": target_rows,
                "latest_target_decision_at": latest_target,
                "current_through_latest_observation": targets_current,
            },
            "gates": gates,
            "retained_barrier_cells": retained,
            "target_definition_hash": thash,
            "note": "Fast status snapshot only; it never rebuilds historical price-path targets.",
        }
