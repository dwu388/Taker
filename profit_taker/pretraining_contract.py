from __future__ import annotations

"""Pre-first-training contracts for the production V24 line.

This module deliberately sits beside the retained V24 implementation.  It owns the
new targets and gates that must be frozen before the first production champion is
minted: ordered triple barriers, economic-collapse labels, friction-aware
counterfactual accounting, collection audits, statistical readiness, simple
baselines, and a reduced first-model profile.
"""

import argparse
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import axiom_peak_structure as peak
from .db import RAW_DB_DEFAULT

TARGET_TABLE = "axiom_v24_pretraining_targets"
BASELINE_TABLE = "axiom_v24_baseline_results"
READINESS_TABLE = "axiom_v24_training_readiness"
AUDIT_TABLE = "axiom_v24_collection_audits"

PRETRAINING_SCHEMA_VERSION = "v24_pretraining_contract_v1"


@dataclass(frozen=True)
class PretrainingConfig:
    barrier_horizons_minutes: tuple[int, ...] = (15, 30, 60, 240)
    up_barriers: tuple[float, ...] = (0.30, 0.50)
    down_barriers: tuple[float, ...] = (-0.20, -0.35)
    economic_collapse_drawdown_pct: float = 0.85
    economic_collapse_sustain_minutes: int = 10
    default_round_trip_bps: float = 100.0
    friction_stress_bps: tuple[float, ...] = (100.0, 300.0, 500.0, 1000.0)
    min_collection_days: float = 14.0
    min_train_tokens: int = 300
    min_confirmed_peak_tokens: int = 50
    min_operational_death_tokens: int = 50
    min_binary_positive_tokens: int = 30
    min_binary_negative_tokens: int = 30
    min_multiclass_tokens_per_class: int = 30
    min_cpcv_blocks: int = 6
    ts2vec_min_tokens: int = 120
    first_model_estimators: int = 150


def _utc(v: Any) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _json(v: Any) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), default=str)


def _hash(v: Any) -> str:
    return hashlib.sha256(_json(v).encode("utf-8")).hexdigest()


def target_contract_payload(cfg: PretrainingConfig) -> dict[str, Any]:
    return {
        "schema": PRETRAINING_SCHEMA_VERSION,
        "triple_barrier": {
            "horizons_minutes": list(cfg.barrier_horizons_minutes),
            "up": list(cfg.up_barriers),
            "down": list(cfg.down_barriers),
            "semantics": "first_observation_strictly_after_decision_then_first_barrier_touch",
            "outcomes": ["up_first", "down_first", "neither", "censored"],
        },
        "economic_collapse": {
            "drawdown_from_trailing_observed_peak": cfg.economic_collapse_drawdown_pct,
            "sustain_minutes": cfg.economic_collapse_sustain_minutes,
            "settlement": "retain_observed_return; total_loss_is_stress_only",
        },
        "counterfactual_friction": {
            "default_round_trip_bps": cfg.default_round_trip_bps,
            "stress_bps": list(cfg.friction_stress_bps),
            "entry": "entry_plus_exit_cost",
            "hold": "compare_next_exit_cost_to_later_exit_cost_without_recharging_sunk_entry",
        },
    }


def target_contract_hash(cfg: PretrainingConfig) -> str:
    return _hash(target_contract_payload(cfg))


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            token_key TEXT NOT NULL,
            decision_at TEXT NOT NULL,
            target_kind TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL DEFAULT 0,
            up_barrier REAL,
            down_barrier REAL,
            outcome TEXT,
            event_at TEXT,
            reference_at TEXT,
            reference_mc REAL,
            terminal_mc REAL,
            gross_return REAL,
            net_return REAL,
            target_ready_at TEXT,
            target_definition_hash TEXT NOT NULL,
            details_json TEXT NOT NULL,
            PRIMARY KEY(token_key,decision_at,target_kind,horizon_minutes,up_barrier,down_barrier)
        );
        CREATE INDEX IF NOT EXISTS idx_{TARGET_TABLE}_kind_ready
            ON {TARGET_TABLE}(target_kind,target_ready_at);

        CREATE TABLE IF NOT EXISTS {BASELINE_TABLE} (
            baseline_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            target_definition_hash TEXT NOT NULL,
            baseline_name TEXT NOT NULL,
            metric_json TEXT NOT NULL,
            details_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {READINESS_TABLE} (
            readiness_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            target_definition_hash TEXT NOT NULL,
            ready INTEGER NOT NULL,
            report_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {AUDIT_TABLE} (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            target_definition_hash TEXT NOT NULL,
            report_json TEXT NOT NULL
        );
        """
    )
    conn.commit()


def _capture_times(conn: sqlite3.Connection) -> list[pd.Timestamp]:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='capture_cycles'").fetchone():
        return []
    rows = conn.execute(
        "SELECT captured_at FROM capture_cycles WHERE completed=1 AND clipboard_valid=1 ORDER BY captured_at"
    ).fetchall()
    return [_utc(r[0]) for r in rows]


def _manual_censors(conn: sqlite3.Connection) -> dict[str, list[pd.Timestamp]]:
    name = "axiom_v24_collection_censors"
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
        return {}
    out: dict[str, list[pd.Timestamp]] = {}
    for token, ts in conn.execute(f"SELECT token_key,censor_at FROM {name}"):
        out.setdefault(str(token), []).append(_utc(ts))
    for vals in out.values():
        vals.sort()
    return out


def _next_observation(g: pd.DataFrame, decision: pd.Timestamp) -> pd.Series | None:
    f = g[g.snapshot_at > decision]
    return None if f.empty else f.iloc[0]


def _crosses_manual_censor(censors: list[pd.Timestamp], start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return any(start < c <= end for c in censors)


def _barrier_outcome(
    g: pd.DataFrame,
    decision: pd.Timestamp,
    horizon: int,
    up: float,
    down: float,
    censors: list[pd.Timestamp],
) -> dict[str, Any]:
    ref = _next_observation(g, decision)
    if ref is None:
        return {"outcome": "censored"}
    ref_at = _utc(ref.snapshot_at)
    ref_mc = float(ref.market_cap_usd)
    deadline = decision + pd.Timedelta(minutes=int(horizon))
    if _crosses_manual_censor(censors, decision, deadline):
        censor_at = min(c for c in censors if decision < c <= deadline)
    else:
        censor_at = None
    end = min(deadline, censor_at) if censor_at is not None else deadline
    path = g[(g.snapshot_at >= ref_at) & (g.snapshot_at <= end)].copy()
    if path.empty or not math.isfinite(ref_mc) or ref_mc <= 0:
        return {"outcome": "censored"}
    upper = ref_mc * (1.0 + float(up))
    lower = ref_mc * (1.0 + float(down))
    hit_up = path[path.market_cap_usd >= upper]
    hit_down = path[path.market_cap_usd <= lower]
    up_at = _utc(hit_up.iloc[0].snapshot_at) if not hit_up.empty else None
    down_at = _utc(hit_down.iloc[0].snapshot_at) if not hit_down.empty else None
    if up_at is not None and (down_at is None or up_at < down_at):
        outcome, event_at = "up_first", up_at
    elif down_at is not None and (up_at is None or down_at < up_at):
        outcome, event_at = "down_first", down_at
    elif up_at is not None and down_at is not None:
        # Same one-minute snapshot cannot truthfully reveal within-minute ordering.
        outcome, event_at = "censored", up_at
    elif censor_at is not None:
        outcome, event_at = "censored", censor_at
    elif _utc(g.snapshot_at.max()) >= deadline:
        outcome, event_at = "neither", deadline
    else:
        outcome, event_at = "censored", _utc(g.snapshot_at.max())
    terminal_mc = float(path.iloc[-1].market_cap_usd)
    gross = terminal_mc / ref_mc - 1.0
    return {
        "outcome": outcome,
        "event_at": event_at,
        "reference_at": ref_at,
        "reference_mc": ref_mc,
        "terminal_mc": terminal_mc,
        "gross_return": gross,
        "target_ready_at": event_at if outcome in {"up_first", "down_first"} else (deadline if outcome == "neither" else None),
    }


def _economic_collapse(g: pd.DataFrame, decision: pd.Timestamp, cfg: PretrainingConfig, censors: list[pd.Timestamp]) -> dict[str, Any]:
    ref = _next_observation(g, decision)
    if ref is None:
        return {"outcome": "censored"}
    ref_at = _utc(ref.snapshot_at)
    path = g[g.snapshot_at >= ref_at].copy()
    if path.empty:
        return {"outcome": "censored"}
    running_high = -np.inf
    run_start: pd.Timestamp | None = None
    for r in path.itertuples(index=False):
        t = _utc(r.snapshot_at)
        if any(ref_at < c <= t for c in censors):
            return {"outcome": "censored", "event_at": min(c for c in censors if ref_at < c <= t)}
        mc = float(r.market_cap_usd)
        running_high = max(running_high, mc)
        collapsed = running_high > 0 and mc <= running_high * (1.0 - cfg.economic_collapse_drawdown_pct)
        if collapsed:
            if run_start is None:
                run_start = t
            if (t - run_start).total_seconds() / 60.0 >= cfg.economic_collapse_sustain_minutes:
                return {
                    "outcome": "economic_collapse",
                    "event_at": t,
                    "reference_at": ref_at,
                    "reference_mc": float(ref.market_cap_usd),
                    "terminal_mc": mc,
                    "gross_return": mc / float(ref.market_cap_usd) - 1.0,
                    "target_ready_at": t,
                    "trailing_peak_mc": running_high,
                }
        else:
            run_start = None
    return {"outcome": "open", "reference_at": ref_at, "reference_mc": float(ref.market_cap_usd)}


def refresh_pretraining_targets(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    with closing(sqlite3.connect(db)) as conn, conn:
        migrate(conn)
        obs, source = peak.load_observations(conn)
        if obs.empty:
            return {"written": 0, "source": source, "target_definition_hash": target_contract_hash(cfg)}
        obs = obs[["token_key", "snapshot_at", "market_cap_usd"]].dropna().copy()
        obs["token_key"] = obs.token_key.astype(str)
        obs["snapshot_at"] = pd.to_datetime(obs.snapshot_at, utc=True)
        obs["market_cap_usd"] = pd.to_numeric(obs.market_cap_usd, errors="coerce")
        obs = obs[np.isfinite(obs.market_cap_usd) & (obs.market_cap_usd > 0)].copy()
        censors = _manual_censors(conn)
        thash = target_contract_hash(cfg)
        written = 0
        for token, g in obs.groupby("token_key", sort=False):
            g = g.sort_values("snapshot_at").reset_index(drop=True)
            token_censors = censors.get(str(token), [])
            for decision in g.snapshot_at:
                decision = _utc(decision)
                for h in cfg.barrier_horizons_minutes:
                    for up in cfg.up_barriers:
                        for down in cfg.down_barriers:
                            rec = _barrier_outcome(g, decision, h, up, down, token_censors)
                            net = None
                            if rec.get("gross_return") is not None:
                                net = float(rec["gross_return"]) - cfg.default_round_trip_bps / 10000.0
                            conn.execute(
                                f"""INSERT INTO {TARGET_TABLE}
                                (token_key,decision_at,target_kind,horizon_minutes,up_barrier,down_barrier,outcome,event_at,
                                 reference_at,reference_mc,terminal_mc,gross_return,net_return,target_ready_at,target_definition_hash,details_json)
                                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                                ON CONFLICT(token_key,decision_at,target_kind,horizon_minutes,up_barrier,down_barrier)
                                DO UPDATE SET outcome=excluded.outcome,event_at=excluded.event_at,reference_at=excluded.reference_at,
                                  reference_mc=excluded.reference_mc,terminal_mc=excluded.terminal_mc,gross_return=excluded.gross_return,
                                  net_return=excluded.net_return,target_ready_at=excluded.target_ready_at,
                                  target_definition_hash=excluded.target_definition_hash,details_json=excluded.details_json""",
                                (str(token), decision.isoformat(), "triple_barrier", int(h), float(up), float(down),
                                 rec.get("outcome"), str(rec.get("event_at")) if rec.get("event_at") is not None else None,
                                 str(rec.get("reference_at")) if rec.get("reference_at") is not None else None,
                                 rec.get("reference_mc"), rec.get("terminal_mc"), rec.get("gross_return"), net,
                                 str(rec.get("target_ready_at")) if rec.get("target_ready_at") is not None else None,
                                 thash, _json(rec)),
                            )
                            written += 1
                collapse = _economic_collapse(g, decision, cfg, token_censors)
                conn.execute(
                    f"""INSERT INTO {TARGET_TABLE}
                    (token_key,decision_at,target_kind,horizon_minutes,up_barrier,down_barrier,outcome,event_at,reference_at,
                     reference_mc,terminal_mc,gross_return,net_return,target_ready_at,target_definition_hash,details_json)
                    VALUES(?,?, 'economic_collapse',0,NULL,NULL,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(token_key,decision_at,target_kind,horizon_minutes,up_barrier,down_barrier)
                    DO UPDATE SET outcome=excluded.outcome,event_at=excluded.event_at,reference_at=excluded.reference_at,
                      reference_mc=excluded.reference_mc,terminal_mc=excluded.terminal_mc,gross_return=excluded.gross_return,
                      net_return=excluded.net_return,target_ready_at=excluded.target_ready_at,
                      target_definition_hash=excluded.target_definition_hash,details_json=excluded.details_json""",
                    (str(token), decision.isoformat(), collapse.get("outcome"),
                     str(collapse.get("event_at")) if collapse.get("event_at") is not None else None,
                     str(collapse.get("reference_at")) if collapse.get("reference_at") is not None else None,
                     collapse.get("reference_mc"), collapse.get("terminal_mc"), collapse.get("gross_return"),
                     collapse.get("gross_return"),
                     str(collapse.get("target_ready_at")) if collapse.get("target_ready_at") is not None else None,
                     thash, _json(collapse)),
                )
                written += 1
        conn.commit()
        return {"written": written, "target_definition_hash": thash, "source": source}


def enrich_counterfactual_friction(conn: sqlite3.Connection, cfg: PretrainingConfig | None = None) -> dict[str, int]:
    """Add gross/net policy returns without changing the retained table's compatibility columns."""
    cfg = cfg or PretrainingConfig()
    table = "axiom_v24_counterfactual_policy_targets"
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
        return {"updated": 0}
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    additions = {
        "entry_return_gross": "REAL", "entry_return_net": "REAL",
        "exit_now_return_gross": "REAL", "exit_now_return_net": "REAL",
        "hold_terminal_return_gross": "REAL", "hold_terminal_return_net": "REAL",
        "hold_advantage_gross": "REAL", "hold_advantage_net": "REAL",
        "friction_bps": "REAL", "friction_definition_hash": "TEXT",
    }
    for name, typ in additions.items():
        if name not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")
    rows = conn.execute(
        f"SELECT rowid,action_kind,entry_execution_return,exit_now_return,hold_terminal_return,hold_advantage_return FROM {table}"
    ).fetchall()
    one_way = cfg.default_round_trip_bps / 20000.0
    round_trip = cfg.default_round_trip_bps / 10000.0
    fhash = _hash(target_contract_payload(cfg)["counterfactual_friction"])
    updated = 0
    for rowid, action, entry, exit_now, hold_ret, hold_adv in rows:
        entry_g = float(entry) if entry is not None else None
        exit_g = float(exit_now) if exit_now is not None else None
        hold_g = float(hold_ret) if hold_ret is not None else None
        adv_g = float(hold_adv) if hold_adv is not None else None
        entry_n = entry_g - round_trip if entry_g is not None else None
        exit_n = exit_g - one_way if exit_g is not None else None
        hold_n = hold_g - one_way if hold_g is not None else None
        # HOLD and EXIT share the already-paid entry cost.  The advantage therefore
        # compares only future exit costs, which cancel under equal fixed friction.
        adv_n = adv_g if adv_g is not None else None
        conn.execute(
            f"""UPDATE {table} SET entry_return_gross=?,entry_return_net=?,exit_now_return_gross=?,exit_now_return_net=?,
               hold_terminal_return_gross=?,hold_terminal_return_net=?,hold_advantage_gross=?,hold_advantage_net=?,
               friction_bps=?,friction_definition_hash=? WHERE rowid=?""",
            (entry_g, entry_n, exit_g, exit_n, hold_g, hold_n, adv_g, adv_n,
             cfg.default_round_trip_bps, fhash, rowid),
        )
        updated += 1
    conn.commit()
    return {"updated": updated}


def _terminal_counts(conn: sqlite3.Connection) -> dict[str, int]:
    table = getattr(peak, "LABEL_TABLE", "axiom_peak_structure_labels_v21")
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
        return {}
    try:
        return {str(k or "open"): int(v) for k, v in conn.execute(
            f"SELECT COALESCE(terminal_reason,'open'),COUNT(DISTINCT token_key) FROM {table} GROUP BY COALESCE(terminal_reason,'open')"
        )}
    except sqlite3.DatabaseError:
        return {}


def collection_audit(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    with closing(sqlite3.connect(db)) as conn, conn:
        migrate(conn)
        obs, source = peak.load_observations(conn)
        if obs.empty:
            report = {"available": False, "reason": "no observations", "source": source}
        else:
            obs = obs.copy(); obs["snapshot_at"] = pd.to_datetime(obs.snapshot_at, utc=True)
            first = obs.groupby("token_key").snapshot_at.min(); last = obs.groupby("token_key").snapshot_at.max()
            rows_per = obs.token_key.astype(str).value_counts()
            caps = pd.Series(_capture_times(conn), dtype="datetime64[ns, UTC]").sort_values()
            gaps = caps.diff().dt.total_seconds().div(60).dropna() if len(caps) else pd.Series(dtype=float)
            field_null = {}
            for col in ("market_cap_usd","volume_usd","fees_sol","txns","holders","pro_traders","kols",
                        "recent_visitors","tracked_dev_status_raw","dex_paid","image_reuse_count","token_address"):
                if col in obs:
                    field_null[col] = float(obs[col].isna().mean())
            report = {
                "available": True,
                "source": source,
                "captures": int(len(caps)),
                "capture_gap_minutes": {
                    "p50": float(gaps.quantile(.50)) if len(gaps) else None,
                    "p95": float(gaps.quantile(.95)) if len(gaps) else None,
                    "p99": float(gaps.quantile(.99)) if len(gaps) else None,
                    "max": float(gaps.max()) if len(gaps) else None,
                },
                "observations": int(len(obs)),
                "tokens": int(obs.token_key.nunique()),
                "collection_span_days": float((obs.snapshot_at.max()-obs.snapshot_at.min()).total_seconds()/86400.0),
                "rows_per_token": {"p50": float(rows_per.quantile(.5)), "p95": float(rows_per.quantile(.95)), "max": int(rows_per.max())},
                "lifetime_minutes": {
                    "p50": float(((last-first).dt.total_seconds()/60).quantile(.5)),
                    "p95": float(((last-first).dt.total_seconds()/60).quantile(.95)),
                },
                "field_null_fraction": field_null,
                "full_mint_rows": int(obs.token_address.notna().sum()) if "token_address" in obs else 0,
                "short_identity_rows": int(obs.token_address.isna().sum()) if "token_address" in obs else int(len(obs)),
                "terminal_token_counts": _terminal_counts(conn),
            }
        conn.execute(
            f"INSERT INTO {AUDIT_TABLE}(target_definition_hash,report_json) VALUES(?,?)",
            (target_contract_hash(cfg), _json(report)),
        )
        conn.commit()
        return report


def _barrier_cell_counts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TARGET_TABLE,)).fetchone():
        return []
    rows = conn.execute(
        f"""SELECT horizon_minutes,up_barrier,down_barrier,outcome,COUNT(DISTINCT token_key)
            FROM {TARGET_TABLE} WHERE target_kind='triple_barrier'
            GROUP BY horizon_minutes,up_barrier,down_barrier,outcome"""
    ).fetchall()
    cells: dict[tuple[int,float,float], dict[str,int]] = {}
    for h,u,d,o,n in rows:
        cells.setdefault((int(h),float(u),float(d)), {})[str(o)] = int(n)
    return [{"horizon_minutes": h, "up": u, "down": d, **counts} for (h,u,d),counts in sorted(cells.items())]


def training_readiness(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    refresh_pretraining_targets(db, cfg)
    with closing(sqlite3.connect(db)) as conn, conn:
        migrate(conn)
        obs, _ = peak.load_observations(conn)
        obs["snapshot_at"] = pd.to_datetime(obs.snapshot_at, utc=True) if not obs.empty else pd.Series(dtype="datetime64[ns, UTC]")
        span = float((obs.snapshot_at.max()-obs.snapshot_at.min()).total_seconds()/86400.0) if len(obs) > 1 else 0.0
        train_tokens = int(obs.token_key.nunique()) if not obs.empty else 0
        terminals = _terminal_counts(conn)
        death_tokens = sum(v for k,v in terminals.items() if "dead_after_valid_capture_absence" in k)
        event_table = getattr(peak, "PEAK_EVENT_TABLE", "axiom_peak_events_v21")
        peak_tokens = 0
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (event_table,)).fetchone():
            peak_tokens = int(conn.execute(f"SELECT COUNT(DISTINCT token_key) FROM {event_table}").fetchone()[0])
        cells = _barrier_cell_counts(conn)
        retained = []
        dropped = []
        for c in cells:
            pos = int(c.get("up_first",0)); neg = int(c.get("down_first",0)); neither = int(c.get("neither",0))
            class_floor = min(pos, neg, neither)
            ok = class_floor >= cfg.min_multiclass_tokens_per_class
            (retained if ok else dropped).append({**c, "minimum_resolved_class_tokens": class_floor})
        gates = {
            "collection_days": {"value": span, "minimum": cfg.min_collection_days, "pass": span >= cfg.min_collection_days},
            "train_tokens": {"value": train_tokens, "minimum": cfg.min_train_tokens, "pass": train_tokens >= cfg.min_train_tokens},
            "confirmed_peak_tokens": {"value": peak_tokens, "minimum": cfg.min_confirmed_peak_tokens, "pass": peak_tokens >= cfg.min_confirmed_peak_tokens},
            "operational_death_tokens": {"value": death_tokens, "minimum": cfg.min_operational_death_tokens, "pass": death_tokens >= cfg.min_operational_death_tokens},
            "retained_barrier_cells": {"value": len(retained), "minimum": 1, "pass": len(retained) >= 1},
        }
        # Event-family readiness is modular.  A sparse death head does not prevent
        # barrier/peak training; it remains explicitly disabled until supported.
        core_ready = all(gates[k]["pass"] for k in ("collection_days","train_tokens","confirmed_peak_tokens","retained_barrier_cells"))
        report = {
            "ready": bool(core_ready),
            "gates": gates,
            "retained_barrier_cells": retained,
            "dropped_barrier_cells": dropped,
            "head_enablement": {
                "peak_and_barrier": bool(core_ready),
                "operational_death": bool(death_tokens >= cfg.min_operational_death_tokens),
                "ts2vec": bool(train_tokens >= cfg.ts2vec_min_tokens),
            },
            "target_definition_hash": target_contract_hash(cfg),
            "note": "Readiness counts independent tokens, never one-minute rows. Sparse heads are disabled rather than poisoning the whole bootstrap.",
        }
        conn.execute(
            f"INSERT INTO {READINESS_TABLE}(target_definition_hash,ready,report_json) VALUES(?,?,?)",
            (target_contract_hash(cfg), int(report["ready"]), _json(report)),
        )
        conn.commit()
        return report


def assert_training_ready(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    report = training_readiness(db, cfg)
    if not report["ready"]:
        failed = [k for k,v in report["gates"].items() if not v["pass"] and k != "operational_death_tokens"]
        raise RuntimeError("V24 production bootstrap refused by pretraining readiness gates: " + ", ".join(failed))
    return report


def _age_bucket(minutes: float) -> str:
    edges = ((15,"00-15m"),(30,"15-30m"),(60,"30-60m"),(120,"01-02h"),(240,"02-04h"),(480,"04-08h"),(720,"08-12h"),(1440,"12-24h"))
    lo = 0
    for hi,name in edges:
        if minutes < hi:
            return name
        lo = hi
    return "24h+"


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p-y)**2)) if len(y) else float("nan")


def evaluate_baselines(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    """Pre-registered simple baselines on resolved 60m +30/-20 triple-barrier targets.

    The split is chronological by token first-seen: oldest 70% development, newest
    30% evaluation.  This is deliberately simple and reproducible; the one-use V24
    promotion stream remains the deployment gate for learned models.
    """
    cfg = cfg or PretrainingConfig()
    refresh_pretraining_targets(db, cfg)
    with closing(sqlite3.connect(db)) as conn, conn:
        migrate(conn)
        t = pd.read_sql_query(
            f"""SELECT token_key,decision_at,outcome FROM {TARGET_TABLE}
                WHERE target_kind='triple_barrier' AND horizon_minutes=60 AND up_barrier=0.3 AND down_barrier=-0.2
                  AND outcome IN ('up_first','down_first','neither')""", conn
        )
        obs, _ = peak.load_observations(conn)
        if t.empty or obs.empty:
            return {"available": False, "reason": "resolved baseline target unavailable"}
        # Persisted ISO timestamps legitimately mix whole and fractional seconds.
        # Parse the ISO family explicitly instead of inferring one fixed format
        # from the first row. Keep strict errors and exact UTC instants.
        obs["snapshot_at"] = pd.to_datetime(obs.snapshot_at, format="ISO8601", utc=True)
        t["decision_at"] = pd.to_datetime(t.decision_at, format="ISO8601", utc=True)
        first = obs.groupby("token_key").snapshot_at.min().sort_values(); tokens = list(first.index.astype(str))
        cut = max(1, int(len(tokens)*0.70)); train_tokens=set(tokens[:cut]); eval_tokens=set(tokens[cut:])
        if not eval_tokens:
            return {"available": False, "reason": "not enough chronological tokens"}
        # Causal age and 15m slope from raw observations only up to each decision.
        by={str(k):g.sort_values("snapshot_at") for k,g in obs.groupby(obs.token_key.astype(str))}
        rows=[]
        for r in t.itertuples(index=False):
            token=str(r.token_key); g=by.get(token)
            if g is None: continue
            d=_utc(r.decision_at); hist=g[g.snapshot_at<=d]
            if hist.empty: continue
            age=(d-_utc(g.snapshot_at.min())).total_seconds()/60.0
            recent=hist[hist.snapshot_at>=d-pd.Timedelta(minutes=15)]
            slope=np.nan
            if len(recent)>=2:
                y=np.log(pd.to_numeric(recent.market_cap_usd,errors="coerce").to_numpy(dtype=float))
                x=(recent.snapshot_at-recent.snapshot_at.iloc[0]).dt.total_seconds().to_numpy()/60.0
                if np.isfinite(y).all() and np.ptp(x)>0: slope=float(np.polyfit(x,y,1)[0])
            rows.append({"token_key":token,"decision_at":d,"outcome":r.outcome,"age_bucket":_age_bucket(age),"slope_15m":slope})
        f=pd.DataFrame(rows); f["y"]=(f.outcome=="up_first").astype(float)
        tr=f[f.token_key.isin(train_tokens)].copy(); ev=f[f.token_key.isin(eval_tokens)].copy()
        if tr.empty or ev.empty:
            return {"available":False,"reason":"chronological split empty"}
        global_rate=float(tr.y.mean())
        rates=tr.groupby("age_bucket").y.mean().to_dict(); ev["p_age"]=[float(rates.get(x,global_rate)) for x in ev.age_bucket]
        # Monotone rank calibration without external dependencies: 10 training
        # quantile bins, then cumulative maximum so higher momentum cannot predict
        # lower upside probability solely from sampling noise.
        valid=tr[np.isfinite(tr.slope_15m)].copy()
        if valid.empty:
            ev["p_slope"]=global_rate
        else:
            q=min(10,max(2,valid.slope_15m.nunique()))
            valid["bin"]=pd.qcut(valid.slope_15m,q=q,duplicates="drop")
            stats=valid.groupby("bin",observed=True).agg(lo=("slope_15m","min"),hi=("slope_15m","max"),p=("y","mean")).sort_values("lo")
            stats["p"]=np.maximum.accumulate(stats.p.to_numpy(dtype=float))
            def mp(x: float) -> float:
                if not math.isfinite(x): return global_rate
                hit=stats[(stats.lo<=x)&(stats.hi>=x)]
                if not hit.empty:return float(hit.iloc[0].p)
                return float(stats.iloc[0].p if x<stats.iloc[0].lo else stats.iloc[-1].p)
            ev["p_slope"]=[mp(float(x)) if pd.notna(x) else global_rate for x in ev.slope_15m]
        def token_metric(pcol: str) -> dict[str,float]:
            e=ev.copy(); e["sq"]=(e[pcol]-e.y)**2; e["ll"]=-(e.y*np.log(np.clip(e[pcol],1e-6,1-1e-6))+(1-e.y)*np.log(np.clip(1-e[pcol],1e-6,1-1e-6)))
            return {"token_balanced_brier":float(e.groupby("token_key").sq.mean().mean()),"token_balanced_log_loss":float(e.groupby("token_key").ll.mean().mean()),"tokens":int(e.token_key.nunique()),"rows":int(len(e))}
        results={"available":True,"target":"60m +30/-20 up-first","age_bucket":token_metric("p_age"),"slope_15m":token_metric("p_slope"),"split":{"train_tokens":len(train_tokens),"eval_tokens":len(eval_tokens)},"target_definition_hash":target_contract_hash(cfg)}
        for name in ("age_bucket","slope_15m"):
            conn.execute(f"INSERT INTO {BASELINE_TABLE}(baseline_id,target_definition_hash,baseline_name,metric_json,details_json) VALUES(?,?,?,?,?)",(_hash([name,pd.Timestamp.now(tz='UTC').isoformat()]),target_contract_hash(cfg),name,_json(results[name]),_json(results["split"])))
        conn.commit(); return results


def first_model_profile(cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    return {
        "name": "first_model",
        "production_readiness_required": True,
        "estimators": min(150, cfg.first_model_estimators),
        "feature_windows_minutes": [1,5,15,30,60,240],
        "hazard_max_minutes": 240,
        "triple_barrier_horizons_minutes": [60,240],
        "up_barriers": [0.30,0.50],
        "down_barriers": [-0.20,-0.35],
        "ts2vec_enabled": False,
        "recurrent_peak_count_enabled": False,
        "later_higher_peak_enabled": False,
        "long_horizon_heads_enabled": False,
        "note": "Reduced capacity is not --allow-small: all statistical readiness and holdout gates remain active.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    p=argparse.ArgumentParser(description="V24 pre-first-training hardening contracts")
    sp=p.add_subparsers(dest="cmd",required=True)
    for name in ("refresh-targets","audit","readiness","baselines","profile","enrich-friction"):
        x=sp.add_parser(name); x.add_argument("--db",default=RAW_DB_DEFAULT)
    args=p.parse_args(argv); cfg=PretrainingConfig()
    if args.cmd=="refresh-targets": out=refresh_pretraining_targets(args.db,cfg)
    elif args.cmd=="audit": out=collection_audit(args.db,cfg)
    elif args.cmd=="readiness": out=training_readiness(args.db,cfg)
    elif args.cmd=="baselines": out=evaluate_baselines(args.db,cfg)
    elif args.cmd=="profile": out=first_model_profile(cfg)
    elif args.cmd=="enrich-friction":
        with closing(sqlite3.connect(args.db)) as conn, conn: out=enrich_counterfactual_friction(conn,cfg)
    else: raise RuntimeError(args.cmd)
    print(json.dumps(out,indent=2,default=str)); return 0


if __name__=="__main__":
    raise SystemExit(main())
