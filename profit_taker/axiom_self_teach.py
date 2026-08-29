from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_pinball_loss

try:
    from . import axiom_peak_structure as peak
except Exception as exc:  # pragma: no cover
    peak = None
    PEAK_IMPORT_ERROR = exc
else:
    PEAK_IMPORT_ERROR = None

SCHEMA_VERSION = "v21_self_teaching_incremental_72h_2_execution_accounting"
POLICY_DIR_DEFAULT = "models/axiom_policy_v21"
PEAK_CHAMPION_DEFAULT = "models/axiom_peak_v21/champion.joblib"
PREDICTIONS_DEFAULT = "data/axiom_predictions_72h.csv"
PEAK_PREDICTIONS_DEFAULT = "data/axiom_peak_structure_predictions_72h.csv"

PREDICTION_EXCLUDE_FRAGMENTS = (
    "actual", "realized", "outcome", "label", "future", "target", "success", "failure",
    "entry_", "exit_", "reward", "closed_", "terminal_", "observed_peak",
)


@dataclass
class PolicyConfig:
    max_open_positions: int = 5
    exploration_fraction: float = 0.15
    candidate_pool: int = 15
    min_hold_minutes: float = 2.0
    max_hold_minutes: float = 72.0 * 60.0
    missed_cycles_to_close: int = 50
    missing_close_minutes: float = 50.0
    reentry_cooldown_minutes: float = 20.0
    friction_bps_round_trip: float = 100.0
    disappearance_profit_recognition_fraction: float = 0.0
    drawdown_penalty: float = 0.50
    peak_capture_weight: float = 0.10
    hold_label_minutes: float = 60.0
    policy_retrain_every_closed: int = 25
    policy_min_closed: int = 80
    forecast_retrain_min_new_mature_rows: int = 1000
    forecast_retrain_cooldown_hours: float = 2.0
    policy_promotion_margin: float = 0.01
    forecast_promotion_margin: float = 0.01


# ------------------------------ basic helpers ------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _loads(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    try:
        out = json.loads(value)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _hash_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_peak() -> None:
    if peak is None:
        raise RuntimeError(
            "V21 requires the axiom_peak_structure module from the V21 patch beside this file. "
            f"Import error: {PEAK_IMPORT_ERROR}"
        )


def _safe_prediction_state(row: pd.Series, extra: dict[str, Any] | None = None) -> dict[str, float]:
    state: dict[str, float] = {}
    for key, value in row.items():
        lk = str(key).lower()
        if key in {"token_key", "snapshot_at", "name", "next_substantial_peak_timing_window"}:
            continue
        if any(fragment in lk for fragment in PREDICTION_EXCLUDE_FRAGMENTS):
            continue
        x = _finite_float(value)
        if x is not None:
            state[str(key)] = x
    for key, value in (extra or {}).items():
        x = _finite_float(value)
        if x is not None:
            state[str(key)] = x
    return state


def _prediction_identity_col(df: pd.DataFrame) -> str:
    for c in ("token_key", "short_address_hint", "token", "mint", "token_address"):
        if c in df.columns:
            return c
    raise RuntimeError("Prediction CSV does not contain a recognized token identity column.")


def _prediction_snapshot(df: pd.DataFrame) -> pd.Timestamp | None:
    for c in ("snapshot_at", "decision_at", "observed_at"):
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True).dropna()
            if len(s):
                return s.max()
    return None


def _load_predictions(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Prediction CSV does not exist: {p}")
    df = pd.read_csv(p)
    if df.empty:
        return df
    key = _prediction_identity_col(df)
    if key != "token_key":
        df = df.rename(columns={key: "token_key"})
    df["token_key"] = df["token_key"].astype(str)
    return df


def _bootstrap_entry_score(state: dict[str, float]) -> float:
    # V19/V21 compatibility first, then V24 coherent-event outputs.
    p = state.get("p_next_substantial_peak_before_terminal_72h")
    if p is None:
        p = state.get("p_first_peak_by_720m", state.get("p_first_peak_by_1440m", state.get("p_first_peak_by_4320m", 0.0)))
    mult = state.get("pred_next_substantial_peak_multiple_q50", state.get("next_peak_multiple_q50", 1.0))
    p_chain = state.get("p_next_then_later_higher_peak_before_terminal_72h")
    if p_chain is None:
        p_chain = state.get("p_later_higher_10pct_by_1440m", state.get("p_later_higher_2pct_by_4320m", 0.0))
    later_mult = state.get("pred_later_higher_peak_multiple_vs_next_q50")
    if later_mult is None:
        later_mult = 1.0 + max(0.0, state.get("second_peak_relative_q50", 0.0))
    timing = state.get("pred_time_to_next_substantial_peak_minutes_q50", state.get("next_gap_q50", 180.0))
    retrace = state.get("pred_post_next_peak_retracement_pct_q50", 0.0)
    if retrace > 1.5: retrace /= 100.0
    primary=max(0.0,float(mult)-1.0)*max(0.0,min(1.0,float(p)))
    secondary=.25*max(0.0,float(later_mult)-1.0)*max(0.0,min(1.0,float(p_chain)))
    # Explicit downside/death heads reduce bootstrap utility when present.
    death=state.get("p_death_by_720m",state.get("p_death_by_1440m",0.0))
    downside=state.get("p_hit_minus50_by_720m",0.0)
    return float(primary+secondary-.0005*max(0.0,float(timing))-.10*max(0.0,float(retrace))-.30*max(0.0,float(death))-.35*max(0.0,float(downside)))


def _bootstrap_hold_value(state: dict[str, float]) -> float:
    p=state.get("p_next_substantial_peak_before_terminal_72h")
    if p is None: p=state.get("p_first_peak_by_720m",state.get("p_first_peak_by_1440m",state.get("p_first_peak_by_4320m",0.0)))
    mult=state.get("pred_next_substantial_peak_multiple_q50",state.get("next_peak_multiple_q50",1.0))
    p_later=state.get("p_later_higher_peak_given_next_before_terminal_72h")
    if p_later is None: p_later=state.get("p_later_higher_10pct_by_1440m",state.get("p_later_higher_2pct_by_4320m",0.0))
    later_mult=state.get("pred_later_higher_peak_multiple_vs_next_q50")
    if later_mult is None: later_mult=1.0+max(0.0,state.get("second_peak_relative_q50",0.0))
    retrace=state.get("pred_post_next_peak_retracement_pct_q50",0.0)
    if retrace>1.5: retrace/=100.0
    death=state.get("p_death_by_720m",0.0)
    return float(float(p)*(float(mult)-1.0)+.15*float(p_later)*max(0.0,float(later_mult)-1.0)-.10*max(0.0,float(retrace))-.30*max(0.0,float(death)))


# ------------------------------ database ------------------------------

def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_column(conn: sqlite3.Connection, table: str, definition: str) -> None:
    name = definition.split()[0]
    if name not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS axiom_self_teach_runs_v20 (
            run_id TEXT PRIMARY KEY,
            run_at TEXT NOT NULL,
            snapshot_at TEXT,
            predictions_path TEXT,
            forecast_model_hash TEXT,
            policy_version TEXT,
            current_tokens INTEGER,
            open_before INTEGER,
            entries INTEGER,
            exits INTEGER,
            open_after INTEGER,
            details_json TEXT
        );

        CREATE TABLE IF NOT EXISTS axiom_paper_positions_v20 (
            position_id TEXT PRIMARY KEY,
            token_key TEXT NOT NULL,
            opened_at TEXT NOT NULL,
            entry_mc REAL NOT NULL,
            entry_state_json TEXT NOT NULL,
            entry_policy_version TEXT,
            entry_forecast_hash TEXT,
            exploration INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            last_mc REAL NOT NULL,
            missed_cycles INTEGER NOT NULL DEFAULT 0,
            mfe_pct REAL NOT NULL DEFAULT 0,
            mae_pct REAL NOT NULL DEFAULT 0,
            closed_at TEXT,
            exit_mc REAL,
            close_reason TEXT,
            gross_return_pct REAL,
            net_return_pct REAL,
            peak_capture_ratio REAL,
            reward REAL,
            config_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_paper_positions_status ON axiom_paper_positions_v20(status);
        CREATE INDEX IF NOT EXISTS idx_paper_positions_token ON axiom_paper_positions_v20(token_key, opened_at);

        CREATE TABLE IF NOT EXISTS axiom_paper_marks_v20 (
            mark_id TEXT PRIMARY KEY,
            position_id TEXT NOT NULL,
            token_key TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            market_cap_usd REAL NOT NULL,
            return_pct REAL NOT NULL,
            mfe_pct REAL NOT NULL,
            mae_pct REAL NOT NULL,
            state_json TEXT NOT NULL,
            action TEXT NOT NULL,
            action_value REAL,
            policy_version TEXT,
            UNIQUE(position_id, snapshot_at)
        );
        CREATE INDEX IF NOT EXISTS idx_paper_marks_token_time ON axiom_paper_marks_v20(token_key, snapshot_at);

        CREATE TABLE IF NOT EXISTS axiom_paper_candidates_v20 (
            snapshot_at TEXT NOT NULL,
            token_key TEXT NOT NULL,
            market_cap_usd REAL NOT NULL,
            state_json TEXT NOT NULL,
            bootstrap_score REAL,
            policy_score REAL,
            chosen INTEGER NOT NULL DEFAULT 0,
            exploration INTEGER NOT NULL DEFAULT 0,
            forecast_hash TEXT,
            policy_version TEXT,
            PRIMARY KEY(snapshot_at, token_key)
        );

        CREATE TABLE IF NOT EXISTS axiom_policy_versions_v20 (
            version_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            model_path TEXT NOT NULL,
            model_hash TEXT,
            status TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            training_closed_trades INTEGER NOT NULL,
            training_hold_samples INTEGER NOT NULL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS axiom_forecast_promotions_v20 (
            promotion_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            candidate_path TEXT NOT NULL,
            candidate_hash TEXT,
            champion_before_path TEXT,
            champion_before_hash TEXT,
            promoted INTEGER NOT NULL,
            mature_rows INTEGER NOT NULL,
            metrics_json TEXT NOT NULL,
            reason TEXT
        );

        CREATE TABLE IF NOT EXISTS axiom_paper_pending_entries_v24 (
            pending_id TEXT PRIMARY KEY,
            token_key TEXT NOT NULL,
            decision_at TEXT NOT NULL,
            decision_mc REAL NOT NULL,
            decision_state_json TEXT NOT NULL,
            policy_version TEXT,
            forecast_hash TEXT,
            exploration INTEGER NOT NULL DEFAULT 0,
            action_probability REAL,
            status TEXT NOT NULL,
            filled_at TEXT,
            fill_mc REAL,
            cancelled_at TEXT,
            cancel_reason TEXT,
            UNIQUE(token_key,decision_at)
        );
        CREATE INDEX IF NOT EXISTS idx_pending_entries_v24_status ON axiom_paper_pending_entries_v24(status,decision_at);
        """
    )
    for definition in (
        "exit_kind TEXT",
        "price_available_at_exit INTEGER",
        "exit_mc_observed REAL",
        "exit_mc_execution_proxy REAL",
        "observed_gross_return_pct REAL",
        "execution_gross_return_pct REAL",
        "observed_net_return_pct REAL",
        "execution_net_return_pct REAL",
        "observed_peak_capture_ratio REAL",
        "execution_peak_capture_ratio REAL",
        "observed_reward REAL",
        "execution_reward REAL",
        "entry_decision_at TEXT",
        "entry_fill_kind TEXT",
        "pending_exit_at TEXT",
        "pending_exit_reason TEXT",
    ):
        _ensure_column(conn, "axiom_paper_positions_v20", definition)
    for definition in (
        "price_available INTEGER",
        "mark_kind TEXT",
        "execution_return_pct REAL",
        "training_eligible INTEGER DEFAULT 1",
        "action_probability REAL",
        "behavior_policy_version TEXT",
    ):
        _ensure_column(conn, "axiom_paper_marks_v20", definition)
    for definition in (
        "action_probability REAL",
        "rank_probability REAL",
        "exploration_probability REAL",
        "behavior_policy_version TEXT",
        "eligible_actions_json TEXT",
        "capital_constraint_state_json TEXT",
    ):
        _ensure_column(conn, "axiom_paper_candidates_v20", definition)
    conn.commit()


def _policy_champion_record(conn: sqlite3.Connection) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM axiom_policy_versions_v20 WHERE status='champion' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()


def _policy_last_training_closed(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT training_closed_trades FROM axiom_policy_versions_v20 ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return int(row[0]) if row else 0


def _closed_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM axiom_paper_positions_v20 WHERE status='closed'").fetchone()[0])


# ------------------------------ policy model ------------------------------

def _dict_frame(json_series: Iterable[Any], features: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    states = [_loads(x) for x in json_series]
    if features is None:
        features = sorted({k for d in states for k, v in d.items() if _finite_float(v) is not None})
    rows = []
    for d in states:
        rows.append({c: _finite_float(d.get(c)) for c in features})
    return pd.DataFrame(rows, columns=features, dtype=float), features


def _token_group_split(df: pd.DataFrame, time_col: str, allow_small: bool) -> tuple[np.ndarray, np.ndarray]:
    first = df.groupby("token_key")[time_col].min().sort_values()
    tokens = list(first.index)
    if len(tokens) < 5:
        order = np.argsort(pd.to_datetime(df[time_col], utc=True).to_numpy())
        cut = max(1, min(len(order) - 1, int(len(order) * 0.8)))
        return order[:cut], order[cut:]
    cut = max(1, min(len(tokens) - 1, int(len(tokens) * 0.8)))
    train_tokens = set(tokens[:cut])
    return (
        np.flatnonzero(df.token_key.isin(train_tokens).to_numpy()),
        np.flatnonzero(~df.token_key.isin(train_tokens).to_numpy()),
    )


def _fit_policy_regressor(df: pd.DataFrame, state_col: str, target: str, time_col: str, allow_small: bool) -> dict[str, Any] | None:
    _require_peak()
    data = df.copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data[data[target].notna() & np.isfinite(data[target])].copy()
    min_rows = 20 if allow_small else 80
    if len(data) < min_rows:
        return None
    X, features = _dict_frame(data[state_col])
    if not features:
        return None
    tr, va = _token_group_split(data, time_col, allow_small)
    if len(tr) < 5 or len(va) < 3:
        return None
    y = data[target].to_numpy(dtype=float)
    models = []
    scores = []
    n_estimators = 160 if allow_small else 500
    for name, model in peak._regressor_components(n_estimators, quantile=None):
        try:
            model.fit(X.iloc[tr], y[tr])
            pred = np.asarray(model.predict(X.iloc[va]), dtype=float)
            score = mean_absolute_error(y[va], pred)
            models.append((name, model))
            scores.append(max(float(score), 1e-9))
        except Exception:
            continue
    if not models:
        return None
    inv = 1.0 / np.asarray(scores)
    blend = (inv / inv.sum()).tolist()
    return {
        "kind": "regressor",
        "target": target,
        "time_col": time_col,
        "features": features,
        "models": models,
        "blend": blend,
        "validation_mae_components": scores,
        "rows": len(data),
        "validation_rows": len(va),
    }


def _predict_policy_head(head: dict[str, Any], states: list[dict[str, Any]]) -> np.ndarray:
    X, _ = _dict_frame(states, head["features"])
    if head.get("kind") == "distributional_return":
        ordered = []
        names = []
        for name, qh in sorted(head.get("quantile_heads", {}).items(), key=lambda kv: float(kv[1].get("quantile", 0.5))):
            comp = []
            for _, model in qh.get("models", []):
                try:
                    comp.append(np.asarray(model.predict(X), dtype=float))
                except Exception:
                    continue
            if comp:
                ordered.append(np.mean(comp, axis=0))
                names.append(name)
        if not ordered:
            return np.full(len(states), np.nan)
        mat = np.sort(np.column_stack(ordered), axis=1)
        values = {name: mat[:, j] for j, name in enumerate(names)}
        tail_parts = [values[x] for x in ("q05", "q10") if x in values]
        tail = np.mean(tail_parts, axis=0) if tail_parts else mat[:, 0]
        median = values.get("q50", mat[:, len(names) // 2])
        upper = values.get("q75", mat[:, min(len(names) - 1, len(names) // 2 + 1)])
        downside = np.maximum(0.0, -tail)
        return median + float(head.get("upside_weight", 0.20)) * np.maximum(0.0, upper - median) - float(head.get("tail_risk_weight", 0.75)) * downside
    preds = []
    for (_, model), weight in zip(head["models"], head["blend"]):
        preds.append(np.asarray(model.predict(X), dtype=float) * float(weight))
    return np.sum(preds, axis=0) if preds else np.full(len(states), np.nan)


def _build_hold_training(conn: sqlite3.Connection, config: PolicyConfig) -> pd.DataFrame:
    _require_peak()
    obs, _ = peak.load_observations(conn)
    if obs.empty:
        return pd.DataFrame()
    latest = obs.snapshot_at.max()
    marks = pd.read_sql_query(
        "SELECT mark_id, position_id, token_key, snapshot_at, market_cap_usd, state_json FROM axiom_paper_marks_v20 WHERE COALESCE(training_eligible,1)=1",
        conn,
    )
    if marks.empty:
        return marks
    marks["snapshot_at"] = pd.to_datetime(marks.snapshot_at, utc=True, errors="coerce")
    horizon = pd.Timedelta(minutes=config.hold_label_minutes)
    rows = []
    grouped = {k: g.sort_values("snapshot_at") for k, g in obs.groupby("token_key")}
    cost = config.friction_bps_round_trip / 10000.0
    for r in marks.itertuples(index=False):
        if pd.isna(r.snapshot_at) or r.token_key not in grouped:
            continue
        end = r.snapshot_at + horizon
        if latest < end:
            continue
        path = grouped[r.token_key]
        path = path[(path.snapshot_at > r.snapshot_at) & (path.snapshot_at <= end)]
        if path.empty:
            continue
        current = float(r.market_cap_usd)
        future = path.market_cap_usd.to_numpy(dtype=float)
        best = float(np.nanmax(future) / current - 1.0)
        worst = float(np.nanmin(future) / current - 1.0)
        terminal = float(future[-1] / current - 1.0)
        # This is a training objective for HOLD vs EXIT NOW. It is explicitly based on
        # realized future clipboard paths and is never available to the live policy.
        reward = 0.70 * best + 0.30 * terminal - config.drawdown_penalty * abs(min(0.0, worst)) - cost
        state = _loads(r.state_json)
        rows.append({
            "mark_id": r.mark_id,
            "token_key": str(r.token_key),
            "snapshot_at": r.snapshot_at,
            "label_ready_at": end,
            "state_json": _json(state),
            "hold_reward": reward,
            "future_best_return_pct": best,
            "future_worst_return_pct": worst,
            "future_terminal_return_pct": terminal,
        })
    return pd.DataFrame(rows)


def _entry_training(conn: sqlite3.Connection) -> pd.DataFrame:
    _backfill_paper_execution_accounting(conn)
    df = pd.read_sql_query(
        """
        SELECT position_id, token_key, opened_at, closed_at, entry_state_json AS state_json,
               reward, net_return_pct, mae_pct, mfe_pct, exploration
        FROM axiom_paper_positions_v20
        WHERE status='closed' AND reward IS NOT NULL
        ORDER BY opened_at
        """,
        conn,
    )
    if not df.empty:
        df["opened_at"] = pd.to_datetime(df.opened_at, utc=True, errors="coerce")
        df["closed_at"] = pd.to_datetime(df.closed_at, utc=True, errors="coerce")
    return df


def _evaluate_policy_head(head: dict[str, Any] | None, df: pd.DataFrame, state_col: str, target: str, time_col: str) -> dict[str, Any]:
    if head is None or df.empty:
        return {"available": False}
    data = df[df[target].notna()].copy()
    if len(data) < 4:
        return {"available": False}
    _, va = _token_group_split(data, time_col, allow_small=True)
    if len(va) < 2:
        return {"available": False}
    actual = data.iloc[va][target].to_numpy(dtype=float)
    states = [_loads(x) for x in data.iloc[va][state_col]]
    pred = _predict_policy_head(head, states)
    mae = float(mean_absolute_error(actual, pred))
    direction = float(np.mean((pred > 0) == (actual > 0)))
    q = max(1, int(math.ceil(len(pred) * 0.25)))
    top_idx = np.argsort(pred)[-q:]
    top_reward = float(np.mean(actual[top_idx]))
    overall_reward = float(np.mean(actual))
    return {
        "available": True,
        "rows": int(len(actual)),
        "mae": mae,
        "direction_accuracy": direction,
        "top_quartile_actual_reward": top_reward,
        "overall_actual_reward": overall_reward,
    }


def _load_policy(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    return joblib.load(p) if p.exists() else None


def _policy_champion_path(conn: sqlite3.Connection) -> tuple[str | None, str | None]:
    row = _policy_champion_record(conn)
    if row is None:
        return None, None
    return row["version_id"], row["model_path"]


def _continue_policy_head(
    head: dict[str, Any],
    df: pd.DataFrame,
    state_col: str,
    target: str,
    time_col: str,
    ready_col: str,
    watermark: pd.Timestamp,
    *,
    append_estimators: int,
    replay_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = df.copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data[data[target].notna() & np.isfinite(data[target])].copy()
    if data.empty or ready_col not in data:
        return head, {"new_rows": 0, "reason": "no ready policy labels"}
    ready = pd.to_datetime(data[ready_col], errors="coerce", utc=True)
    new = data[ready.notna() & (ready > watermark)].copy()
    if new.empty:
        return head, {"new_rows": 0, "reason": "no new policy labels after watermark"}
    _, va = _token_group_split(data, time_col, allow_small=True)
    val_tokens = set(data.iloc[va].token_key.astype(str)) if len(va) else set()
    new_train = new[~new.token_key.astype(str).isin(val_tokens)].copy()
    if new_train.empty:
        return head, {"new_rows": int(len(new)), "trained_new_rows": 0, "reason": "all new policy rows held out"}
    old_pool = data[(ready <= watermark) & (~data.token_key.astype(str).isin(val_tokens))].copy()
    replay = old_pool.sort_values(time_col).tail(replay_rows)
    batch = pd.concat([new_train, replay], ignore_index=False).drop_duplicates(subset=["token_key", time_col], keep="last")
    X, _ = _dict_frame(batch[state_col], head["features"])
    y = batch[target].to_numpy(dtype=float)
    w = np.ones(len(batch), dtype=float)
    val = data[data.token_key.astype(str).isin(val_tokens)].copy()
    if val.empty:
        val = data.tail(max(3, min(100, len(data))))
    Xv, _ = _dict_frame(val[state_col], head["features"])
    yv = val[target].to_numpy(dtype=float)
    models, scores = [], []
    for name, old_model in head.get("models", []):
        try:
            model = peak._continue_component(name, old_model, X, y, w, append_estimators)
            pred = np.asarray(model.predict(Xv), dtype=float)
            score = mean_absolute_error(yv, pred)
            models.append((name, model))
            scores.append(max(float(score), 1e-9))
        except Exception:
            continue
    if not models:
        return head, {"new_rows": int(len(new)), "trained_new_rows": 0, "reason": "policy continuation failed"}
    inv = 1.0 / np.asarray(scores)
    updated = dict(head)
    updated.update({
        "models": models,
        "blend": (inv / inv.sum()).tolist(),
        "validation_mae_components": scores,
        "rows": int(head.get("rows", 0)) + int(len(new_train)),
        "incremental_updates": int(head.get("incremental_updates", 0)) + 1,
    })
    return updated, {
        "new_rows": int(len(new)), "trained_new_rows": int(len(new_train)),
        "replay_rows": int(len(replay)), "validation_rows": int(len(val)), "scores": scores,
    }


def train_policy(db: str, policy_dir: str, config: PolicyConfig, allow_small: bool = False) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        migrate(conn)
        entry = _entry_training(conn)
        hold = _build_hold_training(conn, config)
        champion_id, champion_path = _policy_champion_path(conn)
        champion = _load_policy(champion_path)
        compatible = champion is not None and str(champion.get("schema_version", "")) == SCHEMA_VERSION

        incremental_meta: dict[str, Any] = {}
        if compatible:
            created = _to_ts(champion.get("created_at", _now_iso()))
            entry_wm = _to_ts(champion.get("entry_watermark", created))
            hold_wm = _to_ts(champion.get("hold_watermark", created))
            entry_head = champion.get("entry_head")
            hold_head = champion.get("hold_head")
            if entry_head is not None:
                entry_head, incremental_meta["entry"] = _continue_policy_head(
                    entry_head, entry, "state_json", "reward", "opened_at", "closed_at", entry_wm,
                    append_estimators=10 if allow_small else 25,
                    replay_rows=100 if allow_small else 500,
                )
            if hold_head is not None and not hold.empty:
                hold_head, incremental_meta["hold"] = _continue_policy_head(
                    hold_head, hold, "state_json", "hold_reward", "snapshot_at", "label_ready_at", hold_wm,
                    append_estimators=10 if allow_small else 25,
                    replay_rows=100 if allow_small else 500,
                )
            trained_new = sum(int(v.get("trained_new_rows", 0)) for v in incremental_meta.values())
            if trained_new == 0:
                return {"trained": False, "reason": "no new policy experience eligible for append", "details": incremental_meta}
            training_mode = "incremental_append"
        else:
            entry_head = _fit_policy_regressor(entry, "state_json", "reward", "opened_at", allow_small)
            hold_head = _fit_policy_regressor(hold, "state_json", "hold_reward", "snapshot_at", allow_small) if not hold.empty else None
            if entry_head is None and hold_head is None:
                raise RuntimeError("Not enough mature paper experience to train a policy yet.")
            training_mode = "full_bootstrap_once"

        version_id = datetime.now(timezone.utc).strftime("policy_%Y%m%dT%H%M%S_%fZ")
        out_dir = Path(policy_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = out_dir / f"{version_id}.joblib"
        entry_wm_new = pd.to_datetime(entry.get("closed_at"), errors="coerce", utc=True).max() if not entry.empty else pd.NaT
        hold_wm_new = pd.to_datetime(hold.get("label_ready_at"), errors="coerce", utc=True).max() if not hold.empty else pd.NaT
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _now_iso(),
            "config": asdict(config),
            "entry_head": entry_head,
            "hold_head": hold_head,
            "training_mode": training_mode,
            "parent_policy": champion_path if compatible else None,
            "entry_watermark": entry_wm_new.isoformat() if pd.notna(entry_wm_new) else _now_iso(),
            "hold_watermark": hold_wm_new.isoformat() if pd.notna(hold_wm_new) else _now_iso(),
            "incremental_rounds": int(champion.get("incremental_rounds", 0)) + 1 if compatible else 0,
            "incremental_details": incremental_meta,
        }
        joblib.dump(bundle, candidate_path)

        candidate_metrics = {
            "entry": _evaluate_policy_head(entry_head, entry, "state_json", "reward", "opened_at"),
            "hold": _evaluate_policy_head(hold_head, hold, "state_json", "hold_reward", "snapshot_at"),
        }
        champion_metrics = {
            "entry": _evaluate_policy_head(champion.get("entry_head") if champion else None, entry, "state_json", "reward", "opened_at"),
            "hold": _evaluate_policy_head(champion.get("hold_head") if champion else None, hold, "state_json", "hold_reward", "snapshot_at"),
        }

        promoted = False
        reason = ""
        cand_entry = candidate_metrics["entry"]
        champ_entry = champion_metrics["entry"]
        if champion is None or not compatible:
            promoted = True
            reason = "first compatible V21 policy champion"
        elif cand_entry.get("available") and champ_entry.get("available"):
            improvement = cand_entry["top_quartile_actual_reward"] - champ_entry["top_quartile_actual_reward"]
            direction_ok = cand_entry["direction_accuracy"] >= champ_entry["direction_accuracy"] - 0.03
            threshold = config.policy_promotion_margin * max(0.01, abs(champ_entry["top_quartile_actual_reward"]))
            promoted = improvement >= threshold and direction_ok
            reason = f"entry_top_quartile_delta={improvement:.6f}; direction_ok={direction_ok}"
        else:
            reason = "insufficient common policy validation experience"

        if promoted:
            conn.execute("UPDATE axiom_policy_versions_v20 SET status='retired' WHERE status='champion'")
            champion_file = Path(policy_dir) / "champion.joblib"
            shutil.copy2(candidate_path, champion_file)
            model_path = str(candidate_path)
            status_name = "champion"
        else:
            model_path = str(candidate_path)
            status_name = "challenger_rejected"

        metrics = {"candidate": candidate_metrics, "champion_before": champion_metrics, "promotion_reason": reason, "training_mode": training_mode}
        conn.execute(
            """
            INSERT INTO axiom_policy_versions_v20
            (version_id, created_at, model_path, model_hash, status, metrics_json,
             training_closed_trades, training_hold_samples, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id, _now_iso(), model_path, _hash_file(model_path), status_name, _json(metrics),
                int(len(entry)), int(len(hold)), reason,
            ),
        )
        conn.commit()
    return {
        "trained": True,
        "training_mode": training_mode,
        "version_id": version_id,
        "promoted": promoted,
        "status": status_name,
        "candidate_path": str(candidate_path),
        "active_path": model_path if promoted else champion_path,
        "closed_trades": len(entry),
        "hold_samples": len(hold),
        "incremental_details": incremental_meta,
        "metrics": metrics,
    }


# ------------------------------ paper broker ------------------------------

def _is_unavailable_reason(reason: str | None) -> bool:
    text = str(reason or "").lower()
    return any(k in text for k in ("absence", "missing", "disappear", "dead_after", "unavailable"))


def _execution_proxy_mc(entry_mc: float, observed_mc: float, config: PolicyConfig, *, unavailable: bool) -> float:
    observed = max(0.0, float(observed_mc))
    entry = max(0.0, float(entry_mc))
    if not unavailable or observed <= entry:
        return observed
    frac = min(1.0, max(0.0, float(config.disappearance_profit_recognition_fraction)))
    return entry + frac * (observed - entry)


def _trade_outcome(entry: float, exit_mc: float, mfe: float, mae: float, config: PolicyConfig) -> dict[str, float]:
    gross = float(exit_mc) / float(entry) - 1.0
    net = gross - config.friction_bps_round_trip / 10000.0
    peak_capture = (net / mfe) if mfe > 1e-9 else 0.0
    peak_capture = float(max(-1.0, min(1.5, peak_capture)))
    reward = net - config.drawdown_penalty * abs(min(0.0, mae)) + config.peak_capture_weight * max(0.0, peak_capture)
    return {"gross": gross, "net": net, "peak_capture": peak_capture, "reward": reward}


def _config_from_position(pos: sqlite3.Row) -> PolicyConfig:
    raw = _loads(pos["config_json"])
    defaults = asdict(PolicyConfig())
    defaults.update({k: raw[k] for k in defaults if k in raw})
    return PolicyConfig(**defaults)


def _backfill_paper_execution_accounting(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT * FROM axiom_paper_positions_v20
        WHERE status='closed' AND execution_reward IS NULL
        """
    ).fetchall()
    for pos in rows:
        # Synthetic/legacy rows without a close reason may not have enough audit
        # information to reconstruct the old reward formula exactly. Preserve their
        # existing targets by copying them into both tracks.
        if not pos["close_reason"]:
            gross = _finite_float(pos["gross_return_pct"])
            if gross is None and pos["exit_mc"] is not None:
                gross = float(pos["exit_mc"]) / float(pos["entry_mc"]) - 1.0
            gross = float(gross or 0.0)
            net = _finite_float(pos["net_return_pct"])
            net = float(net if net is not None else gross)
            cap = _finite_float(pos["peak_capture_ratio"])
            cap = float(cap or 0.0)
            reward = _finite_float(pos["reward"])
            reward = float(reward if reward is not None else net)
            observed_mc = float(pos["exit_mc"] if pos["exit_mc"] is not None else pos["last_mc"])
            conn.execute(
                """
                UPDATE axiom_paper_positions_v20
                SET exit_kind='legacy_observed_exit', price_available_at_exit=1,
                    exit_mc_observed=?, exit_mc_execution_proxy=?,
                    observed_gross_return_pct=?, execution_gross_return_pct=?,
                    observed_net_return_pct=?, execution_net_return_pct=?,
                    observed_peak_capture_ratio=?, execution_peak_capture_ratio=?,
                    observed_reward=?, execution_reward=?
                WHERE position_id=?
                """,
                (observed_mc, observed_mc, gross, gross, net, net, cap, cap, reward, reward, pos["position_id"]),
            )
            continue

        config = _config_from_position(pos)
        entry = float(pos["entry_mc"])
        observed_mc = float(pos["exit_mc"] if pos["exit_mc"] is not None else pos["last_mc"])
        unavailable = _is_unavailable_reason(pos["close_reason"])
        execution_mc = _execution_proxy_mc(entry, observed_mc, config, unavailable=unavailable)
        mfe = float(pos["mfe_pct"] or 0.0)
        mae = float(pos["mae_pct"] or 0.0)
        observed = _trade_outcome(entry, observed_mc, mfe, mae, config)
        execution = _trade_outcome(entry, execution_mc, mfe, mae, config)
        kind = "disappearance_terminal" if unavailable else "model_exit_observed_price"
        conn.execute(
            """
            UPDATE axiom_paper_positions_v20
            SET exit_kind=?, price_available_at_exit=?, exit_mc_observed=?, exit_mc_execution_proxy=?,
                observed_gross_return_pct=?, execution_gross_return_pct=?,
                observed_net_return_pct=?, execution_net_return_pct=?,
                observed_peak_capture_ratio=?, execution_peak_capture_ratio=?,
                observed_reward=?, execution_reward=?,
                gross_return_pct=?, net_return_pct=?, peak_capture_ratio=?, reward=?
            WHERE position_id=?
            """,
            (
                kind, int(not unavailable), observed_mc, execution_mc, observed["gross"], execution["gross"],
                observed["net"], execution["net"], observed["peak_capture"], execution["peak_capture"],
                observed["reward"], execution["reward"], execution["gross"], execution["net"],
                execution["peak_capture"], execution["reward"], pos["position_id"],
            ),
        )
    conn.commit()


def _close_position(
    conn: sqlite3.Connection, pos: sqlite3.Row, closed_at: pd.Timestamp, exit_mc: float,
    reason: str, config: PolicyConfig, *, price_available: bool
) -> dict[str, Any]:
    entry = float(pos["entry_mc"])
    observed_mc = float(exit_mc)
    unavailable = not price_available or _is_unavailable_reason(reason)
    execution_mc = _execution_proxy_mc(entry, observed_mc, config, unavailable=unavailable)
    mfe = float(pos["mfe_pct"])
    mae = float(pos["mae_pct"])
    observed = _trade_outcome(entry, observed_mc, mfe, mae, config)
    execution = _trade_outcome(entry, execution_mc, mfe, mae, config)
    exit_kind = "disappearance_terminal" if unavailable else ("max_hold_observed_price" if reason == "max_hold" else "model_exit_observed_price")
    # Compatibility/learning aliases now intentionally point to the execution
    # track, so unavailable last-observed gains cannot teach the active policy.
    conn.execute(
        """
        UPDATE axiom_paper_positions_v20
        SET status='closed', closed_at=?, exit_mc=?, close_reason=?, gross_return_pct=?,
            net_return_pct=?, peak_capture_ratio=?, reward=?, exit_kind=?, price_available_at_exit=?,
            exit_mc_observed=?, exit_mc_execution_proxy=?, observed_gross_return_pct=?,
            execution_gross_return_pct=?, observed_net_return_pct=?, execution_net_return_pct=?,
            observed_peak_capture_ratio=?, execution_peak_capture_ratio=?, observed_reward=?, execution_reward=?
        WHERE position_id=?
        """,
        (
            closed_at.isoformat(), observed_mc, reason, execution["gross"], execution["net"],
            execution["peak_capture"], execution["reward"], exit_kind, int(price_available),
            observed_mc, execution_mc, observed["gross"], execution["gross"], observed["net"],
            execution["net"], observed["peak_capture"], execution["peak_capture"],
            observed["reward"], execution["reward"], pos["position_id"],
        ),
    )
    return {
        "position_id": pos["position_id"], "token_key": pos["token_key"], "reason": reason,
        "exit_kind": exit_kind, "price_available_at_exit": bool(price_available),
        "exit_mc_observed": observed_mc, "exit_mc_execution_proxy": execution_mc,
        "observed_net_return_pct": observed["net"], "execution_net_return_pct": execution["net"],
        "observed_reward": observed["reward"], "execution_reward": execution["reward"],
        "gross_return_pct": execution["gross"], "net_return_pct": execution["net"], "reward": execution["reward"],
    }

def _last_closed_at(conn: sqlite3.Connection, token_key: str) -> pd.Timestamp | None:
    row = conn.execute(
        "SELECT MAX(closed_at) FROM axiom_paper_positions_v20 WHERE token_key=? AND status='closed'",
        (token_key,),
    ).fetchone()
    return _to_ts(row[0]) if row and row[0] else None


def _make_mark_state(pred_state: dict[str, float], pos: sqlite3.Row, current_mc: float, snapshot: pd.Timestamp) -> dict[str, float]:
    entry = float(pos["entry_mc"])
    opened = _to_ts(pos["opened_at"])
    current_return = current_mc / entry - 1.0
    extra = {
        "position_return_pct": current_return,
        "position_minutes_held": max(0.0, (snapshot - opened).total_seconds() / 60.0),
        "position_mfe_pct": max(float(pos["mfe_pct"]), current_return),
        "position_mae_pct": min(float(pos["mae_pct"]), current_return),
    }
    out = dict(pred_state)
    out.update(extra)
    return out


def paper_cycle(
    db: str, predictions_path: str, policy_dir: str, config: PolicyConfig, *,
    allow_stale_predictions: bool = False, auto_train: bool = True, allow_small_policy: bool = False,
) -> dict[str, Any]:
    """Versioned execution dispatch. V24 forecasts use next-observation fills; legacy forecasts retain their historical semantics."""
    try:
        probe=_load_predictions(predictions_path)
        is_v24=bool("v24_model_hash" in probe.columns and probe["v24_model_hash"].notna().any())
    except Exception:
        is_v24=False
    fn=_paper_cycle_v24 if is_v24 else _paper_cycle_legacy
    return fn(db,predictions_path,policy_dir,config,allow_stale_predictions=allow_stale_predictions,auto_train=auto_train,allow_small_policy=allow_small_policy)


def _paper_cycle_legacy(
    db: str,
    predictions_path: str,
    policy_dir: str,
    config: PolicyConfig,
    *,
    allow_stale_predictions: bool = False,
    auto_train: bool = True,
    allow_small_policy: bool = False,
) -> dict[str, Any]:
    _require_peak()
    predictions = _load_predictions(predictions_path)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        _backfill_paper_execution_accounting(conn)
        observations, _ = peak.load_observations(conn)
        if observations.empty:
            raise RuntimeError("No Axiom observations are available.")
        snapshot = observations.snapshot_at.max()
        current_obs = observations[observations.snapshot_at == snapshot].copy()
        current_obs = current_obs.sort_values("token_key").drop_duplicates("token_key", keep="last")
        current_obs = current_obs[["token_key", "market_cap_usd"] + (["name"] if "name" in current_obs.columns else [])]

        pred_snapshot = _prediction_snapshot(predictions)
        if pred_snapshot is not None and not allow_stale_predictions:
            if abs((pred_snapshot - snapshot).total_seconds()) > 180:
                raise RuntimeError(
                    f"Prediction CSV is stale relative to the latest clipboard capture: predictions={pred_snapshot}, capture={snapshot}."
                )
        current = current_obs.merge(predictions, on="token_key", how="left", suffixes=("", "__pred"))
        if "market_cap_usd__pred" in current.columns:
            current = current.drop(columns=["market_cap_usd__pred"])
        current = current[current["market_cap_usd"].notna()].copy()
        current_map = {str(r.token_key): r for r in current.itertuples(index=False)}
        current_series = {str(r["token_key"]): r for _, r in current.iterrows()}

        policy_id, policy_path = _policy_champion_path(conn)
        loaded_policy_bundle = _load_policy(policy_path)
        policy_accounting_incompatible = bool(
            loaded_policy_bundle is not None
            and str(loaded_policy_bundle.get("schema_version", "")) != SCHEMA_VERSION
        )
        # Do not use a policy trained under the old last-observed disappearance
        # reward. Fall back to deterministic bootstrap decisions until the one-time
        # policy-only accounting rebootstrap is complete.
        policy_bundle = None if policy_accounting_incompatible else loaded_policy_bundle
        if policy_accounting_incompatible:
            policy_id = "bootstrap_pending_execution_accounting_rebootstrap"
        forecast_hash = _hash_file(PEAK_CHAMPION_DEFAULT) or _hash_file(predictions_path)
        open_positions = conn.execute(
            "SELECT * FROM axiom_paper_positions_v20 WHERE status='open' ORDER BY opened_at"
        ).fetchall()
        open_before = len(open_positions)
        exits: list[dict[str, Any]] = []
        open_tokens = {str(p["token_key"]) for p in open_positions}

        # First mark/manage existing positions.
        for pos in open_positions:
            token = str(pos["token_key"])
            row = current_series.get(token)
            if row is None:
                misses = int(pos["missed_cycles"]) + 1
                conn.execute("UPDATE axiom_paper_positions_v20 SET missed_cycles=? WHERE position_id=?", (misses, pos["position_id"]))
                absent_minutes = max(0.0, (snapshot - _to_ts(pos["last_seen_at"])).total_seconds() / 60.0)
                observed_mc = float(pos["last_mc"])
                execution_mc = _execution_proxy_mc(float(pos["entry_mc"]), observed_mc, config, unavailable=True)
                observed_return = observed_mc / float(pos["entry_mc"]) - 1.0
                execution_return = execution_mc / float(pos["entry_mc"]) - 1.0
                terminal = absent_minutes >= config.missing_close_minutes
                conn.execute(
                    """
                    INSERT OR REPLACE INTO axiom_paper_marks_v20
                    (mark_id, position_id, token_key, snapshot_at, market_cap_usd, return_pct,
                     mfe_pct, mae_pct, state_json, action, action_value, policy_version,
                     price_available, mark_kind, execution_return_pct, training_eligible)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 0, ?, ?, 0)
                    """,
                    (
                        str(uuid.uuid4()), pos["position_id"], token, snapshot.isoformat(), observed_mc,
                        observed_return, float(pos["mfe_pct"]), float(pos["mae_pct"]),
                        _json({"missing_minutes": absent_minutes, "last_observed_mc": observed_mc}),
                        "DISAPPEARANCE_CLOSE" if terminal else "MISSING_HOLD", policy_id,
                        "disappearance_terminal" if terminal else "stale_last_observed", execution_return,
                    ),
                )
                if terminal:
                    exits.append(_close_position(
                        conn, pos, snapshot, observed_mc, "dead_after_50m_absence", config, price_available=False
                    ))
                    open_tokens.discard(token)
                continue

            mc = float(row["market_cap_usd"])
            pred_state = _safe_prediction_state(row)
            mark_state = _make_mark_state(pred_state, pos, mc, snapshot)
            current_return = mc / float(pos["entry_mc"]) - 1.0
            mfe = max(float(pos["mfe_pct"]), current_return)
            mae = min(float(pos["mae_pct"]), current_return)
            conn.execute(
                """
                UPDATE axiom_paper_positions_v20
                SET last_seen_at=?, last_mc=?, missed_cycles=0, mfe_pct=?, mae_pct=?
                WHERE position_id=?
                """,
                (snapshot.isoformat(), mc, mfe, mae, pos["position_id"]),
            )

            held = (snapshot - _to_ts(pos["opened_at"])).total_seconds() / 60.0
            if policy_bundle and policy_bundle.get("hold_head"):
                hold_value = float(_predict_policy_head(policy_bundle["hold_head"], [mark_state])[0])
                policy_kind = "learned"
            else:
                hold_value = _bootstrap_hold_value(mark_state)
                policy_kind = "bootstrap"
            action = "HOLD"
            close_reason = None
            if held >= config.max_hold_minutes:
                action = "EXIT"
                close_reason = "max_hold"
            elif held >= config.min_hold_minutes and hold_value <= 0.0:
                action = "EXIT"
                close_reason = f"{policy_kind}_hold_value_nonpositive"

            conn.execute(
                """
                INSERT OR REPLACE INTO axiom_paper_marks_v20
                (mark_id, position_id, token_key, snapshot_at, market_cap_usd, return_pct,
                 mfe_pct, mae_pct, state_json, action, action_value, policy_version,
                 price_available, mark_kind, execution_return_pct, training_eligible)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'observed', ?, 1)
                """,
                (
                    str(uuid.uuid4()), pos["position_id"], token, snapshot.isoformat(), mc, current_return,
                    mfe, mae, _json(mark_state), action, hold_value, policy_id, current_return,
                ),
            )
            if action == "EXIT":
                fresh = conn.execute("SELECT * FROM axiom_paper_positions_v20 WHERE position_id=?", (pos["position_id"],)).fetchone()
                exits.append(_close_position(
                    conn, fresh, snapshot, mc, close_reason or "policy_exit", config, price_available=True
                ))
                open_tokens.discard(token)

        # Candidate scoring is logged for every currently visible predicted token.
        candidate_rows: list[dict[str, Any]] = []
        for _, row in current.iterrows():
            token = str(row["token_key"])
            state = _safe_prediction_state(row)
            if not state:
                continue
            bootstrap = _bootstrap_entry_score(state)
            learned = None
            if policy_bundle and policy_bundle.get("entry_head"):
                learned = float(_predict_policy_head(policy_bundle["entry_head"], [state])[0])
            score = learned if learned is not None else bootstrap
            candidate_rows.append({
                "token_key": token,
                "market_cap_usd": float(row["market_cap_usd"]),
                "state": state,
                "bootstrap_score": bootstrap,
                "policy_score": learned,
                "rank_score": score,
            })

        candidate_rows.sort(key=lambda r: r["rank_score"], reverse=True)
        slots = max(0, config.max_open_positions - len(open_tokens))
        eligible = []
        for c in candidate_rows:
            if c["token_key"] in open_tokens:
                continue
            last_close = _last_closed_at(conn, c["token_key"])
            if last_close is not None and (snapshot - last_close).total_seconds() < config.reentry_cooldown_minutes * 60:
                continue
            eligible.append(c)

        selected: list[tuple[dict[str, Any], bool]] = []
        if slots > 0 and eligible:
            exploit_n = min(slots, len(eligible))
            # Deterministic pseudo-randomness per capture makes the exploration audit reproducible.
            seed = int(hashlib.sha256(snapshot.isoformat().encode()).hexdigest()[:12], 16)
            rng = random.Random(seed)
            explore_slots = 1 if exploit_n > 0 and config.exploration_fraction > 0 and rng.random() < config.exploration_fraction else 0
            exploit_slots = max(0, exploit_n - explore_slots)
            selected.extend((c, False) for c in eligible[:exploit_slots])
            remaining = eligible[exploit_slots:max(exploit_slots + config.candidate_pool, exploit_slots + 1)]
            if explore_slots and remaining:
                selected.append((rng.choice(remaining), True))

        selected_keys = {c["token_key"] for c, _ in selected}
        exploration_keys = {c["token_key"] for c, ex in selected if ex}
        for c in candidate_rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO axiom_paper_candidates_v20
                (snapshot_at, token_key, market_cap_usd, state_json, bootstrap_score, policy_score,
                 chosen, exploration, forecast_hash, policy_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.isoformat(), c["token_key"], c["market_cap_usd"], _json(c["state"]),
                    c["bootstrap_score"], c["policy_score"], int(c["token_key"] in selected_keys),
                    int(c["token_key"] in exploration_keys), forecast_hash, policy_id,
                ),
            )

        entries = []
        for c, exploration in selected:
            position_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO axiom_paper_positions_v20
                (position_id, token_key, opened_at, entry_mc, entry_state_json, entry_policy_version,
                 entry_forecast_hash, exploration, status, last_seen_at, last_mc, missed_cycles,
                 mfe_pct, mae_pct, config_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, 0, 0, 0, ?)
                """,
                (
                    position_id, c["token_key"], snapshot.isoformat(), c["market_cap_usd"], _json(c["state"]),
                    policy_id, forecast_hash, int(exploration), snapshot.isoformat(), c["market_cap_usd"], _json(asdict(config)),
                ),
            )
            entries.append({"position_id": position_id, "token_key": c["token_key"], "entry_mc": c["market_cap_usd"], "exploration": exploration})
            open_tokens.add(c["token_key"])

        run_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO axiom_self_teach_runs_v20
            (run_id, run_at, snapshot_at, predictions_path, forecast_model_hash, policy_version,
             current_tokens, open_before, entries, exits, open_after, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, _now_iso(), snapshot.isoformat(), predictions_path, forecast_hash, policy_id,
                len(current), open_before, len(entries), len(exits), len(open_tokens),
                _json({"config": asdict(config)}),
            ),
        )
        conn.commit()
        closed_count = _closed_count(conn)
        last_train_closed = _policy_last_training_closed(conn)

    training = None
    if auto_train and closed_count >= (20 if allow_small_policy else config.policy_min_closed):
        normal_gate = closed_count - last_train_closed >= (5 if allow_small_policy else config.policy_retrain_every_closed)
        if normal_gate or policy_accounting_incompatible:
            try:
                training = train_policy(db, policy_dir, config, allow_small=allow_small_policy)
            except Exception as exc:
                training = {"error": str(exc)}

    return {
        "run_id": run_id,
        "snapshot_at": snapshot.isoformat(),
        "current_tokens": len(current),
        "open_before": open_before,
        "entries": entries,
        "exits": exits,
        "open_after": len(open_tokens),
        "active_policy": policy_id or "bootstrap",
        "closed_paper_trades": closed_count,
        "policy_training": training,
    }


def _paper_cycle_v24(
    db: str,
    predictions_path: str,
    policy_dir: str,
    config: PolicyConfig,
    *,
    allow_stale_predictions: bool = False,
    auto_train: bool = True,
    allow_small_policy: bool = False,
) -> dict[str, Any]:
    """Run one paper-policy decision cycle.

    V24 execution semantics are intentionally delayed by one observation:
    a decision made from snapshot T can only fill at the first token price
    observed strictly after T.  Missing-token terminal handling is based on
    successful capture heartbeats rather than elapsed wall time when V24 is
    active, so collector outages cannot become artificial token deaths.
    """
    _require_peak()
    predictions = _load_predictions(predictions_path)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        _backfill_paper_execution_accounting(conn)
        observations, _ = peak.load_observations(conn)
        if observations.empty:
            raise RuntimeError("No Axiom observations are available.")
        snapshot = observations.snapshot_at.max()
        current_obs = observations[observations.snapshot_at == snapshot].copy()
        current_obs = current_obs.sort_values("token_key").drop_duplicates("token_key", keep="last")
        current_obs = current_obs[["token_key", "market_cap_usd"] + (["name"] if "name" in current_obs.columns else [])]

        pred_snapshot = _prediction_snapshot(predictions)
        if pred_snapshot is not None and not allow_stale_predictions:
            if abs((pred_snapshot - snapshot).total_seconds()) > 180:
                raise RuntimeError(
                    f"Prediction CSV is stale relative to the latest clipboard capture: predictions={pred_snapshot}, capture={snapshot}."
                )
        current = current_obs.merge(predictions, on="token_key", how="left", suffixes=("", "__pred"))
        if "market_cap_usd__pred" in current.columns:
            current = current.drop(columns=["market_cap_usd__pred"])
        current = current[current["market_cap_usd"].notna()].copy()
        current_series = {str(r["token_key"]): r for _, r in current.iterrows()}

        is_v24_forecast = bool("v24_model_hash" in predictions.columns and predictions["v24_model_hash"].notna().any())
        is_v23_forecast = bool("v23_model_hash" in predictions.columns and predictions["v23_model_hash"].notna().any())
        v24mod = None
        if is_v24_forecast:
            try:
                from . import axiom_v24 as v24mod
                v24mod.migrate(conn)
                v24mod.record_capture_heartbeat(
                    conn, snapshot, valid_capture=True, row_count=len(current_obs), source="paper_cycle_v24"
                )
            except Exception:
                v24mod = None

        policy_id, policy_path = _policy_champion_path(conn)
        loaded_policy_bundle = _load_policy(policy_path)
        policy_accounting_incompatible = bool(
            loaded_policy_bundle is not None and str(loaded_policy_bundle.get("schema_version", "")) != SCHEMA_VERSION
        )
        policy_generation_incompatible = False
        if is_v24_forecast:
            policy_generation_incompatible = bool(
                loaded_policy_bundle is None
                or not str(loaded_policy_bundle.get("v24_policy_schema", "")).startswith("v24_")
                or not bool(loaded_policy_bundle.get("oos_only", False))
            )
        elif is_v23_forecast:
            policy_generation_incompatible = bool(
                loaded_policy_bundle is None
                or not str(loaded_policy_bundle.get("v23_policy_schema", "")).startswith("v23_")
                or not bool(loaded_policy_bundle.get("oos_only", False))
            )
        policy_bundle = None if (policy_accounting_incompatible or policy_generation_incompatible) else loaded_policy_bundle
        if policy_accounting_incompatible:
            policy_id = "bootstrap_pending_execution_accounting_rebootstrap"
        elif policy_generation_incompatible:
            policy_id = "bootstrap_pending_v24_oos_policy" if is_v24_forecast else "bootstrap_pending_v23_oos_distributional_policy"

        if is_v24_forecast:
            forecast_hash = str(predictions.loc[predictions["v24_model_hash"].notna(), "v24_model_hash"].iloc[-1])
        elif is_v23_forecast:
            forecast_hash = str(predictions.loc[predictions["v23_model_hash"].notna(), "v23_model_hash"].iloc[-1])
        else:
            forecast_hash = _hash_file(PEAK_CHAMPION_DEFAULT) or _hash_file(predictions_path)

        def terminal_absence(last_seen: pd.Timestamp) -> tuple[bool, float, int]:
            elapsed = max(0.0, (snapshot - last_seen).total_seconds() / 60.0)
            if v24mod is not None:
                run_start, run_end, n = v24mod._contiguous_capture_absence(conn, last_seen, v24mod.V24Config(), upto=snapshot)
                if run_start is None or run_end is None:
                    return False, elapsed, 0
                valid_minutes = max(0.0, (run_end - last_seen).total_seconds() / 60.0)
                return bool(valid_minutes >= config.missing_close_minutes and n >= v24mod.V24Config().heartbeat_min_valid_captures_for_death), valid_minutes, n
            return elapsed >= config.missing_close_minutes, elapsed, 0

        filled_entries: list[dict[str, Any]] = []
        cancelled_pending: list[dict[str, Any]] = []
        # Fill ENTRY decisions only on the first later observation of that token.
        pending = conn.execute(
            "SELECT * FROM axiom_paper_pending_entries_v24 WHERE status='pending' ORDER BY decision_at"
        ).fetchall()
        for pen in pending:
            token = str(pen["token_key"])
            decision_at = _to_ts(pen["decision_at"])
            if snapshot <= decision_at:
                continue
            row = current_series.get(token)
            if row is None:
                terminal, absent_minutes, _ = terminal_absence(decision_at)
                if terminal:
                    conn.execute(
                        "UPDATE axiom_paper_pending_entries_v24 SET status='cancelled',cancelled_at=?,cancel_reason=? WHERE pending_id=?",
                        (snapshot.isoformat(), "unavailable_before_next_observable_fill", pen["pending_id"]),
                    )
                    cancelled_pending.append({"token_key": token, "reason": "unavailable_before_next_observable_fill", "missing_minutes": absent_minutes})
                continue
            fill_mc = float(row["market_cap_usd"])
            position_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO axiom_paper_positions_v20
                (position_id, token_key, opened_at, entry_mc, entry_state_json, entry_policy_version,
                 entry_forecast_hash, exploration, status, last_seen_at, last_mc, missed_cycles,
                 mfe_pct, mae_pct, config_json, entry_decision_at, entry_fill_kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, 0, 0, 0, ?, ?, 'next_observable')
                """,
                (
                    position_id, token, snapshot.isoformat(), fill_mc, pen["decision_state_json"], pen["policy_version"],
                    pen["forecast_hash"], int(pen["exploration"]), snapshot.isoformat(), fill_mc,
                    _json(asdict(config)), pen["decision_at"],
                ),
            )
            conn.execute(
                "UPDATE axiom_paper_pending_entries_v24 SET status='filled',filled_at=?,fill_mc=? WHERE pending_id=?",
                (snapshot.isoformat(), fill_mc, pen["pending_id"]),
            )
            filled_entries.append({"position_id": position_id, "token_key": token, "decision_mc": float(pen["decision_mc"]), "entry_mc": fill_mc, "fill_kind": "next_observable"})

        open_positions = conn.execute(
            "SELECT * FROM axiom_paper_positions_v20 WHERE status='open' ORDER BY opened_at"
        ).fetchall()
        open_before = len(open_positions)
        exits: list[dict[str, Any]] = []
        open_tokens = {str(p["token_key"]) for p in open_positions}

        # Manage positions. Pending EXIT decisions fill only on a later observation.
        for pos in open_positions:
            token = str(pos["token_key"])
            row = current_series.get(token)
            pending_exit_at = _to_ts(pos["pending_exit_at"]) if pos["pending_exit_at"] else None
            if row is None:
                misses = int(pos["missed_cycles"] or 0) + 1
                conn.execute("UPDATE axiom_paper_positions_v20 SET missed_cycles=? WHERE position_id=?", (misses, pos["position_id"]))
                terminal, absent_minutes, _ = terminal_absence(_to_ts(pos["last_seen_at"]))
                observed_mc = float(pos["last_mc"])
                execution_mc = _execution_proxy_mc(float(pos["entry_mc"]), observed_mc, config, unavailable=True)
                observed_return = observed_mc / float(pos["entry_mc"]) - 1.0
                execution_return = execution_mc / float(pos["entry_mc"]) - 1.0
                conn.execute(
                    """
                    INSERT OR REPLACE INTO axiom_paper_marks_v20
                    (mark_id, position_id, token_key, snapshot_at, market_cap_usd, return_pct,
                     mfe_pct, mae_pct, state_json, action, action_value, policy_version,
                     price_available, mark_kind, execution_return_pct, training_eligible,
                     action_probability,behavior_policy_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, 0, ?, ?, 0, 1.0, ?)
                    """,
                    (
                        str(uuid.uuid4()), pos["position_id"], token, snapshot.isoformat(), observed_mc,
                        observed_return, float(pos["mfe_pct"]), float(pos["mae_pct"]),
                        _json({"missing_minutes": absent_minutes, "last_observed_mc": observed_mc}),
                        "DISAPPEARANCE_CLOSE" if terminal else "MISSING_HOLD", policy_id,
                        "disappearance_terminal" if terminal else "stale_last_observed", execution_return, policy_id,
                    ),
                )
                if terminal:
                    reason = str(pos["pending_exit_reason"] or "dead_after_valid_capture_absence")
                    exits.append(_close_position(conn, pos, snapshot, observed_mc, reason, config, price_available=False))
                    open_tokens.discard(token)
                continue

            mc = float(row["market_cap_usd"])
            # A pending exit from a prior snapshot executes now, before a new decision is made.
            if pending_exit_at is not None and snapshot > pending_exit_at:
                fresh = conn.execute("SELECT * FROM axiom_paper_positions_v20 WHERE position_id=?", (pos["position_id"],)).fetchone()
                exits.append(_close_position(
                    conn, fresh, snapshot, mc, str(pos["pending_exit_reason"] or "policy_exit_next_observable"), config, price_available=True
                ))
                open_tokens.discard(token)
                continue

            pred_state = _safe_prediction_state(row)
            mark_state = _make_mark_state(pred_state, pos, mc, snapshot)
            current_return = mc / float(pos["entry_mc"]) - 1.0
            mfe = max(float(pos["mfe_pct"]), current_return)
            mae = min(float(pos["mae_pct"]), current_return)
            conn.execute(
                "UPDATE axiom_paper_positions_v20 SET last_seen_at=?,last_mc=?,missed_cycles=0,mfe_pct=?,mae_pct=? WHERE position_id=?",
                (snapshot.isoformat(), mc, mfe, mae, pos["position_id"]),
            )
            held = (snapshot - _to_ts(pos["opened_at"])).total_seconds() / 60.0
            if policy_bundle and policy_bundle.get("hold_head"):
                hold_value = float(_predict_policy_head(policy_bundle["hold_head"], [mark_state])[0]); policy_kind = "learned"
            else:
                hold_value = _bootstrap_hold_value(mark_state); policy_kind = "bootstrap"
            action = "HOLD"; close_reason = None
            if held >= config.max_hold_minutes:
                action = "EXIT_DECISION"; close_reason = "max_hold"
            elif held >= config.min_hold_minutes and hold_value <= 0.0:
                action = "EXIT_DECISION"; close_reason = f"{policy_kind}_hold_value_nonpositive"
            conn.execute(
                """
                INSERT OR REPLACE INTO axiom_paper_marks_v20
                (mark_id,position_id,token_key,snapshot_at,market_cap_usd,return_pct,mfe_pct,mae_pct,state_json,
                 action,action_value,policy_version,price_available,mark_kind,execution_return_pct,training_eligible,
                 action_probability,behavior_policy_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,'observed',?,1,1.0,?)
                """,
                (str(uuid.uuid4()),pos["position_id"],token,snapshot.isoformat(),mc,current_return,mfe,mae,
                 _json(mark_state),action,hold_value,policy_id,current_return,policy_id),
            )
            if action == "EXIT_DECISION":
                conn.execute(
                    "UPDATE axiom_paper_positions_v20 SET pending_exit_at=?,pending_exit_reason=? WHERE position_id=?",
                    (snapshot.isoformat(), close_reason, pos["position_id"]),
                )

        # Candidate scoring and exact behavior propensities.
        open_tokens = {str(r[0]) for r in conn.execute("SELECT token_key FROM axiom_paper_positions_v20 WHERE status='open'").fetchall()}
        pending_tokens = {str(r[0]) for r in conn.execute("SELECT token_key FROM axiom_paper_pending_entries_v24 WHERE status='pending'").fetchall()}
        candidate_rows: list[dict[str, Any]] = []
        for _, row in current.iterrows():
            token = str(row["token_key"]); state = _safe_prediction_state(row)
            if not state: continue
            bootstrap = _bootstrap_entry_score(state)
            learned = float(_predict_policy_head(policy_bundle["entry_head"], [state])[0]) if policy_bundle and policy_bundle.get("entry_head") else None
            score = learned if learned is not None else bootstrap
            candidate_rows.append({"token_key":token,"market_cap_usd":float(row["market_cap_usd"]),"state":state,
                                   "bootstrap_score":bootstrap,"policy_score":learned,"rank_score":score})
        candidate_rows.sort(key=lambda r:r["rank_score"], reverse=True)
        slots = max(0, config.max_open_positions - len(open_tokens) - len(pending_tokens))
        eligible=[]
        for c in candidate_rows:
            if c["token_key"] in open_tokens or c["token_key"] in pending_tokens: continue
            last_close=_last_closed_at(conn,c["token_key"])
            if last_close is not None and (snapshot-last_close).total_seconds() < config.reentry_cooldown_minutes*60: continue
            eligible.append(c)

        seed=int(hashlib.sha256(snapshot.isoformat().encode()).hexdigest()[:12],16); rng=random.Random(seed)
        e=max(0.0,min(1.0,float(config.exploration_fraction)))
        k=min(slots,len(eligible)); selected=[]; action_probs={c["token_key"]:0.0 for c in candidate_rows}; exploration_probs={c["token_key"]:0.0 for c in candidate_rows}
        if k>0:
            fixed=max(0,k-1)
            for c in eligible[:fixed]:
                selected.append((c,False)); action_probs[c["token_key"]]=1.0
            pool=eligible[fixed:max(fixed+config.candidate_pool,fixed+1)]
            if pool:
                for c in pool:
                    exploration_probs[c["token_key"]]=e/len(pool)
                boundary=eligible[fixed] if fixed < len(eligible) else None
                if boundary is not None:
                    action_probs[boundary["token_key"]]=(1.0-e)+e/len(pool)
                for c in pool:
                    if boundary is None or c["token_key"]!=boundary["token_key"]:
                        action_probs[c["token_key"]]=e/len(pool)
                if rng.random() < e:
                    selected.append((rng.choice(pool),True))
                elif boundary is not None:
                    selected.append((boundary,False))

        selected_keys={c["token_key"] for c,_ in selected}; exploration_keys={c["token_key"] for c,ex in selected if ex}
        for rank,c in enumerate(candidate_rows, start=1):
            rank_prob = action_probs.get(c["token_key"],0.0)
            conn.execute(
                """INSERT OR REPLACE INTO axiom_paper_candidates_v20
                (snapshot_at,token_key,market_cap_usd,state_json,bootstrap_score,policy_score,chosen,exploration,
                 forecast_hash,policy_version,action_probability,rank_probability,exploration_probability,
                 behavior_policy_version,eligible_actions_json,capital_constraint_state_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (snapshot.isoformat(),c["token_key"],c["market_cap_usd"],_json(c["state"]),c["bootstrap_score"],c["policy_score"],
                 int(c["token_key"] in selected_keys),int(c["token_key"] in exploration_keys),forecast_hash,policy_id,
                 float(rank_prob),float(rank_prob),float(exploration_probs.get(c["token_key"],0.0)),policy_id,
                 _json(["PASS","ENTER"]),_json({"slots":slots,"rank":rank,"open":len(open_tokens),"pending":len(pending_tokens)})),
            )

        pending_created=[]
        for c,exploration in selected:
            pending_id=str(uuid.uuid4())
            conn.execute(
                """INSERT INTO axiom_paper_pending_entries_v24
                (pending_id,token_key,decision_at,decision_mc,decision_state_json,policy_version,forecast_hash,
                 exploration,action_probability,status)
                VALUES (?,?,?,?,?,?,?,?,?,'pending')""",
                (pending_id,c["token_key"],snapshot.isoformat(),c["market_cap_usd"],_json(c["state"]),policy_id,forecast_hash,
                 int(exploration),float(action_probs.get(c["token_key"],1.0))),
            )
            pending_created.append({"pending_id":pending_id,"token_key":c["token_key"],"decision_mc":c["market_cap_usd"],
                                    "action_probability":action_probs.get(c["token_key"],1.0),"exploration":exploration})

        open_after=int(conn.execute("SELECT COUNT(*) FROM axiom_paper_positions_v20 WHERE status='open'").fetchone()[0])
        run_id=str(uuid.uuid4())
        conn.execute(
            """INSERT INTO axiom_self_teach_runs_v20
            (run_id,run_at,snapshot_at,predictions_path,forecast_model_hash,policy_version,current_tokens,open_before,entries,exits,open_after,details_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id,_now_iso(),snapshot.isoformat(),predictions_path,forecast_hash,policy_id,len(current),open_before,
             len(filled_entries),len(exits),open_after,_json({"config":asdict(config),"pending_entries_created":len(pending_created)})),
        )
        conn.commit()
        closed_count=_closed_count(conn); last_train_closed=_policy_last_training_closed(conn)

    training=None
    # V24 owns policy promotion/training; never let the legacy V20 learner silently
    # auto-install a policy over a V24 OOS forecaster.
    if auto_train and not is_v24_forecast and closed_count >= (20 if allow_small_policy else config.policy_min_closed):
        normal_gate=closed_count-last_train_closed >= (5 if allow_small_policy else config.policy_retrain_every_closed)
        if normal_gate or policy_accounting_incompatible:
            try: training=train_policy(db,policy_dir,config,allow_small=allow_small_policy)
            except Exception as exc: training={"error":str(exc)}
    return {"run_id":run_id,"snapshot_at":snapshot.isoformat(),"current_tokens":len(current),"open_before":open_before,
            "entries":filled_entries,"pending_entries":pending_created,"cancelled_pending":cancelled_pending,"exits":exits,
            "open_after":open_after,"active_policy":policy_id or "bootstrap","closed_paper_trades":closed_count,
            "policy_training":training,"execution_semantics":"next_observable_fill"}


# ------------------------------ forecaster champion/challenger ------------------------------

def _forecast_val_tokens(frame: pd.DataFrame) -> set[str]:
    firsts = frame.groupby("token_key")["snapshot_at"].min().sort_values()
    tokens = list(firsts.index)
    if not tokens:
        return set()
    cut = max(1, min(len(tokens) - 1, int(len(tokens) * 0.80))) if len(tokens) > 1 else 0
    return set(tokens[cut:])


def _evaluate_peak_bundle(bundle_path: str | Path, frame: pd.DataFrame, val_tokens: set[str]) -> dict[str, Any]:
    _require_peak()
    p = Path(bundle_path)
    if not p.exists():
        return {"available": False}
    bundle = joblib.load(p)
    features = list(bundle.get("feature_columns", []))
    data = frame[frame.token_key.isin(val_tokens)].copy()
    for c in features:
        if c not in data.columns:
            data[c] = np.nan
    X_all = data[features].replace([np.inf, -np.inf], np.nan)
    head_scores = {}
    for output, head in bundle.get("heads", {}).items():
        target = head.get("target")
        if not target or target not in data.columns:
            continue
        mask = pd.to_numeric(data[target], errors="coerce").notna().to_numpy()
        if np.sum(mask) < 5:
            continue
        y = pd.to_numeric(data.loc[mask, target], errors="coerce").to_numpy(dtype=float)
        try:
            pred = peak._predict_head(head, X_all.loc[mask])
            if head.get("kind") == "classifier":
                if len(np.unique(y)) < 2:
                    continue
                score = float(log_loss(y.astype(int), np.clip(pred, 1e-5, 1 - 1e-5), labels=[0, 1]))
                metric = "log_loss"
            elif head.get("quantile") is not None:
                score = float(mean_pinball_loss(y, pred, alpha=float(head["quantile"])))
                metric = f"pinball_q{head['quantile']}"
            else:
                score = float(mean_absolute_error(y, pred))
                metric = "mae"
            head_scores[output] = {"score": score, "metric": metric, "rows": int(len(y))}
        except Exception:
            continue
    return {"available": bool(head_scores), "heads": head_scores, "validation_tokens": len(val_tokens)}


def _compare_forecasts(candidate: dict[str, Any], champion: dict[str, Any], margin: float) -> tuple[bool, dict[str, Any], str]:
    if not champion.get("available"):
        return True, {"common_heads": 0}, "no existing evaluable champion"
    c_heads = candidate.get("heads", {})
    h_heads = champion.get("heads", {})
    common = sorted(set(c_heads) & set(h_heads))
    if not common:
        return False, {"common_heads": 0}, "no common forecast heads for promotion comparison"
    ratios = []
    materially_degraded = 0
    catastrophic = 0
    per_head = {}
    for name in common:
        c = c_heads[name]["score"]
        h = h_heads[name]["score"]
        ratio = c / max(h, 1e-12)
        ratios.append(ratio)
        if ratio > 1.05:
            materially_degraded += 1
        if ratio > 1.20:
            catastrophic += 1
        per_head[name] = {"candidate": c, "champion": h, "ratio": ratio}
    median_ratio = float(np.median(ratios))
    max_degraded = int(len(common) * 0.10)
    promoted = (
        median_ratio <= (1.0 - margin)
        and materially_degraded <= max_degraded
        and catastrophic == 0
    )
    reason = (
        f"median_error_ratio={median_ratio:.4f}; materially_degraded_heads="
        f"{materially_degraded}/{len(common)}; catastrophic_heads={catastrophic}/{len(common)}"
    )
    return promoted, {
        "common_heads": len(common), "median_error_ratio": median_ratio,
        "materially_degraded_heads": materially_degraded, "catastrophic_heads": catastrophic,
        "per_head": per_head,
    }, reason



def _peak_config_from_db(conn: sqlite3.Connection):
    _require_peak()
    if peak.LABEL_TABLE not in peak._all_tables(conn):
        return peak.PeakStructureConfig()
    try:
        row = conn.execute(
            f"SELECT config_json FROM {peak.LABEL_TABLE} WHERE config_json IS NOT NULL ORDER BY decision_at DESC LIMIT 1"
        ).fetchone()
        raw = _loads(row[0]) if row and row[0] else {}
        allowed = set(asdict(peak.PeakStructureConfig()))
        values = {k: v for k, v in raw.items() if k in allowed}
        return peak.PeakStructureConfig(**values)
    except Exception:
        return peak.PeakStructureConfig()

def _mature_peak_rows(conn: sqlite3.Connection) -> int:
    if peak.LABEL_TABLE not in peak._all_tables(conn):
        return 0
    return int(conn.execute(
        f"SELECT COUNT(*) FROM {peak.LABEL_TABLE} WHERE has_next_substantial_peak_before_terminal_72h IS NOT NULL"
    ).fetchone()[0])


def _learning_updates_since(conn: sqlite3.Connection, since: pd.Timestamp | None) -> int:
    if peak.LABEL_TABLE not in peak._all_tables(conn):
        return 0
    if since is None:
        return int(conn.execute(
            f"SELECT COUNT(*) FROM {peak.LABEL_TABLE} WHERE learning_updated_at IS NOT NULL"
        ).fetchone()[0])
    return int(conn.execute(
        f"SELECT COUNT(*) FROM {peak.LABEL_TABLE} WHERE learning_updated_at > ?",
        (since.isoformat(),),
    ).fetchone()[0])


def retrain_forecaster(
    db: str,
    model_root: str,
    champion_path: str,
    config: PolicyConfig,
    *,
    allow_small: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Refresh labels and append trees to the active V21 forecaster.

    A full fit is used only once when no compatible 72h champion exists. Normal
    maintenance calls `peak.incremental_train`, which adds a small number of trees
    using newly matured/revised labels plus a bounded replay buffer.
    """
    _require_peak()
    with sqlite3.connect(db) as config_conn:
        peak_config = _peak_config_from_db(config_conn)
    # Force the new operating assumptions even if the legacy table stored V19 values.
    peak_config = peak.PeakStructureConfig(
        min_runup_pct=peak_config.min_runup_pct,
        confirm_retrace_pct=peak_config.confirm_retrace_pct,
        higher_peak_margin_pct=peak_config.higher_peak_margin_pct,
        horizon_minutes=72 * 60,
        death_missed_cycles=50,
        death_gap_minutes=getattr(peak_config, "death_gap_minutes", 50.0),
        age_out_minutes=71 * 60,
        min_peak_separation_minutes=peak_config.min_peak_separation_minutes,
    )
    refresh_result = peak.refresh_labels(db, peak_config)

    champion = Path(champion_path)
    compatible = False
    if champion.exists():
        try:
            existing = joblib.load(champion)
            compatible = str(existing.get("schema_version", "")).startswith("v21_")
        except Exception:
            compatible = False

    mature_rows = 0
    with sqlite3.connect(db) as conn:
        migrate(conn)
        mature_rows = _mature_peak_rows(conn)
        last = conn.execute(
            "SELECT created_at, mature_rows FROM axiom_forecast_promotions_v20 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        last_at = _to_ts(last[0]) if last else None
        new_updates = _learning_updates_since(conn, last_at)
        cooldown = (pd.Timestamp.now(tz="UTC") - last_at).total_seconds() / 3600.0 if last_at is not None else float("inf")
        if compatible and last and not force:
            if new_updates < config.forecast_retrain_min_new_mature_rows and cooldown < config.forecast_retrain_cooldown_hours:
                return {
                    "trained": False, "reason": "incremental update gate not met", "mature_rows": mature_rows,
                    "new_learning_updates": new_updates, "hours_since_last": cooldown, "refresh": refresh_result,
                }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate_dir = Path(model_root) / "challengers" / stamp
    if not compatible:
        # Required once when moving from the old 24h/V19 target schema to 72h/V21.
        train_result = peak.train(db, str(candidate_dir), allow_small=allow_small)
        candidate_path = Path(train_result["versioned_model"])
        bootstrap = True
    else:
        train_result = peak.incremental_train(
            db, str(champion), str(candidate_dir),
            append_estimators=12 if allow_small else 30,
            replay_rows=250 if allow_small else 1000,
        )
        if not train_result.get("trained"):
            return {
                "trained": False, "reason": train_result.get("reason", "no incremental update"),
                "mature_rows": mature_rows, "refresh": refresh_result, "update": train_result,
            }
        candidate_path = Path(train_result["versioned_model"])
        bootstrap = False

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        frame, _, _ = peak.load_training_frame(conn)
        val_tokens = _forecast_val_tokens(frame)
        candidate_eval = _evaluate_peak_bundle(candidate_path, frame, val_tokens)
        champion_eval = _evaluate_peak_bundle(champion, frame, val_tokens) if compatible else {"available": False}
        if bootstrap:
            promoted, comparison, reason = True, {"bootstrap": True}, "first compatible V21 72h champion"
        else:
            promoted, comparison, reason = _compare_forecasts(candidate_eval, champion_eval, config.forecast_promotion_margin)
        champion_before_hash = _hash_file(champion)
        champion_before_path = str(champion) if champion.exists() else None
        if promoted:
            champion.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_path, champion)
        promotion_id = str(uuid.uuid4())
        metrics = {"candidate": candidate_eval, "champion_before": champion_eval, "comparison": comparison}
        conn.execute(
            """
            INSERT INTO axiom_forecast_promotions_v20
            (promotion_id, created_at, candidate_path, candidate_hash, champion_before_path,
             champion_before_hash, promoted, mature_rows, metrics_json, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                promotion_id, _now_iso(), str(candidate_path), _hash_file(candidate_path), champion_before_path,
                champion_before_hash, int(promoted), mature_rows, _json(metrics), reason,
            ),
        )
        conn.commit()
    return {
        "trained": True,
        "training_mode": "full_bootstrap_once" if bootstrap else "incremental_append",
        "promoted": promoted,
        "candidate": str(candidate_path),
        "champion": str(champion) if promoted or champion.exists() else None,
        "mature_rows": mature_rows,
        "new_learning_updates": new_updates,
        "reason": reason,
        "comparison": comparison,
        "refresh": refresh_result,
        "train": train_result,
    }

def refresh_current_predictions(db: str, peak_model: str, peak_out: str, base_predictions: str | None = None) -> dict[str, Any]:
    _require_peak()
    if not Path(peak_model).exists():
        raise RuntimeError(f"Peak champion model not found: {peak_model}")
    rows = peak.predict(db, peak_model, peak_out)
    augmented = None
    if base_predictions and Path(base_predictions).exists():
        tmp = str(Path(base_predictions).with_suffix(".v21tmp.csv"))
        augmented = peak.augment_csv(base_predictions, peak_out, tmp)
        os.replace(tmp, base_predictions)
    return {"peak_predictions": len(rows), "peak_out": peak_out, "augmented": augmented}


# ------------------------------ reporting ------------------------------

def status(db: str) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        _backfill_paper_execution_accounting(conn)
        def scalar(sql: str) -> Any:
            return conn.execute(sql).fetchone()[0]
        champion = _policy_champion_record(conn)
        closed = int(scalar("SELECT COUNT(*) FROM axiom_paper_positions_v20 WHERE status='closed'"))
        open_ = int(scalar("SELECT COUNT(*) FROM axiom_paper_positions_v20 WHERE status='open'"))
        avg_net = scalar("SELECT AVG(net_return_pct) FROM axiom_paper_positions_v20 WHERE status='closed'")
        avg_reward = scalar("SELECT AVG(reward) FROM axiom_paper_positions_v20 WHERE status='closed'")
        win_rate = scalar("SELECT AVG(CASE WHEN net_return_pct>0 THEN 1.0 ELSE 0.0 END) FROM axiom_paper_positions_v20 WHERE status='closed'")
        observed_avg_net = scalar("SELECT AVG(COALESCE(observed_net_return_pct,net_return_pct)) FROM axiom_paper_positions_v20 WHERE status='closed'")
        observed_avg_reward = scalar("SELECT AVG(COALESCE(observed_reward,reward)) FROM axiom_paper_positions_v20 WHERE status='closed'")
        observed_win_rate = scalar("SELECT AVG(CASE WHEN COALESCE(observed_net_return_pct,net_return_pct)>0 THEN 1.0 ELSE 0.0 END) FROM axiom_paper_positions_v20 WHERE status='closed'")
        unavailable_closes = int(scalar("SELECT COUNT(*) FROM axiom_paper_positions_v20 WHERE status='closed' AND exit_kind='disappearance_terminal'"))
        exploration = int(scalar("SELECT COUNT(*) FROM axiom_paper_positions_v20 WHERE exploration=1"))
        promotions = int(scalar("SELECT COUNT(*) FROM axiom_forecast_promotions_v20 WHERE promoted=1"))
        rejected = int(scalar("SELECT COUNT(*) FROM axiom_forecast_promotions_v20 WHERE promoted=0"))
        recent = [dict(r) for r in conn.execute(
            """
            SELECT token_key, opened_at, closed_at, close_reason, exit_kind, price_available_at_exit,
                   exit_mc_observed, exit_mc_execution_proxy, observed_net_return_pct,
                   execution_net_return_pct, observed_reward, execution_reward, exploration
            FROM axiom_paper_positions_v20 WHERE status='closed' ORDER BY closed_at DESC LIMIT 10
            """
        ).fetchall()]
        forecast_bundle = _load_policy(PEAK_CHAMPION_DEFAULT)
        policy_bundle = _load_policy(champion["model_path"]) if champion else None
        policy_accounting_compatible = bool(
            policy_bundle is None or str(policy_bundle.get("schema_version", "")) == SCHEMA_VERSION
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "native_sampling_minutes": 1,
            "axiom_visibility_hours": 72,
            "absence_terminal_minutes": PolicyConfig().missing_close_minutes,
            "forecast_schema_version": forecast_bundle.get("schema_version") if forecast_bundle else None,
            "forecast_training_mode": forecast_bundle.get("training_mode") if forecast_bundle else None,
            "forecast_incremental_rounds": forecast_bundle.get("incremental_rounds") if forecast_bundle else None,
            "forecast_training_watermark": forecast_bundle.get("training_watermark") if forecast_bundle else None,
            "policy_training_mode": policy_bundle.get("training_mode") if policy_bundle and policy_accounting_compatible else ("bootstrap_pending_execution_accounting_rebootstrap" if champion else "bootstrap"),
            "policy_accounting_compatible": policy_accounting_compatible,
            "policy_incremental_rounds": policy_bundle.get("incremental_rounds") if policy_bundle else None,
            "open_positions": open_,
            "closed_trades": closed,
            # Authoritative policy-learning metrics use execution-conservative outcomes.
            "paper_win_rate": win_rate,
            "mean_net_return_pct": avg_net,
            "mean_reward": avg_reward,
            "effectiveness_accounting": "execution_conservative",
            "execution_conservative": {
                "win_rate": win_rate, "mean_net_return_pct": avg_net, "mean_reward": avg_reward,
                "unavailable_terminal_trades": unavailable_closes,
                "default_disappearance_profit_recognition_fraction": PolicyConfig().disappearance_profit_recognition_fraction,
            },
            "observed_market": {
                "win_rate": observed_win_rate, "mean_net_return_pct": observed_avg_net,
                "mean_reward": observed_avg_reward,
            },
            "exploration_positions_total": exploration,
            "active_policy_version": champion["version_id"] if champion else "bootstrap",
            "active_policy_path": champion["model_path"] if champion else None,
            "forecast_promotions": promotions,
            "forecast_challengers_rejected": rejected,
            "recent_closed_trades": recent,
        }


# ------------------------------ CLI ------------------------------

def _config_from_args(args: argparse.Namespace) -> PolicyConfig:
    defaults = PolicyConfig()
    vals = {}
    for key in asdict(defaults):
        vals[key] = getattr(args, key, getattr(defaults, key))
    return PolicyConfig(**vals)


def _add_config_args(p: argparse.ArgumentParser) -> None:
    d = PolicyConfig()
    p.add_argument("--max-open-positions", type=int, default=d.max_open_positions)
    p.add_argument("--exploration-fraction", type=float, default=d.exploration_fraction)
    p.add_argument("--candidate-pool", type=int, default=d.candidate_pool)
    p.add_argument("--min-hold-minutes", type=float, default=d.min_hold_minutes)
    p.add_argument("--max-hold-minutes", type=float, default=d.max_hold_minutes)
    p.add_argument("--missed-cycles-to-close", type=int, default=d.missed_cycles_to_close, help="Diagnostic compatibility count; elapsed minutes control closure")
    p.add_argument("--missing-close-minutes", type=float, default=d.missing_close_minutes)
    p.add_argument("--reentry-cooldown-minutes", type=float, default=d.reentry_cooldown_minutes)
    p.add_argument("--friction-bps-round-trip", type=float, default=d.friction_bps_round_trip)
    p.add_argument("--disappearance-profit-recognition-fraction", type=float, default=d.disappearance_profit_recognition_fraction)
    p.add_argument("--drawdown-penalty", type=float, default=d.drawdown_penalty)
    p.add_argument("--peak-capture-weight", type=float, default=d.peak_capture_weight)
    p.add_argument("--hold-label-minutes", type=float, default=d.hold_label_minutes)
    p.add_argument("--policy-retrain-every-closed", type=int, default=d.policy_retrain_every_closed)
    p.add_argument("--policy-min-closed", type=int, default=d.policy_min_closed)
    p.add_argument("--forecast-retrain-min-new-mature-rows", type=int, default=d.forecast_retrain_min_new_mature_rows)
    p.add_argument("--forecast-retrain-cooldown-hours", type=float, default=d.forecast_retrain_cooldown_hours)
    p.add_argument("--policy-promotion-margin", type=float, default=d.policy_promotion_margin)
    p.add_argument("--forecast-promotion-margin", type=float, default=d.forecast_promotion_margin)


def main() -> None:
    ap = argparse.ArgumentParser(description="V21 1-minute / 72-hour self-teaching paper-trading environment")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cycle", help="Consume the latest clipboard-backed observations/predictions and run one paper-trading cycle")
    p.add_argument("--db", default="data/live.sqlite")
    p.add_argument("--predictions", default=PREDICTIONS_DEFAULT)
    p.add_argument("--policy-dir", default=POLICY_DIR_DEFAULT)
    p.add_argument("--allow-stale-predictions", action="store_true")
    p.add_argument("--no-auto-train", action="store_true")
    p.add_argument("--allow-small-policy", action="store_true")
    _add_config_args(p)

    p = sub.add_parser("train-policy", help="Train a challenger policy from paper outcomes and promote only if it improves")
    p.add_argument("--db", default="data/live.sqlite")
    p.add_argument("--policy-dir", default=POLICY_DIR_DEFAULT)
    p.add_argument("--allow-small", action="store_true")
    _add_config_args(p)

    p = sub.add_parser("learn-forecaster", help="Incrementally refresh V21 72h labels, append a challenger, and champion-gate it")
    p.add_argument("--db", default="data/live.sqlite")
    p.add_argument("--model-root", default="models/axiom_peak_v21")
    p.add_argument("--champion", default=PEAK_CHAMPION_DEFAULT)
    p.add_argument("--allow-small", action="store_true")
    p.add_argument("--force", action="store_true")
    _add_config_args(p)

    p = sub.add_parser("predict", help="Run the active V21 72h champion and optionally augment another prediction CSV")
    p.add_argument("--db", default="data/live.sqlite")
    p.add_argument("--peak-model", default=PEAK_CHAMPION_DEFAULT)
    p.add_argument("--peak-out", default=PEAK_PREDICTIONS_DEFAULT)
    p.add_argument("--augment", default=PREDICTIONS_DEFAULT)

    p = sub.add_parser("status", help="Show paper-learning and champion/challenger status")
    p.add_argument("--db", default="data/live.sqlite")

    args = ap.parse_args()
    if args.cmd == "cycle":
        result = paper_cycle(
            args.db, args.predictions, args.policy_dir, _config_from_args(args),
            allow_stale_predictions=args.allow_stale_predictions,
            auto_train=not args.no_auto_train,
            allow_small_policy=args.allow_small_policy,
        )
    elif args.cmd == "train-policy":
        result = train_policy(args.db, args.policy_dir, _config_from_args(args), allow_small=args.allow_small)
    elif args.cmd == "learn-forecaster":
        result = retrain_forecaster(
            args.db, args.model_root, args.champion, _config_from_args(args),
            allow_small=args.allow_small, force=args.force,
        )
    elif args.cmd == "predict":
        result = refresh_current_predictions(args.db, args.peak_model, args.peak_out, args.augment)
    elif args.cmd == "status":
        result = status(args.db)
    else:  # pragma: no cover
        raise AssertionError(args.cmd)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
