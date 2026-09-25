from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from . import axiom_peak_structure as peak
    from . import axiom_self_teach as selfteach
    from . import axiom_v24 as v24
except Exception as exc:  # pragma: no cover
    peak = None
    selfteach = None
    v24 = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

SCHEMA_VERSION = "v22_1_dual_execution_accounting"
DEFAULT_SOURCE_DB = "data/live.sqlite"
DEFAULT_BENCHMARK_DB = "data/v22_1000_benchmark.sqlite"
DEFAULT_PREDICTIONS = "data/axiom_predictions_v24.csv"
DEFAULT_PEAK_MODEL = "models/axiom_v24/champion.joblib"
DEFAULT_POLICY_MODEL = "models/axiom_policy_v24/champion.joblib"


@dataclass
class BenchmarkConfig:
    initial_cash_usd: float = 1000.0
    max_open_positions: int = 5
    # Compatibility alias for legacy benchmark models. V24 uses the conviction
    # tiers below, and the legacy default now matches the ordinary 5% tier.
    position_fraction: float = 0.05
    ordinary_position_fraction: float = 0.05
    strong_position_fraction: float = 0.075
    exceptional_position_fraction: float = 0.10
    max_total_exposure_fraction: float = 0.30
    min_cash_reserve_fraction: float = 0.70
    max_correlated_exposure_fraction: float = 0.15
    strong_score_quantile: float = 0.75
    exceptional_score_quantile: float = 0.95
    conviction_calibration_min_scores: int = 40
    conviction_calibration_max_scores: int = 5000
    correlation_lookback_minutes: float = 240.0
    correlation_min_overlap: int = 10
    correlation_threshold: float = 0.80
    min_entry_score: float = 0.0
    min_hold_minutes: float = 2.0
    max_hold_minutes: float = 72.0 * 60.0
    missing_close_minutes: float = 50.0
    reentry_cooldown_minutes: float = 20.0
    friction_bps_round_trip: float = 100.0
    # Recurrent swing overlay. Short-horizon heads answer "buy now"; recurrent
    # lifecycle heads decide whether an approaching peak should be held through
    # or converted into a watched, retracement-gated re-entry opportunity.
    recurrent_swing_enabled: bool = True
    swing_entry_min_probability: float = 0.55
    # Once enough champion-matched 60-minute outcomes mature, the absolute gate
    # above is replaced by an empirical probability region and conservative
    # outcome-regression edge. These bounds affect decisions, not model fitting.
    swing_calibration_min_samples: int = 80
    swing_calibration_min_tokens: int = 12
    swing_calibration_max_samples: int = 5000
    swing_entry_max_occurrence_minutes: float = 60.0
    swing_entry_min_net_upside: float = 0.02
    swing_peak_boundary_minutes: float = 10.0
    swing_peak_boundary_probability: float = 0.55
    swing_exit_min_return: float = 0.03
    swing_hold_later_probability: float = 0.65
    swing_hold_min_second_upside: float = 0.05
    swing_hold_max_second_gap_minutes: float = 30.0
    swing_reentry_min_minutes: float = 3.0
    swing_reentry_min_retrace: float = 0.03
    swing_watch_min_later_probability: float = 0.35
    swing_watch_min_second_upside: float = 0.03
    swing_watch_max_minutes: float = 240.0
    # For unavailable/disappeared tokens, preserve all observed downside but only
    # recognize this fraction of positive MC movement in the execution ledger.
    # 0.0 means a stale/disappearance mark can never create paper profit.
    disappearance_profit_recognition_fraction: float = 0.0


def _require_v21() -> None:
    if peak is None or selfteach is None:
        raise RuntimeError(
            "V22 benchmark requires the V21 axiom_peak_structure.py and axiom_self_teach.py modules. "
            f"Import error: {IMPORT_ERROR}"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _load_bundle(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    obj = joblib.load(p)
    return obj if isinstance(obj, dict) else None


def _prediction_identity_col(df: pd.DataFrame) -> str:
    for c in ("token_key", "short_address_hint", "token", "mint", "token_address"):
        if c in df.columns:
            return c
    raise RuntimeError("Prediction CSV does not contain a recognized token identity column.")


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


def _prediction_snapshot(df: pd.DataFrame) -> pd.Timestamp | None:
    for c in ("snapshot_at", "decision_at", "observed_at"):
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True).dropna()
            if len(s):
                return s.max()
    return None


def _safe_state(row: pd.Series) -> dict[str, float]:
    # The dedicated V21 prediction CSV contains model outputs rather than realized
    # future labels. Keep p_*/pred_* heads, including names containing
    # "before_terminal_72h"; that phrase describes the prediction horizon and is
    # not hindsight. This is intentionally narrower than filtering arbitrary raw
    # database columns.
    state: dict[str, float] = {}
    for key, value in row.items():
        name = str(key)
        if name in {"token_key", "snapshot_at", "decision_at", "observed_at", "name",
                    "next_substantial_peak_timing_window"}:
            continue
        x = _finite_float(value)
        if x is None:
            continue
        if (name == "market_cap_usd" or name.startswith("p_") or name.startswith("pred_")
                or name.startswith("recurrent_") or name.startswith("next_") or name.startswith("second_")
                or name.startswith("v24_adapter_weight") or name == "liquidity_model_active"):
            state[name] = x
    return state


def _entry_score(state: dict[str, float], policy: dict[str, Any] | None) -> tuple[float, str]:
    if policy and policy.get("entry_head"):
        value = float(selfteach._predict_policy_head(policy["entry_head"], [state])[0])
        return value, "learned"
    return float(selfteach._bootstrap_entry_score(state)), "bootstrap"


def _hold_score(state: dict[str, float], policy: dict[str, Any] | None) -> tuple[float, str]:
    if policy and policy.get("hold_head"):
        value = float(selfteach._predict_policy_head(policy["hold_head"], [state])[0])
        return value, "learned"
    return float(selfteach._bootstrap_hold_value(state)), "bootstrap"


def _policy_version(bundle: dict[str, Any] | None, path: str) -> str:
    if bundle is None:
        return "bootstrap"
    return str(bundle.get("version_id") or bundle.get("created_at") or _hash_file(path) or "loaded")


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_column(conn: sqlite3.Connection, table: str, definition: str) -> None:
    name = definition.split()[0]
    if name not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS benchmark_account_v22 (
            benchmark_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            initial_cash_usd REAL NOT NULL,
            cash_usd REAL NOT NULL,
            status TEXT NOT NULL,
            last_snapshot_at TEXT,
            config_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS benchmark_positions_v22 (
            position_id TEXT PRIMARY KEY,
            benchmark_id TEXT NOT NULL,
            token_key TEXT NOT NULL,
            opened_at TEXT NOT NULL,
            entry_mc REAL NOT NULL,
            entry_notional_usd REAL NOT NULL,
            entry_fee_usd REAL NOT NULL,
            entry_cash_spent_usd REAL NOT NULL,
            exposure_units REAL NOT NULL,
            entry_score REAL,
            entry_score_kind TEXT,
            entry_state_json TEXT NOT NULL,
            forecast_model_hash TEXT,
            policy_model_hash TEXT,
            policy_version TEXT,
            status TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            last_mc REAL NOT NULL,
            last_mark_value_usd REAL NOT NULL,
            mfe_pct REAL NOT NULL DEFAULT 0,
            mae_pct REAL NOT NULL DEFAULT 0,
            closed_at TEXT,
            exit_mc REAL,
            exit_fee_usd REAL,
            exit_proceeds_usd REAL,
            realized_pnl_usd REAL,
            realized_return_pct REAL,
            close_reason TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_benchmark_positions_status_v22
          ON benchmark_positions_v22(benchmark_id, status);
        CREATE INDEX IF NOT EXISTS idx_benchmark_positions_token_v22
          ON benchmark_positions_v22(benchmark_id, token_key, opened_at);

        CREATE TABLE IF NOT EXISTS benchmark_marks_v22 (
            mark_id TEXT PRIMARY KEY,
            benchmark_id TEXT NOT NULL,
            position_id TEXT NOT NULL,
            token_key TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            market_cap_usd REAL NOT NULL,
            liquidation_value_usd REAL NOT NULL,
            return_pct REAL NOT NULL,
            mfe_pct REAL NOT NULL,
            mae_pct REAL NOT NULL,
            hold_score REAL,
            action TEXT NOT NULL,
            state_json TEXT NOT NULL,
            forecast_model_hash TEXT,
            policy_model_hash TEXT,
            UNIQUE(position_id, snapshot_at)
        );

        CREATE TABLE IF NOT EXISTS benchmark_equity_v22 (
            benchmark_id TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            cash_usd REAL NOT NULL,
            open_liquidation_value_usd REAL NOT NULL,
            equity_usd REAL NOT NULL,
            realized_pnl_usd REAL NOT NULL,
            unrealized_pnl_usd REAL NOT NULL,
            open_positions INTEGER NOT NULL,
            forecast_model_hash TEXT,
            policy_model_hash TEXT,
            policy_version TEXT,
            PRIMARY KEY(benchmark_id, snapshot_at)
        );

        CREATE TABLE IF NOT EXISTS benchmark_candidates_v22 (
            benchmark_id TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            token_key TEXT NOT NULL,
            market_cap_usd REAL NOT NULL,
            entry_score REAL NOT NULL,
            score_kind TEXT NOT NULL,
            chosen INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            forecast_model_hash TEXT,
            policy_model_hash TEXT,
            PRIMARY KEY(benchmark_id, snapshot_at, token_key)
        );

        CREATE TABLE IF NOT EXISTS benchmark_pending_entries_v24 (
            pending_id TEXT PRIMARY KEY,
            benchmark_id TEXT NOT NULL,
            token_key TEXT NOT NULL,
            decision_at TEXT NOT NULL,
            decision_mc REAL NOT NULL,
            reserved_cash_usd REAL NOT NULL,
            entry_score REAL,
            entry_score_kind TEXT,
            entry_state_json TEXT NOT NULL,
            forecast_model_hash TEXT,
            policy_model_hash TEXT,
            policy_version TEXT,
            status TEXT NOT NULL,
            filled_at TEXT,
            fill_mc REAL,
            cancelled_at TEXT,
            cancel_reason TEXT,
            UNIQUE(benchmark_id,token_key,decision_at)
        );
        CREATE INDEX IF NOT EXISTS idx_benchmark_pending_v24 ON benchmark_pending_entries_v24(benchmark_id,status,decision_at);
        """
    )
    # V22.1 adds a second, execution-conservative accounting track without
    # changing/deleting the original observed-price V22 fields. This migration is
    # additive so an existing benchmark database can be upgraded in place.
    _ensure_column(conn, "benchmark_account_v22", "execution_cash_usd REAL")

    for definition in (
        "entry_decision_at TEXT",
        "entry_fill_kind TEXT",
        "conviction_tier TEXT",
        "target_position_fraction REAL",
        "risk_bucket TEXT",
        "pending_exit_at TEXT",
        "pending_exit_reason TEXT",
        "exit_kind TEXT",
        "price_available_at_exit INTEGER",
        "exit_mc_observed REAL",
        "exit_mc_execution_proxy REAL",
        "observed_exit_fee_usd REAL",
        "execution_exit_fee_usd REAL",
        "observed_exit_proceeds_usd REAL",
        "execution_exit_proceeds_usd REAL",
        "observed_realized_pnl_usd REAL",
        "execution_realized_pnl_usd REAL",
        "observed_realized_return_pct REAL",
        "execution_realized_return_pct REAL",
        "swing_watch_id TEXT",
        "swing_sequence INTEGER NOT NULL DEFAULT 1",
    ):
        _ensure_column(conn, "benchmark_positions_v22", definition)

    for definition in (
        "price_available INTEGER",
        "mark_kind TEXT",
        "execution_liquidation_value_usd REAL",
    ):
        _ensure_column(conn, "benchmark_marks_v22", definition)

    for definition in (
        "conviction_tier TEXT",
        "target_position_fraction REAL",
        "target_cash_usd REAL",
        "risk_bucket TEXT",
        "swing_watch_id TEXT",
        "selection_reason TEXT",
    ):
        _ensure_column(conn, "benchmark_candidates_v22", definition)

    for definition in (
        "conviction_tier TEXT",
        "target_position_fraction REAL",
        "risk_bucket TEXT",
        "swing_watch_id TEXT",
    ):
        _ensure_column(conn, "benchmark_pending_entries_v24", definition)

    for definition in (
        "observed_equity_usd REAL",
        "execution_cash_usd REAL",
        "execution_open_value_usd REAL",
        "execution_equity_usd REAL",
        "execution_realized_pnl_usd REAL",
        "execution_unrealized_pnl_usd REAL",
        "stale_open_positions INTEGER",
    ):
        _ensure_column(conn, "benchmark_equity_v22", definition)
    conn.commit()


def _account(conn: sqlite3.Connection) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT * FROM benchmark_account_v22 WHERE status='active' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()


def init_benchmark(db: str, config: BenchmarkConfig, *, reset: bool = False) -> dict[str, Any]:
    _validate_munger_config(config)
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        existing = _account(conn)
        if existing and not reset:
            return {
                "initialized": False,
                "reason": "active benchmark already exists",
                "benchmark_id": existing["benchmark_id"],
                "cash_usd": existing["cash_usd"],
            }
        if reset:
            conn.execute("UPDATE benchmark_account_v22 SET status='retired' WHERE status='active'")
        benchmark_id = datetime.now(timezone.utc).strftime("benchmark_1000_%Y%m%dT%H%M%S_%fZ")
        conn.execute(
            """
            INSERT INTO benchmark_account_v22
            (benchmark_id, created_at, initial_cash_usd, cash_usd, status, last_snapshot_at, config_json)
            VALUES (?, ?, ?, ?, 'active', NULL, ?)
            """,
            (
                benchmark_id,
                _now_iso(),
                float(config.initial_cash_usd),
                float(config.initial_cash_usd),
                _json(asdict(config)),
            ),
        )
        conn.execute(
            "UPDATE benchmark_account_v22 SET execution_cash_usd=? WHERE benchmark_id=?",
            (float(config.initial_cash_usd), benchmark_id),
        )
        conn.commit()
        return {
            "initialized": True,
            "benchmark_id": benchmark_id,
            "initial_cash_usd": config.initial_cash_usd,
            "mode": "isolated_paper_benchmark",
        }



def _config_from_saved_json(value: str) -> BenchmarkConfig:
    try:
        raw = json.loads(value) if value else {}
    except Exception:
        raw = {}
    defaults = asdict(BenchmarkConfig())
    defaults.update({k: raw[k] for k in defaults if k in raw})
    return BenchmarkConfig(**defaults)


def _validate_munger_config(config: BenchmarkConfig) -> None:
    fractions = (
        config.ordinary_position_fraction,
        config.strong_position_fraction,
        config.exceptional_position_fraction,
    )
    if not (0.0 < fractions[0] <= fractions[1] <= fractions[2] <= 0.10):
        raise ValueError("V24 position tiers must be ordered, positive, and capped at 10%.")
    if not (0.0 < config.max_correlated_exposure_fraction <= config.max_total_exposure_fraction <= 0.30):
        raise ValueError("V24 correlated/total exposure caps must be positive and no more than 30%.")
    if not (0.70 <= config.min_cash_reserve_fraction < 1.0):
        raise ValueError("V24 cash reserve must be at least 70% of execution-conservative equity.")
    if config.max_total_exposure_fraction + config.min_cash_reserve_fraction > 1.0 + 1e-12:
        raise ValueError("V24 exposure cap and cash reserve cannot sum to more than 100%.")
    if not (0.0 < config.strong_score_quantile < config.exceptional_score_quantile < 1.0):
        raise ValueError("Conviction quantiles must satisfy 0 < strong < exceptional < 1.")
    if config.conviction_calibration_min_scores < 1 or config.conviction_calibration_max_scores < 1:
        raise ValueError("Conviction calibration sample limits must be positive.")
    if (
        config.swing_calibration_min_samples < 20
        or config.swing_calibration_min_tokens < 2
        or config.swing_calibration_max_samples < config.swing_calibration_min_samples
    ):
        raise ValueError("Swing calibration needs at least 20 samples, two tokens, and a valid sample cap.")
    if not (0.0 <= config.correlation_threshold <= 1.0):
        raise ValueError("Correlation threshold must be between 0 and 1.")
    if config.correlation_min_overlap < 2 or config.correlation_lookback_minutes <= 0:
        raise ValueError("Correlation lookback and overlap must be positive.")
    probabilities = (
        config.swing_entry_min_probability,
        config.swing_peak_boundary_probability,
        config.swing_hold_later_probability,
        config.swing_watch_min_later_probability,
    )
    if any(not (0.0 <= float(value) <= 1.0) for value in probabilities):
        raise ValueError("Recurrent swing probability thresholds must be between 0 and 1.")
    if any(float(value) < 0.0 for value in (
        config.swing_entry_min_net_upside,
        config.swing_exit_min_return,
        config.swing_hold_min_second_upside,
        config.swing_reentry_min_retrace,
        config.swing_watch_min_second_upside,
    )):
        raise ValueError("Recurrent swing return/retrace thresholds cannot be negative.")
    if any(float(value) <= 0.0 for value in (
        config.swing_entry_max_occurrence_minutes,
        config.swing_peak_boundary_minutes,
        config.swing_hold_max_second_gap_minutes,
        config.swing_reentry_min_minutes,
        config.swing_watch_max_minutes,
    )):
        raise ValueError("Recurrent swing timing thresholds must be positive.")


def _conviction_thresholds(
    conn: sqlite3.Connection,
    benchmark_id: str,
    score_kind: str,
    snapshot: pd.Timestamp,
    config: BenchmarkConfig,
) -> tuple[float, float] | None:
    """Return causal score thresholds from decisions strictly before snapshot."""
    rows = conn.execute(
        """
        SELECT entry_score FROM benchmark_candidates_v22
        WHERE benchmark_id=? AND score_kind=? AND snapshot_at<? AND entry_score>?
        ORDER BY snapshot_at DESC LIMIT ?
        """,
        (
            benchmark_id,
            score_kind,
            snapshot.isoformat(),
            float(config.min_entry_score),
            int(config.conviction_calibration_max_scores),
        ),
    ).fetchall()
    scores = pd.Series([float(r[0]) for r in rows if _finite_float(r[0]) is not None], dtype=float)
    if len(scores) < int(config.conviction_calibration_min_scores):
        return None
    return (
        float(scores.quantile(config.strong_score_quantile)),
        float(scores.quantile(config.exceptional_score_quantile)),
    )


def _conviction_tier(score: float, thresholds: tuple[float, float] | None) -> str:
    if thresholds is None:
        return "ordinary"
    strong, exceptional = thresholds
    # Strict comparisons prevent tied/flat score histories from making every
    # candidate exceptional merely because it equals a quantile boundary.
    if score > exceptional:
        return "exceptional"
    if score > strong:
        return "strong"
    return "ordinary"


def _tier_fraction(tier: str, config: BenchmarkConfig) -> float:
    return {
        "ordinary": float(config.ordinary_position_fraction),
        "strong": float(config.strong_position_fraction),
        "exceptional": float(config.exceptional_position_fraction),
    }[tier]


def _correlation_buckets(
    source_db: str,
    tokens: set[str],
    snapshot: pd.Timestamp,
    config: BenchmarkConfig,
) -> dict[str, str]:
    """Group only statistically obvious recent positive-return correlations.

    Components are based solely on price observations available by the decision
    timestamp. Tokens without enough paired returns remain separate buckets.
    """
    ordered = sorted(str(t) for t in tokens)
    if len(ordered) < 2:
        return {token: token for token in ordered}
    start = snapshot - pd.Timedelta(minutes=float(config.correlation_lookback_minutes))
    placeholders = ",".join("?" for _ in ordered)
    sql = f"""
        SELECT token_key,snapshot_at,market_cap_usd FROM axiom_observations
        WHERE snapshot_at>=? AND snapshot_at<=? AND token_key IN ({placeholders})
          AND market_cap_usd>0
        ORDER BY snapshot_at
    """
    try:
        with sqlite3.connect(source_db, timeout=10.0) as src:
            src.execute("PRAGMA busy_timeout=10000")
            src.execute("PRAGMA query_only=ON")
            frame = pd.read_sql_query(sql, src, params=(start.isoformat(), snapshot.isoformat(), *ordered))
    except (sqlite3.Error, pd.errors.DatabaseError):
        return {token: token for token in ordered}
    if frame.empty:
        return {token: token for token in ordered}
    frame["snapshot_at"] = pd.to_datetime(frame["snapshot_at"], errors="coerce", utc=True)
    frame["market_cap_usd"] = pd.to_numeric(frame["market_cap_usd"], errors="coerce")
    frame = frame.dropna(subset=["snapshot_at", "market_cap_usd"])
    if frame.empty:
        return {token: token for token in ordered}
    prices = frame.pivot_table(
        index="snapshot_at", columns="token_key", values="market_cap_usd", aggfunc="last"
    ).sort_index()
    returns = np.log(prices.where(prices > 0)).diff()
    corr = returns.corr(min_periods=int(config.correlation_min_overlap))
    parent = {token: token for token in ordered}

    def find(token: str) -> str:
        while parent[token] != token:
            parent[token] = parent[parent[token]]
            token = parent[token]
        return token

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for i, left in enumerate(ordered):
        if left not in corr.index:
            continue
        for right in ordered[i + 1:]:
            value = corr.at[left, right] if right in corr.columns else np.nan
            if pd.notna(value) and float(value) >= float(config.correlation_threshold):
                union(left, right)
    return {token: find(token) for token in ordered}


def _allocation_amount(
    *,
    target_cash: float,
    available_cash: float,
    current_cash: float,
    execution_equity: float,
    committed_total: float,
    committed_bucket: float,
    config: BenchmarkConfig,
) -> float:
    """Apply per-position, total, correlated, and reserve limits in dollars."""
    equity = max(0.0, float(execution_equity))
    reserve_headroom = max(
        0.0,
        float(current_cash) - float(config.min_cash_reserve_fraction) * equity,
    )
    total_headroom = max(
        0.0,
        float(config.max_total_exposure_fraction) * equity - float(committed_total),
    )
    bucket_headroom = max(
        0.0,
        float(config.max_correlated_exposure_fraction) * equity - float(committed_bucket),
    )
    return max(
        0.0,
        min(
            float(target_cash),
            float(available_cash),
            reserve_headroom,
            total_headroom,
            bucket_headroom,
        ),
    )

def _entry_exit_rates(config: BenchmarkConfig) -> tuple[float, float]:
    half = max(0.0, config.friction_bps_round_trip) / 20000.0
    return half, half


def _is_disappearance_reason(reason: str | None) -> bool:
    text = str(reason or "").lower()
    return any(k in text for k in ("absence", "missing", "disappear", "dead_after", "unavailable"))


def _execution_proxy_mc(
    pos: sqlite3.Row, observed_mc: float, config: BenchmarkConfig, *, unavailable: bool
) -> float:
    """Execution-conservative MC proxy.

    When a price is actually present, the execution ledger uses it unchanged.
    When the token is unavailable, all downside is preserved, while only a
    configurable fraction of positive movement above entry is recognized. The
    default fraction is 0, so disappearance can never create an execution profit.
    """
    observed = max(0.0, float(observed_mc))
    if not unavailable:
        return observed
    entry = max(0.0, float(pos["entry_mc"]))
    if observed <= entry:
        return observed
    frac = min(1.0, max(0.0, float(config.disappearance_profit_recognition_fraction)))
    return entry + frac * (observed - entry)


def _proceeds_at_mc(pos: sqlite3.Row, mc: float, exit_fee_rate: float) -> tuple[float, float]:
    gross = float(pos["exposure_units"]) * float(mc)
    fee = gross * exit_fee_rate
    return gross - fee, fee


def _classify_exit(reason: str, price_available: bool) -> str:
    if not price_available or _is_disappearance_reason(reason):
        return "disappearance_terminal"
    if reason == "max_hold_72h":
        return "max_hold_observed_price"
    return "model_exit_observed_price"


def _backfill_closed_execution_columns(
    conn: sqlite3.Connection, benchmark_id: str, config: BenchmarkConfig
) -> None:
    _, exit_fee_rate = _entry_exit_rates(config)
    rows = conn.execute(
        """
        SELECT * FROM benchmark_positions_v22
        WHERE benchmark_id=? AND status='closed'
          AND (execution_realized_pnl_usd IS NULL OR exit_kind IS NULL)
        """,
        (benchmark_id,),
    ).fetchall()
    for pos in rows:
        observed_mc = float(pos["exit_mc"] if pos["exit_mc"] is not None else pos["last_mc"])
        reason = str(pos["close_reason"] or "legacy_close")
        unavailable = _is_disappearance_reason(reason)
        kind = _classify_exit(reason, not unavailable)
        exec_mc = _execution_proxy_mc(pos, observed_mc, config, unavailable=unavailable)
        observed_proceeds, observed_fee = _proceeds_at_mc(pos, observed_mc, exit_fee_rate)
        execution_proceeds, execution_fee = _proceeds_at_mc(pos, exec_mc, exit_fee_rate)
        spent = float(pos["entry_cash_spent_usd"])
        observed_pnl = observed_proceeds - spent
        execution_pnl = execution_proceeds - spent
        observed_ret = observed_pnl / spent if spent > 0 else 0.0
        execution_ret = execution_pnl / spent if spent > 0 else 0.0
        conn.execute(
            """
            UPDATE benchmark_positions_v22
            SET exit_kind=?, price_available_at_exit=?, exit_mc_observed=?,
                exit_mc_execution_proxy=?, observed_exit_fee_usd=?, execution_exit_fee_usd=?,
                observed_exit_proceeds_usd=?, execution_exit_proceeds_usd=?, observed_realized_pnl_usd=?,
                execution_realized_pnl_usd=?, observed_realized_return_pct=?,
                execution_realized_return_pct=?
            WHERE position_id=?
            """,
            (
                kind, int(not unavailable), observed_mc, exec_mc, observed_fee, execution_fee,
                observed_proceeds, execution_proceeds, observed_pnl, execution_pnl, observed_ret,
                execution_ret, pos["position_id"],
            ),
        )


def _execution_cash_from_ledger(
    conn: sqlite3.Connection, benchmark_id: str, initial_cash: float
) -> float:
    spent = float(conn.execute(
        "SELECT COALESCE(SUM(entry_cash_spent_usd),0) FROM benchmark_positions_v22 WHERE benchmark_id=?",
        (benchmark_id,),
    ).fetchone()[0] or 0.0)
    proceeds = float(conn.execute(
        """
        SELECT COALESCE(SUM(COALESCE(execution_exit_proceeds_usd, exit_proceeds_usd)),0)
        FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='closed'
        """,
        (benchmark_id,),
    ).fetchone()[0] or 0.0)
    return float(initial_cash) - spent + proceeds


def _liquidation_value(pos: sqlite3.Row, mc: float, exit_fee_rate: float) -> float:
    gross = float(pos["exposure_units"]) * float(mc)
    return gross * (1.0 - exit_fee_rate)


def _close_position(
    conn: sqlite3.Connection,
    pos: sqlite3.Row,
    snapshot: pd.Timestamp,
    exit_mc: float,
    reason: str,
    exit_fee_rate: float,
    config: BenchmarkConfig,
    *,
    price_available: bool,
) -> dict[str, Any]:
    observed_mc = float(exit_mc)
    exit_kind = _classify_exit(reason, price_available)
    unavailable = exit_kind == "disappearance_terminal"
    execution_mc = _execution_proxy_mc(pos, observed_mc, config, unavailable=unavailable)

    observed_proceeds, observed_exit_fee = _proceeds_at_mc(pos, observed_mc, exit_fee_rate)
    execution_proceeds, execution_exit_fee = _proceeds_at_mc(pos, execution_mc, exit_fee_rate)
    spent = float(pos["entry_cash_spent_usd"])
    observed_pnl = observed_proceeds - spent
    execution_pnl = execution_proceeds - spent
    observed_ret = observed_pnl / spent if spent > 0 else 0.0
    execution_ret = execution_pnl / spent if spent > 0 else 0.0

    conn.execute(
        """
        UPDATE benchmark_positions_v22
        SET status='closed', closed_at=?, exit_mc=?, exit_fee_usd=?, exit_proceeds_usd=?,
            realized_pnl_usd=?, realized_return_pct=?, close_reason=?, last_mc=?, last_mark_value_usd=?,
            exit_kind=?, price_available_at_exit=?, exit_mc_observed=?, exit_mc_execution_proxy=?,
            observed_exit_fee_usd=?, execution_exit_fee_usd=?,
            observed_exit_proceeds_usd=?, execution_exit_proceeds_usd=?,
            observed_realized_pnl_usd=?, execution_realized_pnl_usd=?,
            observed_realized_return_pct=?, execution_realized_return_pct=?
        WHERE position_id=?
        """,
        (
            snapshot.isoformat(), observed_mc, observed_exit_fee, observed_proceeds,
            observed_pnl, observed_ret, reason, observed_mc, observed_proceeds,
            exit_kind, int(price_available), observed_mc, execution_mc, observed_exit_fee, execution_exit_fee,
            observed_proceeds, execution_proceeds, observed_pnl, execution_pnl, observed_ret, execution_ret,
            pos["position_id"],
        ),
    )
    return {
        "position_id": pos["position_id"],
        "token_key": pos["token_key"],
        "reason": reason,
        "exit_kind": exit_kind,
        "price_available_at_exit": bool(price_available),
        "exit_mc_observed": observed_mc,
        "exit_mc_execution_proxy": execution_mc,
        "observed_exit_fee_usd": observed_exit_fee,
        "execution_exit_fee_usd": execution_exit_fee,
        "observed_proceeds_usd": observed_proceeds,
        "execution_proceeds_usd": execution_proceeds,
        "observed_realized_pnl_usd": observed_pnl,
        "execution_realized_pnl_usd": execution_pnl,
        "observed_realized_return_pct": observed_ret,
        "execution_realized_return_pct": execution_ret,
        # Compatibility aliases remain observed-price values.
        "exit_mc": observed_mc,
        "proceeds_usd": observed_proceeds,
        "realized_pnl_usd": observed_pnl,
        "realized_return_pct": observed_ret,
    }

def _last_closed_at(conn: sqlite3.Connection, benchmark_id: str, token: str) -> pd.Timestamp | None:
    row = conn.execute(
        """
        SELECT MAX(closed_at) FROM benchmark_positions_v22
        WHERE benchmark_id=? AND token_key=? AND status='closed'
        """,
        (benchmark_id, token),
    ).fetchone()
    return _to_ts(row[0]) if row and row[0] else None


def _state_with_position(state: dict[str, float], pos: sqlite3.Row, mc: float, snapshot: pd.Timestamp) -> dict[str, float]:
    entry = float(pos["entry_mc"])
    current_return = mc / entry - 1.0
    held = max(0.0, (snapshot - _to_ts(pos["opened_at"])).total_seconds() / 60.0)
    out = dict(state)
    out.update(
        {
            "position_return_pct": current_return,
            "position_minutes_held": held,
            "position_mfe_pct": max(float(pos["mfe_pct"]), current_return),
            "position_mae_pct": min(float(pos["mae_pct"]), current_return),
        }
    )
    return out


def _read_current(
    source_db: str,
    predictions_path: str,
    *,
    exact_prediction_snapshot: bool = False,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Read the selected observable board before joining frozen predictions.

    Normal callers select the newest board. Live paper decisions select the
    prediction frame's exact board so concurrent collection cannot mix an older
    forecast with newer prices. Historical features have already been
    incorporated into the prediction file, so this avoids materializing the
    entire observation history a second time on every cycle.
    """
    _require_v21()
    preds = _load_predictions(predictions_path)
    pred_snapshot = _prediction_snapshot(preds)
    with sqlite3.connect(source_db, timeout=10.0) as src:
        src.execute("PRAGMA busy_timeout=10000")
        src.execute("PRAGMA query_only=ON")
        if exact_prediction_snapshot and pred_snapshot is not None:
            # SQLite stores the collector's original ISO spelling, while pandas
            # may normalize ``Z``/``+00:00`` or remove trailing zero fractions.
            # Narrow to the second, then compare parsed instants exactly.
            candidates = src.execute(
                "SELECT DISTINCT snapshot_at FROM axiom_observations "
                "WHERE snapshot_at LIKE ? ORDER BY snapshot_at",
                (pred_snapshot.strftime("%Y-%m-%dT%H:%M:%S") + "%",),
            ).fetchall()
            matches = [row for row in candidates if _to_ts(row[0]) == pred_snapshot]
            latest = matches[0] if len(matches) == 1 else None
        else:
            latest = src.execute(
                "SELECT snapshot_at FROM axiom_observations "
                "ORDER BY snapshot_at DESC LIMIT 1"
            ).fetchone()
        if latest is None and exact_prediction_snapshot:
            raise RuntimeError(
                "Prediction snapshot does not identify exactly one durable Axiom board."
            )
        if latest is None:
            raise RuntimeError("No Axiom observations are available in the source database.")
        latest_snapshot_raw = str(latest[0])
        current_obs = pd.read_sql_query(
            """
            SELECT observation_id,token_key,market_cap_usd,name
            FROM axiom_observations
            WHERE snapshot_at=?
            ORDER BY observation_id
            """,
            src,
            params=(latest_snapshot_raw,),
        )

    if current_obs.empty:
        raise RuntimeError("No Axiom observations are available in the newest source snapshot.")
    snapshot = _to_ts(latest_snapshot_raw)
    current_obs["market_cap_usd"] = pd.to_numeric(
        current_obs["market_cap_usd"], errors="coerce"
    )
    current_obs = (
        current_obs.sort_values("observation_id")
        .drop_duplicates("token_key", keep="last")
        .sort_values("token_key")
    )
    current_obs = current_obs[["token_key", "market_cap_usd", "name"]]
    if pred_snapshot is not None and abs((pred_snapshot - snapshot).total_seconds()) > 180:
        raise RuntimeError(
            f"Prediction CSV is stale: predictions={pred_snapshot.isoformat()}, latest_capture={snapshot.isoformat()}"
        )
    current = current_obs.merge(preds, on="token_key", how="left", suffixes=("", "__pred"))
    if "market_cap_usd__pred" in current.columns:
        current = current.drop(columns=["market_cap_usd__pred"])
    current = current[current["market_cap_usd"].notna()].copy()
    return snapshot, current


def cycle(
    source_db: str, benchmark_db: str, predictions_path: str, forecast_model: str, policy_model: str, config: BenchmarkConfig,
) -> dict[str, Any]:
    """V24 forecasts use next-observation order fills; legacy benchmark models keep their original accounting semantics."""
    is_v24=False
    try:
        b=_load_bundle(forecast_model) or {}
        is_v24=bool(v24 is not None and str(b.get("schema_version",""))==v24.SCHEMA_VERSION)
        if not is_v24 and Path(predictions_path).exists():
            pr=_load_predictions(predictions_path); is_v24=bool("v24_model_hash" in pr.columns and pr["v24_model_hash"].notna().any())
    except Exception:
        is_v24=False
    fn=_cycle_v24 if is_v24 else _cycle_legacy
    return fn(source_db,benchmark_db,predictions_path,forecast_model,policy_model,config)


def _cycle_legacy(
    source_db: str,
    benchmark_db: str,
    predictions_path: str,
    forecast_model: str,
    policy_model: str,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    _require_v21()
    snapshot, current = _read_current(source_db, predictions_path)
    loaded_policy = _load_bundle(policy_model)
    available_policy_hash = _hash_file(policy_model)
    required_policy_schema = getattr(selfteach, "SCHEMA_VERSION", None)
    policy_accounting_compatible = bool(
        loaded_policy is None
        or required_policy_schema is None
        or str(loaded_policy.get("schema_version", "")) == str(required_policy_schema)
    )
    policy = loaded_policy if policy_accounting_compatible else None
    forecast_hash = _hash_file(forecast_model) or _hash_file(predictions_path)
    policy_hash = available_policy_hash if policy_accounting_compatible else None
    policy_version = (
        _policy_version(policy, policy_model)
        if policy_accounting_compatible
        else "bootstrap_pending_execution_accounting_rebootstrap"
    )
    entry_fee_rate, exit_fee_rate = _entry_exit_rates(config)

    with sqlite3.connect(benchmark_db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        acct = _account(conn)
        if acct is None:
            raise RuntimeError("Benchmark is not initialized. Run the init command first.")
        benchmark_id = str(acct["benchmark_id"])
        # Benchmark trading rules are frozen at initialization. A model/policy may
        # improve during the experiment, but position sizing, friction and exit
        # rules do not move the goalposts. Start a new benchmark to change them.
        config = _config_from_saved_json(acct["config_json"])
        _backfill_closed_execution_columns(conn, benchmark_id, config)
        execution_cash = _execution_cash_from_ledger(
            conn, benchmark_id, float(acct["initial_cash_usd"])
        )
        last_snapshot = _to_ts(acct["last_snapshot_at"]) if acct["last_snapshot_at"] else None
        if last_snapshot is not None and snapshot <= last_snapshot:
            return {
                "processed": False,
                "reason": "no new clipboard snapshot",
                "benchmark_id": benchmark_id,
                "snapshot_at": snapshot.isoformat(),
            }

        current_series = {str(r["token_key"]): r for _, r in current.iterrows()}
        open_positions = conn.execute(
            "SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open' ORDER BY opened_at",
            (benchmark_id,),
        ).fetchall()
        exits: list[dict[str, Any]] = []
        cash = float(acct["cash_usd"])
        open_tokens = {str(p["token_key"]) for p in open_positions}

        # Manage existing positions first.
        for pos in open_positions:
            token = str(pos["token_key"])
            row = current_series.get(token)
            if row is None:
                absent = max(0.0, (snapshot - _to_ts(pos["last_seen_at"])).total_seconds() / 60.0)
                observed_mc = float(pos["last_mc"])
                observed_liq = _liquidation_value(pos, observed_mc, exit_fee_rate)
                execution_mc = _execution_proxy_mc(pos, observed_mc, config, unavailable=True)
                execution_liq = _liquidation_value(pos, execution_mc, exit_fee_rate)
                current_return = observed_mc / float(pos["entry_mc"]) - 1.0
                terminal = absent >= config.missing_close_minutes
                conn.execute(
                    """
                    INSERT OR REPLACE INTO benchmark_marks_v22
                    (mark_id, benchmark_id, position_id, token_key, snapshot_at, market_cap_usd,
                     liquidation_value_usd, return_pct, mfe_pct, mae_pct, hold_score, action,
                     state_json, forecast_model_hash, policy_model_hash, price_available, mark_kind,
                     execution_liquidation_value_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()), benchmark_id, pos["position_id"], token, snapshot.isoformat(),
                        observed_mc, observed_liq, current_return, float(pos["mfe_pct"]),
                        float(pos["mae_pct"]), "DISAPPEARANCE_CLOSE" if terminal else "MISSING_HOLD",
                        _json({"missing_minutes": absent, "last_observed_mc": observed_mc}),
                        forecast_hash, policy_hash,
                        "disappearance_terminal" if terminal else "stale_last_observed", execution_liq,
                    ),
                )
                if terminal:
                    closed = _close_position(
                        conn, pos, snapshot, observed_mc, "dead_after_50m_absence", exit_fee_rate,
                        config, price_available=False,
                    )
                    cash += float(closed["observed_proceeds_usd"])
                    execution_cash += float(closed["execution_proceeds_usd"])
                    exits.append(closed)
                    open_tokens.discard(token)
                continue

            mc = float(row["market_cap_usd"])
            pred_state = _safe_state(row)
            mark_state = _state_with_position(pred_state, pos, mc, snapshot)
            current_return = mc / float(pos["entry_mc"]) - 1.0
            mfe = max(float(pos["mfe_pct"]), current_return)
            mae = min(float(pos["mae_pct"]), current_return)
            liquidation = _liquidation_value(pos, mc, exit_fee_rate)
            hold_score, hold_kind = _hold_score(mark_state, policy)
            held = max(0.0, (snapshot - _to_ts(pos["opened_at"])).total_seconds() / 60.0)
            action = "HOLD"
            reason = None
            if held >= config.max_hold_minutes:
                action, reason = "EXIT", "max_hold_72h"
            elif held >= config.min_hold_minutes and hold_score <= 0.0:
                action, reason = "EXIT", f"{hold_kind}_hold_value_nonpositive"

            conn.execute(
                """
                UPDATE benchmark_positions_v22
                SET last_seen_at=?, last_mc=?, last_mark_value_usd=?, mfe_pct=?, mae_pct=?
                WHERE position_id=?
                """,
                (snapshot.isoformat(), mc, liquidation, mfe, mae, pos["position_id"]),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_marks_v22
                (mark_id, benchmark_id, position_id, token_key, snapshot_at, market_cap_usd,
                 liquidation_value_usd, return_pct, mfe_pct, mae_pct, hold_score, action,
                 state_json, forecast_model_hash, policy_model_hash, price_available, mark_kind,
                 execution_liquidation_value_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'observed', ?)
                """,
                (
                    str(uuid.uuid4()), benchmark_id, pos["position_id"], token, snapshot.isoformat(),
                    mc, liquidation, current_return, mfe, mae, hold_score, action, _json(mark_state),
                    forecast_hash, policy_hash, liquidation,
                ),
            )
            if action == "EXIT":
                fresh = conn.execute(
                    "SELECT * FROM benchmark_positions_v22 WHERE position_id=?", (pos["position_id"],)
                ).fetchone()
                closed = _close_position(
                    conn, fresh, snapshot, mc, reason or "policy_exit", exit_fee_rate,
                    config, price_available=True,
                )
                cash += float(closed["observed_proceeds_usd"])
                execution_cash += float(closed["execution_proceeds_usd"])
                exits.append(closed)
                open_tokens.discard(token)

        # Re-query after exits and mark account equity before new allocations.
        # Position sizing is governed by the execution-conservative ledger. This
        # prevents a disappeared token's unverified observed gain from funding
        # later benchmark trades.
        open_positions = conn.execute(
            "SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open'",
            (benchmark_id,),
        ).fetchall()
        open_liq = sum(float(p["last_mark_value_usd"]) for p in open_positions)
        execution_open_before_entries = 0.0
        for p in open_positions:
            if str(p["token_key"]) in current_series:
                execution_open_before_entries += float(p["last_mark_value_usd"])
            else:
                proxy_mc = _execution_proxy_mc(p, float(p["last_mc"]), config, unavailable=True)
                execution_open_before_entries += _liquidation_value(p, proxy_mc, exit_fee_rate)
        effectiveness_equity_before_entries = execution_cash + execution_open_before_entries

        candidates: list[dict[str, Any]] = []
        for _, row in current.iterrows():
            token = str(row["token_key"])
            if token in open_tokens:
                continue
            state = _safe_state(row)
            if not state:
                continue
            score, kind = _entry_score(state, policy)
            if not math.isfinite(score):
                continue
            candidates.append(
                {
                    "token_key": token,
                    "market_cap_usd": float(row["market_cap_usd"]),
                    "state": state,
                    "score": score,
                    "score_kind": kind,
                }
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)

        slots = max(0, config.max_open_positions - len(open_positions))
        eligible: list[dict[str, Any]] = []
        for c in candidates:
            if c["score"] <= config.min_entry_score:
                continue
            last_close = _last_closed_at(conn, benchmark_id, c["token_key"])
            if last_close is not None and (snapshot - last_close).total_seconds() < config.reentry_cooldown_minutes * 60:
                continue
            eligible.append(c)
        selected = eligible[:slots]
        selected_keys = {c["token_key"] for c in selected}

        for c in candidates:
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_candidates_v22
                (benchmark_id, snapshot_at, token_key, market_cap_usd, entry_score, score_kind,
                 chosen, state_json, forecast_model_hash, policy_model_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    benchmark_id, snapshot.isoformat(), c["token_key"], c["market_cap_usd"], c["score"],
                    c["score_kind"], int(c["token_key"] in selected_keys), _json(c["state"]),
                    forecast_hash, policy_hash,
                ),
            )

        entries: list[dict[str, Any]] = []
        for c in selected:
            if cash <= 0.01:
                break
            # The fraction is a fraction of pre-entry equity INCLUDING the entry fee,
            # so five 20% slots can never overdraft a $1,000 account.
            target_cash_spend = max(0.0, effectiveness_equity_before_entries * config.position_fraction)
            cash_spend = min(cash, execution_cash, target_cash_spend)
            if cash_spend < 1.0:
                continue
            notional = cash_spend / (1.0 + entry_fee_rate)
            entry_fee = cash_spend - notional
            mc = float(c["market_cap_usd"])
            units = notional / mc
            position_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO benchmark_positions_v22
                (position_id, benchmark_id, token_key, opened_at, entry_mc, entry_notional_usd,
                 entry_fee_usd, entry_cash_spent_usd, exposure_units, entry_score, entry_score_kind,
                 entry_state_json, forecast_model_hash, policy_model_hash, policy_version, status,
                 last_seen_at, last_mc, last_mark_value_usd, mfe_pct, mae_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, 0, 0)
                """,
                (
                    position_id, benchmark_id, c["token_key"], snapshot.isoformat(), mc, notional,
                    entry_fee, cash_spend, units, c["score"], c["score_kind"], _json(c["state"]),
                    forecast_hash, policy_hash, policy_version, snapshot.isoformat(), mc,
                    notional * (1.0 - exit_fee_rate),
                ),
            )
            cash -= cash_spend
            execution_cash -= cash_spend
            entries.append(
                {
                    "position_id": position_id,
                    "token_key": c["token_key"],
                    "cash_spent_usd": cash_spend,
                    "entry_notional_usd": notional,
                    "entry_fee_usd": entry_fee,
                    "entry_mc": mc,
                    "entry_score": c["score"],
                }
            )
            open_tokens.add(c["token_key"])

        # Final end-of-cycle account mark. The original V22 fields remain the
        # observed-price ledger. The execution ledger discounts stale/unavailable
        # winners so disappearance cannot improve the effectiveness benchmark.
        open_positions = conn.execute(
            "SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open'",
            (benchmark_id,),
        ).fetchall()
        open_liq = 0.0
        execution_open_liq = 0.0
        unrealized = 0.0
        execution_unrealized = 0.0
        stale_open_positions = 0
        for p in open_positions:
            observed_mark = float(p["last_mark_value_usd"])
            open_liq += observed_mark
            unrealized += observed_mark - float(p["entry_cash_spent_usd"])
            if str(p["token_key"]) in current_series:
                execution_mark = observed_mark
            else:
                stale_open_positions += 1
                proxy_mc = _execution_proxy_mc(p, float(p["last_mc"]), config, unavailable=True)
                execution_mark = _liquidation_value(p, proxy_mc, exit_fee_rate)
            execution_open_liq += execution_mark
            execution_unrealized += execution_mark - float(p["entry_cash_spent_usd"])

        observed_realized = float(conn.execute(
            "SELECT COALESCE(SUM(realized_pnl_usd),0) FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='closed'",
            (benchmark_id,),
        ).fetchone()[0] or 0.0)
        execution_realized = float(conn.execute(
            """
            SELECT COALESCE(SUM(COALESCE(execution_realized_pnl_usd, realized_pnl_usd)),0)
            FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='closed'
            """,
            (benchmark_id,),
        ).fetchone()[0] or 0.0)
        # Recompute from the immutable ledger to prevent cash-accounting drift.
        execution_cash = _execution_cash_from_ledger(
            conn, benchmark_id, float(acct["initial_cash_usd"])
        )
        observed_equity = cash + open_liq
        execution_equity = execution_cash + execution_open_liq
        conn.execute(
            """
            INSERT OR REPLACE INTO benchmark_equity_v22
            (benchmark_id, snapshot_at, cash_usd, open_liquidation_value_usd, equity_usd,
             realized_pnl_usd, unrealized_pnl_usd, open_positions, forecast_model_hash,
             policy_model_hash, policy_version, observed_equity_usd, execution_cash_usd,
             execution_open_value_usd, execution_equity_usd, execution_realized_pnl_usd,
             execution_unrealized_pnl_usd, stale_open_positions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                benchmark_id, snapshot.isoformat(), cash, open_liq, observed_equity,
                observed_realized, unrealized, len(open_positions), forecast_hash, policy_hash,
                policy_version, observed_equity, execution_cash, execution_open_liq,
                execution_equity, execution_realized, execution_unrealized, stale_open_positions,
            ),
        )
        conn.execute(
            """
            UPDATE benchmark_account_v22
            SET cash_usd=?, execution_cash_usd=?, last_snapshot_at=?
            WHERE benchmark_id=?
            """,
            (cash, execution_cash, snapshot.isoformat(), benchmark_id),
        )
        conn.commit()

    return {
        "processed": True,
        "benchmark_id": benchmark_id,
        "snapshot_at": snapshot.isoformat(),
        "starting_budget_usd": config.initial_cash_usd,
        "cash_usd": cash,
        "execution_cash_usd": execution_cash,
        # Compatibility: equity_usd/total_return_pct remain observed-price aliases.
        "equity_usd": observed_equity,
        "total_return_pct": observed_equity / config.initial_cash_usd - 1.0,
        "observed_equity_usd": observed_equity,
        "observed_total_return_pct": observed_equity / config.initial_cash_usd - 1.0,
        "execution_conservative_equity_usd": execution_equity,
        "execution_conservative_total_return_pct": execution_equity / config.initial_cash_usd - 1.0,
        "effectiveness_equity_usd": execution_equity,
        "effectiveness_total_return_pct": execution_equity / config.initial_cash_usd - 1.0,
        "effectiveness_accounting": "execution_conservative",
        "stale_open_positions": stale_open_positions,
        "open_positions": len(open_positions),
        "entries": entries,
        "exits": exits,
        "forecast_model_hash": forecast_hash,
        "policy_model_hash": policy_hash,
        "policy_version": policy_version,
        "policy_accounting_compatible": policy_accounting_compatible,
        "ignored_incompatible_policy_hash": available_policy_hash if not policy_accounting_compatible else None,
        "training_feedback": "disabled",
    }


def _cycle_v24(
    source_db:str,benchmark_db:str,predictions_path:str,forecast_model:str,policy_model:str,config:BenchmarkConfig,
)->dict[str,Any]:
    _require_v21(); snapshot,current=_read_current(source_db,predictions_path)
    loaded_policy=_load_bundle(policy_model); forecast_bundle=_load_bundle(forecast_model); available_policy_hash=_hash_file(policy_model)
    required_policy_schema=getattr(selfteach,"SCHEMA_VERSION",None)
    is_v24_forecast=bool((v24 is not None and forecast_bundle is not None and str(forecast_bundle.get("schema_version",""))==v24.SCHEMA_VERSION) or "v24_model_hash" in current.columns)
    policy_compatible=bool(loaded_policy is None or required_policy_schema is None or str(loaded_policy.get("schema_version",""))==str(required_policy_schema))
    if is_v24_forecast and loaded_policy is not None:
        policy_compatible=bool(policy_compatible and str(loaded_policy.get("v24_policy_schema","")).startswith("v24_") and bool(loaded_policy.get("oos_only",False)))
    policy=loaded_policy if policy_compatible else None
    forecast_hash=_hash_file(forecast_model) or _hash_file(predictions_path); policy_hash=available_policy_hash if policy_compatible else None
    policy_version=_policy_version(policy,policy_model) if policy_compatible else "bootstrap_pending_v24_policy_promotion"
    entry_fee_rate,exit_fee_rate=_entry_exit_rates(config)

    # Successful capture heartbeat lives in the source DB, never the benchmark DB.
    if is_v24_forecast and v24 is not None:
        try:
            with sqlite3.connect(source_db) as src:
                v24.migrate(src); v24.record_capture_heartbeat(src,snapshot,valid_capture=True,row_count=len(current),source="benchmark_v24")
        except Exception: pass

    def terminal_absence(last_seen:pd.Timestamp)->tuple[bool,float]:
        elapsed=max(0.0,(snapshot-last_seen).total_seconds()/60.0)
        if is_v24_forecast and v24 is not None:
            try:
                with sqlite3.connect(source_db) as src:
                    v24.migrate(src); rs,re,n=v24._contiguous_capture_absence(src,last_seen,v24.V24Config(),upto=snapshot)
                    if rs is None or re is None:return False,elapsed
                    valid=max(0.0,(re-last_seen).total_seconds()/60.0)
                    return bool(valid>=config.missing_close_minutes and n>=v24.V24Config().heartbeat_min_valid_captures_for_death),valid
            except Exception:return False,elapsed
        return elapsed>=config.missing_close_minutes,elapsed

    with sqlite3.connect(benchmark_db) as conn:
        conn.row_factory=sqlite3.Row; migrate(conn); acct=_account(conn)
        if acct is None: raise RuntimeError("Benchmark is not initialized. Run the init command first.")
        bid=str(acct["benchmark_id"]); config=_config_from_saved_json(acct["config_json"]); entry_fee_rate,exit_fee_rate=_entry_exit_rates(config)
        _backfill_closed_execution_columns(conn,bid,config)
        last_snapshot=_to_ts(acct["last_snapshot_at"]) if acct["last_snapshot_at"] else None
        if last_snapshot is not None and snapshot<=last_snapshot:return {"processed":False,"reason":"no new clipboard snapshot","benchmark_id":bid,"snapshot_at":snapshot.isoformat()}
        current_series={str(r["token_key"]):r for _,r in current.iterrows()}
        cash=float(acct["cash_usd"]); execution_cash=_execution_cash_from_ledger(conn,bid,float(acct["initial_cash_usd"])); entries=[]; exits=[]; cancelled=[]

        # Fill pending entries only at a later observable token value.
        pending=conn.execute("SELECT * FROM benchmark_pending_entries_v24 WHERE benchmark_id=? AND status='pending' ORDER BY decision_at",(bid,)).fetchall()
        for pen in pending:
            if snapshot<=_to_ts(pen["decision_at"]):continue
            token=str(pen["token_key"]); row=current_series.get(token)
            if row is None:
                terminal,mins=terminal_absence(_to_ts(pen["decision_at"]))
                if terminal:
                    conn.execute("UPDATE benchmark_pending_entries_v24 SET status='cancelled',cancelled_at=?,cancel_reason=? WHERE pending_id=?",(snapshot.isoformat(),"unavailable_before_next_observable_fill",pen["pending_id"]))
                    cancelled.append({"token_key":token,"missing_minutes":mins})
                continue
            reserved=min(float(pen["reserved_cash_usd"]),cash,execution_cash)
            if reserved<1.0:
                conn.execute("UPDATE benchmark_pending_entries_v24 SET status='cancelled',cancelled_at=?,cancel_reason='insufficient_cash_at_fill' WHERE pending_id=?",(snapshot.isoformat(),pen["pending_id"]));continue
            notional=reserved/(1.0+entry_fee_rate); fee=reserved-notional; mc=float(row["market_cap_usd"]); units=notional/mc; pid=str(uuid.uuid4())
            conn.execute("""INSERT INTO benchmark_positions_v22
                (position_id,benchmark_id,token_key,opened_at,entry_mc,entry_notional_usd,entry_fee_usd,entry_cash_spent_usd,
                 exposure_units,entry_score,entry_score_kind,entry_state_json,forecast_model_hash,policy_model_hash,policy_version,status,
                 last_seen_at,last_mc,last_mark_value_usd,mfe_pct,mae_pct,entry_decision_at,entry_fill_kind)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?,?,?,0,0,?,'next_observable')""",
                (pid,bid,token,snapshot.isoformat(),mc,notional,fee,reserved,units,pen["entry_score"],pen["entry_score_kind"],pen["entry_state_json"],
                 pen["forecast_model_hash"],pen["policy_model_hash"],pen["policy_version"],snapshot.isoformat(),mc,notional*(1-exit_fee_rate),pen["decision_at"]))
            conn.execute("UPDATE benchmark_pending_entries_v24 SET status='filled',filled_at=?,fill_mc=? WHERE pending_id=?",(snapshot.isoformat(),mc,pen["pending_id"]))
            cash-=reserved; execution_cash-=reserved; entries.append({"position_id":pid,"token_key":token,"decision_mc":float(pen["decision_mc"]),"entry_mc":mc,"cash_spent_usd":reserved,"fill_kind":"next_observable"})

        open_positions=conn.execute("SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open' ORDER BY opened_at",(bid,)).fetchall(); open_tokens={str(p["token_key"]) for p in open_positions}
        # EXIT decisions also fill one observation later.
        for pos in open_positions:
            token=str(pos["token_key"]); row=current_series.get(token); pending_exit=_to_ts(pos["pending_exit_at"]) if pos["pending_exit_at"] else None
            if row is None:
                terminal,absent=terminal_absence(_to_ts(pos["last_seen_at"])); observed_mc=float(pos["last_mc"]); observed_liq=_liquidation_value(pos,observed_mc,exit_fee_rate); proxy=_execution_proxy_mc(pos,observed_mc,config,unavailable=True); exliq=_liquidation_value(pos,proxy,exit_fee_rate); ret=observed_mc/float(pos["entry_mc"])-1
                conn.execute("""INSERT OR REPLACE INTO benchmark_marks_v22
                    (mark_id,benchmark_id,position_id,token_key,snapshot_at,market_cap_usd,liquidation_value_usd,return_pct,mfe_pct,mae_pct,hold_score,action,state_json,forecast_model_hash,policy_model_hash,price_available,mark_kind,execution_liquidation_value_usd)
                    VALUES(?,?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,0,?,?)""",
                    (str(uuid.uuid4()),bid,pos["position_id"],token,snapshot.isoformat(),observed_mc,observed_liq,ret,float(pos["mfe_pct"]),float(pos["mae_pct"]),
                     "DISAPPEARANCE_CLOSE" if terminal else "MISSING_HOLD",_json({"missing_minutes":absent}),forecast_hash,policy_hash,"disappearance_terminal" if terminal else "stale_last_observed",exliq))
                if terminal:
                    closed=_close_position(conn,pos,snapshot,observed_mc,str(pos["pending_exit_reason"] or "dead_after_valid_capture_absence"),exit_fee_rate,config,price_available=False)
                    cash+=closed["observed_proceeds_usd"]; execution_cash+=closed["execution_proceeds_usd"]; exits.append(closed); open_tokens.discard(token)
                continue
            mc=float(row["market_cap_usd"])
            if pending_exit is not None and snapshot>pending_exit:
                fresh=conn.execute("SELECT * FROM benchmark_positions_v22 WHERE position_id=?",(pos["position_id"],)).fetchone(); closed=_close_position(conn,fresh,snapshot,mc,str(pos["pending_exit_reason"] or "policy_exit_next_observable"),exit_fee_rate,config,price_available=True)
                cash+=closed["observed_proceeds_usd"]; execution_cash+=closed["execution_proceeds_usd"]; exits.append(closed); open_tokens.discard(token);continue
            state=_safe_state(row); mark_state=_state_with_position(state,pos,mc,snapshot); ret=mc/float(pos["entry_mc"])-1; mfe=max(float(pos["mfe_pct"]),ret); mae=min(float(pos["mae_pct"]),ret); liq=_liquidation_value(pos,mc,exit_fee_rate); hold_score,kind=_hold_score(mark_state,policy); held=max(0.0,(snapshot-_to_ts(pos["opened_at"])).total_seconds()/60.0)
            action="HOLD";reason=None
            if held>=config.max_hold_minutes:action,reason="EXIT_DECISION","max_hold_72h"
            elif held>=config.min_hold_minutes and hold_score<=0:action,reason="EXIT_DECISION",f"{kind}_hold_value_nonpositive"
            conn.execute("UPDATE benchmark_positions_v22 SET last_seen_at=?,last_mc=?,last_mark_value_usd=?,mfe_pct=?,mae_pct=? WHERE position_id=?",(snapshot.isoformat(),mc,liq,mfe,mae,pos["position_id"]))
            conn.execute("""INSERT OR REPLACE INTO benchmark_marks_v22
                (mark_id,benchmark_id,position_id,token_key,snapshot_at,market_cap_usd,liquidation_value_usd,return_pct,mfe_pct,mae_pct,hold_score,action,state_json,forecast_model_hash,policy_model_hash,price_available,mark_kind,execution_liquidation_value_usd)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,'observed',?)""",
                (str(uuid.uuid4()),bid,pos["position_id"],token,snapshot.isoformat(),mc,liq,ret,mfe,mae,hold_score,action,_json(mark_state),forecast_hash,policy_hash,liq))
            if action=="EXIT_DECISION":conn.execute("UPDATE benchmark_positions_v22 SET pending_exit_at=?,pending_exit_reason=? WHERE position_id=?",(snapshot.isoformat(),reason,pos["position_id"]))

        # Conservative equity before decisions, with pending capital reserved.
        open_positions=conn.execute("SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open'",(bid,)).fetchall(); reserved=float(conn.execute("SELECT COALESCE(SUM(reserved_cash_usd),0) FROM benchmark_pending_entries_v24 WHERE benchmark_id=? AND status='pending'",(bid,)).fetchone()[0] or 0.0)
        exec_open=0.0
        for p in open_positions:
            if str(p["token_key"]) in current_series:exec_open+=float(p["last_mark_value_usd"])
            else:exec_open+=_liquidation_value(p,_execution_proxy_mc(p,float(p["last_mc"]),config,unavailable=True),exit_fee_rate)
        effect_equity=execution_cash+exec_open
        candidates=[]
        open_tokens={str(p["token_key"]) for p in open_positions}; pending_tokens={str(r[0]) for r in conn.execute("SELECT token_key FROM benchmark_pending_entries_v24 WHERE benchmark_id=? AND status='pending'",(bid,)).fetchall()}
        for _,row in current.iterrows():
            token=str(row["token_key"])
            if token in open_tokens or token in pending_tokens:continue
            state=_safe_state(row)
            if not state:continue
            score,kind=_entry_score(state,policy)
            if math.isfinite(score):candidates.append({"token_key":token,"market_cap_usd":float(row["market_cap_usd"]),"state":state,"score":score,"kind":kind})
        candidates.sort(key=lambda x:x["score"],reverse=True); slots=max(0,config.max_open_positions-len(open_positions)-len(pending_tokens)); eligible=[]
        for c in candidates:
            if c["score"]<=config.min_entry_score:continue
            lc=_last_closed_at(conn,bid,c["token_key"])
            if lc is not None and (snapshot-lc).total_seconds()<config.reentry_cooldown_minutes*60:continue
            eligible.append(c)
        selected=eligible[:slots]; selected_keys={c["token_key"] for c in selected}
        for c in candidates:
            conn.execute("""INSERT OR REPLACE INTO benchmark_candidates_v22(benchmark_id,snapshot_at,token_key,market_cap_usd,entry_score,score_kind,chosen,state_json,forecast_model_hash,policy_model_hash) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                         (bid,snapshot.isoformat(),c["token_key"],c["market_cap_usd"],c["score"],c["kind"],int(c["token_key"] in selected_keys),_json(c["state"]),forecast_hash,policy_hash))
        pending_created=[]; available_cash=max(0.0,min(cash,execution_cash)-reserved)
        for c in selected:
            target=max(0.0,effect_equity*config.position_fraction); reserve=min(available_cash,target)
            if reserve<1.0:break
            pid=str(uuid.uuid4()); conn.execute("""INSERT INTO benchmark_pending_entries_v24(pending_id,benchmark_id,token_key,decision_at,decision_mc,reserved_cash_usd,entry_score,entry_score_kind,entry_state_json,forecast_model_hash,policy_model_hash,policy_version,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?, 'pending')""",
                (pid,bid,c["token_key"],snapshot.isoformat(),c["market_cap_usd"],reserve,c["score"],c["kind"],_json(c["state"]),forecast_hash,policy_hash,policy_version))
            pending_created.append({"pending_id":pid,"token_key":c["token_key"],"reserved_cash_usd":reserve,"decision_mc":c["market_cap_usd"]});available_cash-=reserve

        # Final mark. Pending reservations remain cash, but unavailable for new orders.
        open_positions=conn.execute("SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open'",(bid,)).fetchall(); open_liq=exec_open_liq=unreal=exunreal=0.0; stale=0
        for p in open_positions:
            obsval=float(p["last_mark_value_usd"]);open_liq+=obsval;unreal+=obsval-float(p["entry_cash_spent_usd"])
            if str(p["token_key"]) in current_series:exval=obsval
            else:stale+=1;exval=_liquidation_value(p,_execution_proxy_mc(p,float(p["last_mc"]),config,unavailable=True),exit_fee_rate)
            exec_open_liq+=exval;exunreal+=exval-float(p["entry_cash_spent_usd"])
        observed_realized=float(conn.execute("SELECT COALESCE(SUM(realized_pnl_usd),0) FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='closed'",(bid,)).fetchone()[0] or 0.0); execution_realized=float(conn.execute("SELECT COALESCE(SUM(COALESCE(execution_realized_pnl_usd,realized_pnl_usd)),0) FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='closed'",(bid,)).fetchone()[0] or 0.0)
        execution_cash=_execution_cash_from_ledger(conn,bid,float(acct["initial_cash_usd"])); observed_equity=cash+open_liq; execution_equity=execution_cash+exec_open_liq
        conn.execute("""INSERT OR REPLACE INTO benchmark_equity_v22(benchmark_id,snapshot_at,cash_usd,open_liquidation_value_usd,equity_usd,realized_pnl_usd,unrealized_pnl_usd,open_positions,forecast_model_hash,policy_model_hash,policy_version,observed_equity_usd,execution_cash_usd,execution_open_value_usd,execution_equity_usd,execution_realized_pnl_usd,execution_unrealized_pnl_usd,stale_open_positions) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (bid,snapshot.isoformat(),cash,open_liq,observed_equity,observed_realized,unreal,len(open_positions),forecast_hash,policy_hash,policy_version,observed_equity,execution_cash,exec_open_liq,execution_equity,execution_realized,exunreal,stale))
        conn.execute("UPDATE benchmark_account_v22 SET cash_usd=?,execution_cash_usd=?,last_snapshot_at=? WHERE benchmark_id=?",(cash,execution_cash,snapshot.isoformat(),bid));conn.commit()
    return {"processed":True,"benchmark_id":bid,"snapshot_at":snapshot.isoformat(),"cash_usd":cash,"execution_cash_usd":execution_cash,
            "observed_equity_usd":observed_equity,"effectiveness_equity_usd":execution_equity,"effectiveness_total_return_pct":execution_equity/config.initial_cash_usd-1,
            "effectiveness_accounting":"execution_conservative_next_observable","open_positions":len(open_positions),"entries":entries,"pending_entries":pending_created,
            "cancelled_pending":cancelled,"exits":exits,"forecast_model_hash":forecast_hash,"policy_model_hash":policy_hash,"policy_version":policy_version,
            "policy_accounting_compatible":policy_compatible,"training_feedback":"disabled"}


def _max_drawdown(equity: pd.Series) -> float | None:
    if equity.empty:
        return None
    peak_s = equity.cummax()
    dd = equity / peak_s - 1.0
    return float(dd.min())



def friction_stress_curves_from_closed(closed: pd.DataFrame, initial_cash: float, bps_values=(100.0,300.0,500.0,1000.0)) -> dict[str, Any]:
    """Replay the same closed trade set under alternative round-trip friction.

    This does not pretend position selection/sizing would be identical under a
    different cost regime; it is an intentionally transparent fixed-trade stress.
    """
    out={}
    if closed.empty:
        return {f"{int(b)}bps": {"equity_usd": float(initial_cash), "return_pct": 0.0, "trades": 0} for b in bps_values}
    for b in bps_values:
        half=max(0.0,float(b))/20000.0; pnl=0.0
        for _,r in closed.iterrows():
            notional=float(r.get("entry_notional_usd") or 0.0); units=float(r.get("exposure_units") or 0.0)
            exit_mc=r.get("exit_mc_execution_proxy")
            if pd.isna(exit_mc): exit_mc=r.get("exit_mc")
            if notional<=0 or units<=0 or pd.isna(exit_mc): continue
            stressed_spent=notional*(1.0+half); stressed_proceeds=units*float(exit_mc)*(1.0-half); pnl+=stressed_proceeds-stressed_spent
        equity=float(initial_cash)+pnl; out[f"{int(b)}bps"]={"equity_usd":equity,"pnl_usd":pnl,"return_pct":equity/float(initial_cash)-1.0,"trades":int(len(closed))}
    return out

def status(db: str) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        acct = _account(conn)
        if acct is None:
            return {"initialized": False, "schema_version": SCHEMA_VERSION}
        bid = str(acct["benchmark_id"])
        config = _config_from_saved_json(acct["config_json"])
        _backfill_closed_execution_columns(conn, bid, config)
        execution_cash = _execution_cash_from_ledger(conn, bid, float(acct["initial_cash_usd"]))
        conn.execute(
            "UPDATE benchmark_account_v22 SET execution_cash_usd=? WHERE benchmark_id=?",
            (execution_cash, bid),
        )
        conn.commit()

        eq = pd.read_sql_query(
            "SELECT * FROM benchmark_equity_v22 WHERE benchmark_id=? ORDER BY snapshot_at",
            conn, params=(bid,),
        )
        closed = pd.read_sql_query(
            "SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='closed' ORDER BY closed_at",
            conn, params=(bid,),
        )
        open_ = pd.read_sql_query(
            "SELECT * FROM benchmark_positions_v22 WHERE benchmark_id=? AND status='open' ORDER BY opened_at",
            conn, params=(bid,),
        )

        observed_equity = (
            float(eq.iloc[-1]["observed_equity_usd"])
            if not eq.empty and pd.notna(eq.iloc[-1].get("observed_equity_usd"))
            else (float(eq.iloc[-1]["equity_usd"]) if not eq.empty else float(acct["cash_usd"]))
        )
        initial = float(acct["initial_cash_usd"])
        last_snapshot = _to_ts(acct["last_snapshot_at"]) if acct["last_snapshot_at"] else None
        _, exit_fee_rate = _entry_exit_rates(config)

        execution_open_value = 0.0
        stale_open = 0
        if not open_.empty:
            # Convert rows to dict-like SQLite-compatible values for the proxy helper.
            for _, row in open_.iterrows():
                entry_mc = float(row["entry_mc"])
                observed_mc = float(row["last_mc"])
                last_seen = _to_ts(row["last_seen_at"])
                unavailable = last_snapshot is not None and last_seen < last_snapshot
                if unavailable:
                    stale_open += 1
                    if observed_mc > entry_mc:
                        frac = min(1.0, max(0.0, float(config.disappearance_profit_recognition_fraction)))
                        proxy_mc = entry_mc + frac * (observed_mc - entry_mc)
                    else:
                        proxy_mc = observed_mc
                else:
                    proxy_mc = observed_mc
                gross = float(row["exposure_units"]) * proxy_mc
                execution_open_value += gross * (1.0 - exit_fee_rate)

        execution_equity = execution_cash + execution_open_value
        observed_realized = float(closed["realized_pnl_usd"].sum()) if not closed.empty else 0.0
        execution_realized = (
            float(closed["execution_realized_pnl_usd"].fillna(closed["realized_pnl_usd"]).sum())
            if not closed.empty else 0.0
        )

        def _trade_metrics(pnl_col: str, ret_col: str) -> dict[str, Any]:
            if closed.empty:
                return {"win_rate": None, "mean_return_pct": None, "median_return_pct": None, "profit_factor": None}
            pnl = pd.to_numeric(closed[pnl_col], errors="coerce").fillna(0.0)
            ret = pd.to_numeric(closed[ret_col], errors="coerce")
            gross_profit = float(pnl[pnl > 0].sum())
            gross_loss = float(-pnl[pnl < 0].sum())
            pf = gross_profit / gross_loss if gross_loss > 1e-12 else (None if gross_profit == 0 else float("inf"))
            return {
                "win_rate": float((pnl > 0).mean()),
                "mean_return_pct": float(ret.mean()) if ret.notna().any() else None,
                "median_return_pct": float(ret.median()) if ret.notna().any() else None,
                "profit_factor": pf,
            }

        observed_metrics = _trade_metrics("realized_pnl_usd", "realized_return_pct")
        if not closed.empty:
            closed["_exec_pnl"] = closed["execution_realized_pnl_usd"].fillna(closed["realized_pnl_usd"])
            closed["_exec_ret"] = closed["execution_realized_return_pct"].fillna(closed["realized_return_pct"])
            exec_pnl = closed["_exec_pnl"]
            exec_ret = closed["_exec_ret"]
            gp = float(exec_pnl[exec_pnl > 0].sum())
            gl = float(-exec_pnl[exec_pnl < 0].sum())
            execution_metrics = {
                "win_rate": float((exec_pnl > 0).mean()),
                "mean_return_pct": float(exec_ret.mean()),
                "median_return_pct": float(exec_ret.median()),
                "profit_factor": gp / gl if gl > 1e-12 else (None if gp == 0 else float("inf")),
            }
        else:
            execution_metrics = {"win_rate": None, "mean_return_pct": None, "median_return_pct": None, "profit_factor": None}

        recent_cols = [
            "token_key", "opened_at", "closed_at", "entry_cash_spent_usd", "close_reason", "exit_kind",
            "price_available_at_exit", "exit_mc_observed", "exit_mc_execution_proxy",
            "observed_realized_pnl_usd", "execution_realized_pnl_usd",
            "observed_realized_return_pct", "execution_realized_return_pct",
        ]
        recent = [] if closed.empty else closed.tail(10)[recent_cols].to_dict("records")

        by_model = []
        if not closed.empty:
            grouped = closed.groupby(["forecast_model_hash", "policy_model_hash"], dropna=False).agg(
                trades=("position_id", "count"),
                observed_pnl_usd=("realized_pnl_usd", "sum"),
                execution_pnl_usd=("_exec_pnl", "sum"),
                observed_mean_return_pct=("realized_return_pct", "mean"),
                execution_mean_return_pct=("_exec_ret", "mean"),
            ).reset_index()
            by_model = grouped.to_dict("records")

        disappearance = closed[closed["exit_kind"] == "disappearance_terminal"] if not closed.empty else closed
        disappearance_count = int(len(disappearance)) if not closed.empty else 0
        disappearance_observed_pnl = float(disappearance["realized_pnl_usd"].sum()) if disappearance_count else 0.0
        disappearance_execution_pnl = float(disappearance["_exec_pnl"].sum()) if disappearance_count else 0.0

        observed_dd = _max_drawdown(eq["equity_usd"]) if not eq.empty else None
        execution_dd = None
        if not eq.empty and "execution_equity_usd" in eq.columns:
            exec_eq = pd.to_numeric(eq["execution_equity_usd"], errors="coerce").dropna()
            execution_dd = _max_drawdown(exec_eq) if not exec_eq.empty else None
        friction_stress=friction_stress_curves_from_closed(closed,initial,(100.0,300.0,500.0,1000.0))
        pending_row = conn.execute(
            """SELECT COUNT(*),COALESCE(SUM(reserved_cash_usd),0)
               FROM benchmark_pending_entries_v24 WHERE benchmark_id=? AND status='pending'""",
            (bid,),
        ).fetchone()
        pending_count = int(pending_row[0])
        pending_exposure = float(pending_row[1] or 0.0)
        open_cost_exposure = (
            float(pd.to_numeric(open_["entry_cash_spent_usd"], errors="coerce").fillna(0.0).sum())
            if not open_.empty else 0.0
        )
        committed_exposure = open_cost_exposure + pending_exposure
        committed_fraction = committed_exposure / execution_equity if execution_equity > 0 else None
        swing_summary = {
            "enabled": bool(config.recurrent_swing_enabled),
            "active_watches": 0,
            "reentries_completed": 0,
            "tokens_with_multiple_swings": 0,
            "maximum_swing_sequence": 1,
        }
        watch_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='benchmark_swing_watch_v24'"
        ).fetchone()
        if watch_exists:
            watch_counts = dict(conn.execute(
                """SELECT status,COUNT(*) FROM benchmark_swing_watch_v24
                WHERE benchmark_id=? GROUP BY status""",
                (bid,),
            ).fetchall())
            swing_summary["active_watches"] = int(watch_counts.get("active", 0))
            swing_summary["reentries_completed"] = int(watch_counts.get("reentered", 0))
        if "swing_sequence" in open_.columns or "swing_sequence" in closed.columns:
            sequences = pd.concat([
                pd.to_numeric(open_.get("swing_sequence", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(closed.get("swing_sequence", pd.Series(dtype=float)), errors="coerce"),
            ]).dropna()
            if not sequences.empty:
                swing_summary["maximum_swing_sequence"] = int(sequences.max())
            repeated = conn.execute(
                """SELECT COUNT(*) FROM (
                    SELECT token_key FROM benchmark_positions_v22
                    WHERE benchmark_id=? GROUP BY token_key HAVING MAX(swing_sequence)>1
                )""",
                (bid,),
            ).fetchone()
            swing_summary["tokens_with_multiple_swings"] = int(repeated[0] if repeated else 0)

        return {
            "initialized": True,
            "schema_version": SCHEMA_VERSION,
            "benchmark_id": bid,
            "created_at": acct["created_at"],
            "initial_budget_usd": initial,
            # Compatibility fields retain the original observed-price interpretation.
            "cash_usd": float(acct["cash_usd"]),
            "equity_usd": observed_equity,
            "total_pnl_usd": observed_equity - initial,
            "total_return_pct": observed_equity / initial - 1.0,
            "realized_pnl_usd": observed_realized,
            "win_rate": observed_metrics["win_rate"],
            "mean_trade_return_pct": observed_metrics["mean_return_pct"],
            "median_trade_return_pct": observed_metrics["median_return_pct"],
            "profit_factor": observed_metrics["profit_factor"],
            "max_equity_drawdown_pct": observed_dd,

            "observed_market": {
                "cash_usd": float(acct["cash_usd"]),
                "equity_usd": observed_equity,
                "total_pnl_usd": observed_equity - initial,
                "total_return_pct": observed_equity / initial - 1.0,
                "realized_pnl_usd": observed_realized,
                **observed_metrics,
                "max_equity_drawdown_pct": observed_dd,
            },
            "execution_conservative": {
                "cash_usd": execution_cash,
                "open_value_usd": execution_open_value,
                "equity_usd": execution_equity,
                "total_pnl_usd": execution_equity - initial,
                "total_return_pct": execution_equity / initial - 1.0,
                "realized_pnl_usd": execution_realized,
                **execution_metrics,
                "max_equity_drawdown_pct": execution_dd,
                "stale_open_positions": stale_open,
                "disappearance_profit_recognition_fraction": config.disappearance_profit_recognition_fraction,
            },
            # This is the authoritative effectiveness track.
            "effectiveness_accounting": "execution_conservative_next_observable",
            "effectiveness_equity_usd": execution_equity,
            "effectiveness_total_pnl_usd": execution_equity - initial,
            "effectiveness_total_return_pct": execution_equity / initial - 1.0,
            "open_positions": int(len(open_)),
            "pending_entry_orders": pending_count,
            "munger_sizing": {
                "ordinary_position_fraction": config.ordinary_position_fraction,
                "strong_position_fraction": config.strong_position_fraction,
                "exceptional_position_fraction": config.exceptional_position_fraction,
                "max_total_exposure_fraction": config.max_total_exposure_fraction,
                "max_correlated_exposure_fraction": config.max_correlated_exposure_fraction,
                "min_cash_reserve_fraction": config.min_cash_reserve_fraction,
                "open_cost_exposure_usd": open_cost_exposure,
                "pending_reserved_exposure_usd": pending_exposure,
                "committed_exposure_usd": committed_exposure,
                "committed_exposure_fraction": committed_fraction,
            },
            "recurrent_swing_policy": swing_summary,
            "friction_stress_fixed_trade_replay": friction_stress,
            "closed_trades": int(len(closed)),
            "disappearance_terminal_trades": disappearance_count,
            "disappearance_observed_pnl_usd": disappearance_observed_pnl,
            "disappearance_execution_pnl_usd": disappearance_execution_pnl,
            "last_snapshot_at": acct["last_snapshot_at"],
            "training_feedback": "disabled; benchmark data is isolated from V24 training",
            "model_version_performance": by_model,
            "recent_closed_trades": recent,
        }

def export(db: str, out_dir: str) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        acct = _account(conn)
        if acct is None:
            raise RuntimeError("Benchmark is not initialized.")
        bid = str(acct["benchmark_id"])
        tables = {
            "equity": "benchmark_equity_v22",
            "positions": "benchmark_positions_v22",
            "marks": "benchmark_marks_v22",
            "candidates": "benchmark_candidates_v22",
        }
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='benchmark_swing_watch_v24'"
        ).fetchone():
            tables["swing_watches"] = "benchmark_swing_watch_v24"
        paths: dict[str, str] = {}
        for name, table in tables.items():
            df = pd.read_sql_query(f"SELECT * FROM {table} WHERE benchmark_id=?", conn, params=(bid,))
            path = out / f"v22_1000_{name}.csv"
            df.to_csv(path, index=False)
            paths[name] = str(path)
        summary_path = out / "v22_1000_status.json"
        summary_path.write_text(json.dumps(status(db), indent=2, default=str), encoding="utf-8")
        paths["status"] = str(summary_path)
        return paths


def refresh_predictions(
    source_db: str,
    forecast_model: str,
    predictions_path: str,
    *,
    require_latest: bool = True,
) -> dict[str, Any]:
    _require_v21()
    if not Path(forecast_model).exists():
        raise RuntimeError(f"Forecast champion does not exist: {forecast_model}")
    try:
        bundle = joblib.load(forecast_model)
    except Exception:
        bundle = {}
    if v24 is not None and str(bundle.get("schema_version", "")) == v24.SCHEMA_VERSION:
        prediction_file = Path(predictions_path)
        if prediction_file.exists():
            existing = _load_predictions(predictions_path)
            predicted_at = _prediction_snapshot(existing)
            current_at = v24._latest_observation_timestamp(source_db)
            expected_hash = _hash_file(forecast_model)
            stored_hashes = set(
                existing.get("v24_model_hash", pd.Series(dtype=object)).dropna().astype(str)
            )
            if (
                predicted_at is not None
                and current_at is not None
                and predicted_at == current_at
                and stored_hashes == {str(expected_hash)}
            ):
                return {
                    "rows": len(existing),
                    "out": predictions_path,
                    "forecaster": "v24",
                    "skipped": True,
                    "reason": "prediction_already_current_for_frozen_model",
                    "snapshot_at": predicted_at.isoformat(),
                }
        raw_cfg = bundle.get("config") if isinstance(bundle, dict) else None
        allowed = set(v24.V24Config.__dataclass_fields__)
        fallback = v24.V24Config()
        cfg_values = {}
        for name, value in (raw_cfg.items() if isinstance(raw_cfg, dict) else ()):
            if name not in allowed:
                continue
            if isinstance(getattr(fallback, name, None), tuple) and isinstance(value, list):
                value = tuple(value)
            cfg_values[name] = value
        cfg = v24.V24Config(**cfg_values)
        # This isolated benchmark has training feedback disabled. Its prediction
        # refresh must therefore remain read-only against the collector database;
        # cycle() records benchmark decisions in benchmark_db instead.
        prediction_kwargs = {"persist_source": False}
        if not require_latest:
            prediction_kwargs["require_latest"] = False
        rows = v24.predict_current(
            source_db, forecast_model, predictions_path, cfg, **prediction_kwargs,
        )
        snapshots = set(pd.to_datetime(rows["snapshot_at"], format="ISO8601", utc=True))
        if len(snapshots) != 1:
            raise RuntimeError("V24 prediction did not produce exactly one snapshot.")
        return {
            "rows": len(rows),
            "out": predictions_path,
            "forecaster": "v24",
            "skipped": False,
            "snapshot_at": next(iter(snapshots)).isoformat(),
            "source_db_writes": False,
            "training_feedback": "disabled",
        }
    return selfteach.refresh_current_predictions(source_db, forecast_model, predictions_path, None)


def loop(
    source_db: str,
    benchmark_db: str,
    predictions_path: str,
    forecast_model: str,
    policy_model: str,
    config: BenchmarkConfig,
    interval_seconds: float,
) -> None:
    while True:
        started = time.monotonic()
        try:
            pred = refresh_predictions(source_db, forecast_model, predictions_path)
            result = cycle(source_db, benchmark_db, predictions_path, forecast_model, policy_model, config)
            print(json.dumps({"predictions": pred, "benchmark": result}, indent=2, default=str), flush=True)
        except Exception as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2), flush=True)
        elapsed = time.monotonic() - started
        time.sleep(max(1.0, float(interval_seconds) - elapsed))


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    d = BenchmarkConfig()
    vals = {k: getattr(args, k, getattr(d, k)) for k in asdict(d)}
    return BenchmarkConfig(**vals)


def _add_config_args(p: argparse.ArgumentParser) -> None:
    d = BenchmarkConfig()
    p.add_argument("--initial-cash-usd", type=float, default=d.initial_cash_usd)
    p.add_argument("--max-open-positions", type=int, default=d.max_open_positions)
    p.add_argument("--position-fraction", type=float, default=d.position_fraction)
    p.add_argument("--ordinary-position-fraction", type=float, default=d.ordinary_position_fraction)
    p.add_argument("--strong-position-fraction", type=float, default=d.strong_position_fraction)
    p.add_argument("--exceptional-position-fraction", type=float, default=d.exceptional_position_fraction)
    p.add_argument("--max-total-exposure-fraction", type=float, default=d.max_total_exposure_fraction)
    p.add_argument("--min-cash-reserve-fraction", type=float, default=d.min_cash_reserve_fraction)
    p.add_argument("--max-correlated-exposure-fraction", type=float, default=d.max_correlated_exposure_fraction)
    p.add_argument("--strong-score-quantile", type=float, default=d.strong_score_quantile)
    p.add_argument("--exceptional-score-quantile", type=float, default=d.exceptional_score_quantile)
    p.add_argument(
        "--conviction-calibration-min-scores", type=int,
        default=d.conviction_calibration_min_scores,
    )
    p.add_argument(
        "--conviction-calibration-max-scores", type=int,
        default=d.conviction_calibration_max_scores,
    )
    p.add_argument("--correlation-lookback-minutes", type=float, default=d.correlation_lookback_minutes)
    p.add_argument("--correlation-min-overlap", type=int, default=d.correlation_min_overlap)
    p.add_argument("--correlation-threshold", type=float, default=d.correlation_threshold)
    p.add_argument("--min-entry-score", type=float, default=d.min_entry_score)
    p.add_argument("--min-hold-minutes", type=float, default=d.min_hold_minutes)
    p.add_argument("--max-hold-minutes", type=float, default=d.max_hold_minutes)
    p.add_argument("--missing-close-minutes", type=float, default=d.missing_close_minutes)
    p.add_argument("--reentry-cooldown-minutes", type=float, default=d.reentry_cooldown_minutes)
    p.add_argument("--friction-bps-round-trip", type=float, default=d.friction_bps_round_trip)
    p.add_argument(
        "--no-recurrent-swing", dest="recurrent_swing_enabled", action="store_false",
        default=d.recurrent_swing_enabled,
    )
    p.add_argument("--swing-entry-min-probability", type=float, default=d.swing_entry_min_probability)
    p.add_argument("--swing-calibration-min-samples", type=int, default=d.swing_calibration_min_samples)
    p.add_argument("--swing-calibration-min-tokens", type=int, default=d.swing_calibration_min_tokens)
    p.add_argument("--swing-calibration-max-samples", type=int, default=d.swing_calibration_max_samples)
    p.add_argument("--swing-entry-max-occurrence-minutes", type=float, default=d.swing_entry_max_occurrence_minutes)
    p.add_argument("--swing-entry-min-net-upside", type=float, default=d.swing_entry_min_net_upside)
    p.add_argument("--swing-peak-boundary-minutes", type=float, default=d.swing_peak_boundary_minutes)
    p.add_argument("--swing-peak-boundary-probability", type=float, default=d.swing_peak_boundary_probability)
    p.add_argument("--swing-exit-min-return", type=float, default=d.swing_exit_min_return)
    p.add_argument("--swing-hold-later-probability", type=float, default=d.swing_hold_later_probability)
    p.add_argument("--swing-hold-min-second-upside", type=float, default=d.swing_hold_min_second_upside)
    p.add_argument("--swing-hold-max-second-gap-minutes", type=float, default=d.swing_hold_max_second_gap_minutes)
    p.add_argument("--swing-reentry-min-minutes", type=float, default=d.swing_reentry_min_minutes)
    p.add_argument("--swing-reentry-min-retrace", type=float, default=d.swing_reentry_min_retrace)
    p.add_argument("--swing-watch-min-later-probability", type=float, default=d.swing_watch_min_later_probability)
    p.add_argument("--swing-watch-min-second-upside", type=float, default=d.swing_watch_min_second_upside)
    p.add_argument("--swing-watch-max-minutes", type=float, default=d.swing_watch_max_minutes)
    p.add_argument(
        "--disappearance-profit-recognition-fraction", type=float,
        default=d.disappearance_profit_recognition_fraction,
        help="Fraction of positive last-observed MC movement recognized in execution accounting when unavailable (default 0).",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="V22.1 isolated $1,000 benchmark using the active V24 leakage-hardened 1-minute / 72-hour champion models"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="Start a fresh isolated $1,000 benchmark account")
    p.add_argument("--benchmark-db", default=DEFAULT_BENCHMARK_DB)
    p.add_argument("--reset", action="store_true")
    _add_config_args(p)

    p = sub.add_parser("cycle", help="Run one benchmark decision cycle from the latest V21 predictions")
    p.add_argument("--source-db", default=DEFAULT_SOURCE_DB)
    p.add_argument("--benchmark-db", default=DEFAULT_BENCHMARK_DB)
    p.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    p.add_argument("--forecast-model", default=DEFAULT_PEAK_MODEL)
    p.add_argument("--policy-model", default=DEFAULT_POLICY_MODEL)
    p.add_argument("--refresh-predictions", action="store_true")
    _add_config_args(p)

    p = sub.add_parser("loop", help="Continuously refresh V21 predictions and run the isolated benchmark")
    p.add_argument("--source-db", default=DEFAULT_SOURCE_DB)
    p.add_argument("--benchmark-db", default=DEFAULT_BENCHMARK_DB)
    p.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    p.add_argument("--forecast-model", default=DEFAULT_PEAK_MODEL)
    p.add_argument("--policy-model", default=DEFAULT_POLICY_MODEL)
    p.add_argument("--interval-seconds", type=float, default=60.0)
    _add_config_args(p)

    p = sub.add_parser("status", help="Show dollar P&L and effectiveness statistics")
    p.add_argument("--benchmark-db", default=DEFAULT_BENCHMARK_DB)

    p = sub.add_parser("export", help="Export equity curve, trades, marks, candidates and summary")
    p.add_argument("--benchmark-db", default=DEFAULT_BENCHMARK_DB)
    p.add_argument("--out-dir", default="data/v22_1000_benchmark")

    args = ap.parse_args()
    if args.cmd == "init":
        result = init_benchmark(args.benchmark_db, _config_from_args(args), reset=args.reset)
    elif args.cmd == "cycle":
        if args.refresh_predictions:
            refresh_predictions(args.source_db, args.forecast_model, args.predictions)
        result = cycle(
            args.source_db, args.benchmark_db, args.predictions,
            args.forecast_model, args.policy_model, _config_from_args(args),
        )
    elif args.cmd == "loop":
        loop(
            args.source_db, args.benchmark_db, args.predictions,
            args.forecast_model, args.policy_model, _config_from_args(args), args.interval_seconds,
        )
        return
    elif args.cmd == "status":
        result = status(args.benchmark_db)
    elif args.cmd == "export":
        result = export(args.benchmark_db, args.out_dir)
    else:  # pragma: no cover
        raise AssertionError(args.cmd)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
