from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - runtime dependency check
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - runtime dependency check
    XGBClassifier = None
    XGBRegressor = None

from sklearn.metrics import log_loss, mean_absolute_error, mean_pinball_loss


SCHEMA_VERSION = "v21_peak_structure_72h_incremental_v1"
LABEL_TABLE = "axiom_peak_structure_labels_v21"
DEFAULT_MODEL = "models/axiom_peak_v21/champion.joblib"
DEFAULT_OUT = "data/axiom_peak_structure_predictions_72h.csv"


@dataclass(frozen=True)
class PeakStructureConfig:
    # A substantial swing must first rise meaningfully from its local trough...
    min_runup_pct: float = 0.20
    # ...and then be confirmed by a meaningful retracement from the candidate high.
    confirm_retrace_pct: float = 0.15
    # Later peak must clear the first peak by this much to count as truly higher.
    higher_peak_margin_pct: float = 0.02
    # Axiom now exposes up to 72h; forecast the remaining visible lifecycle.
    horizon_minutes: int = 72 * 60
    # Operational death rule already used by the project.
    death_missed_cycles: int = 50
    # Elapsed-time death guard preserves the old ~50 minute meaning at 1m sampling.
    death_gap_minutes: float = 50.0
    # Axiom natural age-out should be a terminal boundary, not a death event.
    age_out_minutes: int = 71 * 60
    # Successful capture heartbeat guards token-death inference against collector outages.
    heartbeat_max_gap_minutes: float = 5.0
    heartbeat_min_valid_captures: int = 10
    # Ignore pathological duplicate timestamps/noisy instantaneous wiggles.
    min_peak_separation_minutes: float = 1.0


@dataclass
class SwingPeak:
    trough_at: str
    trough_price: float
    peak_at: str
    peak_price: float
    confirmed_at: str
    confirmation_price: float
    runup_pct: float
    confirmation_retrace_pct: float


TOKEN_CANDIDATES = ("token_key", "token", "token_id", "mint", "token_address")
TIME_CANDIDATES = ("snapshot_at", "decision_at", "observed_at", "captured_at", "as_of", "timestamp")
MC_CANDIDATES = ("market_cap_usd", "mc_usd", "market_cap", "market_capitalization_usd")
AGE_CANDIDATES = ("age_minutes", "token_age_minutes", "age_mins")
NAME_CANDIDATES = ("name", "token_name", "ticker")

LEAKAGE_PATTERNS = (
    "target", "label", "future", "outcome", "success", "failure", "hit_plus",
    "hit_minus", "return_", "max_return", "min_return", "time_to_peak",
    "peak_multiple", "dead_", "death", "terminal", "drawdown_before_peak",
)
EXCLUDED_OPERATIONAL_PATTERNS = (
    "capture_count", "visibility_bonus", "consecutive_capture", "reappearance_count",
    "observation_index", "obs_count_", "interval_minutes", "row_index", "screenshot_rows",
)


def _utc_iso(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _to_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="ISO8601", utc=True, errors="coerce")


def _table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]


def _all_tables(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )]


def _pick(cols: Iterable[str], candidates: Iterable[str]) -> str | None:
    cset = {c.lower(): c for c in cols}
    for candidate in candidates:
        if candidate.lower() in cset:
            return cset[candidate.lower()]
    return None


def discover_observation_source(conn: sqlite3.Connection) -> dict[str, str]:
    scored: list[tuple[int, str, dict[str, str]]] = []
    for table in _all_tables(conn):
        cols = _table_columns(conn, table)
        token = _pick(cols, TOKEN_CANDIDATES)
        time_col = _pick(cols, TIME_CANDIDATES)
        mc = _pick(cols, MC_CANDIDATES)
        if not (token and time_col and mc):
            continue
        age = _pick(cols, AGE_CANDIDATES)
        name = _pick(cols, NAME_CANDIDATES)
        lname = table.lower()
        score = 0
        if "axiom" in lname:
            score += 8
        if "migrated" in lname:
            score += 5
        if "observation" in lname or "observations" in lname:
            score += 6
        if "raw" in lname:
            score += 2
        if "feature" in lname:
            score -= 8
        if "label" in lname or "outcome" in lname or "prediction" in lname:
            score -= 20
        if table == LABEL_TABLE:
            score -= 100
        mapping = {"table": table, "token": token, "time": time_col, "mc": mc}
        if age:
            mapping["age"] = age
        if name:
            mapping["name"] = name
        scored.append((score, table, mapping))
    if not scored:
        details = {t: _table_columns(conn, t) for t in _all_tables(conn)}
        raise RuntimeError(
            "Could not discover an Axiom observation table with token, timestamp, and market-cap columns. "
            f"Available schema: {json.dumps(details, default=str)[:8000]}"
        )
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def load_observations(conn: sqlite3.Connection) -> tuple[pd.DataFrame, dict[str, str]]:
    source = discover_observation_source(conn)
    table = source["table"]
    cols = _table_columns(conn, table)
    # Pull all ordinary columns; this gives fallback feature engineering access to the
    # rich clipboard values without relying on any particular V18 table name.
    quoted = ", ".join(f'"{c}"' for c in cols)
    df = pd.read_sql_query(f'SELECT {quoted} FROM "{table}"', conn)
    rename = {
        source["token"]: "token_key",
        source["time"]: "snapshot_at",
        source["mc"]: "market_cap_usd",
    }
    if "age" in source:
        rename[source["age"]] = "age_minutes"
    if "name" in source:
        rename[source["name"]] = "name"
    df = df.rename(columns=rename)
    df["snapshot_at"] = _to_timestamp(df["snapshot_at"])
    df["market_cap_usd"] = pd.to_numeric(df["market_cap_usd"], errors="coerce")
    df = df[df["token_key"].notna() & df["snapshot_at"].notna() & (df["market_cap_usd"] > 0)].copy()
    df["token_key"] = df["token_key"].astype(str)
    if "age_minutes" in df.columns:
        df["age_minutes"] = pd.to_numeric(df["age_minutes"], errors="coerce")
    df = df.sort_values(["token_key", "snapshot_at"]).drop_duplicates(
        ["token_key", "snapshot_at"], keep="last"
    )
    return df.reset_index(drop=True), source


def find_substantial_peaks(
    path: pd.DataFrame,
    config: PeakStructureConfig,
) -> list[SwingPeak]:
    """Detect retracement-confirmed swing peaks.

    A candidate high is not called a peak simply because the next sample is lower.
    It must (a) have risen at least ``min_runup_pct`` from the local trough and
    (b) subsequently retrace at least ``confirm_retrace_pct``. This makes the
    target stable across 1m and historical 5m sampling and allows several peaks
    in one lifecycle.
    """
    if path.empty:
        return []
    p = path[["snapshot_at", "market_cap_usd"]].dropna().sort_values("snapshot_at")
    if len(p) < 3:
        return []

    first = p.iloc[0]
    trough_price = float(first.market_cap_usd)
    trough_at = first.snapshot_at
    high_price = trough_price
    high_at = trough_at
    peaks: list[SwingPeak] = []
    last_peak_time: pd.Timestamp | None = None

    for row in p.iloc[1:].itertuples(index=False):
        ts = row.snapshot_at
        price = float(row.market_cap_usd)

        if high_price <= trough_price and price < trough_price:
            trough_price = price
            trough_at = ts
            high_price = price
            high_at = ts
            continue

        if price > high_price:
            high_price = price
            high_at = ts
            continue

        runup = high_price / trough_price - 1.0 if trough_price > 0 else 0.0
        retrace = 1.0 - price / high_price if high_price > 0 else 0.0
        separation_ok = (
            last_peak_time is None
            or (high_at - last_peak_time).total_seconds() / 60.0 >= config.min_peak_separation_minutes
        )

        if runup >= config.min_runup_pct and retrace >= config.confirm_retrace_pct and separation_ok:
            peaks.append(
                SwingPeak(
                    trough_at=_utc_iso(trough_at),
                    trough_price=trough_price,
                    peak_at=_utc_iso(high_at),
                    peak_price=high_price,
                    confirmed_at=_utc_iso(ts),
                    confirmation_price=price,
                    runup_pct=runup,
                    confirmation_retrace_pct=retrace,
                )
            )
            last_peak_time = high_at
            # Start the next swing from the price that confirmed the retracement.
            trough_price = price
            trough_at = ts
            high_price = price
            high_at = ts
            continue

        # Before a qualifying swing exists, a new low resets the local trough.
        if runup < config.min_runup_pct and price < trough_price:
            trough_price = price
            trough_at = ts
            high_price = price
            high_at = ts

    return peaks


def _capture_index(observations: pd.DataFrame) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, set[str]]]:
    groups = observations.groupby("snapshot_at")["token_key"].agg(lambda x: set(map(str, x)))
    times = list(groups.index.sort_values())
    return times, {t: groups.loc[t] for t in times}


def _contiguous_absence_run(
    last_seen: pd.Timestamp,
    capture_times: list[pd.Timestamp],
    config: PeakStructureConfig,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    """Return the latest contiguous run of successful captures after ``last_seen``.

    A collector outage is a gap in successful capture heartbeats, not evidence that
    a token disappeared.  If the first post-token capture is itself separated from
    ``last_seen`` by more than ``heartbeat_max_gap_minutes``, the missing clock
    starts at that first successful capture rather than at the stale token mark.
    """
    post = sorted(_to_timestamp(pd.Series(capture_times)).dropna()) if capture_times else []
    post = [t for t in post if t > last_seen]
    if not post:
        return None, None, 0
    max_gap = pd.Timedelta(minutes=float(config.heartbeat_max_gap_minutes))
    run_start = post[0]
    count = 1
    # A normal next capture can continue from last_seen; a long outage cannot.
    if post[0] - last_seen <= max_gap:
        run_start = last_seen
    prev = post[0]
    for t in post[1:]:
        if t - prev > max_gap:
            run_start = t
            count = 1
        else:
            count += 1
        prev = t
    return run_start, prev, count


def infer_token_terminal(
    token_df: pd.DataFrame,
    capture_times: list[pd.Timestamp],
    presence: dict[pd.Timestamp, set[str]],
    config: PeakStructureConfig,
) -> tuple[pd.Timestamp | None, str | None]:
    """Infer terminal state only from a contiguous run of *successful* captures.

    Elapsed wall time alone is insufficient: a browser/collector outage must be
    censored rather than converted into hundreds of simultaneous token deaths.
    """
    if token_df.empty:
        return None, None
    last = token_df.sort_values("snapshot_at").iloc[-1]
    last_seen = pd.Timestamp(last.snapshot_at)
    age = None
    if "age_minutes" in token_df.columns:
        raw = pd.to_numeric(pd.Series([last.get("age_minutes")]), errors="coerce").iloc[0]
        if pd.notna(raw):
            age = float(raw)
    if age is not None and age >= config.age_out_minutes:
        return last_seen, "age_out_72h_window"

    run_start, run_end, valid_count = _contiguous_absence_run(last_seen, capture_times, config)
    if run_start is None or run_end is None:
        return None, None
    absent_minutes = max(0.0, (run_end - run_start).total_seconds() / 60.0)
    if (
        absent_minutes >= config.death_gap_minutes
        and valid_count >= int(config.heartbeat_min_valid_captures)
    ):
        return run_start + pd.Timedelta(minutes=config.death_gap_minutes), "dead_after_valid_capture_absence"
    return None, None


PEAK_EVENT_TABLE = "axiom_peak_events_v21"
STATE_TABLE = "axiom_peak_structure_state_v21"
FEATURE_CACHE_TABLE = "axiom_peak_features_v21"
TARGET_COLUMNS_V21 = (
    "has_next_substantial_peak_before_terminal_72h",
    "time_to_next_substantial_peak_minutes_72h",
    "next_substantial_peak_market_cap_usd_72h",
    "next_substantial_peak_multiple_72h",
    "next_substantial_peak_prominence_pct_72h",
    "next_substantial_peak_confirmation_retrace_pct_72h",
    "next_peak_post_retracement_max_pct_72h",
    "substantial_peaks_count_before_terminal_72h",
    "later_higher_peak_before_terminal_72h",
    "later_higher_peak_within_1h_after_next",
    "later_higher_peak_within_4h_after_next",
    "later_higher_peak_within_12h_after_next",
    "later_higher_peak_within_24h_after_next",
    "later_higher_peak_within_48h_after_next",
    "time_from_next_to_later_higher_peak_minutes_72h",
    "later_higher_peak_multiple_vs_next_72h",
    "later_higher_peak_multiple_vs_decision_72h",
)


def _ensure_column(conn: sqlite3.Connection, table: str, name: str, decl: str) -> None:
    cols = set(_table_columns(conn, table))
    if name not in cols:
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {decl}')


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {LABEL_TABLE} (
            token_key TEXT NOT NULL,
            decision_at TEXT NOT NULL,
            decision_market_cap_usd REAL NOT NULL,
            schema_version TEXT NOT NULL,
            label_status_next_peak TEXT NOT NULL,
            terminal_at TEXT,
            terminal_reason TEXT,
            path_end_at TEXT,
            has_next_substantial_peak_before_terminal_72h INTEGER,
            next_substantial_peak_at TEXT,
            next_substantial_peak_confirmed_at TEXT,
            time_to_next_substantial_peak_minutes_72h REAL,
            next_substantial_peak_market_cap_usd_72h REAL,
            next_substantial_peak_multiple_72h REAL,
            next_substantial_peak_prominence_pct_72h REAL,
            next_substantial_peak_confirmation_retrace_pct_72h REAL,
            next_peak_post_retracement_max_pct_72h REAL,
            substantial_peaks_count_before_terminal_72h INTEGER,
            later_higher_peak_before_terminal_72h INTEGER,
            later_higher_peak_within_1h_after_next INTEGER,
            later_higher_peak_within_4h_after_next INTEGER,
            later_higher_peak_within_12h_after_next INTEGER,
            later_higher_peak_within_24h_after_next INTEGER,
            later_higher_peak_within_48h_after_next INTEGER,
            later_higher_peak_at TEXT,
            time_from_next_to_later_higher_peak_minutes_72h REAL,
            later_higher_peak_multiple_vs_next_72h REAL,
            later_higher_peak_multiple_vs_decision_72h REAL,
            label_ready_at TEXT,
            label_finalized INTEGER NOT NULL DEFAULT 0,
            learning_updated_at TEXT,
            target_fingerprint TEXT,
            config_json TEXT NOT NULL,
            PRIMARY KEY (token_key, decision_at)
        )
        """
    )
    # Safe upgrades when an early V21 database already exists.
    _ensure_column(conn, LABEL_TABLE, "later_higher_peak_within_24h_after_next", "INTEGER")
    _ensure_column(conn, LABEL_TABLE, "later_higher_peak_within_48h_after_next", "INTEGER")
    _ensure_column(conn, LABEL_TABLE, "label_finalized", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, LABEL_TABLE, "learning_updated_at", "TEXT")
    _ensure_column(conn, LABEL_TABLE, "target_fingerprint", "TEXT")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{LABEL_TABLE}_decision ON {LABEL_TABLE}(decision_at)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{LABEL_TABLE}_token ON {LABEL_TABLE}(token_key)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{LABEL_TABLE}_learn ON {LABEL_TABLE}(learning_updated_at)")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PEAK_EVENT_TABLE} (
            token_key TEXT NOT NULL,
            peak_at TEXT NOT NULL,
            trough_at TEXT NOT NULL,
            trough_price REAL NOT NULL,
            peak_price REAL NOT NULL,
            confirmed_at TEXT NOT NULL,
            confirmation_price REAL NOT NULL,
            runup_pct REAL NOT NULL,
            confirmation_retrace_pct REAL NOT NULL,
            config_json TEXT NOT NULL,
            PRIMARY KEY(token_key, peak_at)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
            token_key TEXT PRIMARY KEY,
            last_observation_at TEXT,
            last_peak_refresh_at TEXT,
            config_json TEXT
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FEATURE_CACHE_TABLE} (
            token_key TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            features_json TEXT NOT NULL,
            feature_schema_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(token_key, snapshot_at)
        )
        """
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{FEATURE_CACHE_TABLE}_time ON {FEATURE_CACHE_TABLE}(snapshot_at)")
    conn.commit()


def _event_known(
    first_peak_time: pd.Timestamp,
    later_peak_time: pd.Timestamp | None,
    window_minutes: int,
    path_end: pd.Timestamp,
    terminal_complete: bool,
) -> int | None:
    if later_peak_time is not None:
        delta = (later_peak_time - first_peak_time).total_seconds() / 60.0
        if delta <= window_minutes:
            return 1
    deadline = first_peak_time + pd.Timedelta(minutes=window_minutes)
    if path_end >= deadline or terminal_complete:
        return 0
    return None


def _target_fingerprint(record: dict[str, Any]) -> str:
    payload = {k: record.get(k) for k in TARGET_COLUMNS_V21}
    # Event identity matters even when a numerical target is unchanged.
    payload.update({
        "next_substantial_peak_at": record.get("next_substantial_peak_at"),
        "later_higher_peak_at": record.get("later_higher_peak_at"),
        "terminal_reason": record.get("terminal_reason"),
        "label_finalized": record.get("label_finalized"),
    })
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _peaks_for_token(conn: sqlite3.Connection, token: str) -> list[SwingPeak]:
    rows = conn.execute(
        f"""SELECT trough_at,trough_price,peak_at,peak_price,confirmed_at,confirmation_price,
                   runup_pct,confirmation_retrace_pct
            FROM {PEAK_EVENT_TABLE} WHERE token_key=? ORDER BY peak_at""",
        (token,),
    ).fetchall()
    return [SwingPeak(*r) for r in rows]


def _refresh_token_peaks(
    conn: sqlite3.Connection,
    token: str,
    token_df: pd.DataFrame,
    config: PeakStructureConfig,
) -> tuple[int, int]:
    peaks = find_substantial_peaks(token_df.sort_values("snapshot_at"), config)
    old = conn.execute(f"SELECT COUNT(*) FROM {PEAK_EVENT_TABLE} WHERE token_key=?", (token,)).fetchone()[0]
    # Recompute only this changed token. Confirmed historical events are deterministic;
    # replacing the token slice is cheap and avoids touching unrelated history.
    conn.execute(f"DELETE FROM {PEAK_EVENT_TABLE} WHERE token_key=?", (token,))
    cfg = json.dumps(asdict(config), sort_keys=True)
    if peaks:
        conn.executemany(
            f"""INSERT INTO {PEAK_EVENT_TABLE}
                (token_key,trough_at,trough_price,peak_at,peak_price,confirmed_at,confirmation_price,
                 runup_pct,confirmation_retrace_pct,config_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [(
                token, p.trough_at, p.trough_price, p.peak_at, p.peak_price, p.confirmed_at,
                p.confirmation_price, p.runup_pct, p.confirmation_retrace_pct, cfg,
            ) for p in peaks],
        )
    last_obs = _utc_iso(token_df.snapshot_at.max())
    conn.execute(
        f"""INSERT INTO {STATE_TABLE}(token_key,last_observation_at,last_peak_refresh_at,config_json)
            VALUES(?,?,?,?)
            ON CONFLICT(token_key) DO UPDATE SET
              last_observation_at=excluded.last_observation_at,
              last_peak_refresh_at=excluded.last_peak_refresh_at,
              config_json=excluded.config_json""",
        (token, last_obs, datetime.now(timezone.utc).isoformat(), cfg),
    )
    return int(old), len(peaks)


def label_decision_from_events(
    token_df: pd.DataFrame,
    decision_idx: int,
    latest_capture: pd.Timestamp,
    terminal_at: pd.Timestamp | None,
    terminal_reason: str | None,
    config: PeakStructureConfig,
    peaks: list[SwingPeak],
) -> dict[str, Any]:
    row = token_df.iloc[decision_idx]
    decision_at = pd.Timestamp(row.snapshot_at)
    entry = float(row.market_cap_usd)
    horizon_end = decision_at + pd.Timedelta(minutes=config.horizon_minutes)
    effective_end = horizon_end
    terminal_complete = False
    if terminal_at is not None and decision_at <= terminal_at <= horizon_end:
        effective_end = terminal_at
        terminal_complete = True
    elif latest_capture >= horizon_end:
        terminal_complete = True

    path_end = min(latest_capture, effective_end)
    # Strict horizon semantics: a retracement-confirmed substantial peak belongs to
    # the target only when BOTH the high and the confirmation occur inside the
    # decision's effective horizon.  A high at +71h confirmed at +75h is therefore
    # right-censored for a 72h substantial-peak target, not back-dated as positive.
    future_peaks = [
        p for p in peaks
        if decision_at < pd.Timestamp(p.peak_at) <= effective_end
        and decision_at < pd.Timestamp(p.confirmed_at) <= effective_end
    ]
    future_peaks.sort(key=lambda p: pd.Timestamp(p.peak_at))
    confirmed_future = [p for p in future_peaks if pd.Timestamp(p.confirmed_at) <= path_end]
    next_peak = confirmed_future[0] if confirmed_future else None
    later_higher = None
    if next_peak is not None:
        threshold = next_peak.peak_price * (1.0 + config.higher_peak_margin_pct)
        for p in confirmed_future[1:]:
            if p.peak_price > threshold:
                later_higher = p
                break

    if next_peak is not None:
        label_status, has_next, ready_at = "positive_confirmed", 1, next_peak.confirmed_at
    elif terminal_complete:
        label_status, has_next, ready_at = "complete_no_peak", 0, _utc_iso(effective_end)
    else:
        label_status, has_next, ready_at = "immature", None, None
    out: dict[str, Any] = {
        "token_key": str(row.token_key),
        "decision_at": _utc_iso(decision_at),
        "decision_market_cap_usd": entry,
        "schema_version": SCHEMA_VERSION,
        "label_status_next_peak": label_status,
        "terminal_at": _utc_iso(terminal_at) if terminal_at is not None and terminal_at >= decision_at else None,
        "terminal_reason": terminal_reason if terminal_at is not None and terminal_at >= decision_at else None,
        "path_end_at": _utc_iso(path_end),
        "has_next_substantial_peak_before_terminal_72h": has_next,
        "next_substantial_peak_at": None,
        "next_substantial_peak_confirmed_at": None,
        "time_to_next_substantial_peak_minutes_72h": None,
        "next_substantial_peak_market_cap_usd_72h": None,
        "next_substantial_peak_multiple_72h": None,
        "next_substantial_peak_prominence_pct_72h": None,
        "next_substantial_peak_confirmation_retrace_pct_72h": None,
        "next_peak_post_retracement_max_pct_72h": None,
        "substantial_peaks_count_before_terminal_72h": len(confirmed_future) if terminal_complete else (len(confirmed_future) if confirmed_future else None),
        "later_higher_peak_before_terminal_72h": None,
        "later_higher_peak_within_1h_after_next": None,
        "later_higher_peak_within_4h_after_next": None,
        "later_higher_peak_within_12h_after_next": None,
        "later_higher_peak_within_24h_after_next": None,
        "later_higher_peak_within_48h_after_next": None,
        "later_higher_peak_at": None,
        "time_from_next_to_later_higher_peak_minutes_72h": None,
        "later_higher_peak_multiple_vs_next_72h": None,
        "later_higher_peak_multiple_vs_decision_72h": None,
        "label_ready_at": ready_at,
        "label_finalized": int(terminal_complete),
        "learning_updated_at": None,
        "target_fingerprint": None,
        "config_json": json.dumps(asdict(config), sort_keys=True),
    }
    if next_peak is None:
        out["target_fingerprint"] = _target_fingerprint(out)
        return out

    next_peak_at = pd.Timestamp(next_peak.peak_at)
    out.update({
        "next_substantial_peak_at": next_peak.peak_at,
        "next_substantial_peak_confirmed_at": next_peak.confirmed_at,
        "time_to_next_substantial_peak_minutes_72h": (next_peak_at - decision_at).total_seconds() / 60.0,
        "next_substantial_peak_market_cap_usd_72h": next_peak.peak_price,
        "next_substantial_peak_multiple_72h": next_peak.peak_price / entry,
        "next_substantial_peak_prominence_pct_72h": next_peak.runup_pct,
        "next_substantial_peak_confirmation_retrace_pct_72h": next_peak.confirmation_retrace_pct,
    })

    # Realized valley after the first peak, bounded by the next confirmed swing peak.
    path = token_df[(token_df.snapshot_at > next_peak_at) & (token_df.snapshot_at <= path_end)]
    next_event = next((
        p for p in confirmed_future[1:]
        if pd.Timestamp(p.peak_at) <= path_end and pd.Timestamp(p.confirmed_at) <= path_end
    ), None)
    if next_event is not None:
        path = path[path.snapshot_at <= pd.Timestamp(next_event.peak_at)]
    if not path.empty and (next_event is not None or terminal_complete):
        out["next_peak_post_retracement_max_pct_72h"] = max(0.0, 1.0 - float(path.market_cap_usd.min()) / next_peak.peak_price)

    later_time = pd.Timestamp(later_higher.peak_at) if later_higher else None
    later_confirmed = pd.Timestamp(later_higher.confirmed_at) if later_higher else None
    for hours in (1, 4, 12, 24, 48):
        deadline = next_peak_at + pd.Timedelta(hours=hours)
        if later_time is not None and later_confirmed is not None and later_time <= deadline and later_confirmed <= deadline:
            out[f"later_higher_peak_within_{hours}h_after_next"] = 1
        elif path_end >= deadline or terminal_complete:
            out[f"later_higher_peak_within_{hours}h_after_next"] = 0
        else:
            out[f"later_higher_peak_within_{hours}h_after_next"] = None
    if later_time is not None and later_confirmed is not None and later_confirmed <= path_end:
        out.update({
            "later_higher_peak_before_terminal_72h": 1,
            "later_higher_peak_at": later_higher.peak_at,
            "time_from_next_to_later_higher_peak_minutes_72h": (later_time - next_peak_at).total_seconds() / 60.0,
            "later_higher_peak_multiple_vs_next_72h": later_higher.peak_price / next_peak.peak_price,
            "later_higher_peak_multiple_vs_decision_72h": later_higher.peak_price / entry,
        })
    elif terminal_complete:
        out["later_higher_peak_before_terminal_72h"] = 0

    out["target_fingerprint"] = _target_fingerprint(out)
    return out


def label_decision(
    token_df: pd.DataFrame,
    decision_idx: int,
    latest_capture: pd.Timestamp,
    terminal_at: pd.Timestamp | None,
    terminal_reason: str | None,
    config: PeakStructureConfig,
) -> dict[str, Any]:
    """Compatibility helper; V21 refresh uses cached global lifecycle peak events."""
    peaks = find_substantial_peaks(token_df.sort_values("snapshot_at"), config)
    return label_decision_from_events(
        token_df.sort_values("snapshot_at").reset_index(drop=True), decision_idx,
        latest_capture, terminal_at, terminal_reason, config, peaks,
    )


def refresh_labels(db: str, config: PeakStructureConfig, full_rebuild: bool = False) -> dict[str, Any]:
    """Incrementally refresh peak events and labels.

    New observations are appended.  Only tokens with new observations have their
    O(N) swing-event cache refreshed; only new/unfinalized decisions are relabeled.
    Existing finalized labels are left untouched unless --full-rebuild is requested.
    """
    with sqlite3.connect(db) as conn:
        migrate(conn)
        observations, source = load_observations(conn)
        if observations.empty:
            return {"stored": 0, "source": source, "mode": "incremental"}
        observations = observations.sort_values(["token_key", "snapshot_at"]).reset_index(drop=True)
        latest_capture = observations.snapshot_at.max()
        capture_times, presence = _capture_index(observations)
        cfg_json = json.dumps(asdict(config), sort_keys=True)

        state = {
            str(r[0]): {"last_observation_at": r[1], "config_json": r[2]}
            for r in conn.execute(f"SELECT token_key,last_observation_at,config_json FROM {STATE_TABLE}").fetchall()
        }
        changed_tokens: set[str] = set()
        for token, g in observations.groupby("token_key", sort=False):
            token = str(token)
            last_obs = _utc_iso(g.snapshot_at.max())
            prior = state.get(token)
            if full_rebuild or prior is None or prior.get("last_observation_at") != last_obs or prior.get("config_json") != cfg_json:
                changed_tokens.add(token)

        peak_events_before = int(conn.execute(f"SELECT COUNT(*) FROM {PEAK_EVENT_TABLE}").fetchone()[0])
        peak_tokens_refreshed = 0
        for token in changed_tokens:
            g = observations[observations.token_key.astype(str) == token].sort_values("snapshot_at").reset_index(drop=True)
            _refresh_token_peaks(conn, token, g, config)
            peak_tokens_refreshed += 1

        existing = pd.read_sql_query(
            f"SELECT token_key,decision_at,label_finalized,target_fingerprint,config_json FROM {LABEL_TABLE}", conn
        )
        existing_keys = set()
        unfinalized_tokens: set[str] = set()
        existing_map: dict[tuple[str, str], tuple[int, str | None, str | None]] = {}
        if not existing.empty:
            for r in existing.itertuples(index=False):
                key = (str(r.token_key), str(r.decision_at))
                existing_keys.add(key)
                existing_map[key] = (int(r.label_finalized or 0), r.target_fingerprint, r.config_json)
                if not int(r.label_finalized or 0):
                    unfinalized_tokens.add(str(r.token_key))

        tokens_to_label = set(changed_tokens) | unfinalized_tokens
        if full_rebuild:
            tokens_to_label = set(map(str, observations.token_key.unique()))

        now = datetime.now(timezone.utc).isoformat()
        inserted = updated = unchanged = 0
        terminal_counts: dict[str, int] = {}
        for token in tokens_to_label:
            g = observations[observations.token_key.astype(str) == token].sort_values("snapshot_at").reset_index(drop=True)
            if g.empty:
                continue
            terminal_at, terminal_reason = infer_token_terminal(g, capture_times, presence, config)
            terminal_counts[terminal_reason or "open"] = terminal_counts.get(terminal_reason or "open", 0) + 1
            peaks = _peaks_for_token(conn, token)
            for i in range(len(g)):
                decision_iso = _utc_iso(g.iloc[i].snapshot_at)
                key = (token, decision_iso)
                old = existing_map.get(key)
                if old and old[0] and old[2] == cfg_json and not full_rebuild:
                    continue
                rec = label_decision_from_events(g, i, latest_capture, terminal_at, terminal_reason, config, peaks)
                old_fp = old[1] if old else None
                learning_changed = old_fp != rec["target_fingerprint"]
                rec["learning_updated_at"] = now if learning_changed or old is None else None
                cols = list(rec)
                placeholders = ",".join("?" for _ in cols)
                col_sql = ",".join(f'"{c}"' for c in cols)
                updates = ",".join(
                    f'"{c}"=CASE WHEN excluded."{c}" IS NULL AND "{LABEL_TABLE}"."{c}" IS NOT NULL '
                    f'THEN "{LABEL_TABLE}"."{c}" ELSE excluded."{c}" END'
                    if c == "learning_updated_at" else f'"{c}"=excluded."{c}"'
                    for c in cols if c not in ("token_key", "decision_at")
                )
                conn.execute(
                    f"INSERT INTO {LABEL_TABLE} ({col_sql}) VALUES ({placeholders}) "
                    f"ON CONFLICT(token_key,decision_at) DO UPDATE SET {updates}",
                    [rec[c] for c in cols],
                )
                if old is None:
                    inserted += 1
                elif learning_changed:
                    updated += 1
                else:
                    unchanged += 1
        conn.commit()

        def scalar(sql: str) -> int:
            return int(conn.execute(sql).fetchone()[0])
        peak_events_after = scalar(f"SELECT COUNT(*) FROM {PEAK_EVENT_TABLE}")
        return {
            "mode": "full" if full_rebuild else "incremental",
            "source": source,
            "latest_capture": _utc_iso(latest_capture),
            "changed_tokens": len(changed_tokens),
            "peak_tokens_refreshed": peak_tokens_refreshed,
            "peak_events_before": peak_events_before,
            "peak_events_after": peak_events_after,
            "labels_inserted": inserted,
            "labels_learning_updated": updated,
            "labels_rechecked_unchanged": unchanged,
            "total_labels": scalar(f"SELECT COUNT(*) FROM {LABEL_TABLE}"),
            "next_peak_positive": scalar(f"SELECT COUNT(*) FROM {LABEL_TABLE} WHERE has_next_substantial_peak_before_terminal_72h=1"),
            "next_peak_negative": scalar(f"SELECT COUNT(*) FROM {LABEL_TABLE} WHERE has_next_substantial_peak_before_terminal_72h=0"),
            "next_peak_immature": scalar(f"SELECT COUNT(*) FROM {LABEL_TABLE} WHERE has_next_substantial_peak_before_terminal_72h IS NULL"),
            "later_higher_positive": scalar(f"SELECT COUNT(*) FROM {LABEL_TABLE} WHERE later_higher_peak_before_terminal_72h=1"),
            "terminal_tokens_touched": terminal_counts,
            "config": asdict(config),
        }

def _discover_feature_table(conn: sqlite3.Connection) -> dict[str, str] | None:
    best: tuple[int, dict[str, str]] | None = None
    for table in _all_tables(conn):
        lname = table.lower()
        if "feature" not in lname or table in {LABEL_TABLE, FEATURE_CACHE_TABLE}:
            continue
        cols = _table_columns(conn, table)
        token = _pick(cols, TOKEN_CANDIDATES)
        time_col = _pick(cols, TIME_CANDIDATES)
        if not (token and time_col):
            continue
        score = 10 + (5 if "axiom" in lname else 0) + (3 if "v18" in lname else 0)
        mapping = {"table": table, "token": token, "time": time_col}
        for candidate in ("features_json", "feature_json", "features", "feature_values_json"):
            if candidate in cols:
                mapping["json"] = candidate
                score += 5
                break
        if best is None or score > best[0]:
            best = (score, mapping)
    return best[1] if best else None


def _flatten_numeric_json(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    try:
        obj = json.loads(value) if isinstance(value, str) else value
    except MemoryError:
        raise
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, float] = {}
    stack = [("", obj)]
    while stack:
        prefix, cur = stack.pop()
        for k, v in cur.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}__{k}"
            if isinstance(v, dict):
                stack.append((key, v))
            elif isinstance(v, bool):
                out[key] = float(v)
            elif isinstance(v, (int, float)) and np.isfinite(float(v)):
                out[key] = float(v)
    return out


def _safe_feature_name(name: str) -> bool:
    lname = name.lower()
    if lname in {"token_key", "snapshot_at", "decision_at", "id", "rowid"}:
        return False
    if any(p in lname for p in LEAKAGE_PATTERNS):
        return False
    if any(p in lname for p in EXCLUDED_OPERATIONAL_PATTERNS):
        return False
    return True


def _load_existing_features(conn: sqlite3.Connection) -> pd.DataFrame | None:
    source = _discover_feature_table(conn)
    if not source:
        return None
    table = source["table"]
    quote = lambda value: '"' + str(value).replace('"', '""') + '"'
    if "json" in source:
        selected = [source["token"], source["time"], source["json"]]
        query = f"SELECT {','.join(quote(c) for c in selected)} FROM {quote(table)}"
    else:
        query = f"SELECT * FROM {quote(table)}"

    parts: list[pd.DataFrame] = []
    for df in pd.read_sql_query(query, conn, chunksize=2048):
        if df.empty:
            continue
        df = df.rename(columns={source["token"]: "token_key", source["time"]: "snapshot_at"})
        base = df[["token_key", "snapshot_at"]].copy()
        base["snapshot_at"] = _to_timestamp(base["snapshot_at"])

        if "json" in source:
            expanded = pd.DataFrame.from_records(
                [_flatten_numeric_json(v) for v in df[source["json"]]]
            )
            keep = [c for c in expanded.columns if _safe_feature_name(c)]
            numeric = expanded.reindex(columns=keep)
        else:
            numeric = df.select_dtypes(include=[np.number, "bool"])
            keep = [c for c in numeric.columns if _safe_feature_name(c)]
            numeric = numeric.reindex(columns=keep)
        if len(numeric.columns):
            numeric = numeric.apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)
        parts.append(
            pd.concat([base.reset_index(drop=True), numeric.reset_index(drop=True)], axis=1, copy=False)
        )
    if not parts:
        return None
    result = pd.concat(parts, ignore_index=True, sort=False, copy=False)
    return result if len(result.columns) > 2 else None


def _numeric_observation_columns(obs: pd.DataFrame) -> list[str]:
    candidates = []
    for c in obs.columns:
        if c in {"token_key", "snapshot_at"} or not _safe_feature_name(c):
            continue
        converted = pd.to_numeric(obs[c], errors="coerce")
        if converted.notna().sum() >= max(3, int(len(obs) * 0.01)):
            obs[c] = converted
            candidates.append(c)
    return candidates


def _nearest_past_value(times_ns: np.ndarray, vals: np.ndarray, target_ns: int) -> float:
    idx = int(np.searchsorted(times_ns, target_ns, side="right") - 1)
    if idx < 0:
        return np.nan
    return float(vals[idx]) if np.isfinite(vals[idx]) else np.nan


def build_fallback_features(
    observations: pd.DataFrame,
    *,
    emit_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build causal observation features.

    ``emit_at`` keeps the normal training behavior unchanged while allowing the
    live inference path to calculate only the newest rows.  Historical rows are
    still scanned to preserve lifetime highs, first values and trailing-window
    lookups, but their wide feature dictionaries are never materialized.  This is
    important for minute-by-minute prediction on a mature database.
    """
    obs = observations.copy()
    emit_timestamp = _to_timestamp(pd.Series([emit_at])).iloc[0] if emit_at is not None else None
    numeric_cols = _numeric_observation_columns(obs)
    if emit_timestamp is not None:
        active_tokens = set(
            obs.loc[obs["snapshot_at"] == emit_timestamp, "token_key"].astype(str)
        )
        obs = obs[obs["token_key"].astype(str).isin(active_tokens)].copy()
    # Prefer core market/trader series for trajectory engineering; retain all numeric
    # current-state fields as well.
    trajectory_bases = [
        c for c in (
            "market_cap_usd", "volume_usd", "fees_sol", "txns", "holders",
            "pro_traders", "kols", "recent_visitors", "top10_holders_pct",
            "sniper_pct", "insider_pct", "bundler_pct",
        ) if c in numeric_cols
    ]
    windows = (1, 2, 3, 5, 10, 15, 30, 60, 90, 180)
    rows: list[dict[str, Any]] = []

    for token, g in obs.groupby("token_key", sort=False):
        g = g.sort_values("snapshot_at").reset_index(drop=True)
        times_ns = g.snapshot_at.astype("int64").to_numpy()
        arrays = {c: pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float) for c in trajectory_bases}
        first_vals: dict[str, float] = {c: np.nan for c in trajectory_bases}
        running_high: dict[str, float] = {c: -np.inf for c in trajectory_bases}

        for i, src in g.iterrows():
            now_ns = int(times_ns[i])
            for c in trajectory_bases:
                cur = arrays[c][i]
                if np.isfinite(cur):
                    running_high[c] = max(running_high[c], cur)
                    if not np.isfinite(first_vals[c]) and cur != 0:
                        first_vals[c] = float(cur)

            if emit_timestamp is not None and pd.Timestamp(src.snapshot_at) != emit_timestamp:
                continue

            feat: dict[str, Any] = {"token_key": token, "snapshot_at": src.snapshot_at}
            for c in numeric_cols:
                v = pd.to_numeric(pd.Series([src[c]]), errors="coerce").iloc[0]
                feat[f"raw__{c}"] = float(v) if pd.notna(v) else np.nan

            for c in trajectory_bases:
                cur = arrays[c][i]
                first = first_vals[c]
                feat[f"life__{c}__multiple_from_first"] = cur / first if np.isfinite(cur) and np.isfinite(first) and first != 0 else np.nan
                feat[f"life__{c}__fraction_of_high"] = cur / running_high[c] if np.isfinite(cur) and running_high[c] > 0 else np.nan

                for w in windows:
                    target_ns = now_ns - int(w * 60 * 1e9)
                    past = _nearest_past_value(times_ns[: i + 1], arrays[c][: i + 1], target_ns)
                    if np.isfinite(cur) and np.isfinite(past) and past != 0:
                        feat[f"chg__{c}__{w}m"] = cur / past - 1.0
                        feat[f"slope__{c}__{w}m"] = (cur / past - 1.0) / max(w, 1)
                    else:
                        feat[f"chg__{c}__{w}m"] = np.nan
                        feat[f"slope__{c}__{w}m"] = np.nan

            def ratio(a: str, b: str, name: str) -> None:
                av = feat.get(f"raw__{a}")
                bv = feat.get(f"raw__{b}")
                feat[f"ratio__{name}"] = av / bv if av is not None and bv not in (None, 0) and np.isfinite(av) and np.isfinite(bv) else np.nan

            ratio("volume_usd", "market_cap_usd", "volume_mc")
            ratio("fees_sol", "txns", "fees_tx")
            ratio("volume_usd", "txns", "volume_tx")
            ratio("pro_traders", "holders", "pro_holders")
            ratio("kols", "holders", "kol_holders")
            ratio("recent_visitors", "holders", "visitors_holders")

            # Curvature / acceleration contrasts. They are causal because every
            # component uses only timestamps <= the decision.
            for c in trajectory_bases:
                feat[f"curve__{c}__1v5"] = feat.get(f"slope__{c}__1m", np.nan) - feat.get(f"slope__{c}__5m", np.nan)
                feat[f"curve__{c}__2v10"] = feat.get(f"slope__{c}__2m", np.nan) - feat.get(f"slope__{c}__10m", np.nan)
                feat[f"curve__{c}__5v30"] = feat.get(f"slope__{c}__5m", np.nan) - feat.get(f"slope__{c}__30m", np.nan)
                feat[f"curve__{c}__15v60"] = feat.get(f"slope__{c}__15m", np.nan) - feat.get(f"slope__{c}__60m", np.nan)
            rows.append(feat)

    return pd.DataFrame(rows)


def _feature_schema_hash(columns: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, columns))).encode("utf-8")).hexdigest()


def _serialize_feature_row(row: pd.Series) -> tuple[str, str]:
    payload: dict[str, float | None] = {}
    for c, v in row.items():
        if c in {"token_key", "snapshot_at"}:
            continue
        x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        payload[str(c)] = float(x) if pd.notna(x) and np.isfinite(float(x)) else None
    return json.dumps(payload, sort_keys=True, separators=(",", ":")), _feature_schema_hash(payload.keys())


def refresh_feature_cache(conn: sqlite3.Connection, observations: pd.DataFrame, force: bool = False) -> dict[str, Any]:
    """Append causal feature rows for new observations only.

    Historical feature rows are immutable because every engineered value uses only
    timestamps at or before its observation. This makes the one-minute hot path
    append-only rather than rebuilding all 72 hours on every prediction.
    """
    migrate(conn)
    cached_max = {
        str(r[0]): pd.Timestamp(r[1])
        for r in conn.execute(
            f"SELECT token_key, MAX(snapshot_at) FROM {FEATURE_CACHE_TABLE} GROUP BY token_key"
        ).fetchall() if r[1]
    }
    existing = _load_existing_features(conn)
    inserted = 0
    changed_tokens = 0
    now = datetime.now(timezone.utc).isoformat()
    for token, g in observations.groupby("token_key", sort=False):
        token = str(token)
        g = g.sort_values("snapshot_at").reset_index(drop=True)
        last_cached = cached_max.get(token)
        if not force and last_cached is not None and g.snapshot_at.max() <= last_cached:
            continue
        changed_tokens += 1
        fallback = build_fallback_features(g)
        if last_cached is not None and not force:
            fallback = fallback[fallback.snapshot_at > last_cached].copy()
        if fallback.empty:
            continue
        merged = fallback
        if existing is not None and not existing.empty:
            ext = existing[existing.token_key.astype(str) == token].copy()
            if last_cached is not None and not force:
                ext = ext[ext.snapshot_at > last_cached]
            if not ext.empty:
                merged = fallback.merge(ext, on=["token_key", "snapshot_at"], how="left", suffixes=("", "__v18"))
        for _, row in merged.iterrows():
            js, schema_hash = _serialize_feature_row(row)
            conn.execute(
                f"""INSERT INTO {FEATURE_CACHE_TABLE}(token_key,snapshot_at,features_json,feature_schema_hash,created_at)
                    VALUES(?,?,?,?,?)
                    ON CONFLICT(token_key,snapshot_at) DO NOTHING""",
                (token, _utc_iso(row.snapshot_at), js, schema_hash, now),
            )
            inserted += int(conn.execute("SELECT changes()").fetchone()[0] > 0)
    conn.commit()
    return {"changed_tokens": changed_tokens, "inserted": inserted}


def _load_cached_feature_frame(conn: sqlite3.Connection, current_at: pd.Timestamp | None = None) -> pd.DataFrame:
    if current_at is None:
        query = f"SELECT token_key,snapshot_at,features_json FROM {FEATURE_CACHE_TABLE} ORDER BY snapshot_at"
        params = None
    else:
        query = f"SELECT token_key,snapshot_at,features_json FROM {FEATURE_CACHE_TABLE} WHERE snapshot_at=?"
        params = (_utc_iso(current_at),)
    parts: list[pd.DataFrame] = []
    for df in pd.read_sql_query(query, conn, params=params, chunksize=2048):
        if df.empty:
            continue
        expanded = pd.DataFrame.from_records([_flatten_numeric_json(v) for v in df.features_json])
        if len(expanded.columns):
            expanded = expanded.apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)
        base = df[["token_key", "snapshot_at"]].copy()
        base["snapshot_at"] = _to_timestamp(base["snapshot_at"])
        parts.append(pd.concat([base.reset_index(drop=True), expanded.reset_index(drop=True)], axis=1, copy=False))
    if not parts:
        return pd.DataFrame(columns=["token_key", "snapshot_at"])
    return pd.concat(parts, ignore_index=True, sort=False, copy=False)


def build_feature_frame(conn: sqlite3.Connection, observations: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cache = refresh_feature_cache(conn, observations)
    features = _load_cached_feature_frame(conn)
    source = "v21_append_only_feature_cache"
    if _discover_feature_table(conn):
        source += "+existing_feature_enrichment_at_insert"
    return features, source


def build_current_feature_frame(conn: sqlite3.Connection, observations: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    refresh_feature_cache(conn, observations)
    latest = observations.snapshot_at.max()
    return _load_cached_feature_frame(conn, current_at=latest), "v21_append_only_feature_cache_current"

def load_training_frame(
    conn: sqlite3.Connection,
    observations: pd.DataFrame | None = None,
    observation_source: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, str, dict[str, str]]:
    if observations is None:
        observations, obs_source = load_observations(conn)
    else:
        obs_source = observation_source or discover_observation_source(conn)
    features, feature_source = build_feature_frame(conn, observations)
    labels = pd.read_sql_query(f"SELECT * FROM {LABEL_TABLE}", conn)
    if labels.empty:
        raise RuntimeError("Peak labels are empty. Run refresh first.")
    labels["decision_at"] = _to_timestamp(labels["decision_at"])
    frame = features.merge(
        labels,
        left_on=["token_key", "snapshot_at"],
        right_on=["token_key", "decision_at"],
        how="inner",
    )
    return frame, feature_source, obs_source


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked = {
        "token_key", "snapshot_at", "decision_at", "schema_version", "label_status_next_peak",
        "terminal_at", "terminal_reason", "path_end_at", "next_substantial_peak_at",
        "next_substantial_peak_confirmed_at", "later_higher_peak_at", "label_ready_at", "config_json",
    }
    cols = []
    for c in frame.columns:
        if c in blocked or not _safe_feature_name(c):
            continue
        s = pd.to_numeric(frame[c], errors="coerce")
        if s.notna().sum() >= 5:
            frame[c] = s
            cols.append(c)
    return sorted(set(cols))


def _token_weights(tokens: pd.Series) -> np.ndarray:
    counts = tokens.value_counts()
    return np.array([1.0 / counts[t] for t in tokens], dtype=float)


def _group_split(frame: pd.DataFrame, allow_small: bool) -> tuple[np.ndarray, np.ndarray]:
    firsts = frame.groupby("token_key")["snapshot_at"].min().sort_values()
    tokens = list(firsts.index)
    if len(tokens) < 4:
        cut = max(1, int(len(frame) * 0.8))
        order = np.argsort(frame.snapshot_at.to_numpy())
        return order[:cut], order[cut:]
    cut = max(1, min(len(tokens) - 1, int(len(tokens) * 0.80)))
    train_tokens = set(tokens[:cut])
    val_tokens = set(tokens[cut:])
    tr = np.flatnonzero(frame.token_key.isin(train_tokens).to_numpy())
    va = np.flatnonzero(frame.token_key.isin(val_tokens).to_numpy())
    return tr, va


def _classifier_components(n_estimators: int) -> list[tuple[str, Any]]:
    components: list[tuple[str, Any]] = []
    if LGBMClassifier is not None:
        components.append(("lightgbm", LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.035, num_leaves=31,
            max_depth=-1, subsample=0.9, colsample_bytree=0.9,
            random_state=17, verbosity=-1,
        )))
    if XGBClassifier is not None:
        components.append(("xgboost", XGBClassifier(
            n_estimators=n_estimators, learning_rate=0.035, max_depth=5,
            min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
            random_state=19, n_jobs=4,
        )))
    if not components:
        raise RuntimeError("V21 requires lightgbm and/or xgboost. Install project requirements first.")
    return components


def _regressor_components(n_estimators: int, quantile: float | None = None) -> list[tuple[str, Any]]:
    components: list[tuple[str, Any]] = []
    if LGBMRegressor is not None:
        kwargs = dict(
            n_estimators=n_estimators, learning_rate=0.035, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=23, verbosity=-1,
        )
        if quantile is not None:
            kwargs.update(objective="quantile", alpha=quantile)
        else:
            kwargs.update(objective="regression_l1")
        components.append(("lightgbm", LGBMRegressor(**kwargs)))
    if XGBRegressor is not None:
        kwargs = dict(
            n_estimators=n_estimators, learning_rate=0.035, max_depth=5,
            min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=29, n_jobs=4,
        )
        if quantile is not None:
            kwargs.update(objective="reg:quantileerror", quantile_alpha=quantile)
        else:
            kwargs.update(objective="reg:absoluteerror")
        try:
            components.append(("xgboost", XGBRegressor(**kwargs)))
        except Exception:
            pass
    if not components:
        raise RuntimeError("V21 requires lightgbm and/or xgboost. Install project requirements first.")
    return components


def _transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.log(np.clip(values, 1e-9, None))
    if transform == "log1p":
        return np.log1p(np.clip(values, 0, None))
    return values


def _inverse_target(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.exp(values)
    if transform == "log1p":
        return np.expm1(values)
    return values


def _fit_classifier(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    allow_small: bool,
    n_estimators: int,
) -> dict[str, Any] | None:
    data = frame[frame[target].notna()].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data[data[target].isin([0, 1])]
    min_rows = 30 if allow_small else 300
    if len(data) < min_rows or data[target].nunique() < 2:
        return None
    tr, va = _group_split(data, allow_small)
    if len(va) == 0 or data.iloc[tr][target].nunique() < 2:
        return None
    X = data[features].replace([np.inf, -np.inf], np.nan)
    y = data[target].astype(int).to_numpy()
    w = _token_weights(data.token_key)
    models = []
    scores = []
    for name, model in _classifier_components(n_estimators):
        try:
            model.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
            p = model.predict_proba(X.iloc[va])[:, 1]
            score = log_loss(y[va], np.clip(p, 1e-5, 1 - 1e-5), labels=[0, 1])
            models.append((name, model))
            scores.append(max(score, 1e-6))
        except Exception:
            continue
    if not models:
        return None
    inv = 1.0 / np.array(scores)
    blend = (inv / inv.sum()).tolist()
    return {
        "kind": "classifier", "target": target, "condition": None,
        "models": models, "blend": blend,
        "validation_metric": "log_loss", "validation_scores": scores,
        "rows": len(data), "positives": int(data[target].sum()),
    }


def _fit_regression(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    allow_small: bool,
    n_estimators: int,
    *,
    quantile: float | None = None,
    transform: str = "none",
    condition: str | None = None,
) -> dict[str, Any] | None:
    data = frame.copy()
    if condition:
        data = data.query(condition)
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data[data[target].notna() & np.isfinite(data[target])]
    min_rows = 25 if allow_small else 250
    if len(data) < min_rows:
        return None
    tr, va = _group_split(data, allow_small)
    if len(va) == 0:
        return None
    X = data[features].replace([np.inf, -np.inf], np.nan)
    raw_y = data[target].to_numpy(dtype=float)
    y = _transform_target(raw_y, transform)
    w = _token_weights(data.token_key)
    models = []
    scores = []
    for name, model in _regressor_components(n_estimators, quantile=quantile):
        try:
            model.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
            pred = _inverse_target(np.asarray(model.predict(X.iloc[va]), dtype=float), transform)
            if quantile is not None:
                score = mean_pinball_loss(raw_y[va], pred, alpha=quantile)
                metric = f"pinball_q{quantile}"
            else:
                score = mean_absolute_error(raw_y[va], pred)
                metric = "mae"
            models.append((name, model))
            scores.append(max(float(score), 1e-9))
        except Exception:
            continue
    if not models:
        return None
    inv = 1.0 / np.array(scores)
    blend = (inv / inv.sum()).tolist()
    return {
        "kind": "regressor", "target": target, "quantile": quantile,
        "transform": transform, "condition": condition, "models": models, "blend": blend,
        "validation_metric": metric, "validation_scores": scores,
        "rows": len(data),
    }


def train(db: str, output_dir: str, allow_small: bool = False) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        migrate(conn)
        frame, feature_source, obs_source = load_training_frame(conn)
    features = _feature_columns(frame)
    if not features:
        raise RuntimeError("No usable causal numeric features were found.")

    n_estimators = 160 if allow_small else 700
    heads: dict[str, Any] = {}
    specs: list[tuple[str, str]] = [
        ("p_next_substantial_peak_before_terminal_72h", "has_next_substantial_peak_before_terminal_72h"),
        ("p_later_higher_peak_given_next_before_terminal_72h", "later_higher_peak_before_terminal_72h"),
        ("p_later_higher_peak_given_next_within_1h", "later_higher_peak_within_1h_after_next"),
        ("p_later_higher_peak_given_next_within_4h", "later_higher_peak_within_4h_after_next"),
        ("p_later_higher_peak_given_next_within_12h", "later_higher_peak_within_12h_after_next"),
        ("p_later_higher_peak_given_next_within_24h", "later_higher_peak_within_24h_after_next"),
        ("p_later_higher_peak_given_next_within_48h", "later_higher_peak_within_48h_after_next"),
    ]
    for output, target in specs:
        fit = _fit_classifier(frame, features, target, allow_small, n_estimators)
        if fit:
            heads[output] = fit

    for q in (0.25, 0.50, 0.75):
        suffix = f"q{int(q * 100)}"
        fit = _fit_regression(
            frame, features, "time_to_next_substantial_peak_minutes_72h",
            allow_small, n_estimators, quantile=q, transform="log1p",
            condition="has_next_substantial_peak_before_terminal_72h == 1",
        )
        if fit:
            heads[f"pred_time_to_next_substantial_peak_minutes_{suffix}"] = fit
        fit = _fit_regression(
            frame, features, "next_substantial_peak_multiple_72h",
            allow_small, n_estimators, quantile=q, transform="log",
            condition="has_next_substantial_peak_before_terminal_72h == 1",
        )
        if fit:
            heads[f"pred_next_substantial_peak_multiple_{suffix}"] = fit

    fit = _fit_regression(
        frame, features, "next_peak_post_retracement_max_pct_72h",
        allow_small, n_estimators, quantile=0.50, transform="none",
        condition="has_next_substantial_peak_before_terminal_72h == 1",
    )
    if fit:
        heads["pred_post_next_peak_retracement_pct_q50"] = fit

    fit = _fit_regression(
        frame, features, "later_higher_peak_multiple_vs_next_72h",
        allow_small, n_estimators, quantile=0.50, transform="log",
        condition="later_higher_peak_before_terminal_72h == 1",
    )
    if fit:
        heads["pred_later_higher_peak_multiple_vs_next_q50"] = fit

    fit = _fit_regression(
        frame, features, "time_from_next_to_later_higher_peak_minutes_72h",
        allow_small, n_estimators, quantile=0.50, transform="log1p",
        condition="later_higher_peak_before_terminal_72h == 1",
    )
    if fit:
        heads["pred_time_from_next_to_later_higher_peak_minutes_q50"] = fit

    if not heads:
        raise RuntimeError("No peak-structure heads had enough mature target data to train.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_source": feature_source,
        "observation_source": obs_source,
        "feature_columns": features,
        "heads": heads,
        "training_mode": "full_bootstrap",
        "training_watermark": (
            pd.to_datetime(frame.get("learning_updated_at"), errors="coerce", format="ISO8601", utc=True).max().isoformat()
            if "learning_updated_at" in frame and pd.to_datetime(frame.get("learning_updated_at"), errors="coerce", format="ISO8601", utc=True).notna().any()
            else datetime.now(timezone.utc).isoformat()
        ),
        "incremental_rounds": 0,
        "appended_estimators_total": 0,
    }
    versioned = out_dir / f"peak_structure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.joblib"
    joblib.dump(bundle, versioned)
    latest = out_dir / "latest.joblib"
    joblib.dump(bundle, latest)
    return {
        "model": str(latest), "versioned_model": str(versioned),
        "heads_trained": len(heads), "head_names": sorted(heads),
        "features": len(features), "feature_source": feature_source,
        "rows": len(frame),
    }



def _head_valid_data(frame: pd.DataFrame, head: dict[str, Any]) -> pd.DataFrame:
    data = frame.copy()
    condition = head.get("condition")
    if condition:
        try:
            data = data.query(condition)
        except Exception:
            return data.iloc[0:0].copy()
    target = head.get("target")
    if not target or target not in data:
        return data.iloc[0:0].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data[data[target].notna() & np.isfinite(data[target])].copy()
    if head.get("kind") == "classifier":
        data = data[data[target].isin([0, 1])].copy()
    return data


def _continue_component(name: str, old_model: Any, X: pd.DataFrame, y: np.ndarray,
                        sample_weight: np.ndarray, append_estimators: int) -> Any:
    params = dict(old_model.get_params())
    params["n_estimators"] = int(append_estimators)
    model = type(old_model)(**params)
    if name == "lightgbm":
        init_model = getattr(old_model, "booster_", None)
        if init_model is None:
            raise RuntimeError("LightGBM component has no booster_ for continuation")
        model.fit(X, y, sample_weight=sample_weight, init_model=init_model)
    elif name == "xgboost":
        init_model = old_model.get_booster()
        model.fit(X, y, sample_weight=sample_weight, xgb_model=init_model)
    else:
        raise RuntimeError(f"Unsupported incremental component: {name}")
    return model


def _balanced_replay(data: pd.DataFrame, target: str, rows: int, classifier: bool) -> pd.DataFrame:
    if data.empty or rows <= 0:
        return data.iloc[0:0].copy()
    # Favor recent experience but keep token breadth. One row per token is sampled
    # first, then the remaining budget is filled from the recent tail.
    recent = data.sort_values("snapshot_at")
    per_token = recent.groupby("token_key", sort=False).tail(1)
    pieces = [per_token.tail(min(rows // 2, len(per_token)))]
    remaining = rows - sum(len(x) for x in pieces)
    if remaining > 0:
        pieces.append(recent.tail(remaining))
    replay = pd.concat(pieces, ignore_index=False).drop_duplicates(["token_key", "snapshot_at"], keep="last")
    if classifier and replay[target].nunique() < 2 and data[target].nunique() >= 2:
        extras = []
        for cls in (0, 1):
            if cls not in set(replay[target].astype(int)):
                cand = data[data[target] == cls].tail(1)
                if not cand.empty:
                    extras.append(cand)
        if extras:
            replay = pd.concat([replay] + extras).drop_duplicates(["token_key", "snapshot_at"], keep="last")
    return replay.tail(rows)


def _continue_head(
    head: dict[str, Any],
    frame: pd.DataFrame,
    features: list[str],
    watermark: pd.Timestamp,
    append_estimators: int,
    replay_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _head_valid_data(frame, head)
    target = head.get("target")
    if data.empty or target is None:
        return head, {"new_rows": 0, "reason": "no valid target rows"}
    learn_ts = pd.to_datetime(data.get("learning_updated_at"), errors="coerce", format="ISO8601", utc=True)
    new_mask = learn_ts.notna() & (learn_ts > watermark)
    new = data[new_mask].copy()
    if new.empty:
        return head, {"new_rows": 0, "reason": "no target revisions after watermark"}

    # Current newest token cohort is a shadow validation set and is deliberately not
    # appended. A periodic/manual full compaction later rotates these held-out rows.
    _, va = _group_split(data, allow_small=True)
    val_tokens = set(data.iloc[va].token_key.astype(str)) if len(va) else set()
    new_train = new[~new.token_key.astype(str).isin(val_tokens)].copy()
    if new_train.empty:
        return head, {"new_rows": int(len(new)), "trained_new_rows": 0, "reason": "all new rows held out for shadow validation"}

    old_pool = data[(~new_mask) & (~data.token_key.astype(str).isin(val_tokens))].copy()
    replay = _balanced_replay(old_pool, target, replay_rows, head.get("kind") == "classifier")
    batch = pd.concat([new_train, replay], ignore_index=False).drop_duplicates(["token_key", "snapshot_at"], keep="last")
    if head.get("kind") == "classifier" and batch[target].nunique() < 2:
        return head, {"new_rows": int(len(new)), "trained_new_rows": 0, "reason": "incremental classifier batch has one class"}

    X = batch[features].replace([np.inf, -np.inf], np.nan)
    raw_y = batch[target].to_numpy(dtype=float)
    y = raw_y.astype(int) if head.get("kind") == "classifier" else _transform_target(raw_y, head.get("transform", "none"))
    w = _token_weights(batch.token_key)

    val = data[data.token_key.astype(str).isin(val_tokens)].copy()
    if val.empty:
        val = data.tail(max(3, min(200, len(data))))
    Xv = val[features].replace([np.inf, -np.inf], np.nan)
    raw_yv = val[target].to_numpy(dtype=float)

    models = []
    scores = []
    for name, old_model in head.get("models", []):
        try:
            model = _continue_component(name, old_model, X, y, w, append_estimators)
            if head.get("kind") == "classifier":
                pred = model.predict_proba(Xv)[:, 1]
                score = log_loss(raw_yv.astype(int), np.clip(pred, 1e-5, 1 - 1e-5), labels=[0, 1])
            else:
                pred = _inverse_target(np.asarray(model.predict(Xv), dtype=float), head.get("transform", "none"))
                q = head.get("quantile")
                if q is not None:
                    score = mean_pinball_loss(raw_yv, pred, alpha=float(q))
                else:
                    score = mean_absolute_error(raw_yv, pred)
            models.append((name, model))
            scores.append(max(float(score), 1e-9))
        except Exception:
            continue
    if not models:
        return head, {"new_rows": int(len(new)), "trained_new_rows": 0, "reason": "all continuation components failed"}
    inv = 1.0 / np.asarray(scores)
    updated = dict(head)
    updated.update({
        "models": models,
        "blend": (inv / inv.sum()).tolist(),
        "validation_scores": scores,
        "rows": int(head.get("rows", 0)) + int(len(new_train)),
        "incremental_updates": int(head.get("incremental_updates", 0)) + 1,
        "last_incremental_new_rows": int(len(new_train)),
        "last_replay_rows": int(len(replay)),
    })
    return updated, {
        "new_rows": int(len(new)), "trained_new_rows": int(len(new_train)),
        "replay_rows": int(len(replay)), "validation_rows": int(len(val)),
        "components": len(models), "scores": scores,
    }


def incremental_train(
    db: str,
    base_model_path: str,
    output_dir: str,
    *,
    append_estimators: int = 30,
    replay_rows: int = 1000,
) -> dict[str, Any]:
    """Append trees to an existing V21 bundle using only newly matured/revised labels.

    This is the default ongoing-learning path. It does not fit the model bank from
    scratch. A bounded replay buffer is mixed with new rows to reduce catastrophic
    drift; the full historical dataset is used only for evaluation/feature lookup.
    """
    base_path = Path(base_model_path)
    if not base_path.exists():
        raise RuntimeError("No V21 base model exists. Run one full bootstrap train first.")
    bundle = joblib.load(base_path)
    if not str(bundle.get("schema_version", "")).startswith("v21_"):
        raise RuntimeError("Incremental continuation requires a V21 72h model. Bootstrap V21 once from the current database.")
    watermark_raw = bundle.get("training_watermark")
    if not watermark_raw:
        raise RuntimeError("Base V21 model has no training watermark; run one full V21 bootstrap train.")
    watermark = pd.Timestamp(watermark_raw)
    if watermark.tzinfo is None:
        watermark = watermark.tz_localize("UTC")
    else:
        watermark = watermark.tz_convert("UTC")

    with sqlite3.connect(db) as conn:
        migrate(conn)
        frame, feature_source, obs_source = load_training_frame(conn)
    features = _feature_columns(frame)
    if features != list(bundle.get("feature_columns", [])):
        raise RuntimeError(
            "Feature schema changed; boosted-tree continuation cannot safely alter feature order. "
            "Run an explicit full compaction/bootstrap instead."
        )

    learn_ts = pd.to_datetime(frame.get("learning_updated_at"), errors="coerce", format="ISO8601", utc=True)
    if learn_ts.notna().sum() == 0 or not bool((learn_ts > watermark).any()):
        return {"trained": False, "reason": "no new mature/revised targets after model watermark", "watermark": watermark.isoformat()}
    new_watermark = learn_ts.max()

    new_heads: dict[str, Any] = {}
    details: dict[str, Any] = {}
    total_trained_new = 0
    for name, head in bundle.get("heads", {}).items():
        updated, meta = _continue_head(head, frame, features, watermark, append_estimators, replay_rows)
        new_heads[name] = updated
        details[name] = meta
        total_trained_new += int(meta.get("trained_new_rows", 0))
    if total_trained_new == 0:
        return {"trained": False, "reason": "new rows existed but none were eligible outside shadow validation", "heads": details}

    candidate = dict(bundle)
    candidate.update({
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_source": feature_source,
        "observation_source": obs_source,
        "heads": new_heads,
        "training_mode": "incremental_append",
        "parent_model": str(base_path),
        "parent_watermark": watermark.isoformat(),
        "training_watermark": new_watermark.isoformat(),
        "incremental_rounds": int(bundle.get("incremental_rounds", 0)) + 1,
        "appended_estimators_total": int(bundle.get("appended_estimators_total", 0)) + int(append_estimators),
        "last_append_estimators": int(append_estimators),
        "last_replay_limit": int(replay_rows),
        "last_incremental_details": details,
    })
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned = out_dir / f"peak_structure_incremental_{stamp}.joblib"
    latest = out_dir / "latest.joblib"
    joblib.dump(candidate, versioned)
    joblib.dump(candidate, latest)
    return {
        "trained": True,
        "mode": "incremental_append",
        "model": str(latest),
        "versioned_model": str(versioned),
        "parent": str(base_path),
        "watermark_before": watermark.isoformat(),
        "watermark_after": new_watermark.isoformat(),
        "append_estimators": int(append_estimators),
        "replay_rows_limit": int(replay_rows),
        "trained_new_rows_sum_across_heads": int(total_trained_new),
        "heads": details,
    }

def _predict_head(head: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    preds = []
    for (_, model), weight in zip(head["models"], head["blend"]):
        if head["kind"] == "classifier":
            pred = model.predict_proba(X)[:, 1]
        else:
            pred = np.asarray(model.predict(X), dtype=float)
            pred = _inverse_target(pred, head.get("transform", "none"))
        preds.append(float(weight) * np.asarray(pred, dtype=float))
    return np.sum(preds, axis=0)


def _current_rows(observations: pd.DataFrame) -> pd.DataFrame:
    # The collector writes one common snapshot_at per Axiom capture. Use the newest
    # capture rather than stale latest-per-token values from tokens that disappeared.
    latest = observations.snapshot_at.max()
    return observations[observations.snapshot_at == latest].copy()


def predict(db: str, model_path: str, out_path: str) -> list[dict[str, Any]]:
    bundle = joblib.load(model_path)
    with sqlite3.connect(db) as conn:
        observations, _ = load_observations(conn)
        features, _ = build_current_feature_frame(conn, observations)
    current_obs = _current_rows(observations)
    current = features.merge(
        current_obs[["token_key", "snapshot_at", "market_cap_usd"] + (["name"] if "name" in current_obs.columns else [])],
        on=["token_key", "snapshot_at"], how="inner", suffixes=("", "__obs"),
    )
    if current.empty:
        return []
    feature_cols = bundle["feature_columns"]
    for c in feature_cols:
        if c not in current.columns:
            current[c] = np.nan
    X = current[feature_cols].replace([np.inf, -np.inf], np.nan)

    result = current[["token_key", "snapshot_at", "market_cap_usd"] + (["name"] if "name" in current.columns else [])].copy()
    for output, head in bundle["heads"].items():
        result[output] = _predict_head(head, X)

    # Quantile monotonicity correction, separately for timing and height.
    for stem in ("pred_time_to_next_substantial_peak_minutes", "pred_next_substantial_peak_multiple"):
        qs = [f"{stem}_q25", f"{stem}_q50", f"{stem}_q75"]
        present = [q for q in qs if q in result.columns]
        if len(present) >= 2:
            vals = np.sort(result[present].to_numpy(dtype=float), axis=1)
            for i, q in enumerate(present):
                result[q] = vals[:, i]

    if "pred_next_substantial_peak_multiple_q25" in result:
        for q in (25, 50, 75):
            mult = f"pred_next_substantial_peak_multiple_q{q}"
            if mult in result:
                result[f"pred_next_substantial_peak_market_cap_usd_q{q}"] = result.market_cap_usd * result[mult]

    p_next = "p_next_substantial_peak_before_terminal_72h"
    p_later = "p_later_higher_peak_given_next_before_terminal_72h"
    if p_next in result and p_later in result:
        result["p_next_then_later_higher_peak_before_terminal_72h"] = result[p_next] * result[p_later]

    # A human-readable timing interval is useful in the CSV without replacing the
    # underlying quantiles.
    if all(c in result for c in (
        "pred_time_to_next_substantial_peak_minutes_q25",
        "pred_time_to_next_substantial_peak_minutes_q50",
        "pred_time_to_next_substantial_peak_minutes_q75",
    )):
        result["next_substantial_peak_timing_window"] = result.apply(
            lambda r: f"{max(0, r['pred_time_to_next_substantial_peak_minutes_q25']):.0f}-"
                      f"{max(0, r['pred_time_to_next_substantial_peak_minutes_q75']):.0f}m "
                      f"(median {max(0, r['pred_time_to_next_substantial_peak_minutes_q50']):.0f}m)",
            axis=1,
        )

    # Prefer the new peak-existence probability for this dedicated output.
    if p_next in result:
        result = result.sort_values(p_next, ascending=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    serial = result.copy()
    serial["snapshot_at"] = serial.snapshot_at.astype(str)
    return serial.replace({np.nan: None}).to_dict(orient="records")


def augment_csv(base_path: str, peak_path: str, out_path: str) -> dict[str, Any]:
    base = pd.read_csv(base_path)
    peak = pd.read_csv(peak_path)
    key = "token_key" if "token_key" in base.columns and "token_key" in peak.columns else None
    if key is None:
        for candidate in ("short_address_hint", "token", "mint", "token_address"):
            if candidate in base.columns and candidate in peak.columns:
                key = candidate
                break
    if key is None:
        raise RuntimeError("Could not find a common token identity column to augment V18 predictions.")
    drop_cols = [c for c in peak.columns if c in base.columns and c != key]
    merged = base.merge(peak.drop(columns=drop_cols), on=key, how="left")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return {"base_rows": len(base), "peak_rows": len(peak), "merged_rows": len(merged), "key": key, "out": out_path}


def status(db: str) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        migrate(conn)
        source = discover_observation_source(conn)
        total = conn.execute(f"SELECT COUNT(*) FROM {LABEL_TABLE}").fetchone()[0]
        def count(where: str) -> int:
            return conn.execute(f"SELECT COUNT(*) FROM {LABEL_TABLE} WHERE {where}").fetchone()[0]
        config_row = conn.execute(f"SELECT config_json FROM {LABEL_TABLE} LIMIT 1").fetchone()
        availability = {}
        for col in (
            "has_next_substantial_peak_before_terminal_72h",
            "time_to_next_substantial_peak_minutes_72h",
            "next_substantial_peak_multiple_72h",
            "later_higher_peak_before_terminal_72h",
            "later_higher_peak_within_1h_after_next",
            "later_higher_peak_within_4h_after_next",
            "later_higher_peak_within_12h_after_next",
            "later_higher_peak_within_24h_after_next",
            "later_higher_peak_within_48h_after_next",
            "later_higher_peak_multiple_vs_next_72h",
        ):
            availability[col] = count(f'"{col}" IS NOT NULL')
        return {
            "schema_version": SCHEMA_VERSION,
            "observation_source": source,
            "rows": total,
            "next_peak_positive": count("has_next_substantial_peak_before_terminal_72h = 1"),
            "next_peak_negative": count("has_next_substantial_peak_before_terminal_72h = 0"),
            "next_peak_immature": count("has_next_substantial_peak_before_terminal_72h IS NULL"),
            "later_higher_positive": count("later_higher_peak_before_terminal_72h = 1"),
            "tokens_with_multiple_substantial_peaks": conn.execute(
                f"SELECT COUNT(DISTINCT token_key) FROM {LABEL_TABLE} WHERE substantial_peaks_count_before_terminal_72h >= 2"
            ).fetchone()[0],
            "target_availability": availability,
            "config": json.loads(config_row[0]) if config_row else None,
        }


def _parse_config(args: argparse.Namespace) -> PeakStructureConfig:
    return PeakStructureConfig(
        min_runup_pct=args.min_runup_pct,
        confirm_retrace_pct=args.confirm_retrace_pct,
        higher_peak_margin_pct=args.higher_peak_margin_pct,
        horizon_minutes=args.horizon_minutes,
        death_missed_cycles=args.death_missed_cycles,
        death_gap_minutes=args.death_gap_minutes,
        age_out_minutes=args.age_out_minutes,
        min_peak_separation_minutes=args.min_peak_separation_minutes,
    )


def _add_config_args(ap: argparse.ArgumentParser) -> None:
    defaults = PeakStructureConfig()
    ap.add_argument("--min-runup-pct", type=float, default=defaults.min_runup_pct)
    ap.add_argument("--confirm-retrace-pct", type=float, default=defaults.confirm_retrace_pct)
    ap.add_argument("--higher-peak-margin-pct", type=float, default=defaults.higher_peak_margin_pct)
    ap.add_argument("--horizon-minutes", type=int, default=defaults.horizon_minutes)
    ap.add_argument("--death-missed-cycles", type=int, default=defaults.death_missed_cycles, help="Compatibility/diagnostic count; elapsed time controls death in V21")
    ap.add_argument("--death-gap-minutes", type=float, default=defaults.death_gap_minutes)
    ap.add_argument("--age-out-minutes", type=int, default=defaults.age_out_minutes)
    ap.add_argument("--min-peak-separation-minutes", type=float, default=defaults.min_peak_separation_minutes)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="V21 next-substantial-peak and later-higher-peak model bank"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("refresh", help="Incrementally refresh 72h peak events and lifecycle labels")
    p.add_argument("--db", default="data/live.sqlite")
    p.add_argument("--full-rebuild", action="store_true", help="Explicit maintenance-only relabel of all rows")
    _add_config_args(p)

    p = sub.add_parser("status", help="Show peak-structure target availability")
    p.add_argument("--db", default="data/live.sqlite")

    p = sub.add_parser("train", help="One-time/full-compaction LightGBM/XGBoost bootstrap")
    p.add_argument("--db", default="data/live.sqlite")
    p.add_argument("--output-dir", default="models/axiom_peak_v21")
    p.add_argument("--allow-small", action="store_true")

    p = sub.add_parser("update", help="Append trees from newly matured/revised labels; does not retrain from scratch")
    p.add_argument("--db", default="data/live.sqlite")
    p.add_argument("--base-model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default="models/axiom_peak_v21/challengers")
    p.add_argument("--append-estimators", type=int, default=30)
    p.add_argument("--replay-rows", type=int, default=1000)

    p = sub.add_parser("predict", help="Predict next substantial peak and later higher peak")
    p.add_argument("--db", default="data/live.sqlite")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--augment", help="Optional existing V18 prediction CSV to augment")
    p.add_argument("--augmented-out", help="Where to write augmented V18 CSV; defaults to --augment path")

    args = ap.parse_args()
    if args.cmd == "refresh":
        result = refresh_labels(args.db, _parse_config(args), full_rebuild=args.full_rebuild)
    elif args.cmd == "status":
        result = status(args.db)
    elif args.cmd == "train":
        result = train(args.db, args.output_dir, args.allow_small)
    elif args.cmd == "update":
        result = incremental_train(
            args.db, args.base_model, args.output_dir,
            append_estimators=args.append_estimators, replay_rows=args.replay_rows,
        )
    elif args.cmd == "predict":
        rows = predict(args.db, args.model, args.out)
        result = {"predictions": len(rows), "out": args.out}
        if args.augment:
            augmented_out = args.augmented_out or args.augment
            result["augmentation"] = augment_csv(args.augment, args.out, augmented_out)
    else:  # pragma: no cover
        raise AssertionError(args.cmd)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
