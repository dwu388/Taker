from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import itertools
import json
import math
import os
import shutil
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_pinball_loss
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

from . import axiom_peak_structure as peak

SCHEMA_VERSION = "v24_lifetime_purged_adaptive_event_policy_v1"
MODEL_ROOT_DEFAULT = "models/axiom_v24"
CHAMPION_DEFAULT = "models/axiom_v24/champion.joblib"
POLICY_ROOT_DEFAULT = "models/axiom_policy_v24"
POLICY_CHAMPION_DEFAULT = "models/axiom_policy_v24/champion.joblib"
PREDICTIONS_DEFAULT = "data/axiom_predictions_v24.csv"

# Live inference is deliberately read-mostly.  The collector owns the canonical
# raw write path; prediction waits briefly and retries only its small provenance
# transaction instead of running training-frame refreshes while collection is
# active.
LIVE_SQLITE_BUSY_TIMEOUT_MS = 750
LIVE_SQLITE_WRITE_RETRIES = 8
LIVE_SQLITE_RETRY_BASE_SECONDS = 0.20
LIVE_PREDICTION_SNAPSHOT_RETRIES = 3

COHORT_TABLE = "axiom_v24_calendar_cohorts"
LIFETIME_TABLE = "axiom_v24_token_lifetimes"
PREDICTION_LEDGER = "axiom_v24_prediction_ledger"
MODEL_REGISTRY = "axiom_v24_model_registry"
PROMOTION_TABLE = "axiom_v24_promotions"
POLICY_REGISTRY = "axiom_v24_policy_registry"
LIQUIDITY_TABLE = "axiom_v24_liquidity_observations"
AUDIT_RESULTS_TABLE = "axiom_v24_audit_results"
SEQUENCE_CACHE_TABLE = "axiom_v24_sequence_fingerprint"
TOKEN_ASSIGNMENT_TABLE = "axiom_v24_token_assignment"
CAPTURE_HEARTBEAT_TABLE = "axiom_v24_capture_heartbeat"
POLICY_COHORT_TABLE = "axiom_v24_policy_cohorts"
POLICY_PROMOTION_TABLE = "axiom_v24_policy_promotions"
CALIBRATION_STATE_TABLE = "axiom_v24_calibration_state"
DATA_VINTAGE_TABLE = "axiom_v24_data_vintage"
COUNTERFACTUAL_TABLE = "axiom_v24_counterfactual_policy_targets"
SEQUENCE_MODEL_REGISTRY = "axiom_v24_sequence_challengers"
CALIBRATION_UPDATES_TABLE = "axiom_v24_calibration_updates"

EVENT_NONE = 0
EVENT_PEAK = 1
EVENT_DEATH = 2
EVENT_NAMES = {EVENT_NONE: "none", EVENT_PEAK: "substantial_peak", EVENT_DEATH: "operational_death"}


@dataclass
class V24Config:
    horizon_minutes: int = 72 * 60
    operational_gap_minutes: float = 50.0
    age_out_minutes: float = 71 * 60.0

    # Calendar validation. Warm-up blocks are permanently ordinary training blocks.
    cohort_hours: int = 24
    warmup_blocks: int = 7
    # Leave ordinary learning blocks between one-use promotion tests so 72h labels
    # can mature before the next evaluation. With 24h blocks, 4 means one
    # evaluation slot every four days.
    promotion_every_n_blocks: int = 4
    # Every Nth evaluation slot is sealed audit instead of promotion.
    audit_every_n_blocks: int = 5
    audit_min_age_days: int = 14
    promotion_purge_hours: int = 72
    promotion_embargo_hours: int = 6

    # CPCV is used inside the development/training history. Promotion cohorts are
    # never part of CPCV and are consumed exactly once.
    cpcv_blocks: int = 6
    cpcv_test_blocks: int = 2
    cpcv_max_splits: int = 12

    # Survival bins are deliberately sparse at long horizons to avoid expanding
    # every one-minute decision into thousands of person-period rows.
    survival_bins_minutes: tuple[int, ...] = (
        5, 15, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320
    )

    upside_thresholds: tuple[float, ...] = (0.30, 0.50, 1.00, 2.00, 4.00)
    probability_horizons_minutes: tuple[int, ...] = (60, 240, 720, 1440, 2880, 4320)
    higher_peak_margin_pct: float = 0.02
    higher_peak_mark_margins: tuple[float, ...] = (0.02, 0.10, 0.30, 0.50)

    # Long-history sequence fingerprint/encoder.
    sequence_windows_minutes: tuple[int, ...] = (360, 720, 1440, 2880, 4320)
    sequence_segments: int = 6
    sequence_components: int = 20

    # Stable batch + online adapter.
    stable_estimators: int = 450
    small_estimators: int = 90
    adapter_estimators: int = 90
    adapter_max_weight: float = 0.35
    adapter_recent_days: int = 10
    compaction_every_adapter_rounds: int = 8
    compaction_every_days: int = 14

    # Promotion rule. Lower metric is better.
    promotion_margin: float = 0.005
    max_material_degradation: float = 0.06

    # Return-distribution policy.
    policy_quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    policy_tail_risk_weight: float = 0.75
    policy_upside_weight: float = 0.20
    policy_min_rows: int = 80

    # OOS provenance.
    max_live_prediction_lag_minutes: float = 10.0

    # Promotion uncertainty and fixed coverage.
    promotion_bootstrap_samples: int = 1000
    promotion_confidence: float = 0.95
    promotion_min_tokens: int = 12
    promotion_required_horizons_minutes: tuple[int, ...] = (240, 720, 1440, 4320)
    promotion_required_thresholds: tuple[float, ...] = (0.50, 1.00)

    # Collector heartbeat: a token is absent only during contiguous successful captures.
    heartbeat_max_gap_minutes: float = 5.0
    heartbeat_min_valid_captures_for_death: int = 10

    # Adapter: right-censored recent data + drift-aware family weights.
    adapter_min_censor_minutes: int = 5
    adapter_validation_fraction: float = 0.20
    drift_feature_sample_per_token: int = 12
    drift_weight_floor: float = 0.02
    drift_weight_grid_steps: int = 8

    # Adaptive probability calibration.
    calibration_learning_rate: float = 0.02
    calibration_clip_logit: float = 2.5
    conformal_window: int = 500
    conformal_alpha: float = 0.10

    # Counterfactual policy / OPE.
    counterfactual_horizons_minutes: tuple[int, ...] = (5, 15, 30, 60, 240)
    propensity_floor: float = 0.02
    dr_clip_weight: float = 20.0
    policy_promotion_bootstrap_samples: int = 1000
    policy_promotion_confidence: float = 0.95
    policy_min_promotion_tokens: int = 12

    # Sequence challenger. Token-balanced, time-aware TS2Vec-style encoder.
    sequence_balance_rows_per_token: int = 24
    sequence_grid_points: int = 96
    sequence_challenger_dim: int = 32
    sequence_challenger_epochs: int = 8
    sequence_challenger_batch_size: int = 32
    sequence_challenger_min_tokens: int = 24

    # Minute observations are strongly autocorrelated.  Bound the rows presented
    # to the estimators while retaining every token and evenly covering each
    # lifecycle.  External promotion/audit evaluation remains completely
    # unsampled.
    model_max_training_rows: int = 50_000
    model_max_rows_per_token: int = 384

    # Slippage remains dormant until enough executed-liquidity observations exist.
    liquidity_min_rows: int = 100
    liquidity_validation_fraction: float = 0.20
    liquidity_min_validation_rows: int = 30
    fallback_round_trip_bps: float = 100.0
    friction_stress_bps: tuple[float, ...] = (100.0, 300.0, 500.0, 1000.0)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _iso(value: Any) -> str:
    return _utc(value).isoformat()


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    try:
        obj = json.loads(value)
        return obj if isinstance(obj, dict) else {}
    except MemoryError:
        raise
    except Exception:
        return {}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_column(conn: sqlite3.Connection, table: str, definition: str) -> None:
    name = definition.split()[0]
    if name not in _table_columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def migrate(conn: sqlite3.Connection) -> None:
    peak.migrate(conn)
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {COHORT_TABLE} (
            cohort_id TEXT PRIMARY KEY,
            ordinal INTEGER NOT NULL UNIQUE,
            start_at TEXT NOT NULL,
            end_at TEXT NOT NULL,
            role TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            consumed_at TEXT,
            promotion_id TEXT,
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_{COHORT_TABLE}_role_status
            ON {COHORT_TABLE}(role,status,start_at);

        CREATE TABLE IF NOT EXISTS {LIFETIME_TABLE} (
            lifetime_id TEXT PRIMARY KEY,
            token_key TEXT NOT NULL,
            episode_index INTEGER NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            terminal_at TEXT,
            terminal_reason TEXT,
            observation_count INTEGER NOT NULL,
            UNIQUE(token_key, episode_index)
        );
        CREATE INDEX IF NOT EXISTS idx_{LIFETIME_TABLE}_token
            ON {LIFETIME_TABLE}(token_key,first_seen_at);

        CREATE TABLE IF NOT EXISTS {PREDICTION_LEDGER} (
            prediction_id TEXT PRIMARY KEY,
            token_key TEXT NOT NULL,
            decision_at TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            forecaster_model_hash TEXT,
            forecaster_training_cutoff TEXT,
            provenance TEXT NOT NULL,
            fold_id TEXT,
            prediction_json TEXT NOT NULL,
            policy_training_eligible INTEGER NOT NULL,
            ineligibility_reason TEXT,
            UNIQUE(token_key,decision_at,provenance,forecaster_model_hash)
        );
        CREATE INDEX IF NOT EXISTS idx_{PREDICTION_LEDGER}_policy
            ON {PREDICTION_LEDGER}(policy_training_eligible,decision_at);

        CREATE TABLE IF NOT EXISTS {SEQUENCE_CACHE_TABLE} (
            token_key TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            fingerprint_json TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(token_key,snapshot_at)
        );
        CREATE INDEX IF NOT EXISTS idx_{SEQUENCE_CACHE_TABLE}_time
            ON {SEQUENCE_CACHE_TABLE}(snapshot_at);

        CREATE TABLE IF NOT EXISTS {MODEL_REGISTRY} (
            version_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            model_path TEXT NOT NULL,
            model_hash TEXT,
            status TEXT NOT NULL,
            stable_training_cutoff TEXT NOT NULL,
            stable_generation INTEGER NOT NULL,
            adapter_round INTEGER NOT NULL,
            adapter_training_start TEXT,
            adapter_training_cutoff TEXT,
            metrics_json TEXT NOT NULL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS {PROMOTION_TABLE} (
            promotion_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            cohort_id TEXT NOT NULL,
            candidate_path TEXT NOT NULL,
            candidate_hash TEXT,
            champion_before_path TEXT,
            champion_before_hash TEXT,
            promoted INTEGER NOT NULL,
            metrics_json TEXT NOT NULL,
            reason TEXT,
            FOREIGN KEY(cohort_id) REFERENCES {COHORT_TABLE}(cohort_id)
        );

        CREATE TABLE IF NOT EXISTS {POLICY_REGISTRY} (
            version_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            model_path TEXT NOT NULL,
            model_hash TEXT,
            status TEXT NOT NULL,
            training_rows_entry INTEGER NOT NULL,
            training_rows_hold INTEGER NOT NULL,
            oos_only INTEGER NOT NULL,
            metrics_json TEXT NOT NULL,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS {AUDIT_RESULTS_TABLE} (
            audit_result_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            model_family TEXT NOT NULL,
            prediction_rows INTEGER NOT NULL,
            metric_json TEXT NOT NULL,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS {LIQUIDITY_TABLE} (
            observation_id TEXT PRIMARY KEY,
            token_key TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            trade_size_usd REAL NOT NULL,
            quoted_price REAL,
            executed_price REAL,
            slippage_bps REAL,
            liquidity_usd REAL,
            volume_usd REAL,
            market_cap_usd REAL,
            source TEXT,
            extra_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_{LIQUIDITY_TABLE}_time
            ON {LIQUIDITY_TABLE}(observed_at);

        CREATE TABLE IF NOT EXISTS {TOKEN_ASSIGNMENT_TABLE} (
            token_key TEXT PRIMARY KEY,
            first_seen_at TEXT NOT NULL,
            birth_ordinal INTEGER NOT NULL,
            forecast_cohort_id TEXT NOT NULL,
            forecast_role TEXT NOT NULL,
            policy_cohort_id TEXT NOT NULL,
            policy_role TEXT NOT NULL,
            assigned_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_{TOKEN_ASSIGNMENT_TABLE}_forecast
            ON {TOKEN_ASSIGNMENT_TABLE}(forecast_role,birth_ordinal);
        CREATE INDEX IF NOT EXISTS idx_{TOKEN_ASSIGNMENT_TABLE}_policy
            ON {TOKEN_ASSIGNMENT_TABLE}(policy_role,birth_ordinal);

        CREATE TABLE IF NOT EXISTS {POLICY_COHORT_TABLE} (
            cohort_id TEXT PRIMARY KEY,
            ordinal INTEGER NOT NULL UNIQUE,
            start_at TEXT NOT NULL,
            end_at TEXT NOT NULL,
            role TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            consumed_at TEXT,
            promotion_id TEXT,
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_{POLICY_COHORT_TABLE}_role_status
            ON {POLICY_COHORT_TABLE}(role,status,start_at);

        CREATE TABLE IF NOT EXISTS {POLICY_PROMOTION_TABLE} (
            promotion_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            cohort_id TEXT NOT NULL,
            candidate_path TEXT NOT NULL,
            candidate_hash TEXT,
            champion_before_path TEXT,
            champion_before_hash TEXT,
            promoted INTEGER NOT NULL,
            metrics_json TEXT NOT NULL,
            reason TEXT
        );

        CREATE TABLE IF NOT EXISTS {CAPTURE_HEARTBEAT_TABLE} (
            capture_at TEXT PRIMARY KEY,
            completed_at TEXT NOT NULL,
            valid_capture INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            source TEXT,
            details_json TEXT,
            first_ingested_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_{CAPTURE_HEARTBEAT_TABLE}_valid_time
            ON {CAPTURE_HEARTBEAT_TABLE}(valid_capture,capture_at);

        CREATE TABLE IF NOT EXISTS {CALIBRATION_STATE_TABLE} (
            calibration_key TEXT PRIMARY KEY,
            bias_logit REAL NOT NULL DEFAULT 0,
            n_updates INTEGER NOT NULL DEFAULT 0,
            last_updated_at TEXT,
            residual_scores_json TEXT,
            config_hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {DATA_VINTAGE_TABLE} (
            token_key TEXT NOT NULL,
            event_time TEXT NOT NULL,
            first_ingested_at TEXT NOT NULL,
            last_corrected_at TEXT NOT NULL,
            value_version INTEGER NOT NULL DEFAULT 1,
            row_fingerprint TEXT NOT NULL,
            PRIMARY KEY(token_key,event_time)
        );
        CREATE INDEX IF NOT EXISTS idx_{DATA_VINTAGE_TABLE}_ingested
            ON {DATA_VINTAGE_TABLE}(first_ingested_at);

        CREATE TABLE IF NOT EXISTS {COUNTERFACTUAL_TABLE} (
            token_key TEXT NOT NULL,
            decision_at TEXT NOT NULL,
            action_kind TEXT NOT NULL,
            horizon_minutes INTEGER NOT NULL,
            decision_mc REAL NOT NULL,
            next_observed_at TEXT,
            next_observed_mc REAL,
            exit_now_return REAL,
            hold_terminal_return REAL,
            hold_best_return REAL,
            hold_worst_return REAL,
            target_ready_at TEXT,
            source_fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(token_key,decision_at,action_kind,horizon_minutes)
        );

        CREATE TABLE IF NOT EXISTS {CALIBRATION_UPDATES_TABLE} (
            prediction_id TEXT NOT NULL,
            calibration_key TEXT NOT NULL,
            resolved_at TEXT NOT NULL,
            observed_target REAL NOT NULL,
            raw_probability REAL NOT NULL,
            PRIMARY KEY(prediction_id,calibration_key)
        );

        CREATE TABLE IF NOT EXISTS {SEQUENCE_MODEL_REGISTRY} (
            version_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            model_kind TEXT NOT NULL,
            status TEXT NOT NULL,
            training_cutoff TEXT NOT NULL,
            training_tokens INTEGER NOT NULL,
            metrics_json TEXT NOT NULL,
            model_path TEXT,
            model_hash TEXT,
            notes TEXT
        );
        """
    )
    # New registry/provenance fields are additive for safe upgrades.
    _ensure_column(conn, SEQUENCE_CACHE_TABLE, "source_fingerprint TEXT")
    for definition in (
        "target_definition_hash TEXT",
        "feature_definition_hash TEXT",
        "execution_definition_hash TEXT",
        "data_vintage_hash TEXT",
        "oos_valid INTEGER NOT NULL DEFAULT 0",
    ):
        _ensure_column(conn, PREDICTION_LEDGER, definition)
    for definition in (
        "stable_created_at TEXT",
        "last_compaction_at TEXT",
        "target_definition_hash TEXT",
        "feature_definition_hash TEXT",
        "execution_definition_hash TEXT",
        "training_data_hash TEXT",
    ):
        _ensure_column(conn, MODEL_REGISTRY, definition)
    for definition in (
        "hold_advantage_return REAL",
        "entry_execution_return REAL",
    ):
        _ensure_column(conn, COUNTERFACTUAL_TABLE, definition)
    for definition in (
        "side TEXT",
        "first_ingested_at TEXT",
    ):
        _ensure_column(conn, LIQUIDITY_TABLE, definition)
    _ensure_column(conn, DATA_VINTAGE_TABLE, "ingestion_provenance TEXT")
    for definition in (
        "target_definition_hash TEXT",
        "feature_definition_hash TEXT",
        "execution_definition_hash TEXT",
        "forecast_schema_hash TEXT",
    ):
        _ensure_column(conn, POLICY_REGISTRY, definition)
    conn.commit()


# ---------------------------------------------------------------------------
# Lifetime segmentation and calendar cohorts
# ---------------------------------------------------------------------------

def refresh_capture_heartbeats(conn: sqlite3.Connection, observations: pd.DataFrame | None = None) -> dict[str, int]:
    """Persist successful Axiom captures separately from token presence.

    Distinct observation snapshots are indisputable successful captures.  A future
    collector can call ``record_capture_heartbeat`` even for an empty-but-valid
    page; the model never infers a heartbeat merely from elapsed wall time.
    """
    migrate(conn)
    if observations is None:
        observations, _ = peak.load_observations(conn)
    if observations.empty:
        return {"inserted": 0, "total": int(conn.execute(f"SELECT COUNT(*) FROM {CAPTURE_HEARTBEAT_TABLE}").fetchone()[0])}
    counts = observations.groupby("snapshot_at").token_key.nunique().sort_index()
    now = _now_iso(); inserted = 0
    for t, n in counts.items():
        conn.execute(
            f"""INSERT OR IGNORE INTO {CAPTURE_HEARTBEAT_TABLE}
                (capture_at,completed_at,valid_capture,row_count,source,details_json,first_ingested_at)
                VALUES(?,?,?,?,?,?,?)""",
            (_iso(t), _iso(t), 1, int(n), "inferred_from_observation_snapshot", _json({}), now),
        )
        inserted += int(conn.execute("SELECT changes()").fetchone()[0] > 0)
    conn.commit()
    return {"inserted": inserted, "total": int(conn.execute(f"SELECT COUNT(*) FROM {CAPTURE_HEARTBEAT_TABLE}").fetchone()[0])}


def _upsert_capture_heartbeat(
    conn: sqlite3.Connection,
    capture_at: Any,
    *,
    valid_capture: bool,
    row_count: int,
    source: str = "collector",
    details: dict[str, Any] | None = None,
) -> None:
    now = _now_iso()
    conn.execute(
        f"""INSERT INTO {CAPTURE_HEARTBEAT_TABLE}
            (capture_at,completed_at,valid_capture,row_count,source,details_json,first_ingested_at)
            VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(capture_at) DO UPDATE SET
              completed_at=excluded.completed_at,valid_capture=excluded.valid_capture,
              row_count=excluded.row_count,source=excluded.source,details_json=excluded.details_json""",
        (_iso(capture_at), now, int(valid_capture), int(row_count), source, _json(details or {}), now),
    )


def record_capture_heartbeat(
    conn: sqlite3.Connection,
    capture_at: Any,
    *,
    valid_capture: bool,
    row_count: int,
    source: str = "collector",
    details: dict[str, Any] | None = None,
) -> None:
    migrate(conn)
    _upsert_capture_heartbeat(
        conn,
        capture_at,
        valid_capture=valid_capture,
        row_count=row_count,
        source=source,
        details=details,
    )
    conn.commit()


def _contiguous_capture_absence(
    conn: sqlite3.Connection,
    last_seen: pd.Timestamp,
    cfg: V24Config,
    *,
    upto: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    upto = _utc(upto or pd.Timestamp.now(tz="UTC"))
    rows = conn.execute(
        f"""SELECT capture_at FROM {CAPTURE_HEARTBEAT_TABLE}
            WHERE valid_capture=1 AND capture_at>? AND capture_at<=? ORDER BY capture_at""",
        (last_seen.isoformat(), upto.isoformat()),
    ).fetchall()
    times = [_utc(r[0]) for r in rows]
    if not times:
        return None, None, 0
    max_gap = pd.Timedelta(minutes=cfg.heartbeat_max_gap_minutes)
    run_start = last_seen if times[0] - last_seen <= max_gap else times[0]
    prev = times[0]; count = 1
    for t in times[1:]:
        if t - prev > max_gap:
            run_start = t; count = 1
        else:
            count += 1
        prev = t
    return run_start, prev, count


def refresh_data_vintage(conn: sqlite3.Connection, observations: pd.DataFrame) -> dict[str, int]:
    """Track when the *current value version* became available.

    Canonical V24 observations are immutable hard inserts.  Their original
    ``created_at`` is therefore valid knowledge time when it is corroborated by a
    completed clipboard-valid capture, a successful matching capture attempt, an
    exact archived payload, and the production collection-session contract.  This
    lets a first model reconstruct vintage even if the derived vintage table was
    not materialized continuously during collection.  Imported/replayed rows and
    uncorroborated legacy history remain unknown and cannot be replayed through a
    historical cutoff.
    """
    migrate(conn)
    _ensure_column(conn, DATA_VINTAGE_TABLE, "ingestion_provenance TEXT")
    now = _now_iso(); inserted = corrected = reconstructed = 0
    if observations.empty:
        return {"inserted": 0, "corrected": 0, "reconstructed": 0}

    trusted_cycles: dict[int, pd.Timestamp] = {}
    required_tables = {
        "collection_sessions", "capture_cycles", "capture_attempts", "capture_payloads"
    }
    if all(_table_exists(conn, table) for table in required_tables):
        rows = conn.execute(
            """SELECT DISTINCT c.cycle_id,c.captured_at
               FROM capture_cycles c
               JOIN collection_sessions s ON s.session_id=c.session_id
               JOIN capture_attempts a ON a.cycle_id=c.cycle_id
               JOIN capture_payloads p ON p.cycle_id=c.cycle_id
               WHERE c.completed=1 AND c.clipboard_valid=1
                 AND s.purpose='v24_production_raw_collection'
                 AND s.collector_schema='v24_clipboard_raw_v2'
                 AND a.success=1 AND a.clipboard_valid=1
                 AND a.raw_payload_sha256=p.sha256
                 AND a.raw_payload_bytes=p.byte_count"""
        ).fetchall()
        trusted_cycles = {int(cycle): _utc(captured) for cycle, captured in rows}

    def durable_ingestion(rec: dict[str, Any], event_ts: pd.Timestamp) -> pd.Timestamp | None:
        cycle = rec.get("cycle_id")
        created = rec.get("created_at")
        if cycle is None or pd.isna(cycle) or created is None or pd.isna(created):
            return None
        captured = trusted_cycles.get(int(cycle))
        if captured is None or abs((captured - event_ts).total_seconds()) > 1.0:
            return None
        try:
            ingested = _utc(created)
        except (TypeError, ValueError):
            return None
        # SQLite CURRENT_TIMESTAMP is second-precision UTC, while capture times
        # may contain microseconds.  Permit that truncation and ordinary capture
        # processing latency, but never reinterpret a historical replay as live.
        delay = (ingested - event_ts).total_seconds()
        return ingested if -1.0 <= delay <= 15.0 * 60.0 else None

    safe_cols = [c for c in observations.columns if c not in {"_rowid_"}]
    for r in observations[safe_cols].itertuples(index=False, name=None):
        rec = dict(zip(safe_cols, r))
        token = str(rec.get("token_key")); event = _iso(rec.get("snapshot_at"))
        event_ts = _utc(event)
        durable = durable_ingestion(rec, event_ts)
        fp = _stable_hash(rec)
        prior = conn.execute(
            f"""SELECT row_fingerprint,value_version,first_ingested_at,last_corrected_at,
                       ingestion_provenance
                FROM {DATA_VINTAGE_TABLE} WHERE token_key=? AND event_time=?""",
            (token, event),
        ).fetchone()
        if prior is None:
            known_at = durable.isoformat() if durable is not None else now
            provenance = "canonical_prospective_insert" if durable is not None else "legacy_unknown_vintage"
            conn.execute(
                f"""INSERT INTO {DATA_VINTAGE_TABLE}
                    (token_key,event_time,first_ingested_at,last_corrected_at,value_version,row_fingerprint,ingestion_provenance)
                    VALUES(?,?,?,?,?,?,?)""",
                (token, event, known_at, known_at, 1, fp, provenance),
            ); inserted += 1
        elif str(prior[0]) != fp:
            conn.execute(
                f"""UPDATE {DATA_VINTAGE_TABLE}
                    SET last_corrected_at=?,value_version=?,row_fingerprint=?,
                        ingestion_provenance=?
                    WHERE token_key=? AND event_time=?""",
                (now, int(prior[1]) + 1, fp, "corrected_after_ingestion", token, event),
            ); corrected += 1
        elif durable is not None and str(prior[4] or "") == "legacy_unknown_vintage":
            # The value is byte-for-byte unchanged since vintage was first
            # observed, and the authoritative capture tables independently prove
            # when the immutable canonical row was inserted.
            known_at = durable.isoformat()
            conn.execute(
                f"""UPDATE {DATA_VINTAGE_TABLE}
                    SET first_ingested_at=?,last_corrected_at=?,ingestion_provenance=?
                    WHERE token_key=? AND event_time=?""",
                (known_at, known_at, "canonical_prospective_reconstructed", token, event),
            ); reconstructed += 1
    conn.commit()
    return {"inserted": inserted, "corrected": corrected, "reconstructed": reconstructed}


def refresh_lifetimes(
    conn: sqlite3.Connection,
    cfg: V24Config,
    observations: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Track token episodes using successful-capture absence, never wall-clock gaps alone."""
    migrate(conn)
    obs = observations
    if obs is None:
        obs, _ = peak.load_observations(conn)
    if obs.empty:
        return {"tokens": 0, "lifetimes": 0, "new_lifetimes": 0}
    refresh_capture_heartbeats(conn, obs)
    new_lifetimes = 0
    latest_capture = _utc(obs.snapshot_at.max())
    for token, g in obs.groupby("token_key", sort=False):
        token = str(token); g = g.sort_values("snapshot_at").reset_index(drop=True)
        # Rebuild only this token's episode segmentation deterministically.  The table
        # is derived metadata; raw observations are untouched.
        old_rows = conn.execute(f"SELECT lifetime_id FROM {LIFETIME_TABLE} WHERE token_key=?", (token,)).fetchall()
        conn.execute(f"DELETE FROM {LIFETIME_TABLE} WHERE token_key=?", (token,))
        episode = 0; start = _utc(g.iloc[0].snapshot_at); last = start; count = 1
        current_rows = [g.iloc[0]]
        segments: list[tuple[pd.Timestamp,pd.Timestamp,int,str | None,pd.Timestamp | None]] = []
        for j in range(1, len(g)):
            t = _utc(g.iloc[j].snapshot_at)
            run_start, run_end, ncap = _contiguous_capture_absence(conn, last, cfg, upto=t)
            true_terminal = None
            if run_start is not None and run_end is not None:
                if (run_end-run_start).total_seconds()/60.0 >= cfg.operational_gap_minutes and ncap >= cfg.heartbeat_min_valid_captures_for_death:
                    true_terminal = run_start + pd.Timedelta(minutes=cfg.operational_gap_minutes)
            if true_terminal is not None and true_terminal < t:
                segments.append((start,last,count,"operational_gap_then_reappearance",true_terminal))
                episode += 1; start=t; count=1; current_rows=[g.iloc[j]]
            else:
                count += 1; current_rows.append(g.iloc[j])
            last=t
        # Open/current segment may already be terminal if the token is still absent.
        terminal_reason = None; terminal_at = None
        age = pd.to_numeric(pd.Series([g.iloc[-1].get("age_minutes")]), errors="coerce").iloc[0] if "age_minutes" in g.columns else np.nan
        if pd.notna(age) and float(age) >= cfg.age_out_minutes:
            terminal_reason="natural_axiom_age_out"; terminal_at=last
        else:
            run_start,run_end,ncap=_contiguous_capture_absence(conn,last,cfg,upto=latest_capture)
            if run_start is not None and run_end is not None and (run_end-run_start).total_seconds()/60.0 >= cfg.operational_gap_minutes and ncap >= cfg.heartbeat_min_valid_captures_for_death:
                terminal_reason="dead_after_valid_capture_absence"; terminal_at=run_start+pd.Timedelta(minutes=cfg.operational_gap_minutes)
        segments.append((start,last,count,terminal_reason,terminal_at))
        for ep,(first_seen,last_seen,n,reason,term) in enumerate(segments):
            lid=hashlib.sha256(f"{token}|{ep}|{first_seen.isoformat()}".encode()).hexdigest()[:24]
            conn.execute(
                f"""INSERT INTO {LIFETIME_TABLE}
                    (lifetime_id,token_key,episode_index,first_seen_at,last_seen_at,terminal_at,terminal_reason,observation_count)
                    VALUES(?,?,?,?,?,?,?,?)""",
                (lid,token,ep,first_seen.isoformat(),last_seen.isoformat(),term.isoformat() if term is not None else None,reason,int(n)),
            )
        new_lifetimes += max(0, len(segments)-len(old_rows))
    conn.commit()
    return {
        "tokens": int(obs.token_key.nunique()),
        "lifetimes": int(conn.execute(f"SELECT COUNT(*) FROM {LIFETIME_TABLE}").fetchone()[0]),
        "new_lifetimes": int(new_lifetimes),
    }


def _block_floor(ts: pd.Timestamp, hours: int) -> pd.Timestamp:
    ts = _utc(ts)
    epoch_hours = int(ts.timestamp() // 3600)
    block = (epoch_hours // hours) * hours
    return pd.Timestamp(block * 3600, unit="s", tz="UTC")


def _forecast_cohort_role_status(ordinal: int, cfg: V24Config) -> tuple[str, str]:
    if ordinal < cfg.warmup_blocks:
        return "train", "available"
    post = ordinal - cfg.warmup_blocks + 1
    is_eval = cfg.promotion_every_n_blocks <= 1 or post % cfg.promotion_every_n_blocks == 0
    if not is_eval:
        return "train", "available"
    eval_no = max(1, post // max(1, cfg.promotion_every_n_blocks))
    if cfg.audit_every_n_blocks > 0 and eval_no % cfg.audit_every_n_blocks == 0:
        return "audit", "sealed"
    return "promotion", "available"


def _policy_cohort_role_status(
    ordinal: int,
    forecast_role: str,
    cfg: V24Config,
) -> tuple[str, str]:
    if ordinal < cfg.warmup_blocks:
        return "train", "available"
    if forecast_role == "audit":
        return "audit", "sealed"
    n = max(1, int(cfg.promotion_every_n_blocks))
    offset = max(1, n // 2)
    post = ordinal - cfg.warmup_blocks + 1
    return ("promotion", "available") if (post % n) == offset else ("train", "available")


def refresh_calendar_cohorts(conn: sqlite3.Connection, cfg: V24Config) -> dict[str, int]:
    """Create immutable calendar cohort roles.

    The first warmup_blocks are ordinary training history. Thereafter every
    audit_every_n_blocks-th block is permanently sealed audit data; all other blocks
    are one-use promotion cohorts. Promotion cohorts become training-eligible only
    after they have been consumed once.
    """
    migrate(conn)
    labels = pd.read_sql_query(
        f"SELECT decision_at,path_end_at FROM {peak.LABEL_TABLE} ORDER BY decision_at", conn
    )
    if labels.empty:
        return {"created": 0, "total": 0}
    decisions = pd.to_datetime(labels.decision_at, format="ISO8601", utc=True, errors="coerce").dropna()
    if decisions.empty:
        return {"created": 0, "total": 0}
    first = _block_floor(decisions.min(), cfg.cohort_hours)
    last = _block_floor(decisions.max(), cfg.cohort_hours)
    existing = {
        int(r[0]): (str(r[1]), str(r[2]))
        for r in conn.execute(f"SELECT ordinal,role,status FROM {COHORT_TABLE}").fetchall()
    }
    created = 0
    ordinal = 0
    cur = first
    while cur <= last:
        end = cur + pd.Timedelta(hours=cfg.cohort_hours)
        if ordinal not in existing:
            role, status = _forecast_cohort_role_status(ordinal, cfg)
            cohort_id = f"c{ordinal:06d}_{cur.strftime('%Y%m%dT%H%M%SZ')}"
            conn.execute(
                f"""INSERT INTO {COHORT_TABLE}
                    (cohort_id,ordinal,start_at,end_at,role,status,created_at)
                    VALUES(?,?,?,?,?,?,?)""",
                (cohort_id, ordinal, cur.isoformat(), end.isoformat(), role, status, _now_iso()),
            )
            created += 1
        ordinal += 1
        cur = end
    conn.commit()
    total = int(conn.execute(f"SELECT COUNT(*) FROM {COHORT_TABLE}").fetchone()[0])
    return {"created": created, "total": total}


def refresh_policy_cohorts(conn: sqlite3.Connection, cfg: V24Config) -> dict[str, int]:
    """Create a policy-only one-use schedule offset from forecaster promotion blocks."""
    migrate(conn)
    forecast = _cohort_rows(conn)
    if forecast.empty:
        return {"created": 0, "total": 0}
    existing = {int(r[0]) for r in conn.execute(f"SELECT ordinal FROM {POLICY_COHORT_TABLE}").fetchall()}
    created = 0
    for r in forecast.itertuples(index=False):
        ordinal=int(r.ordinal)
        if ordinal in existing:
            continue
        role, status = _policy_cohort_role_status(ordinal, str(r.role), cfg)
        cid=f"p{ordinal:06d}_{_utc(r.start_at).strftime('%Y%m%dT%H%M%SZ')}"
        conn.execute(
            f"""INSERT INTO {POLICY_COHORT_TABLE}
                (cohort_id,ordinal,start_at,end_at,role,status,created_at) VALUES(?,?,?,?,?,?,?)""",
            (cid,ordinal,r.start_at,r.end_at,role,status,_now_iso()),
        ); created+=1
    conn.commit()
    return {"created":created,"total":int(conn.execute(f"SELECT COUNT(*) FROM {POLICY_COHORT_TABLE}").fetchone()[0])}


def refresh_token_assignments(conn: sqlite3.Connection, cfg: V24Config) -> dict[str, int]:
    """Permanently assign every token lifetime to holdout roles at first observation.

    The role is immutable: later decisions never cross from train to promotion/audit
    merely because a 72-hour token lived across a calendar boundary.
    """
    migrate(conn); refresh_calendar_cohorts(conn,cfg); refresh_policy_cohorts(conn,cfg)
    obs,_=peak.load_observations(conn)
    if obs.empty: return {"assigned":0,"total":0}
    firsts=obs.groupby(obs.token_key.astype(str)).snapshot_at.min().sort_values()
    forecast=_cohort_rows(conn)
    policy_rows=pd.read_sql_query(f"SELECT * FROM {POLICY_COHORT_TABLE} ORDER BY ordinal",conn)
    assigned=0
    for token, first_seen in firsts.items():
        if conn.execute(f"SELECT 1 FROM {TOKEN_ASSIGNMENT_TABLE} WHERE token_key=?",(str(token),)).fetchone():
            continue
        t=_utc(first_seen)
        f=forecast[(pd.to_datetime(forecast.start_at,format="ISO8601", utc=True)<=t)&(pd.to_datetime(forecast.end_at,format="ISO8601", utc=True)>t)]
        p=policy_rows[(pd.to_datetime(policy_rows.start_at,format="ISO8601", utc=True)<=t)&(pd.to_datetime(policy_rows.end_at,format="ISO8601", utc=True)>t)]
        if f.empty or p.empty:
            continue
        fr=f.iloc[-1]; pr=p.iloc[-1]
        conn.execute(
            f"""INSERT INTO {TOKEN_ASSIGNMENT_TABLE}
                (token_key,first_seen_at,birth_ordinal,forecast_cohort_id,forecast_role,
                 policy_cohort_id,policy_role,assigned_at) VALUES(?,?,?,?,?,?,?,?)""",
            (str(token),t.isoformat(),int(fr.ordinal),str(fr.cohort_id),str(fr.role),str(pr.cohort_id),str(pr.role),_now_iso()),
        ); assigned+=1
    conn.commit()
    return {"assigned":assigned,"total":int(conn.execute(f"SELECT COUNT(*) FROM {TOKEN_ASSIGNMENT_TABLE}").fetchone()[0])}


def _ensure_live_cohorts_through(
    conn: sqlite3.Connection,
    timestamp: pd.Timestamp,
    cfg: V24Config,
) -> None:
    """Extend immutable cohort schedules without requiring current labels."""
    first = conn.execute(
        f"SELECT ordinal,start_at FROM {COHORT_TABLE} ORDER BY ordinal LIMIT 1"
    ).fetchone()
    if first is None:
        start = _block_floor(timestamp, cfg.cohort_hours)
        first_ordinal = 0
    else:
        first_ordinal = int(first[0])
        start = _utc(first[1])
    if first_ordinal != 0:
        raise RuntimeError("V24 calendar cohort schedule does not begin at ordinal zero.")

    target = _block_floor(timestamp, cfg.cohort_hours)
    ordinal = int((target - start).total_seconds() // (cfg.cohort_hours * 3600))
    if ordinal < 0:
        raise RuntimeError("Live token predates the immutable V24 cohort schedule.")
    now = _now_iso()
    for number in range(ordinal + 1):
        block_start = start + pd.Timedelta(hours=number * cfg.cohort_hours)
        block_end = block_start + pd.Timedelta(hours=cfg.cohort_hours)
        forecast_role, forecast_status = _forecast_cohort_role_status(number, cfg)
        forecast_id = f"c{number:06d}_{block_start.strftime('%Y%m%dT%H%M%SZ')}"
        conn.execute(
            f"""INSERT OR IGNORE INTO {COHORT_TABLE}
                (cohort_id,ordinal,start_at,end_at,role,status,created_at)
                VALUES(?,?,?,?,?,?,?)""",
            (
                forecast_id,
                number,
                block_start.isoformat(),
                block_end.isoformat(),
                forecast_role,
                forecast_status,
                now,
            ),
        )
        policy_role, policy_status = _policy_cohort_role_status(number, forecast_role, cfg)
        policy_id = f"p{number:06d}_{block_start.strftime('%Y%m%dT%H%M%SZ')}"
        conn.execute(
            f"""INSERT OR IGNORE INTO {POLICY_COHORT_TABLE}
                (cohort_id,ordinal,start_at,end_at,role,status,created_at)
                VALUES(?,?,?,?,?,?,?)""",
            (
                policy_id,
                number,
                block_start.isoformat(),
                block_end.isoformat(),
                policy_role,
                policy_status,
                now,
            ),
        )


def _ensure_live_token_assignments(
    conn: sqlite3.Connection,
    token_keys: Iterable[str],
    cfg: V24Config,
) -> int:
    """Assign current tokens once using raw first-seen time and immutable roles."""
    tokens = sorted(set(map(str, token_keys)))
    if not tokens:
        return 0
    source = peak.discover_observation_source(conn)
    quote = lambda value: '"' + str(value).replace('"', '""') + '"'
    first_seen: dict[str, pd.Timestamp] = {}
    for offset in range(0, len(tokens), 500):
        batch = tokens[offset: offset + 500]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT {quote(source['token'])},MIN({quote(source['time'])}) "
            f"FROM {quote(source['table'])} "
            f"WHERE {quote(source['token'])} IN ({placeholders}) "
            f"GROUP BY {quote(source['token'])}",
            batch,
        ).fetchall()
        first_seen.update({str(row[0]): _utc(row[1]) for row in rows if row[1]})
    if not first_seen:
        return 0

    if conn.execute(f"SELECT 1 FROM {COHORT_TABLE} LIMIT 1").fetchone() is None:
        _ensure_live_cohorts_through(conn, min(first_seen.values()), cfg)
    _ensure_live_cohorts_through(conn, max(first_seen.values()), cfg)
    forecast = _cohort_rows(conn)
    policy = pd.read_sql_query(
        f"SELECT * FROM {POLICY_COHORT_TABLE} ORDER BY ordinal", conn
    )
    forecast_start = pd.to_datetime(forecast.start_at, format="ISO8601", utc=True)
    forecast_end = pd.to_datetime(forecast.end_at, format="ISO8601", utc=True)
    policy_start = pd.to_datetime(policy.start_at, format="ISO8601", utc=True)
    policy_end = pd.to_datetime(policy.end_at, format="ISO8601", utc=True)
    assigned = 0
    now = _now_iso()
    for token, timestamp in first_seen.items():
        if conn.execute(
            f"SELECT 1 FROM {TOKEN_ASSIGNMENT_TABLE} WHERE token_key=?", (token,)
        ).fetchone():
            continue
        forecast_match = forecast[(forecast_start <= timestamp) & (forecast_end > timestamp)]
        policy_match = policy[(policy_start <= timestamp) & (policy_end > timestamp)]
        if forecast_match.empty or policy_match.empty:
            raise RuntimeError(
                f"Could not assign live token {token!r} to the immutable V24 cohort schedule."
            )
        fr = forecast_match.iloc[-1]
        pr = policy_match.iloc[-1]
        conn.execute(
            f"""INSERT INTO {TOKEN_ASSIGNMENT_TABLE}
                (token_key,first_seen_at,birth_ordinal,forecast_cohort_id,forecast_role,
                 policy_cohort_id,policy_role,assigned_at) VALUES(?,?,?,?,?,?,?,?)""",
            (
                token,
                timestamp.isoformat(),
                int(fr.ordinal),
                str(fr.cohort_id),
                str(fr.role),
                str(pr.cohort_id),
                str(pr.role),
                now,
            ),
        )
        assigned += 1
    return assigned


def _token_assignments(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {TOKEN_ASSIGNMENT_TABLE}",conn)


def _latest_capture(conn: sqlite3.Connection) -> pd.Timestamp | None:
    obs, _ = peak.load_observations(conn)
    return obs.snapshot_at.max() if not obs.empty else None


def next_one_use_promotion_cohort(conn: sqlite3.Connection, cfg: V24Config) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    latest = _latest_capture(conn)
    if latest is None:
        return None
    # Token membership is fixed at first-seen. A token born near the end of the
    # cohort can remain observable for ~72h, and its late-life decisions need a
    # further 72h outcome window. Reserve cohorts therefore mature only after
    # lifetime + outcome, while the online adapter uses right-censored recent data.
    deadline = latest - pd.Timedelta(minutes=2*cfg.horizon_minutes)
    return conn.execute(
        f"""SELECT * FROM {COHORT_TABLE}
            WHERE role='promotion' AND status='available' AND end_at <= ?
            ORDER BY ordinal LIMIT 1""",
        (deadline.isoformat(),),
    ).fetchone()


def _cohort_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {COHORT_TABLE} ORDER BY ordinal", conn)


def _row_cohort_ordinal(times: pd.Series, cohorts: pd.DataFrame) -> pd.Series:
    if cohorts.empty:
        return pd.Series([-1] * len(times), index=times.index, dtype=int)
    starts = pd.to_datetime(cohorts.start_at, format="ISO8601", utc=True).astype("int64").to_numpy()
    ords = cohorts.ordinal.to_numpy(dtype=int)
    vals = pd.to_datetime(times, format="ISO8601", utc=True).astype("int64").to_numpy()
    idx = np.searchsorted(starts, vals, side="right") - 1
    out = np.where(idx >= 0, ords[np.clip(idx, 0, len(ords) - 1)], -1)
    return pd.Series(out, index=times.index, dtype=int)


def _attach_calendar_and_lifetime(conn: sqlite3.Connection, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cohorts = _cohort_rows(conn)
    out["decision_calendar_ordinal"] = _row_cohort_ordinal(out.snapshot_at, cohorts)
    assignments=_token_assignments(conn)
    if not assignments.empty:
        assignments=assignments.rename(columns={"birth_ordinal":"calendar_cohort_ordinal"})
        keep=["token_key","calendar_cohort_ordinal","forecast_cohort_id","forecast_role","policy_cohort_id","policy_role","first_seen_at"]
        out=out.merge(assignments[keep],on="token_key",how="left")
    else:
        out["calendar_cohort_ordinal"]=-1; out["forecast_role"]="unassigned"; out["policy_role"]="unassigned"
    out["calendar_cohort_ordinal"]=pd.to_numeric(out["calendar_cohort_ordinal"],errors="coerce").fillna(-1).astype(int)
    # lifetime mapping by interval; token_key remains the hard grouping key.
    life = pd.read_sql_query(f"SELECT * FROM {LIFETIME_TABLE}", conn)
    out["lifetime_id"] = None
    if not life.empty:
        # Stored ISO timestamps legitimately mix whole and fractional seconds.
        # Explicit ISO parsing preserves both precisions and still rejects bad data.
        life["first_seen_at"] = pd.to_datetime(life.first_seen_at, format="ISO8601", utc=True)
        life["last_seen_at"] = pd.to_datetime(life.last_seen_at, format="ISO8601", utc=True)
        for token, idxs in out.groupby("token_key").groups.items():
            episodes = life[life.token_key.astype(str) == str(token)].sort_values("first_seen_at")
            if episodes.empty: continue
            for i in idxs:
                t = out.at[i, "snapshot_at"]
                candidates = episodes[(episodes.first_seen_at <= t) & (episodes.last_seen_at >= t)]
                if candidates.empty: candidates = episodes[episodes.first_seen_at <= t].tail(1)
                if not candidates.empty: out.at[i,"lifetime_id"]=candidates.iloc[-1].lifetime_id
    out["lifetime_id"] = out["lifetime_id"].fillna(out.token_key.astype(str))
    return out


# ---------------------------------------------------------------------------
# Long-history causal sequence fingerprint + train-only encoder
# ---------------------------------------------------------------------------

_SEQUENCE_BASES = (
    "market_cap_usd", "volume_usd", "fees_sol", "txns", "holders",
    "pro_traders", "kols", "recent_visitors", "top10_holders_pct",
    "sniper_pct", "insider_pct", "bundler_pct",
)


def _obs_numeric(obs: pd.DataFrame, name: str) -> np.ndarray:
    if name not in obs:
        return np.full(len(obs), np.nan)
    return pd.to_numeric(obs[name], errors="coerce").to_numpy(dtype=float)


def _path_stats(times:np.ndarray,vals:np.ndarray,i:int,window_minutes:int,segments:int) -> dict[str,float]:
    now=int(times[i]); window_ns=int(window_minutes*60*1e9); start=now-window_ns
    a=int(np.searchsorted(times[:i+1],start,side="left")); x=vals[a:i+1]; t=times[a:i+1]
    mask=np.isfinite(x); x=x[mask]; t=t[mask]
    out={"has_window":float(len(x)>=2),"coverage":0.0,"points":float(len(x))}
    if len(x)<2: return out
    eps=1e-9; first,last=float(x[0]),float(x[-1]); duration=max(0,int(t[-1])-int(t[0])); out["coverage"]=min(1.0,duration/max(1,window_ns))
    change=last/first-1.0 if abs(first)>eps else np.nan; diffs=np.diff(np.log(np.clip(np.abs(x),eps,None))); total=float(np.sum(np.abs(diffs))); direct=float(abs(np.log(max(abs(last),eps)/max(abs(first),eps))))
    out.update({"change":change,"log_vol":float(np.std(diffs)) if len(diffs) else 0.0,"positive_ratio":float(np.mean(diffs>0)) if len(diffs) else 0.0,"path_efficiency":direct/total if total>eps else 0.0,"fraction_of_high":last/float(np.nanmax(x)) if np.nanmax(x)!=0 else np.nan,"fraction_of_low":last/float(np.nanmin(x)) if np.nanmin(x)!=0 else np.nan})
    # Equal elapsed-time segments, not equal observation-count segments.
    edges=np.linspace(start,now,segments+1)
    for si in range(segments):
        lo,hi=edges[si],edges[si+1]; sm=(t>=lo)&(t<hi if si<segments-1 else t<=hi); sx=x[sm]
        out[f"seg{si}_present"]=float(len(sx)>=1)
        if len(sx)>=2 and abs(float(sx[0]))>eps: out[f"seg{si}_return"]=float(sx[-1])/float(sx[0])-1.0
        else: out[f"seg{si}_return"]=np.nan
    return out


def build_long_history_fingerprint(observations: pd.DataFrame, cfg: V24Config) -> pd.DataFrame:
    """Encode up to the full 72h past without future access.

    This intentionally separates raw causal path construction from the learned PCA
    encoder. PCA/scaling are fit only on the permitted training history, preventing
    validation/audit distribution information from entering the encoder.
    """
    rows: list[dict[str, Any]] = []
    bases = [c for c in _SEQUENCE_BASES if c in observations.columns]
    for token, g in observations.groupby("token_key", sort=False):
        g = g.sort_values("snapshot_at").reset_index(drop=True)
        times = pd.to_datetime(g.snapshot_at, format="ISO8601", utc=True).astype("int64").to_numpy()
        arrays = {c: _obs_numeric(g, c) for c in bases}
        for i, r in g.iterrows():
            rec: dict[str, Any] = {"token_key": str(token), "snapshot_at": r.snapshot_at}
            for c in bases:
                for w in cfg.sequence_windows_minutes:
                    stats = _path_stats(times, arrays[c], i, int(w), cfg.sequence_segments)
                    for k, v in stats.items():
                        rec[f"seqraw__{c}__{w}m__{k}"] = v
            rows.append(rec)
    return pd.DataFrame(rows)


def _sequence_config_hash(cfg: V24Config) -> str:
    payload = {"windows": list(cfg.sequence_windows_minutes), "segments": cfg.sequence_segments, "bases": list(_SEQUENCE_BASES)}
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _fingerprint_for_index(token: str, g: pd.DataFrame, times: np.ndarray, arrays: dict[str,np.ndarray], i: int, cfg: V24Config) -> dict[str, Any]:
    r=g.iloc[i]; rec={"token_key":str(token),"snapshot_at":r.snapshot_at}
    for c,vals in arrays.items():
        for w in cfg.sequence_windows_minutes:
            for k,v in _path_stats(times,vals,i,int(w),cfg.sequence_segments).items():
                rec[f"seqraw__{c}__{w}m__{k}"]=v
    return rec


def refresh_sequence_fingerprint_cache(conn:sqlite3.Connection,observations:pd.DataFrame,cfg:V24Config,force:bool=False) -> dict[str,int]:
    migrate(conn); conf=_sequence_config_hash(cfg)
    existing_hashes={str(r[0]) for r in conn.execute(f"SELECT DISTINCT config_hash FROM {SEQUENCE_CACHE_TABLE}").fetchall() if r[0]}
    if existing_hashes and existing_hashes!={conf} and not force:
        raise RuntimeError("V24 long-history sequence schema changed. Rebuild the sequence cache; mixed encoder schemas are forbidden.")
    if force: conn.execute(f"DELETE FROM {SEQUENCE_CACHE_TABLE}"); conn.commit()
    inserted=deleted=tokens=0; now=_now_iso(); bases=[c for c in _SEQUENCE_BASES if c in observations.columns]
    vintage=pd.read_sql_query(f"SELECT token_key,event_time,row_fingerprint,last_corrected_at FROM {DATA_VINTAGE_TABLE}",conn) if _table_exists(conn,DATA_VINTAGE_TABLE) else pd.DataFrame()
    if not vintage.empty:
        vintage["event_time"]=pd.to_datetime(vintage.event_time,format="ISO8601", utc=True); vmap={(str(r.token_key),_utc(r.event_time)):(str(r.row_fingerprint),_utc(r.last_corrected_at)) for r in vintage.itertuples(index=False)}
    else: vmap={}
    for token,g in observations.groupby("token_key",sort=False):
        token=str(token); g=g.sort_values("snapshot_at").reset_index(drop=True)
        cache=pd.read_sql_query(f"SELECT snapshot_at,source_fingerprint,created_at FROM {SEQUENCE_CACHE_TABLE} WHERE token_key=? ORDER BY snapshot_at",conn,params=(token,))
        dirty=None
        if not cache.empty:
            cache["snapshot_at"]=pd.to_datetime(cache.snapshot_at,format="ISO8601", utc=True); cache_by={_utc(r.snapshot_at):(r.source_fingerprint,_utc(r.created_at)) for r in cache.itertuples(index=False)}
            for t in pd.to_datetime(g.snapshot_at,format="ISO8601", utc=True):
                vt=vmap.get((token,_utc(t))); cr=cache_by.get(_utc(t))
                if cr is None: continue
                if vt is not None and (str(cr[0] or "")!=str(vt[0]) or vt[1]>cr[1]): dirty=_utc(t); break
        if dirty is not None:
            conn.execute(f"DELETE FROM {SEQUENCE_CACHE_TABLE} WHERE token_key=? AND snapshot_at>=?",(token,dirty.isoformat())); deleted+=int(conn.execute("SELECT changes()").fetchone()[0]); conn.commit()
        lastrow=conn.execute(f"SELECT MAX(snapshot_at) FROM {SEQUENCE_CACHE_TABLE} WHERE token_key=?",(token,)).fetchone(); last=_utc(lastrow[0]) if lastrow and lastrow[0] else None
        new_idx=[i for i,t in enumerate(g.snapshot_at) if last is None or _utc(t)>last]
        if not new_idx: continue
        tokens+=1; times=pd.to_datetime(g.snapshot_at,format="ISO8601", utc=True).astype("int64").to_numpy(); arrays={c:_obs_numeric(g,c) for c in bases}
        for i in new_idx:
            rec=_fingerprint_for_index(token,g,times,arrays,i,cfg); payload={k:(float(v) if _finite(v) is not None else None) for k,v in rec.items() if k not in {"token_key","snapshot_at"}}
            src=vmap.get((token,_utc(rec["snapshot_at"])),(None,None))[0]
            conn.execute(f"INSERT OR REPLACE INTO {SEQUENCE_CACHE_TABLE}(token_key,snapshot_at,fingerprint_json,config_hash,created_at,source_fingerprint) VALUES(?,?,?,?,?,?)",(token,_iso(rec["snapshot_at"]),_json(payload),conf,now,src)); inserted+=1
    conn.commit(); return {"changed_tokens":tokens,"inserted":inserted,"invalidated_rows":deleted}


@dataclass(frozen=True)
class SequenceFingerprintCache:
    """Connection-local, lazy view of the wide sequence-fingerprint cache.

    The durable JSON rows are intentionally not materialized together.  A mature
    cache can contain hundreds of thousands of rows and more than one thousand
    numeric fields per row; expanding every JSON document before pandas allocates
    the numeric matrix temporarily requires several copies of the dataset.
    """

    conn: sqlite3.Connection
    row_count: int
    feature_columns: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.row_count)


SequenceFingerprintSource = pd.DataFrame | SequenceFingerprintCache


def _sequence_feature_columns(observations: pd.DataFrame, cfg: V24Config) -> tuple[str, ...]:
    bases = [c for c in _SEQUENCE_BASES if c in observations.columns]
    scalar = (
        "has_window", "coverage", "points", "change", "log_vol",
        "positive_ratio", "path_efficiency", "fraction_of_high", "fraction_of_low",
    )
    names: list[str] = []
    for base in bases:
        for window in cfg.sequence_windows_minutes:
            names.extend(f"seqraw__{base}__{int(window)}m__{name}" for name in scalar)
            for segment in range(int(cfg.sequence_segments)):
                names.append(f"seqraw__{base}__{int(window)}m__seg{segment}_present")
                names.append(f"seqraw__{base}__{int(window)}m__seg{segment}_return")
    return tuple(names)


def load_sequence_fingerprint_cache(
    conn: sqlite3.Connection,
    observations: pd.DataFrame,
    cfg: V24Config,
) -> SequenceFingerprintCache:
    refresh_sequence_fingerprint_cache(conn,observations,cfg,force=False)
    row_count = int(conn.execute(f"SELECT COUNT(*) FROM {SEQUENCE_CACHE_TABLE}").fetchone()[0])
    return SequenceFingerprintCache(
        conn=conn,
        row_count=row_count,
        feature_columns=_sequence_feature_columns(observations, cfg),
    )


def _sequence_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["token_key", "snapshot_at"])
    keys = frame[["token_key", "snapshot_at"]].copy()
    keys["token_key"] = keys.token_key.astype(str)
    keys["snapshot_at"] = pd.to_datetime(keys.snapshot_at, format="ISO8601", utc=True)
    return keys.drop_duplicates(["token_key", "snapshot_at"], keep="last").reset_index(drop=True)


def _raw_sequence_frame(
    rows: list[tuple[Any, Any, Any]],
    feature_columns: Sequence[str] | None,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["token_key", "snapshot_at"])
    records = [_loads(row[2]) for row in rows]
    expanded = pd.DataFrame.from_records(records, columns=feature_columns)
    if len(expanded.columns):
        expanded = expanded.apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)
    base = pd.DataFrame({
        "token_key": [str(row[0]) for row in rows],
        "snapshot_at": pd.to_datetime([row[1] for row in rows], format="ISO8601", utc=True),
    })
    return pd.concat([base, expanded], axis=1, copy=False)


def _iter_sequence_raw_by_token(
    source: SequenceFingerprintSource,
    keys: pd.DataFrame,
) -> Iterable[pd.DataFrame]:
    """Yield one token at a time, never a full cache of decoded JSON objects."""
    wanted = _sequence_keys(keys)
    if wanted.empty:
        return
    if isinstance(source, pd.DataFrame):
        for token, token_keys in wanted.groupby("token_key", sort=False):
            raw = token_keys.merge(source, on=["token_key", "snapshot_at"], how="left")
            raw["token_key"] = str(token)
            yield raw.sort_values("snapshot_at").reset_index(drop=True)
        return

    table = f"_v24_sequence_keys_{uuid.uuid4().hex}"
    quoted = '"' + table.replace('"', '""') + '"'
    source.conn.execute(
        f"CREATE TEMP TABLE {quoted}(token_key TEXT NOT NULL,snapshot_at TEXT NOT NULL,"
        "PRIMARY KEY(token_key,snapshot_at)) WITHOUT ROWID"
    )
    try:
        source.conn.executemany(
            f"INSERT OR IGNORE INTO {quoted}(token_key,snapshot_at) VALUES(?,?)",
            ((str(r.token_key), _iso(r.snapshot_at)) for r in wanted.itertuples(index=False)),
        )
        cursor = source.conn.execute(
            f"SELECT k.token_key,k.snapshot_at,c.fingerprint_json FROM {quoted} k "
            f"LEFT JOIN {SEQUENCE_CACHE_TABLE} c "
            "ON c.token_key=k.token_key AND c.snapshot_at=k.snapshot_at "
            "ORDER BY k.token_key,k.snapshot_at"
        )
        current_token: str | None = None
        token_rows: list[tuple[Any, Any, Any]] = []
        for row in cursor:
            token = str(row[0])
            if current_token is not None and token != current_token:
                yield _raw_sequence_frame(token_rows, source.feature_columns)
                token_rows = []
            current_token = token
            token_rows.append((row[0], row[1], row[2]))
        if token_rows:
            yield _raw_sequence_frame(token_rows, source.feature_columns)
    finally:
        source.conn.execute(f"DROP TABLE IF EXISTS {quoted}")


def _sequence_raw_for_keys(
    source: SequenceFingerprintSource,
    keys: pd.DataFrame,
) -> pd.DataFrame:
    parts = list(_iter_sequence_raw_by_token(source, keys))
    if not parts:
        return pd.DataFrame(columns=["token_key", "snapshot_at"])
    return pd.concat(parts, ignore_index=True, copy=False)


def _token_balanced_rows(df:pd.DataFrame,max_per_token:int) -> pd.DataFrame:
    if df.empty or "token_key" not in df: return df
    parts=[]
    for _,g in df.groupby(df.token_key.astype(str),sort=False):
        g=g.sort_values("snapshot_at")
        if len(g)>max_per_token:
            idx=np.unique(np.linspace(0,len(g)-1,max_per_token).round().astype(int)); g=g.iloc[idx]
        parts.append(g)
    return pd.concat(parts,ignore_index=True) if parts else df.iloc[0:0].copy()


def _bounded_model_training_rows(frame: pd.DataFrame, cfg: V24Config) -> pd.DataFrame:
    """Deterministically cap correlated minute rows without dropping any token."""
    if frame.empty:
        return frame.copy()
    groups = [(str(token), g.sort_values("snapshot_at")) for token, g in frame.groupby(frame.token_key.astype(str), sort=True)]
    caps = {token: min(len(g), max(1, int(cfg.model_max_rows_per_token))) for token, g in groups}
    budget = max(int(cfg.model_max_training_rows), len(groups))
    if sum(caps.values()) <= budget:
        quotas = caps
    else:
        low, high = 1, max(caps.values())
        while low < high:
            mid = (low + high + 1) // 2
            if sum(min(cap, mid) for cap in caps.values()) <= budget:
                low = mid
            else:
                high = mid - 1
        quotas = {token: min(cap, low) for token, cap in caps.items()}
        remaining = budget - sum(quotas.values())
        for token in sorted(quotas):
            if remaining <= 0:
                break
            if quotas[token] < caps[token]:
                quotas[token] += 1
                remaining -= 1
    pieces = []
    for token, group in groups:
        quota = int(quotas[token])
        if len(group) > quota:
            index = np.unique(np.linspace(0, len(group) - 1, quota).round().astype(int))
            group = group.iloc[index]
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()


def fit_sequence_encoder(train_raw:pd.DataFrame,cfg:V24Config) -> dict[str,Any]:
    cols=sorted(c for c in train_raw.columns if c.startswith("seqraw__"))
    if not cols: return {"columns":[],"components":0,"scaler":None,"pca":None,"impute":{}}
    balanced=_token_balanced_rows(train_raw,cfg.sequence_balance_rows_per_token)
    numeric=balanced[cols].replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce")
    impute={c:float(numeric[c].median()) if numeric[c].notna().any() else 0.0 for c in cols}
    X=numeric.fillna(impute).to_numpy(dtype=float); scaler=StandardScaler(); Z=scaler.fit_transform(X)
    ncomp=max(1,min(cfg.sequence_components,Z.shape[1],max(1,Z.shape[0]-1))); pca=PCA(n_components=ncomp,random_state=83); pca.fit(Z)
    return {"columns":cols,"components":ncomp,"scaler":scaler,"pca":pca,"impute":impute,"training_rows":len(balanced),"training_tokens":int(balanced.token_key.nunique())}


def apply_sequence_encoder(raw:pd.DataFrame,encoder:dict[str,Any]) -> pd.DataFrame:
    out=raw[["token_key","snapshot_at"]].copy(); cols=encoder.get("columns",[])
    if not cols: return out
    numeric=raw.reindex(columns=cols).replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors="coerce")
    X=numeric.fillna(encoder.get("impute",{})).fillna(0.0).to_numpy(dtype=float); Z=encoder["scaler"].transform(X); emb=encoder["pca"].transform(Z)
    for j in range(emb.shape[1]): out[f"seqenc__{j:02d}"]=emb[:,j]
    return out


def _fit_sequence_encoder_from_source(
    source: SequenceFingerprintSource,
    frame: pd.DataFrame,
    cfg: V24Config,
) -> dict[str, Any]:
    keys = _token_balanced_rows(
        _sequence_keys(frame),
        max(1, int(cfg.sequence_balance_rows_per_token)),
    )
    return fit_sequence_encoder(_sequence_raw_for_keys(source, keys), cfg)



if nn is not None:
    class _CausalConvEncoder(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            hidden=max(32,out_dim)
            self.layers=nn.ModuleList([
                nn.Conv1d(in_dim,hidden,3,dilation=1),
                nn.Conv1d(hidden,hidden,3,dilation=2),
                nn.Conv1d(hidden,out_dim,3,dilation=4),
            ])
        @staticmethod
        def _causal(conv: nn.Conv1d, x: Any) -> Any:
            pad=(conv.kernel_size[0]-1)*conv.dilation[0]
            x=F.pad(x,(pad,0)); return conv(x)
        def forward(self,x:Any)->Any:
            # x: batch,time,features
            y=x.transpose(1,2)
            for j,conv in enumerate(self.layers):
                y=self._causal(conv,y)
                if j<len(self.layers)-1: y=F.gelu(y)
            return y.transpose(1,2)


def _torch_state_to_numpy(model:Any)->dict[str,np.ndarray]:
    return {k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}


def _torch_state_from_numpy(model:Any,state:dict[str,Any])->None:
    model.load_state_dict({k:torch.as_tensor(v) for k,v in state.items()})


def _hierarchical_contrastive_loss(z1:Any,z2:Any,temperature:float=.2)->Any:
    losses=[]
    a,b=z1,z2
    for _ in range(3):
        n=a.shape[1]
        if n<2: break
        aa=F.normalize(a.reshape(-1,a.shape[-1]),dim=-1); bb=F.normalize(b.reshape(-1,b.shape[-1]),dim=-1)
        logits=aa@bb.T/temperature; labels=torch.arange(logits.shape[0],device=logits.device)
        losses.append((F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2)
        if n<4: break
        a=F.avg_pool1d(a.transpose(1,2),2,2).transpose(1,2); b=F.avg_pool1d(b.transpose(1,2),2,2).transpose(1,2)
    return torch.stack(losses).mean() if losses else ((z1-z2)**2).mean()


def fit_ts2vec_style_encoder(train_raw:pd.DataFrame,cfg:V24Config,*,allow_small:bool=False)->dict[str,Any]|None:
    """Token-balanced causal contrastive sequence challenger inspired by TS2Vec.

    This is intentionally labelled ``ts2vec_style`` rather than claiming a byte-for-byte
    reproduction of the research implementation.  It keeps the central ideas needed
    here: timestamp representations, causal temporal convolutions, two augmented views,
    hierarchical contrastive loss, and token-balanced training.
    """
    if torch is None or nn is None or train_raw.empty: return None
    cols=sorted(c for c in train_raw.columns if c.startswith('seqraw__'))
    if not cols or train_raw.token_key.astype(str).nunique() < (6 if allow_small else cfg.sequence_challenger_min_tokens): return None
    balanced=_token_balanced_rows(train_raw,max(cfg.sequence_balance_rows_per_token*4,48))
    num=balanced[cols].replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors='coerce')
    impute={c:float(num[c].median()) if num[c].notna().any() else 0.0 for c in cols}; X=num.fillna(impute).to_numpy(dtype=np.float32)
    scaler=StandardScaler(); Z=scaler.fit_transform(X).astype(np.float32)
    pre_dim=min(48,Z.shape[1],max(2,Z.shape[0]-1)); pre=PCA(n_components=pre_dim,random_state=271); P=pre.fit_transform(Z).astype(np.float32)
    work=balanced[['token_key','snapshot_at']].copy();
    for j in range(P.shape[1]): work[f'__p{j}']=P[:,j]
    pcols=[c for c in work.columns if c.startswith('__p')]
    sequences=[]
    for _,g in work.groupby(work.token_key.astype(str),sort=False):
        arr=g.sort_values('snapshot_at')[pcols].to_numpy(dtype=np.float32)
        if len(arr)>=8: sequences.append(arr)
    if len(sequences)<(4 if allow_small else 12): return None
    torch.manual_seed(277); np.random.seed(277)
    out_dim=max(8,int(cfg.sequence_challenger_dim)); model=_CausalConvEncoder(pre_dim,out_dim); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    epochs=1 if allow_small else max(1,int(cfg.sequence_challenger_epochs)); win=32 if allow_small else 64
    for _ in range(epochs):
        order=np.random.permutation(len(sequences))
        for q in order:
            seq=sequences[int(q)]
            if len(seq)>win:
                st=np.random.randint(0,len(seq)-win+1); seq=seq[st:st+win]
            x=torch.as_tensor(seq[None,:,:],dtype=torch.float32)
            def aug(v:Any)->Any:
                noise=torch.randn_like(v)*0.02; mask=(torch.rand(v.shape[:2]+(1,))>0.12).to(v.dtype); return (v+noise)*mask
            z1=model(aug(x)); z2=model(aug(x)); loss=_hierarchical_contrastive_loss(z1,z2)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
    return {'kind':'ts2vec_style_causal','columns':cols,'impute':impute,'scaler':scaler,'pre_pca':pre,'in_dim':pre_dim,'out_dim':out_dim,'state':_torch_state_to_numpy(model),'training_tokens':len(sequences),'training_rows':len(balanced)}


def apply_ts2vec_style_encoder(
    raw: pd.DataFrame,
    encoder: dict[str, Any] | None,
    *,
    model: Any | None = None,
) -> pd.DataFrame:
    out=raw[['token_key','snapshot_at']].copy()
    if not encoder or torch is None: return out
    cols=encoder['columns']; num=raw.reindex(columns=cols).replace([np.inf,-np.inf],np.nan).apply(pd.to_numeric,errors='coerce')
    X=num.fillna(encoder.get('impute',{})).fillna(0.0).to_numpy(dtype=np.float32); Z=encoder['scaler'].transform(X).astype(np.float32); P=encoder['pre_pca'].transform(Z).astype(np.float32)
    if model is None:
        model=_CausalConvEncoder(int(encoder['in_dim']),int(encoder['out_dim']))
        _torch_state_from_numpy(model,encoder['state'])
        model.eval()
    emb=np.full((len(raw),int(encoder['out_dim'])),np.nan,dtype=np.float32)
    work=raw[['token_key','snapshot_at']].copy(); work['__idx']=np.arange(len(raw))
    with torch.no_grad():
        for _,g in work.groupby(work.token_key.astype(str),sort=False):
            g=g.sort_values('snapshot_at'); ids=g['__idx'].to_numpy(dtype=int); x=torch.as_tensor(P[ids][None,:,:],dtype=torch.float32); z=model(x).squeeze(0).cpu().numpy(); emb[ids]=z
    for j in range(emb.shape[1]): out[f'ts2enc__{j:02d}']=emb[:,j]
    return out


def _fit_ts2vec_from_source(
    source: SequenceFingerprintSource,
    frame: pd.DataFrame,
    cfg: V24Config,
    *,
    allow_small: bool,
) -> dict[str, Any] | None:
    keys = _token_balanced_rows(
        _sequence_keys(frame),
        max(int(cfg.sequence_balance_rows_per_token) * 4, 48),
    )
    return fit_ts2vec_style_encoder(
        _sequence_raw_for_keys(source, keys), cfg, allow_small=allow_small
    )


def _encode_sequence_keys(
    source: SequenceFingerprintSource,
    keys: pd.DataFrame,
    encoder: dict[str, Any],
    sequence_challenger: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Decode and transform one token at a time, retaining only compact embeddings."""
    # In-memory sources are already bounded (live inference contains one row per
    # currently visible token). Transforming those rows together avoids hundreds
    # of tiny DataFrame.apply/to_numeric calls. The challenger remains causal:
    # apply_ts2vec_style_encoder still groups and orders every token separately.
    if isinstance(source, pd.DataFrame):
        raw = _sequence_raw_for_keys(source, keys)
        if raw.empty:
            return pd.DataFrame(columns=["token_key", "snapshot_at"])
        encoded = apply_sequence_encoder(raw, encoder)
        if sequence_challenger:
            challenger = apply_ts2vec_style_encoder(raw, sequence_challenger)
            extra = [c for c in challenger.columns if c.startswith("ts2enc__")]
            if extra:
                encoded = pd.concat(
                    [encoded.reset_index(drop=True), challenger[extra].reset_index(drop=True)],
                    axis=1,
                    copy=False,
                )
        return encoded

    parts: list[pd.DataFrame] = []
    challenger_model = None
    if sequence_challenger and torch is not None:
        challenger_model = _CausalConvEncoder(
            int(sequence_challenger['in_dim']), int(sequence_challenger['out_dim'])
        )
        _torch_state_from_numpy(challenger_model, sequence_challenger['state'])
        challenger_model.eval()
    for raw in _iter_sequence_raw_by_token(source, keys):
        encoded = apply_sequence_encoder(raw, encoder)
        if sequence_challenger:
            challenger = apply_ts2vec_style_encoder(
                raw, sequence_challenger, model=challenger_model
            )
            extra = [c for c in challenger.columns if c.startswith("ts2enc__")]
            if extra:
                encoded = pd.concat(
                    [encoded.reset_index(drop=True), challenger[extra].reset_index(drop=True)],
                    axis=1,
                    copy=False,
                )
        parts.append(encoded)
    if not parts:
        return pd.DataFrame(columns=["token_key", "snapshot_at"])
    return pd.concat(parts, ignore_index=True, copy=False)

# ---------------------------------------------------------------------------
# Leakage-safe training frame and CPCV
# ---------------------------------------------------------------------------

def load_v24_frame(
    conn: sqlite3.Connection,
    cfg: V24Config,
) -> tuple[pd.DataFrame, SequenceFingerprintSource, dict[str, Any]]:
    migrate(conn)
    obs, obs_source = peak.load_observations(conn)
    refresh_capture_heartbeats(conn, obs)
    refresh_data_vintage(conn, obs)
    refresh_lifetimes(conn, cfg, observations=obs)
    refresh_calendar_cohorts(conn, cfg)
    refresh_policy_cohorts(conn, cfg)
    refresh_token_assignments(conn, cfg)
    frame, feature_source, obs_source = peak.load_training_frame(
        conn,
        observations=obs,
        observation_source=obs_source,
    )
    frame = _attach_calendar_and_lifetime(conn, frame)
    vint=pd.read_sql_query(f"SELECT token_key,event_time,first_ingested_at,last_corrected_at,value_version,ingestion_provenance FROM {DATA_VINTAGE_TABLE}",conn)
    if not vint.empty:
        vint["event_time"]=pd.to_datetime(vint.event_time,format="ISO8601", utc=True)
        frame=frame.merge(vint,left_on=["token_key","snapshot_at"],right_on=["token_key","event_time"],how="left").drop(columns=["event_time"],errors="ignore")
    # label interval is what must be purged against validation intervals.
    frame["label_interval_start"] = pd.to_datetime(frame.decision_at, format="ISO8601", utc=True)
    end = pd.to_datetime(frame.path_end_at, format="ISO8601", utc=True, errors="coerce")
    fallback = frame.label_interval_start + pd.Timedelta(minutes=cfg.horizon_minutes)
    frame["label_interval_end"] = end.fillna(fallback)
    seqraw = load_sequence_fingerprint_cache(conn, obs, cfg)
    return frame, seqraw, {"feature_source": feature_source, "observation_source": obs_source}


def _audit_ordinals(conn: sqlite3.Connection) -> set[int]:
    return {
        int(r[0]) for r in conn.execute(
            f"SELECT ordinal FROM {COHORT_TABLE} WHERE role='audit'"
        ).fetchall()
    }


def _eligible_train_ordinals(conn:sqlite3.Connection,cutoff:pd.Timestamp)->set[int]:
    rows=conn.execute(f"SELECT ordinal,role,status,end_at,consumed_at FROM {COHORT_TABLE} WHERE end_at<? ORDER BY ordinal",(_utc(cutoff).isoformat(),)).fetchall(); allowed=set()
    for ordinal,role,status,_end,consumed_at in rows:
        if role=="audit": continue
        if role=="train": allowed.add(int(ordinal))
        elif role=="promotion" and status=="consumed" and consumed_at and _utc(consumed_at)<_utc(cutoff): allowed.add(int(ordinal))
    return allowed


def _eligible_policy_ordinals(conn:sqlite3.Connection,cutoff:pd.Timestamp)->set[int]:
    rows=conn.execute(f"SELECT ordinal,role,status,end_at,consumed_at FROM {POLICY_COHORT_TABLE} WHERE end_at<? ORDER BY ordinal",(_utc(cutoff).isoformat(),)).fetchall(); allowed=set()
    for ordinal,role,status,_end,consumed_at in rows:
        if role=="audit": continue
        if role=="train": allowed.add(int(ordinal))
        elif role=="promotion" and status=="consumed" and consumed_at and _utc(consumed_at)<_utc(cutoff): allowed.add(int(ordinal))
    return allowed


def _vintage_known_by(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """A historical row is replayable only if its current value version existed then."""
    if "first_ingested_at" not in frame.columns or "last_corrected_at" not in frame.columns:
        return pd.Series(False,index=frame.index)
    first=pd.to_datetime(frame.first_ingested_at,format="ISO8601", utc=True,errors="coerce")
    corrected=pd.to_datetime(frame.last_corrected_at,format="ISO8601", utc=True,errors="coerce")
    return first.notna() & corrected.notna() & (first<=cutoff) & (corrected<=cutoff)


def _training_history_selection(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: V24Config,
    exclude_tokens: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cutoff = _utc(cutoff)
    allowed = _eligible_train_ordinals(conn, cutoff)
    embargo_cutoff = cutoff - pd.Timedelta(hours=cfg.promotion_embargo_hours)
    allowed_mask = frame.calendar_cohort_ordinal.isin(allowed)
    embargo_mask = frame.snapshot_at < embargo_cutoff
    label_mask = frame.label_interval_end <= cutoff
    vintage_mask = _vintage_known_by(frame, cutoff)
    before_exclusion_mask = allowed_mask & embargo_mask & label_mask & vintage_mask
    before_exclusion = int(before_exclusion_mask.sum())
    excluded = pd.Series(False, index=frame.index)
    if exclude_tokens:
        excluded = frame.token_key.astype(str).isin(set(map(str, exclude_tokens)))
    cumulative = before_exclusion_mask & ~excluded
    provenance = (
        frame.loc[allowed_mask, "ingestion_provenance"].fillna("missing").astype(str).value_counts().to_dict()
        if "ingestion_provenance" in frame.columns else {"missing": int(allowed_mask.sum())}
    )
    diagnostics: dict[str, Any] = {
        "cutoff": cutoff.isoformat(),
        "embargo_cutoff": embargo_cutoff.isoformat(),
        "total_frame_rows": int(len(frame)),
        "allowed_train_ordinals": sorted(int(v) for v in allowed),
        "allowed_cohort_rows": int(allowed_mask.sum()),
        "before_embargo_rows": int((allowed_mask & embargo_mask).sum()),
        "mature_label_rows": int((allowed_mask & embargo_mask & label_mask).sum()),
        "vintage_known_rows": before_exclusion,
        "excluded_evaluation_rows": int((before_exclusion_mask & excluded).sum()),
        "eligible_training_rows": int(cumulative.sum()),
        "allowed_cohort_vintage_provenance": {str(k): int(v) for k, v in provenance.items()},
    }
    return frame[cumulative].copy(), diagnostics


def training_history_diagnostics(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: V24Config,
    exclude_tokens: set[str] | None = None,
) -> dict[str, Any]:
    """Explain every cumulative gate used by stable batch training."""
    _, diagnostics = _training_history_selection(
        conn, frame, cutoff, cfg, exclude_tokens=exclude_tokens
    )
    return diagnostics


def training_history_before(conn: sqlite3.Connection, frame: pd.DataFrame, cutoff: pd.Timestamp, cfg: V24Config, exclude_tokens: set[str] | None = None) -> pd.DataFrame:
    """Fully mature history for stable batch training.

    Unlike V23 this does not subtract another 72h from the decision time: requiring
    the actual label interval to end before the cutoff already performs the causal
    purge.  A small embargo remains around the external evaluation boundary.
    """
    out, _ = _training_history_selection(
        conn, frame, cutoff, cfg, exclude_tokens=exclude_tokens
    )
    return out


def adapter_history_before(conn: sqlite3.Connection, frame: pd.DataFrame, cutoff: pd.Timestamp, stable_cutoff: pd.Timestamp, cfg: V24Config, exclude_tokens: set[str] | None=None) -> pd.DataFrame:
    """Recent adapter history including legitimate right-censored observations.

    This is the High-#17 fix: a six-hour-old decision may contribute six hours of
    event-free survival information at cutoff T instead of waiting three more days.
    Only information available at cutoff is constructed downstream.
    """
    cutoff=_utc(cutoff); allowed=_eligible_train_ordinals(conn,cutoff)
    recent_start=max(_utc(stable_cutoff),cutoff-pd.Timedelta(days=cfg.adapter_recent_days))
    min_age=pd.Timedelta(minutes=cfg.adapter_min_censor_minutes)
    mask=(
        frame.calendar_cohort_ordinal.isin(allowed)
        & (frame.snapshot_at>recent_start)
        & (frame.snapshot_at<=cutoff-min_age)
        & _vintage_known_by(frame,cutoff)
    )
    out=frame[mask].copy()
    if exclude_tokens: out=out[~out.token_key.astype(str).isin(set(map(str,exclude_tokens)))].copy()
    return out


def _calendar_blocks_from_frame(frame: pd.DataFrame, max_blocks: int) -> list[int]:
    blocks = sorted(int(x) for x in frame.calendar_cohort_ordinal.dropna().unique() if int(x) >= 0)
    return blocks[-max_blocks:] if max_blocks > 0 else blocks


def purged_cpcv_splits(frame: pd.DataFrame, cfg: V24Config) -> list[dict[str, Any]]:
    """Token-group + calendar-group combinatorial purged CV.

    No token in a test fold may occur anywhere in that fold's training set. In
    addition, a training label interval may not overlap any test calendar interval
    expanded by the configured purge/embargo windows.
    """
    blocks = _calendar_blocks_from_frame(frame, cfg.cpcv_blocks)
    if len(blocks) < 3:
        return []
    k = min(cfg.cpcv_test_blocks, max(1, len(blocks) - 2))
    combos = list(itertools.combinations(blocks, k))[: cfg.cpcv_max_splits]
    result: list[dict[str, Any]] = []
    for fold_no, test_blocks in enumerate(combos):
        test_mask = frame.calendar_cohort_ordinal.isin(test_blocks)
        test = frame[test_mask]
        if test.empty:
            continue
        test_tokens = set(test.token_key.astype(str))
        train_mask = ~frame.token_key.astype(str).isin(test_tokens)
        train_mask &= ~test_mask
        # Purge overlaps independently around each selected calendar block.
        for b in test_blocks:
            b_rows = test[test.calendar_cohort_ordinal == b]
            if b_rows.empty:
                continue
            start = b_rows.label_interval_start.min() - pd.Timedelta(hours=cfg.promotion_purge_hours)
            end = b_rows.label_interval_end.max() + pd.Timedelta(hours=cfg.promotion_embargo_hours)
            overlap = (frame.label_interval_start <= end) & (frame.label_interval_end >= start)
            train_mask &= ~overlap
        tr = np.flatnonzero(train_mask.to_numpy())
        te = np.flatnonzero(test_mask.to_numpy())
        if len(tr) and len(te):
            result.append({
                "fold_id": f"cpcv_{fold_no:03d}", "test_blocks": tuple(test_blocks),
                "train_idx": tr, "test_idx": te,
                "train_tokens": int(frame.iloc[tr].token_key.nunique()),
                "test_tokens": int(frame.iloc[te].token_key.nunique()),
            })
    return result


def validate_no_overlap(frame: pd.DataFrame, split: dict[str, Any]) -> None:
    tr = frame.iloc[split["train_idx"]]
    te = frame.iloc[split["test_idx"]]
    shared = set(tr.token_key.astype(str)) & set(te.token_key.astype(str))
    if shared:
        raise AssertionError(f"Token leakage in CPCV fold: {sorted(shared)[:5]}")
    for t in te.itertuples(index=False):
        overlap = tr[(tr.label_interval_start <= t.label_interval_end) & (tr.label_interval_end >= t.label_interval_start)]
        # Calendar purging can intentionally keep far-away intervals while the test
        # token itself is fully excluded; any direct interval overlap is forbidden.
        if not overlap.empty:
            raise AssertionError("Label interval leakage in CPCV fold")


# ---------------------------------------------------------------------------
# Competing-risk survival + recurrent event targets
# ---------------------------------------------------------------------------

def _death_event(row: pd.Series) -> pd.Timestamp | None:
    reason = str(row.get("terminal_reason") or "").lower()
    if any(x in reason for x in ("death", "dead", "disappear", "operational_gap")):
        val = row.get("terminal_at")
        if val not in (None, "") and pd.notna(val):
            return _utc(val)
    return None


def _event_sources(conn: sqlite3.Connection) -> tuple[dict[str,list[tuple[pd.Timestamp,pd.Timestamp,float]]],dict[str,list[tuple[pd.Timestamp,str]]]]:
    peaks_by: dict[str,list[tuple[pd.Timestamp,pd.Timestamp,float]]]={}
    if _table_exists(conn,peak.PEAK_EVENT_TABLE):
        e=pd.read_sql_query(f"SELECT token_key,peak_at,confirmed_at,peak_price FROM {peak.PEAK_EVENT_TABLE} ORDER BY token_key,peak_at",conn)
        if not e.empty:
            e["peak_at"]=pd.to_datetime(e.peak_at,format="ISO8601", utc=True); e["confirmed_at"]=pd.to_datetime(e.confirmed_at,format="ISO8601", utc=True)
            for k,g in e.groupby("token_key"):
                peaks_by[str(k)]=[(r.peak_at,r.confirmed_at,float(r.peak_price)) for r in g.itertuples(index=False)]
    life_by: dict[str,list[tuple[pd.Timestamp,str]]]={}
    if _table_exists(conn,LIFETIME_TABLE):
        l=pd.read_sql_query(f"SELECT token_key,terminal_at,terminal_reason FROM {LIFETIME_TABLE} WHERE terminal_at IS NOT NULL",conn)
        if not l.empty:
            l["terminal_at"]=pd.to_datetime(l.terminal_at,format="ISO8601", utc=True)
            for k,g in l.groupby("token_key"):
                life_by[str(k)]=[(r.terminal_at,str(r.terminal_reason or "")) for r in g.itertuples(index=False)]
    return peaks_by,life_by


def first_competing_event_asof(
    row: pd.Series, cfg: V24Config, as_of: pd.Timestamp,
    peaks_by: dict[str,list[tuple[pd.Timestamp,pd.Timestamp,float]]],
    life_by: dict[str,list[tuple[pd.Timestamp,str]]],
) -> tuple[int,float,float]:
    """Return event class, event/censor minutes, and observed follow-up minutes as known at cutoff.

    Confirmed substantial peaks are timed at confirmation for survival purposes.
    ``peak_at`` remains a separate mark used by the recurrent-event model.
    """
    decision=_utc(row.decision_at); as_of=_utc(as_of); horizon_end=decision+pd.Timedelta(minutes=cfg.horizon_minutes)
    observed_end=min(as_of,horizon_end)
    events: list[tuple[pd.Timestamp,int]]=[]
    for peak_at,confirmed_at,_price in peaks_by.get(str(row.token_key),[]):
        if decision < peak_at <= horizon_end and decision < confirmed_at <= horizon_end and confirmed_at <= as_of:
            events.append((confirmed_at,EVENT_PEAK))
    for terminal_at,reason in life_by.get(str(row.token_key),[]):
        if not (decision < terminal_at <= horizon_end and terminal_at <= as_of): continue
        if any(x in reason.lower() for x in ("dead","operational_gap","disappear","valid_capture_absence")):
            events.append((terminal_at,EVENT_DEATH))
        elif any(x in reason.lower() for x in ("age_out","natural")):
            observed_end=min(observed_end,terminal_at)
    if events:
        when,kind=min(events,key=lambda x:x[0])
        if when<=observed_end:
            mins=max(0.0,(when-decision).total_seconds()/60.0)
            return kind,mins,mins
    mins=max(0.0,(observed_end-decision).total_seconds()/60.0)
    return EVENT_NONE,mins,mins


def first_competing_event(row: pd.Series, cfg: V24Config) -> tuple[int,float,bool]:
    """Compatibility path for fully materialized labels."""
    decision=_utc(row.decision_at); horizon_end=decision+pd.Timedelta(minutes=cfg.horizon_minutes)
    candidates=[]
    p=row.get("next_substantial_peak_confirmed_at")
    pa=row.get("next_substantial_peak_at")
    if p not in (None,"") and pd.notna(p) and pa not in (None,"") and pd.notna(pa):
        pt=_utc(p); pat=_utc(pa)
        if decision<pat<=horizon_end and decision<pt<=horizon_end: candidates.append((pt,EVENT_PEAK))
    d=_death_event(row)
    if d is not None and decision<d<=horizon_end: candidates.append((d,EVENT_DEATH))
    if candidates:
        when,kind=min(candidates,key=lambda x:x[0]); return kind,max(0.0,(when-decision).total_seconds()/60.0),True
    path_end=row.get("path_end_at"); end=_utc(path_end) if path_end not in (None,"") and pd.notna(path_end) else horizon_end
    end=min(end,horizon_end); finalized=bool(int(row.get("label_finalized",0) or 0)) or end>=horizon_end
    return EVENT_NONE,max(0.0,(end-decision).total_seconds()/60.0),finalized


def build_survival_person_period(
    frame: pd.DataFrame, cfg: V24Config, *, conn: sqlite3.Connection | None=None, as_of: pd.Timestamp | None=None,
) -> pd.DataFrame:
    rows: list[dict[str,Any]]=[]; bins=list(cfg.survival_bins_minutes)
    peaks_by=life_by=None
    if conn is not None and as_of is not None:
        peaks_by,life_by=_event_sources(conn)
    for idx,r in frame.iterrows():
        if peaks_by is not None and life_by is not None:
            event,event_min,_=first_competing_event_asof(r,cfg,_utc(as_of),peaks_by,life_by); known_end=True
        else:
            event,event_min,known_end=first_competing_event(r,cfg)
        prev=0.0
        for bidx,bend in enumerate(bins):
            if event_min<=prev and event==EVENT_NONE: break
            if event!=EVENT_NONE and event_min<=prev: break
            cls=EVENT_NONE
            if event!=EVENT_NONE and prev<event_min<=bend:
                cls=event
            elif bend>event_min:
                # Right-censored inside this bin: do not invent a completed no-event interval.
                break
            rows.append({
                "source_index":idx,"token_key":str(r.token_key),"decision_at":r.decision_at,
                "bin_index":bidx,"bin_start_minutes":prev,"bin_end_minutes":float(bend),"event_class":cls,
                "censored_at_minutes":float(event_min) if cls==EVENT_NONE else None,
            })
            if cls!=EVENT_NONE: break
            prev=float(bend)
    return pd.DataFrame(rows)


def _future_peak_lists(conn: sqlite3.Connection) -> dict[str,list[dict[str,Any]]]:
    if not _table_exists(conn,peak.PEAK_EVENT_TABLE): return {}
    df=pd.read_sql_query(
        f"SELECT token_key,peak_at,peak_price,confirmed_at,confirmation_price,runup_pct,confirmation_retrace_pct FROM {peak.PEAK_EVENT_TABLE} ORDER BY token_key,peak_at",conn)
    if df.empty: return {}
    df["peak_at"]=pd.to_datetime(df.peak_at,format="ISO8601", utc=True); df["confirmed_at"]=pd.to_datetime(df.confirmed_at,format="ISO8601", utc=True)
    out={}
    for k,g in df.groupby("token_key"):
        ev=[]; prev=None
        for ordinal,r in enumerate(g.itertuples(index=False),start=1):
            price=float(r.peak_price); rel=(price/prev-1.0) if prev and prev>0 else None
            ev.append({"peak_at":r.peak_at,"confirmed_at":r.confirmed_at,"peak_price":price,"ordinal":ordinal,"relative_to_previous":rel,
                       "confirmation_price":float(r.confirmation_price),"runup_pct":float(r.runup_pct),"confirmation_retrace_pct":float(r.confirmation_retrace_pct)})
            prev=price
        out[str(k)]=ev
    return out


def _known_followup_end(row: pd.Series,cfg: V24Config,as_of: pd.Timestamp | None) -> pd.Timestamp:
    decision=_utc(row.decision_at); horizon=decision+pd.Timedelta(minutes=cfg.horizon_minutes)
    end=_utc(as_of) if as_of is not None else (_utc(row.path_end_at) if pd.notna(row.get("path_end_at")) else horizon)
    return min(end,horizon)


def add_recurrent_targets(conn: sqlite3.Connection,frame: pd.DataFrame,cfg: V24Config,*,as_of: pd.Timestamp | None=None) -> pd.DataFrame:
    """Marked recurrent-event targets with strict confirmation-time boundaries."""
    peaks=_future_peak_lists(conn); out=frame.copy()
    horizons=(240,720,1440,4320)
    for h in horizons:
        out[f"recurrent_peak_count_{h}m"]=np.nan
        out[f"second_peak_by_{h}m"]=np.nan
        for margin in cfg.higher_peak_mark_margins:
            out[f"later_higher_{int(round(margin*100))}pct_by_{h}m"]=np.nan
    for c in ("prior_confirmed_peak_count","minutes_since_last_confirmed_peak","last_confirmed_peak_multiple_vs_decision",
              "recurrent_next_gap_minutes","recurrent_next_occurrence_gap_minutes","recurrent_second_gap_minutes",
              "recurrent_next_peak_multiple","recurrent_second_peak_relative_to_first","recurrent_second_peak_is_higher"):
        out[c]=np.nan
    for i,r in out.iterrows():
        decision=_utc(r.decision_at); known_end=_known_followup_end(r,cfg,as_of)
        events=peaks.get(str(r.token_key),[])
        prior=[e for e in events if e["confirmed_at"]<=decision]
        if prior:
            last=prior[-1]; out.at[i,"prior_confirmed_peak_count"]=float(len(prior)); out.at[i,"minutes_since_last_confirmed_peak"]=(decision-last["confirmed_at"]).total_seconds()/60.0
            entry=float(r.decision_market_cap_usd) if _finite(r.get("decision_market_cap_usd")) else np.nan
            if np.isfinite(entry) and entry>0: out.at[i,"last_confirmed_peak_multiple_vs_decision"]=last["peak_price"]/entry
        else:
            out.at[i,"prior_confirmed_peak_count"]=0.0
        eligible=[e for e in events if decision<e["peak_at"]<=decision+pd.Timedelta(minutes=cfg.horizon_minutes) and decision<e["confirmed_at"]<=known_end]
        eligible.sort(key=lambda e:e["peak_at"])
        if eligible:
            first=eligible[0]; out.at[i,"recurrent_next_occurrence_gap_minutes"]=(first["peak_at"]-decision).total_seconds()/60.0; out.at[i,"recurrent_next_gap_minutes"]=(first["confirmed_at"]-decision).total_seconds()/60.0
            entry=float(r.decision_market_cap_usd) if _finite(r.get("decision_market_cap_usd")) else np.nan
            if np.isfinite(entry) and entry>0: out.at[i,"recurrent_next_peak_multiple"]=first["peak_price"]/entry
        if len(eligible)>=2:
            first,second=eligible[0],eligible[1]; out.at[i,"recurrent_second_gap_minutes"]=(second["confirmed_at"]-first["confirmed_at"]).total_seconds()/60.0
            out.at[i,"recurrent_second_peak_relative_to_first"]=second["peak_price"]/first["peak_price"]-1.0
            out.at[i,"recurrent_second_peak_is_higher"]=float(second["peak_price"]>first["peak_price"]*(1+cfg.higher_peak_margin_pct))
        for h in horizons:
            deadline=decision+pd.Timedelta(minutes=h); cutoff=min(deadline,known_end)
            inside=[e for e in events if decision<e["peak_at"]<=deadline and decision<e["confirmed_at"]<=deadline and e["confirmed_at"]<=known_end]
            # Negative/count outcomes are known only after the entire horizon (or terminal label finalization).
            complete=known_end>=deadline or (bool(int(r.get("label_finalized",0) or 0)) and as_of is None)
            if complete:
                out.at[i,f"recurrent_peak_count_{h}m"]=float(len(inside)); out.at[i,f"second_peak_by_{h}m"]=float(len(inside)>=2)
            elif len(inside)>=2:
                out.at[i,f"second_peak_by_{h}m"]=1.0
            if inside:
                first=inside[0]
                for margin in cfg.higher_peak_mark_margins:
                    col=f"later_higher_{int(round(margin*100))}pct_by_{h}m"
                    hit=any(e["peak_price"]>first["peak_price"]*(1.0+margin) for e in inside[1:])
                    if hit: out.at[i,col]=1.0
                    elif complete: out.at[i,col]=0.0
            elif complete:
                for margin in cfg.higher_peak_mark_margins:
                    out.at[i,f"later_higher_{int(round(margin*100))}pct_by_{h}m"]=0.0
    return out


def _future_window_max(times_ns: np.ndarray,values: np.ndarray,horizon_minutes:int,*,as_of_ns:int | None=None) -> np.ndarray:
    """O(n) future-window maxima, optionally censored at an historical as-of cutoff."""
    n=len(values); out=np.full(n,np.nan); q=deque(); r=1; horizon_ns=int(horizon_minutes*60*1e9)
    for i in range(n):
        if r<i+1: r=i+1
        while q and q[0]<=i: q.popleft()
        end=times_ns[i]+horizon_ns
        if as_of_ns is not None: end=min(end,int(as_of_ns))
        while r<n and times_ns[r]<=end:
            if np.isfinite(values[r]):
                while q and (not np.isfinite(values[q[-1]]) or values[q[-1]]<=values[r]): q.pop()
                q.append(r)
            r+=1
        while q and q[0]<=i: q.popleft()
        if q: out[i]=values[q[0]]
    return out


def _threshold_tag(x: float) -> str:
    return str(int(round(x*100)))


def add_barrier_targets(conn: sqlite3.Connection,frame: pd.DataFrame,cfg: V24Config,*,as_of: pd.Timestamp | None=None) -> pd.DataFrame:
    """Future upside barriers using only observations available at ``as_of``."""
    out=frame.copy()
    source=peak.discover_observation_source(conn)
    table='"'+source["table"].replace('"','""')+'"'
    selected=[source["token"],source["time"],source["mc"]]
    quoted=", ".join('"'+c.replace('"','""')+'"' for c in selected)
    obs=pd.read_sql_query(f"SELECT {quoted} FROM {table}",conn).rename(columns={
        source["token"]:"token_key",source["time"]:"snapshot_at",source["mc"]:"market_cap_usd",
    })
    obs["snapshot_at"]=pd.to_datetime(obs.snapshot_at,format="ISO8601",utc=True)
    obs["market_cap_usd"]=pd.to_numeric(obs.market_cap_usd,errors="coerce")
    obs=obs.dropna(subset=["token_key","snapshot_at","market_cap_usd"])
    obs=obs[obs.market_cap_usd>0].drop_duplicates(["token_key","snapshot_at"],keep="last")
    asof=_utc(as_of) if as_of is not None else None; asof_ns=int(asof.value) if asof is not None else None
    for thr in cfg.upside_thresholds:
        for h in cfg.probability_horizons_minutes: out[f"hit_plus{_threshold_tag(thr)}_by_{h}m"]=np.nan
    index_map={(str(r.token_key),_utc(r.snapshot_at)):i for i,r in out.iterrows()}
    for token,g in obs.groupby("token_key",sort=False):
        g=g.sort_values("snapshot_at").reset_index(drop=True); times=pd.to_datetime(g.snapshot_at,format="ISO8601", utc=True); times_ns=times.astype("int64").to_numpy(); mc=pd.to_numeric(g.market_cap_usd,errors="coerce").to_numpy(dtype=float)
        for h in cfg.probability_horizons_minutes:
            mx=_future_window_max(times_ns,mc,int(h),as_of_ns=asof_ns)
            for j,t in enumerate(times):
                oi=index_map.get((str(token),_utc(t)))
                if oi is None or not np.isfinite(mc[j]): continue
                decision=_utc(t); deadline=decision+pd.Timedelta(minutes=int(h)); row=out.loc[oi]
                if asof is not None and decision>=asof: continue
                terminal=None
                tv=row.get("terminal_at")
                if tv not in (None,"") and pd.notna(tv):
                    tt=_utc(tv)
                    if decision<tt<=deadline and (asof is None or tt<=asof): terminal=tt
                complete=(asof is None and (bool(int(row.get("label_finalized",0) or 0)) or _utc(row.get("path_end_at"))>=deadline if pd.notna(row.get("path_end_at")) else False)) or (asof is not None and (asof>=deadline or terminal is not None))
                for thr in cfg.upside_thresholds:
                    col=f"hit_plus{_threshold_tag(thr)}_by_{h}m"; hit=bool(np.isfinite(mx[j]) and mx[j]>=mc[j]*(1.0+thr))
                    if hit: out.at[oi,col]=1.0
                    elif complete: out.at[oi,col]=0.0
    return out


# ---------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------

def _safe_feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked_fragments = (
        "label", "future", "target", "terminal", "path_end", "learning_updated",
        "fingerprint", "next_substantial_peak", "later_higher_peak", "recurrent_",
        "event_", "interval_", "calendar_cohort", "lifetime_id", "decision_market_cap", "hit_plus",
    )
    blocked_exact = {
        "token_key", "snapshot_at", "decision_at", "schema_version", "config_json",
        "label_status_next_peak", "next_substantial_peak_at", "next_substantial_peak_confirmed_at",
        "later_higher_peak_at", "label_ready_at",
    }
    cols: list[str] = []
    for c in frame.columns:
        lc = str(c).lower()
        if c in blocked_exact or any(x in lc for x in blocked_fragments):
            continue
        s = pd.to_numeric(frame[c], errors="coerce")
        if s.notna().sum() >= 5:
            frame[c] = s
            cols.append(c)
    return sorted(set(cols))


def _token_weights(tokens: pd.Series) -> np.ndarray:
    counts = tokens.astype(str).value_counts()
    return np.asarray([1.0 / counts[str(x)] for x in tokens], dtype=float)


def _reg_components(n_estimators: int, quantile: float | None = None, poisson: bool = False) -> list[tuple[str, Any]]:
    models: list[tuple[str, Any]] = []
    if LGBMRegressor is not None:
        kwargs: dict[str, Any] = dict(
            n_estimators=n_estimators, learning_rate=0.035, num_leaves=31,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            random_state=101, verbosity=-1, n_jobs=2,
        )
        if poisson:
            kwargs["objective"] = "poisson"
        elif quantile is not None:
            kwargs.update(objective="quantile", alpha=float(quantile))
        else:
            kwargs["objective"] = "regression_l1"
        models.append(("lightgbm", LGBMRegressor(**kwargs)))
    if XGBRegressor is not None and not poisson:
        kwargs = dict(
            n_estimators=n_estimators, learning_rate=0.035, max_depth=5,
            min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=103, n_jobs=1,
        )
        if quantile is not None:
            kwargs.update(objective="reg:quantileerror", quantile_alpha=float(quantile))
        else:
            kwargs["objective"] = "reg:absoluteerror"
        models.append(("xgboost", XGBRegressor(**kwargs)))
    return models


def _multiclass_components(n_estimators: int) -> list[tuple[str, Any]]:
    models: list[tuple[str, Any]] = []
    if LGBMClassifier is not None:
        models.append(("lightgbm", LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=n_estimators,
            learning_rate=0.035, num_leaves=31, min_child_samples=20,
            subsample=0.9, colsample_bytree=0.9, random_state=107, verbosity=-1, n_jobs=2,
        )))
    if XGBClassifier is not None:
        models.append(("xgboost", XGBClassifier(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            n_estimators=n_estimators, learning_rate=0.035, max_depth=5,
            min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=109, n_jobs=1,
        )))
    if not models:
        raise RuntimeError("V24 requires LightGBM and/or XGBoost.")
    return models


def _binary_components(n_estimators: int) -> list[tuple[str,Any]]:
    out=[]
    if LGBMClassifier is not None:
        out.append(("lightgbm",LGBMClassifier(n_estimators=n_estimators,learning_rate=.035,num_leaves=31,min_child_samples=20,subsample=.9,colsample_bytree=.9,random_state=113,verbosity=-1,n_jobs=2)))
    if XGBClassifier is not None:
        out.append(("xgboost",XGBClassifier(n_estimators=n_estimators,learning_rate=.035,max_depth=5,min_child_weight=3,subsample=.9,colsample_bytree=.9,reg_lambda=1.,objective="binary:logistic",eval_metric="logloss",random_state=127,n_jobs=1)))
    return out


def _fit_binary_head(data: pd.DataFrame, features: list[str], target: str, n_estimators: int) -> dict[str,Any] | None:
    d=data[["token_key",*features,target]].copy(); d[target]=pd.to_numeric(d[target],errors="coerce"); d=d[d[target].isin([0,1])].copy()
    if len(d)<20 or d[target].nunique()<2: return None
    X=d[features].replace([np.inf,-np.inf],np.nan); y=d[target].astype(int).to_numpy(); w=_token_weights(d.token_key)
    fitted=[]
    for name,m in _binary_components(n_estimators):
        try: m.fit(X,y,sample_weight=w); fitted.append((name,m))
        except MemoryError: raise
        except Exception: continue
    return {"features":features,"target":target,"models":fitted} if fitted else None


def _predict_binary_head(head: dict[str,Any], frame: pd.DataFrame) -> np.ndarray:
    X=frame.reindex(columns=head["features"]).replace([np.inf,-np.inf],np.nan); ps=[]
    for _,m in head.get("models",[]):
        try: ps.append(np.asarray(m.predict_proba(X)[:,1],dtype=float))
        except MemoryError: raise
        except Exception: continue
    return np.mean(ps,axis=0) if ps else np.full(len(frame),np.nan)


def _fit_blended_regression(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    n_estimators: int,
    *,
    quantile: float | None = None,
    poisson: bool = False,
) -> dict[str, Any] | None:
    d = data[["token_key", *features, target]].copy()
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d[d[target].notna() & np.isfinite(d[target])].copy()
    if len(d) < 15:
        return None
    X = d[features].replace([np.inf, -np.inf], np.nan)
    y = d[target].to_numpy(dtype=float)
    w = _token_weights(d.token_key)
    fitted = []
    for name, model in _reg_components(n_estimators, quantile=quantile, poisson=poisson):
        try:
            model.fit(X, y, sample_weight=w)
            fitted.append((name, model))
        except MemoryError:
            raise
        except Exception:
            continue
    if not fitted:
        return None
    return {"features": features, "target": target, "models": fitted, "quantile": quantile, "poisson": poisson}


def _predict_blended(head: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    X = frame.reindex(columns=head["features"]).replace([np.inf, -np.inf], np.nan)
    preds = []
    for _, m in head.get("models", []):
        try:
            preds.append(np.asarray(m.predict(X), dtype=float))
        except MemoryError:
            raise
        except Exception:
            continue
    return np.mean(preds, axis=0) if preds else np.full(len(frame), np.nan)


def _hazard_token_weights(person_period: pd.DataFrame) -> np.ndarray:
    """Each token gets total survival-training weight exactly one."""
    counts=person_period.token_key.astype(str).value_counts()
    return np.asarray([1.0/counts[str(t)] for t in person_period.token_key],dtype=float)


def _fit_hazard_model(
    frame: pd.DataFrame,
    survival: pd.DataFrame,
    features: list[str],
    n_estimators: int,
) -> dict[str, Any]:
    if survival.empty:
        raise RuntimeError("No survival person-period rows are available.")
    s = survival.merge(
        frame[["token_key", "decision_at"] + features],
        on=["token_key", "decision_at"], how="left",
    )
    s["hazard__bin_index"] = s.bin_index.astype(float)
    s["hazard__log_end"] = np.log1p(s.bin_end_minutes.astype(float))
    hz_features = features + ["hazard__bin_index", "hazard__log_end"]
    X = s[hz_features].replace([np.inf, -np.inf], np.nan)
    y = s.event_class.to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Competing-risk target currently has only one event class.")
    # Hierarchical token weighting: each token contributes total weight 1 across
    # all of its decision timestamps and person-period intervals.  No event-class
    # upweighting is applied because that would distort calibrated hazard priors.
    w=_hazard_token_weights(s)
    fitted = []
    for name, model in _multiclass_components(n_estimators):
        try:
            model.fit(X, y, sample_weight=w)
            fitted.append((name, model))
        except MemoryError:
            raise
        except Exception:
            continue
    if not fitted:
        raise RuntimeError("All V24 hazard model components failed.")
    return {"features": hz_features, "models": fitted, "bins": list(sorted(set(s.bin_end_minutes.astype(float))))}


def _predict_hazard_probs(hazard: dict[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(frame)
    bins = [float(x) for x in hazard["bins"]]
    survival = np.ones(n, dtype=float)
    cif_peak = np.zeros(n, dtype=float)
    cif_death = np.zeros(n, dtype=float)
    by_bin: dict[str, np.ndarray] = {}
    base_features = [c for c in hazard["features"] if not c.startswith("hazard__")]
    for bidx, bend in enumerate(bins):
        x = frame.reindex(columns=base_features).copy()
        x["hazard__bin_index"] = float(bidx)
        x["hazard__log_end"] = math.log1p(bend)
        X = x.reindex(columns=hazard["features"]).replace([np.inf, -np.inf], np.nan)
        components = []
        for _, model in hazard.get("models", []):
            try:
                p = np.asarray(model.predict_proba(X), dtype=float)
                # Some components can omit an absent class. Re-map by classes_.
                full = np.zeros((n, 3), dtype=float)
                for j, cls in enumerate(getattr(model, "classes_", range(p.shape[1]))):
                    if int(cls) in (0, 1, 2):
                        full[:, int(cls)] = p[:, j]
                components.append(full)
            except MemoryError:
                raise
            except Exception:
                continue
        if not components:
            probs = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
        else:
            probs = np.mean(components, axis=0)
            denom = probs.sum(axis=1, keepdims=True)
            probs = probs / np.where(denom <= 0, 1.0, denom)
        peak_h = np.clip(probs[:, EVENT_PEAK], 0.0, 1.0)
        death_h = np.clip(probs[:, EVENT_DEATH], 0.0, 1.0)
        total_h = np.clip(peak_h + death_h, 0.0, 1.0)
        cif_peak += survival * peak_h
        cif_death += survival * death_h
        survival *= 1.0 - total_h
        by_bin[f"p_first_peak_by_{int(bend)}m"] = np.clip(cif_peak.copy(), 0.0, 1.0)
        by_bin[f"p_death_by_{int(bend)}m"] = np.clip(cif_death.copy(), 0.0, 1.0)
        by_bin[f"p_event_free_through_{int(bend)}m"] = np.clip(survival.copy(), 0.0, 1.0)
    return by_bin


def _isotonic_1d_observed(values:np.ndarray,increasing:bool) -> np.ndarray:
    arr=np.asarray(values,dtype=float).copy(); mask=np.isfinite(arr)
    if mask.sum()<=1: return arr
    x=np.arange(len(arr),dtype=float)[mask]; y=arr[mask]
    iso=IsotonicRegression(increasing=increasing,out_of_bounds="clip")
    arr[mask]=iso.fit_transform(x,y)
    return arr


def monotonic_probability_projection(matrix:np.ndarray) -> np.ndarray:
    """NaN-preserving threshold/horizon isotonic projection.

    Missing heads impose no artificial zero constraint.
    """
    x=np.asarray(matrix,dtype=float).copy()
    if x.ndim!=2: raise ValueError("probability matrix must be 2-dimensional")
    for _ in range(4):
        for r in range(x.shape[0]): x[r,:]=_isotonic_1d_observed(x[r,:],True)
        for c in range(x.shape[1]): x[:,c]=_isotonic_1d_observed(x[:,c],False)
    x[np.isfinite(x)]=np.clip(x[np.isfinite(x)],0.0,1.0)
    return x


def _hazard_person_period_logloss(hazard: dict[str,Any], frame: pd.DataFrame, survival: pd.DataFrame) -> float | None:
    if survival.empty: return None
    s=survival.merge(frame[["token_key","decision_at"]+[c for c in hazard["features"] if not c.startswith("hazard__")]],on=["token_key","decision_at"],how="left")
    probs=[]; ys=[]
    for bidx,g in s.groupby("bin_index"):
        x=g.copy(); x["hazard__bin_index"]=float(bidx); x["hazard__log_end"]=np.log1p(g.bin_end_minutes.astype(float))
        X=x.reindex(columns=hazard["features"]).replace([np.inf,-np.inf],np.nan); comps=[]
        for _,m in hazard.get("models",[]):
            try:
                p=np.asarray(m.predict_proba(X),dtype=float); full=np.zeros((len(g),3))
                for j,cls in enumerate(m.classes_): full[:,int(cls)]=p[:,j]
                comps.append(full)
            except MemoryError: raise
            except Exception: continue
        if comps:
            pp=np.mean(comps,axis=0); pp=np.clip(pp,1e-7,None); pp/=pp.sum(axis=1,keepdims=True)
            probs.append(pp); ys.extend(g.event_class.astype(int).tolist())
    if not probs or len(set(ys))<2: return None
    return float(log_loss(np.asarray(ys),np.vstack(probs),labels=[0,1,2]))


def run_cpcv_diagnostics(conn: sqlite3.Connection, train: pd.DataFrame, seqraw: SequenceFingerprintSource, cfg: V24Config, allow_small: bool) -> dict[str,Any]:
    folds=[]
    for split in purged_cpcv_splits(train,cfg):
        tr=train.iloc[split["train_idx"]].copy(); te=train.iloc[split["test_idx"]].copy()
        if len(tr)<(25 if allow_small else 120) or len(te)<5: continue
        encoder=_fit_sequence_encoder_from_source(seqraw,tr,cfg)
        fold_keys=pd.concat(
            [tr[["token_key","snapshot_at"]],te[["token_key","snapshot_at"]]],
            ignore_index=True,
        ).drop_duplicates(["token_key","snapshot_at"])
        enc=_encode_sequence_keys(seqraw,fold_keys,encoder)
        trm=add_barrier_targets(conn,add_recurrent_targets(conn,tr.merge(enc,on=["token_key","snapshot_at"],how="left"),cfg),cfg)
        tem=add_barrier_targets(conn,add_recurrent_targets(conn,te.merge(enc,on=["token_key","snapshot_at"],how="left"),cfg),cfg)
        features=_safe_feature_columns(trm)
        # Test frame gets the same ordered feature schema; missing values are allowed.
        for c in features:
            if c not in tem: tem[c]=np.nan
        try:
            hazard=_fit_hazard_model(trm,build_survival_person_period(trm,cfg),features,max(15,cfg.small_estimators if allow_small else 100))
            score=_hazard_person_period_logloss(hazard,tem,build_survival_person_period(tem,cfg))
        except MemoryError:
            raise
        except Exception:
            score=None
        folds.append({"fold_id":split["fold_id"],"test_blocks":split["test_blocks"],"train_rows":len(tr),"test_rows":len(te),"train_tokens":tr.token_key.nunique(),"test_tokens":te.token_key.nunique(),"hazard_logloss":score})
    scores=[float(x["hazard_logloss"]) for x in folds if x.get("hazard_logloss") is not None and np.isfinite(x["hazard_logloss"])]
    return {"folds":folds,"mean_hazard_logloss":float(np.mean(scores)) if scores else None,"scored_folds":len(scores)}



def _fit_isotonic_from_samples(samples:dict[str,list[tuple[float,float,str]]])->dict[str,Any]:
    out={}
    for key,vals in samples.items():
        if len(vals)<20: continue
        p=np.asarray([v[0] for v in vals],dtype=float); y=np.asarray([v[1] for v in vals],dtype=float); tok=pd.Series([v[2] for v in vals],dtype=str)
        mask=np.isfinite(p)&np.isfinite(y)
        p=p[mask]; y=y[mask]; tok=tok[mask]
        if len(p)<20 or len(np.unique(y))<2: continue
        w=_token_weights(tok)
        try:
            iso=IsotonicRegression(increasing=True,out_of_bounds='clip'); iso.fit(p,y,sample_weight=w); out[key]=iso
        except MemoryError: raise
        except Exception: continue
    return out


def fit_cpcv_calibrators(conn:sqlite3.Connection,train:pd.DataFrame,seqraw:SequenceFingerprintSource,cfg:V24Config,allow_small:bool)->dict[str,Any]:
    """Fit probability calibrators only from purged out-of-fold predictions."""
    samples:dict[str,list[tuple[float,float,str]]]={}
    splits=purged_cpcv_splits(train,cfg)
    for split in splits[:(3 if allow_small else len(splits))]:
        tr=train.iloc[split['train_idx']].copy(); te=train.iloc[split['test_idx']].copy()
        if len(tr)<(25 if allow_small else 120) or len(te)<5: continue
        enc=_fit_sequence_encoder_from_source(seqraw,tr,cfg)
        fold_keys=pd.concat(
            [tr[['token_key','snapshot_at']],te[['token_key','snapshot_at']]],
            ignore_index=True,
        ).drop_duplicates(['token_key','snapshot_at'])
        allenc=_encode_sequence_keys(seqraw,fold_keys,enc)
        trm=tr.merge(allenc,on=['token_key','snapshot_at'],how='left'); tem=te.merge(allenc,on=['token_key','snapshot_at'],how='left')
        trm=add_barrier_targets(conn,add_recurrent_targets(conn,trm,cfg),cfg); tem=add_barrier_targets(conn,add_recurrent_targets(conn,tem,cfg),cfg)
        feats=_safe_feature_columns(trm)
        for c in feats:
            if c not in tem: tem[c]=np.nan
        n=max(20,cfg.small_estimators if allow_small else 100)
        try:
            hz=_fit_hazard_model(trm,build_survival_person_period(trm,cfg),feats,n); hp=_predict_hazard_probs(hz,tem)
            for j,r in tem.reset_index(drop=True).iterrows():
                event,event_min,known=first_competing_event(r,cfg)
                if not known: continue
                for h in cfg.promotion_required_horizons_minutes:
                    for kind,name in ((EVENT_PEAK,'p_first_peak_by_'),(EVENT_DEATH,'p_death_by_')):
                        key=f'{name}{int(h)}m'
                        if key in hp:
                            y=float(event==kind and event_min<=h); samples.setdefault(key,[]).append((float(hp[key][j]),y,str(r.token_key)))
        except MemoryError: raise
        except Exception: pass
        jb=_fit_shared_binary_grid(trm,feats,_barrier_specs(cfg),n,'jointbarrier')
        for target,pred in _predict_shared_binary_grid(jb,tem).items() if jb else []:
            y=pd.to_numeric(tem[target],errors='coerce').to_numpy(dtype=float)
            for j in np.flatnonzero(np.isfinite(y)&np.isfinite(pred)):
                samples.setdefault(f'p_{target}',[]).append((float(pred[j]),float(y[j]),str(tem.iloc[j].token_key)))
        mh=_fit_shared_binary_grid(trm,feats,_higher_specs(cfg),n,'markedhigher')
        for target,pred in _predict_shared_binary_grid(mh,tem).items() if mh else []:
            y=pd.to_numeric(tem[target],errors='coerce').to_numpy(dtype=float)
            for j in np.flatnonzero(np.isfinite(y)&np.isfinite(pred)):
                samples.setdefault(f'p_{target}',[]).append((float(pred[j]),float(y[j]),str(tem.iloc[j].token_key)))
    return _fit_isotonic_from_samples(samples)


def _adaptive_state(conn:sqlite3.Connection,key:str,cfg:V24Config)->tuple[float,list[float],int]:
    row=conn.execute(f'SELECT bias_logit,residual_scores_json,n_updates,config_hash FROM {CALIBRATION_STATE_TABLE} WHERE calibration_key=?',(key,)).fetchone()
    conf=_stable_hash({'lr':cfg.calibration_learning_rate,'clip':cfg.calibration_clip_logit,'alpha':cfg.conformal_alpha})
    if not row or str(row[3])!=conf: return 0.0,[],0
    vals=json.loads(row[1]) if row[1] else []
    return float(row[0]),[float(x) for x in vals[-cfg.conformal_window:]],int(row[2])


def _apply_probability_calibration(conn:sqlite3.Connection,pred:dict[str,np.ndarray],bundle:dict[str,Any],cfg:V24Config)->dict[str,float]:
    base=bundle.get('probability_calibrators') or {}; radii=[]
    for key,arr in list(pred.items()):
        if not key.startswith('p_'): continue
        x=np.asarray(arr,dtype=float); mask=np.isfinite(x)
        if key in base and mask.any():
            try: x[mask]=base[key].predict(np.clip(x[mask],0,1))
            except Exception: pass
        bias,residuals,_=_adaptive_state(conn,key,cfg)
        if mask.any() and abs(bias)>1e-12:
            pp=np.clip(x[mask],1e-6,1-1e-6); logits=np.log(pp/(1-pp))+bias; x[mask]=1/(1+np.exp(-logits))
        pred[key]=np.clip(x,0,1)
        if residuals:
            radii.append(float(np.quantile(residuals,min(1.0,max(0.0,1-cfg.conformal_alpha)))))
    return {'adaptive_calibration_radius_mean':float(np.mean(radii)) if radii else float('nan')}


def _resolved_truth_for_prediction(row:pd.Series,key:str,cfg:V24Config)->float|None:
    if key.startswith('p_hit_plus'):
        col=key[2:]; v=row.get(col); return float(v) if v in (0,1,0.0,1.0) else None
    if key.startswith('p_later_higher_'):
        col=key[2:]; v=row.get(col); return float(v) if v in (0,1,0.0,1.0) else None
    if key.startswith('p_first_peak_by_') or key.startswith('p_death_by_'):
        try: h=int(key.rsplit('_',1)[-1].rstrip('m'))
        except Exception: return None
        event,event_min,known=first_competing_event(row,cfg)
        if not known and event_min<h: return None
        kind=EVENT_PEAK if key.startswith('p_first_peak') else EVENT_DEATH
        return float(event==kind and event_min<=h)
    return None


def update_adaptive_calibration(conn:sqlite3.Connection,frame:pd.DataFrame,cfg:V24Config)->dict[str,int]:
    """Online intercept calibration from resolved development-eligible OOS predictions."""
    if frame.empty: return {'updates':0}
    enriched=add_barrier_targets(conn,add_recurrent_targets(conn,frame,cfg),cfg)
    fmap={(str(r.token_key),_utc(r.decision_at)):r for _,r in enriched.iterrows()}
    led=pd.read_sql_query(f"SELECT prediction_id,token_key,decision_at,prediction_json FROM {PREDICTION_LEDGER} WHERE policy_training_eligible=1 ORDER BY decision_at",conn)
    if led.empty: return {'updates':0}
    conf=_stable_hash({'lr':cfg.calibration_learning_rate,'clip':cfg.calibration_clip_logit,'alpha':cfg.conformal_alpha}); updates=0; now=_now_iso()
    for lr in led.itertuples(index=False):
        truth_row=fmap.get((str(lr.token_key),_utc(lr.decision_at)))
        if truth_row is None: continue
        state=_loads(lr.prediction_json)
        for key,pv in state.items():
            if not str(key).startswith('p_'): continue
            p=_finite(pv); y=_resolved_truth_for_prediction(truth_row,str(key),cfg)
            if p is None or y is None: continue
            if conn.execute(f'SELECT 1 FROM {CALIBRATION_UPDATES_TABLE} WHERE prediction_id=? AND calibration_key=?',(lr.prediction_id,key)).fetchone(): continue
            bias,residuals,n=_adaptive_state(conn,key,cfg); pp=float(np.clip(p,1e-6,1-1e-6)); cal=1/(1+math.exp(-(math.log(pp/(1-pp))+bias))); bias=float(np.clip(bias+cfg.calibration_learning_rate*(y-cal),-cfg.calibration_clip_logit,cfg.calibration_clip_logit)); residuals=(residuals+[abs(y-cal)])[-cfg.conformal_window:]
            conn.execute(f"""INSERT INTO {CALIBRATION_STATE_TABLE}(calibration_key,bias_logit,n_updates,last_updated_at,residual_scores_json,config_hash) VALUES(?,?,?,?,?,?) ON CONFLICT(calibration_key) DO UPDATE SET bias_logit=excluded.bias_logit,n_updates=excluded.n_updates,last_updated_at=excluded.last_updated_at,residual_scores_json=excluded.residual_scores_json,config_hash=excluded.config_hash""",(key,bias,n+1,now,_json(residuals),conf))
            conn.execute(f'INSERT INTO {CALIBRATION_UPDATES_TABLE}(prediction_id,calibration_key,resolved_at,observed_target,raw_probability) VALUES(?,?,?,?,?)',(lr.prediction_id,key,now,y,p)); updates+=1
    conn.commit(); return {'updates':updates}

# ---------------------------------------------------------------------------
# Batch forecaster, adapter, compaction, prediction
# ---------------------------------------------------------------------------

def _prepare_model_frame(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    seqraw: SequenceFingerprintSource,
    cfg: V24Config,
    *,
    encoder: dict[str, Any] | None = None,
    sequence_challenger: dict[str, Any] | None = None,
    fit_encoder_on: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
    include_targets: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if encoder is None:
        if fit_encoder_on is None:
            fit_encoder_on = frame
        encoder = _fit_sequence_encoder_from_source(seqraw, fit_encoder_on, cfg)
    enc = _encode_sequence_keys(
        seqraw,
        frame[["token_key", "snapshot_at"]],
        encoder,
        sequence_challenger=sequence_challenger,
    )
    out = frame.merge(enc, on=["token_key", "snapshot_at"], how="left")
    if include_targets:
        out = add_recurrent_targets(conn, out, cfg, as_of=as_of)
        out = add_barrier_targets(conn, out, cfg, as_of=as_of)
    return out, encoder



def _shared_grid_long(data: pd.DataFrame,features:list[str],specs:list[tuple[str,float,float]],prefix:str) -> tuple[pd.DataFrame,list[str]]:
    parts=[]
    base=data[["token_key",*features]].copy()
    if features:
        base[features]=base[features].apply(pd.to_numeric,errors="coerce").astype(np.float32,copy=False)
    for target,a,b in specs:
        if target not in data.columns: continue
        target_values=pd.to_numeric(data[target],errors="coerce")
        valid=target_values.isin([0,1])
        d=base.loc[valid].copy(); d["__y"]=target_values.loc[valid].astype(np.int8).to_numpy()
        if d.empty: continue
        d[f"{prefix}__a"]=np.float32(a); d[f"{prefix}__log_b"]=np.float32(math.log1p(float(b))); d[f"{prefix}__a_log_b"]=np.float32(float(a)*math.log1p(float(b)))
        parts.append(d)
    if not parts: return pd.DataFrame(),[]
    long=pd.concat(parts,ignore_index=True); meta=[f"{prefix}__a",f"{prefix}__log_b",f"{prefix}__a_log_b"]
    return long,features+meta


def _fit_shared_binary_grid(data: pd.DataFrame,features:list[str],specs:list[tuple[str,float,float]],n_estimators:int,prefix:str) -> dict[str,Any] | None:
    long,all_features=_shared_grid_long(data,features,specs,prefix)
    if len(long)<30 or long["__y"].nunique()<2: return None
    X=long[all_features].replace([np.inf,-np.inf],np.nan); y=long["__y"].astype(int).to_numpy()
    counts=long.token_key.astype(str).value_counts(); w=np.asarray([1.0/counts[str(t)] for t in long.token_key],dtype=float)
    fitted=[]
    for name,m in _binary_components(n_estimators):
        try: m.fit(X,y,sample_weight=w); fitted.append((name,m))
        except MemoryError: raise
        except Exception: continue
    return {"features":all_features,"base_features":features,"specs":specs,"prefix":prefix,"models":fitted} if fitted else None


def _predict_shared_binary_grid(head:dict[str,Any],frame:pd.DataFrame) -> dict[str,np.ndarray]:
    if not head: return {}
    out={}; prefix=head["prefix"]
    for target,a,b in head["specs"]:
        x=frame.reindex(columns=head.get("base_features",[])).copy(); x[f"{prefix}__a"]=float(a); x[f"{prefix}__log_b"]=math.log1p(float(b)); x[f"{prefix}__a_log_b"]=float(a)*math.log1p(float(b))
        X=x.reindex(columns=head["features"]).replace([np.inf,-np.inf],np.nan); ps=[]
        for _,m in head.get("models",[]):
            try: ps.append(np.asarray(m.predict_proba(X)[:,1],dtype=float))
            except MemoryError: raise
            except Exception: continue
        out[target]=np.mean(ps,axis=0) if ps else np.full(len(frame),np.nan)
    return out


def _barrier_specs(cfg:V24Config) -> list[tuple[str,float,float]]:
    return [(f"hit_plus{_threshold_tag(thr)}_by_{h}m",float(thr),float(h)) for thr in cfg.upside_thresholds for h in cfg.probability_horizons_minutes]


def _higher_specs(cfg:V24Config) -> list[tuple[str,float,float]]:
    return [(f"later_higher_{int(round(m*100))}pct_by_{h}m",float(m),float(h)) for m in cfg.higher_peak_mark_margins for h in (240,720,1440,4320)]


def _feature_reference(frame:pd.DataFrame,features:list[str],cfg:V24Config)->dict[str,dict[str,float]]:
    sample=_token_balanced_rows(frame[["token_key","snapshot_at"]+features],cfg.drift_feature_sample_per_token)
    ref={}
    for c in features:
        x=pd.to_numeric(sample[c],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
        if len(x)>=5:
            ref[c]={"mean":float(x.mean()),"std":float(max(x.std(ddof=0),1e-6)),"median":float(x.median())}
    return ref


def _feature_drift_score(frame:pd.DataFrame,reference:dict[str,dict[str,float]],cfg:V24Config)->float:
    if not reference or frame.empty: return 0.0
    sample=_token_balanced_rows(frame[["token_key","snapshot_at"]+[c for c in reference if c in frame]],cfg.drift_feature_sample_per_token)
    shifts=[]
    for c,r in reference.items():
        if c not in sample: continue
        x=pd.to_numeric(sample[c],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
        if len(x)>=3: shifts.append(abs(float(x.mean())-float(r["mean"]))/max(float(r["std"]),1e-6))
    return float(np.median(shifts)) if shifts else 0.0


def _adapter_family_weights(drift:float,cfg:V24Config)->dict[str,float]:
    lo=float(cfg.drift_weight_floor); hi=float(cfg.adapter_max_weight); strength=min(1.0,max(0.0,math.tanh(max(0.0,drift)/1.5))); base=lo+(hi-lo)*strength
    return {"hazard":base,"short_barrier":min(hi,base*1.15),"long_barrier":max(lo,base*0.65),"marked_higher":max(lo,base*0.80),"recurrent":max(lo,base*0.70)}


def _adapter_weight_for_key(adapter:dict[str,Any],key:str,cfg:V24Config)->float:
    key=str(key); head_weights=adapter.get("head_weights") or {}
    if key in head_weights:
        return float(np.clip(head_weights[key],0.0,cfg.adapter_max_weight))
    weights=adapter.get("weights") or {}
    if key.startswith("p_first_peak") or key.startswith("p_death") or key.startswith("p_event_free"): fam="hazard"
    elif key.startswith("p_hit_plus"):
        try: h=int(key.rsplit("_",1)[-1].rstrip("m"))
        except Exception: h=4320
        fam="short_barrier" if h<=720 else "long_barrier"
    elif key.startswith("p_later_higher"): fam="marked_higher"
    else: fam="recurrent"
    return float(np.clip(weights.get(fam,adapter.get("weight",0.0)),0.0,cfg.adapter_max_weight))



def _derive_adapter_head_weights(adapter:dict[str,Any],cfg:V24Config)->dict[str,float]:
    """Materialize drift-aware weights for every forecast head.

    Short-horizon heads are allowed to adapt faster; long-horizon heads are
    shrunk toward the stable model because their recent supervision is more
    censored.  This is head-specific even when several heads share one fitted
    multi-task classifier.
    """
    out={}; base=adapter.get("weights") or {}
    for h in cfg.probability_horizons_minutes:
        horizon_scale=max(0.35,min(1.20,math.sqrt(720.0/max(float(h),60.0))))
        hz=float(base.get("hazard",0.0))*horizon_scale
        out[f"p_first_peak_by_{h}m"]=min(cfg.adapter_max_weight,hz); out[f"p_death_by_{h}m"]=min(cfg.adapter_max_weight,hz); out[f"p_event_free_by_{h}m"]=min(cfg.adapter_max_weight,hz)
        bf=float(base.get("short_barrier" if h<=720 else "long_barrier",0.0))*horizon_scale
        for thr in cfg.upside_thresholds: out[f"p_hit_plus{_threshold_tag(thr)}_by_{h}m"]=min(cfg.adapter_max_weight,bf)
    for h in (240,720,1440,4320):
        hs=max(0.35,min(1.15,math.sqrt(720.0/max(float(h),240.0))))
        for m in cfg.higher_peak_mark_margins: out[f"p_later_higher_{int(round(m*100))}pct_by_{h}m"]=min(cfg.adapter_max_weight,float(base.get("marked_higher",0.0))*hs)
        out[f"recurrent_peak_count_{h}m"]=min(cfg.adapter_max_weight,float(base.get("recurrent",0.0))*hs)
    for k in ("next_gap_q25","next_gap_q50","next_gap_q75","next_peak_multiple_q25","next_peak_multiple_q50","next_peak_multiple_q75","second_gap_q50","second_peak_relative_q50"):
        out[k]=float(base.get("recurrent",0.0))
    return out

def target_definition_hash(cfg:V24Config)->str:
    payload={"horizon_minutes":cfg.horizon_minutes,"operational_gap_minutes":cfg.operational_gap_minutes,"age_out_minutes":cfg.age_out_minutes,"survival_bins":list(cfg.survival_bins_minutes),"upside_thresholds":list(cfg.upside_thresholds),"probability_horizons":list(cfg.probability_horizons_minutes),"higher_peak_margin_pct":cfg.higher_peak_margin_pct,"higher_peak_mark_margins":list(cfg.higher_peak_mark_margins),"peak_min_runup_pct":0.20,"peak_confirm_retrace_pct":0.15}
    return _stable_hash(payload)


def feature_definition_hash(features:list[str],cfg:V24Config,sequence_challenger:dict[str,Any]|None)->str:
    payload={"features":list(features),"sequence_config":_sequence_config_hash(cfg),"sequence_challenger":(sequence_challenger or {}).get("kind"),"sequence_dim":(sequence_challenger or {}).get("out_dim")}
    return _stable_hash(payload)


def execution_definition_hash(cfg:V24Config)->str:
    return _stable_hash({"fill_rule":"first_observation_strictly_after_decision","disappearance_accounting":"observed_plus_execution_conservative","fallback_round_trip_bps":cfg.fallback_round_trip_bps,"friction_stress_bps":list(cfg.friction_stress_bps)})


def training_data_hash(frame:pd.DataFrame)->str:
    cols=[c for c in ("token_key","snapshot_at","value_version","row_fingerprint","last_corrected_at") if c in frame.columns]
    if not cols: cols=["token_key","snapshot_at"]
    rec=frame[cols].copy().sort_values([c for c in ("token_key","snapshot_at") if c in cols]).astype(str).to_dict("records")
    return _stable_hash(rec)


def bundle_identity_hash(bundle:dict[str,Any])->str:
    try: return str(joblib.hash(bundle,hash_name="sha1"))
    except Exception: return _stable_hash({"schema":bundle.get("schema_version"),"cutoff":bundle.get("stable_training_cutoff"),"target":bundle.get("target_definition_hash"),"feature":bundle.get("feature_definition_hash"),"execution":bundle.get("execution_definition_hash")})

def fit_batch_bundle(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    seqraw: SequenceFingerprintSource,
    cutoff: pd.Timestamp,
    cfg: V24Config,
    *,
    allow_small: bool = False,
    generation: int = 1,
    exclude_tokens: set[str] | None = None,
) -> dict[str, Any]:
    eligible_train, eligibility = _training_history_selection(
        conn, frame, cutoff, cfg, exclude_tokens=exclude_tokens
    )
    if len(eligible_train) < (40 if allow_small else 200):
        raise RuntimeError(
            "Insufficient leakage-safe V24 training history: "
            f"{len(eligible_train)} rows; diagnostics={_json(eligibility)}"
        )
    train = _bounded_model_training_rows(eligible_train, cfg)
    sequence_challenger=_fit_ts2vec_from_source(
        seqraw, train, cfg, allow_small=allow_small
    )
    model_frame, encoder = _prepare_model_frame(conn, train, seqraw, cfg, fit_encoder_on=train, as_of=cutoff, sequence_challenger=sequence_challenger)
    features = _safe_feature_columns(model_frame)
    if not features:
        raise RuntimeError("No usable V24 feature columns.")
    n_estimators = cfg.small_estimators if allow_small else cfg.stable_estimators
    survival = build_survival_person_period(model_frame, cfg, conn=conn, as_of=cutoff)
    hazard = _fit_hazard_model(model_frame, survival, features, n_estimators)

    recurrent: dict[str,Any]={}
    recurrent_horizons=tuple(
        int(h) for h in cfg.probability_horizons_minutes
        if 240 <= int(h) <= int(cfg.horizon_minutes)
    )
    for h in recurrent_horizons:
        name=f"recurrent_peak_count_{h}m"
        fit=_fit_blended_regression(model_frame,features,name,n_estimators,poisson=True)
        if fit: recurrent[name]=fit
    for q in (0.25,0.50,0.75):
        fit=_fit_blended_regression(model_frame[model_frame.recurrent_next_gap_minutes.notna()],features,"recurrent_next_gap_minutes",n_estimators,quantile=q)
        if fit: recurrent[f"next_gap_q{int(q*100)}"]=fit
        fit=_fit_blended_regression(model_frame[model_frame.recurrent_next_peak_multiple.notna()],features,"recurrent_next_peak_multiple",n_estimators,quantile=q)
        if fit: recurrent[f"next_peak_multiple_q{int(q*100)}"]=fit
    second=_fit_blended_regression(model_frame[model_frame.recurrent_second_gap_minutes.notna()],features,"recurrent_second_gap_minutes",n_estimators,quantile=0.50)
    if second: recurrent["second_gap_q50"]=second
    second_rel=_fit_blended_regression(model_frame[model_frame.recurrent_second_peak_relative_to_first.notna()],features,"recurrent_second_peak_relative_to_first",n_estimators,quantile=0.50)
    if second_rel: recurrent["second_peak_relative_q50"]=second_rel

    joint_barrier=_fit_shared_binary_grid(model_frame,features,_barrier_specs(cfg),n_estimators,"jointbarrier")
    marked_higher=_fit_shared_binary_grid(model_frame,features,_higher_specs(cfg),n_estimators,"markedhigher")

    # CPCV fits real fold-specific hazard models. The external one-use promotion
    # cohort is not included in these folds and remains the deployment gate.
    cpcv = run_cpcv_diagnostics(conn, train, seqraw, cfg, allow_small)
    try:
        probability_calibrators=fit_cpcv_calibrators(conn,train,seqraw,cfg,allow_small)
    except MemoryError:
        raise
    except Exception:
        probability_calibrators={}
    feature_reference=_feature_reference(model_frame,features,cfg)
    stable_created=_now_iso(); target_hash=target_definition_hash(cfg); feature_hash=feature_definition_hash(features,cfg,sequence_challenger); execution_hash=execution_definition_hash(cfg); data_hash=training_data_hash(train)

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": stable_created,
        "stable_created_at": stable_created,
        "last_compaction_at": stable_created,
        "config": asdict(cfg),
        "target_definition_hash":target_hash,"feature_definition_hash":feature_hash,"execution_definition_hash":execution_hash,"training_data_hash":data_hash,
        "stable_training_cutoff": _iso(cutoff),
        "stable_generation": int(generation),
        "adapter_round": 0,
        "adapter": None,
        "sequence_encoder": encoder,
        "sequence_challenger": sequence_challenger,
        "features": features,
        "feature_reference": feature_reference,
        "hazard": hazard,
        "recurrent": recurrent,
        "joint_barrier": joint_barrier,
        "marked_higher": marked_higher,
        "probability_calibrators": probability_calibrators,
        "liquidity_model": fit_liquidity_model(conn, cfg, allow_small=allow_small, cutoff=cutoff),
        "training_rows": int(len(model_frame)),
        "eligible_training_rows": int(len(eligible_train)),
        "training_tokens": int(model_frame.token_key.nunique()),
        "training_row_sampling": {
            "method": "deterministic_even_lifecycle_token_balanced",
            "max_rows": int(cfg.model_max_training_rows),
            "max_rows_per_token": int(cfg.model_max_rows_per_token),
        },
        "cpcv_diagnostics": cpcv,
    }


def _adapter_training_frame(
    conn: sqlite3.Connection, frame: pd.DataFrame, cutoff: pd.Timestamp, stable_cutoff: pd.Timestamp,
    cfg: V24Config, exclude_tokens: set[str] | None = None,
) -> pd.DataFrame:
    return adapter_history_before(conn,frame,cutoff,stable_cutoff,cfg,exclude_tokens=exclude_tokens)


def fit_online_adapter(
    conn: sqlite3.Connection,
    champion: dict[str, Any],
    frame: pd.DataFrame,
    seqraw: SequenceFingerprintSource,
    cutoff: pd.Timestamp,
    cfg: V24Config,
    *,
    allow_small: bool = False,
    exclude_tokens: set[str] | None = None,
) -> dict[str, Any]:
    stable_cutoff = _utc(champion["stable_training_cutoff"])
    eligible_recent = _adapter_training_frame(conn, frame, cutoff, stable_cutoff, cfg, exclude_tokens=exclude_tokens)
    if len(eligible_recent) < (20 if allow_small else 100):
        raise RuntimeError(f"Insufficient recent rows for V24 online adapter: {len(eligible_recent)}")
    recent = _bounded_model_training_rows(eligible_recent, cfg)
    model_frame, _ = _prepare_model_frame(
        conn, recent, seqraw, cfg, encoder=champion["sequence_encoder"], sequence_challenger=champion.get("sequence_challenger"), as_of=cutoff
    )
    features = champion["features"]
    n_estimators = max(30, cfg.adapter_estimators // (2 if allow_small else 1))
    survival = build_survival_person_period(model_frame, cfg, conn=conn, as_of=cutoff)
    hazard = _fit_hazard_model(model_frame, survival, features, n_estimators)
    recurrent: dict[str,Any]={}
    recurrent_horizons=tuple(
        int(h) for h in cfg.probability_horizons_minutes
        if 240 <= int(h) <= int(cfg.horizon_minutes)
    )
    for h in recurrent_horizons:
        name=f"recurrent_peak_count_{h}m"
        fit=_fit_blended_regression(model_frame,features,name,n_estimators,poisson=True)
        if fit: recurrent[name]=fit
    for q in (0.25,0.50,0.75):
        fit=_fit_blended_regression(model_frame[model_frame.recurrent_next_gap_minutes.notna()],features,"recurrent_next_gap_minutes",n_estimators,quantile=q)
        if fit: recurrent[f"next_gap_q{int(q*100)}"]=fit
        fit=_fit_blended_regression(model_frame[model_frame.recurrent_next_peak_multiple.notna()],features,"recurrent_next_peak_multiple",n_estimators,quantile=q)
        if fit: recurrent[f"next_peak_multiple_q{int(q*100)}"]=fit
    second=_fit_blended_regression(model_frame[model_frame.recurrent_second_gap_minutes.notna()],features,"recurrent_second_gap_minutes",n_estimators,quantile=.50)
    if second: recurrent["second_gap_q50"]=second
    second_rel=_fit_blended_regression(model_frame[model_frame.recurrent_second_peak_relative_to_first.notna()],features,"recurrent_second_peak_relative_to_first",n_estimators,quantile=.50)
    if second_rel: recurrent["second_peak_relative_q50"]=second_rel
    joint_barrier=_fit_shared_binary_grid(model_frame,features,_barrier_specs(cfg),n_estimators,"jointbarrier")
    marked_higher=_fit_shared_binary_grid(model_frame,features,_higher_specs(cfg),n_estimators,"markedhigher")
    drift=_feature_drift_score(model_frame,champion.get("feature_reference",{}),cfg)
    weights=_adapter_family_weights(drift,cfg); weight=float(max(weights.values()) if weights else 0.0)
    adapter = {
        "created_at": _now_iso(), "adapter_created_at": _now_iso(), "training_start": recent.snapshot_at.min().isoformat(),
        "training_cutoff": _iso(cutoff), "rows": int(len(model_frame)),
        "eligible_rows": int(len(eligible_recent)),
        "tokens": int(model_frame.token_key.nunique()), "weight": weight, "weights":weights,"drift_score":drift,
        "hazard": hazard, "recurrent": recurrent, "joint_barrier": joint_barrier,
        "marked_higher": marked_higher,
    }
    adapter["head_weights"]=_derive_adapter_head_weights(adapter,cfg)
    return adapter


def should_compact(bundle:dict[str,Any],cfg:V24Config,now:pd.Timestamp|None=None)->bool:
    now=_utc(now or pd.Timestamp.now(tz="UTC")); rounds=int(bundle.get("adapter_round",0)); stable_created=_utc(bundle.get("stable_created_at") or bundle.get("last_compaction_at") or bundle["created_at"]); age_days=(now-stable_created).total_seconds()/86400.0
    return rounds>=cfg.compaction_every_adapter_rounds or age_days>=cfg.compaction_every_days


def _blend_predictions(stable:dict[str,np.ndarray],adapter_pred:dict[str,np.ndarray]|None,adapter_bundle:dict[str,Any]|None,cfg:V24Config)->dict[str,np.ndarray]:
    if not adapter_pred or not adapter_bundle: return stable
    out={}; keys=set(stable)|set(adapter_pred)
    for k in keys:
        sv=stable.get(k); av=adapter_pred.get(k); w=_adapter_weight_for_key(adapter_bundle,k,cfg)
        if sv is None: out[k]=av
        elif av is None: out[k]=sv
        else: out[k]=(1.0-w)*sv+w*av
    return out


def _predict_bundle_parts(bundle_part:dict[str,Any],frame:pd.DataFrame) -> dict[str,np.ndarray]:
    out=_predict_hazard_probs(bundle_part["hazard"],frame)
    for name,head in bundle_part.get("recurrent",{}).items(): out[name]=_predict_blended(head,frame)
    for target,p in _predict_shared_binary_grid(bundle_part.get("joint_barrier"),frame).items(): out[f"p_{target}"]=p
    for target,p in _predict_shared_binary_grid(bundle_part.get("marked_higher"),frame).items(): out[f"p_{target}"]=p
    return out


def project_recurrent_outputs(pred: dict[str, np.ndarray]) -> None:
    counts = [k for k in ("recurrent_peak_count_240m", "recurrent_peak_count_720m", "recurrent_peak_count_1440m", "recurrent_peak_count_4320m") if k in pred]
    if counts:
        mat = np.column_stack([np.clip(pred[k], 0.0, None) for k in counts])
        mat = np.maximum.accumulate(mat, axis=1)
        for j, k in enumerate(counts):
            pred[k] = mat[:, j]
    qs = [k for k in ("next_gap_q25", "next_gap_q50", "next_gap_q75") if k in pred]
    if len(qs) == 3:
        mat = np.sort(np.column_stack([pred[k] for k in qs]), axis=1)
        for j, k in enumerate(qs):
            pred[k] = np.clip(mat[:, j], 0.0, None)
    qs = [k for k in ("next_peak_multiple_q25", "next_peak_multiple_q50", "next_peak_multiple_q75") if k in pred]
    if len(qs) == 3:
        mat = np.sort(np.column_stack([pred[k] for k in qs]), axis=1)
        for j, k in enumerate(qs):
            pred[k] = np.clip(mat[:, j], 0.0, None)


def predict_frame(
    conn: sqlite3.Connection,
    bundle: dict[str, Any],
    base_frame: pd.DataFrame,
    seqraw: SequenceFingerprintSource,
    cfg: V24Config,
) -> pd.DataFrame:
    model_frame, _ = _prepare_model_frame(
        conn,
        base_frame,
        seqraw,
        cfg,
        encoder=bundle["sequence_encoder"],
        sequence_challenger=bundle.get("sequence_challenger"),
        include_targets=False,
    )
    stable = _predict_bundle_parts(bundle, model_frame)
    adapter_bundle = bundle.get("adapter")
    adapter_pred = None
    if adapter_bundle:
        adapter_pred = _predict_bundle_parts(adapter_bundle, model_frame)
    pred = _blend_predictions(stable, adapter_pred, adapter_bundle, cfg)
    weight=float(max((adapter_bundle or {}).get("weights",{}).values(),default=float((adapter_bundle or {}).get("weight",0.0))))
    calibration_meta=_apply_probability_calibration(conn,pred,bundle,cfg)
    project_recurrent_outputs(pred)
    # Joint monotonic projection across gain thresholds and time horizons.
    n=len(model_frame)
    for i in range(n):
        mat=np.full((len(cfg.upside_thresholds),len(cfg.probability_horizons_minutes)),np.nan)
        for ti,thr in enumerate(cfg.upside_thresholds):
            for hi,h in enumerate(cfg.probability_horizons_minutes):
                key=f"p_hit_plus{_threshold_tag(thr)}_by_{h}m"
                if key in pred: mat[ti,hi]=pred[key][i]
        if np.isfinite(mat).any():
            proj=monotonic_probability_projection(mat)
            for ti,thr in enumerate(cfg.upside_thresholds):
                for hi,h in enumerate(cfg.probability_horizons_minutes):
                    key=f"p_hit_plus{_threshold_tag(thr)}_by_{h}m"
                    if key in pred and np.isfinite(mat[ti,hi]): pred[key][i]=proj[ti,hi]
    result = pd.concat(
        [
            model_frame[["token_key", "snapshot_at"]].reset_index(drop=True),
            pd.DataFrame({k: np.asarray(v) for k, v in pred.items()}),
        ],
        axis=1,
        copy=False,
    )
    # Direct marked higher-peak probabilities; no Poisson count proxy.
    margins=list(cfg.higher_peak_mark_margins); horizons=(240,720,1440,4320)
    for i in range(len(result)):
        mat=np.full((len(margins),len(horizons)),np.nan)
        for mi,m in enumerate(margins):
            for hi,h in enumerate(horizons):
                key=f"p_later_higher_{int(round(m*100))}pct_by_{h}m"
                if key in result: mat[mi,hi]=result.at[result.index[i],key]
        if np.isfinite(mat).any():
            proj=monotonic_probability_projection(mat)
            for mi,m in enumerate(margins):
                for hi,h in enumerate(horizons):
                    key=f"p_later_higher_{int(round(m*100))}pct_by_{h}m"
                    if key in result and np.isfinite(mat[mi,hi]): result.at[result.index[i],key]=proj[mi,hi]
    liquidity_model = bundle.get("liquidity_model") or {"active": False, "fallback_round_trip_bps": cfg.fallback_round_trip_bps}
    extra_columns: dict[str, Any] = {
        "liquidity_model_active": np.full(
            len(result), 1.0 if liquidity_model.get("active") else 0.0
        )
    }
    for size in (100.0, 200.0, 500.0):
        vals = []
        for _, r in model_frame.iterrows():
            def rv(name: str) -> float | None:
                for key in (f"raw__{name}", name):
                    if key in r.index:
                        x = _finite(r.get(key))
                        if x is not None:
                            return x
                return None
            vals.append(estimate_slippage_bps(
                liquidity_model, size, rv("liquidity_usd"), rv("volume_usd"), rv("market_cap_usd")
            ))
        extra_columns[f"pred_one_way_slippage_bps_{int(size)}usd"] = vals
    extra_columns["v24_adapter_weight_max"] = np.full(len(result), weight)
    for fam, w in (adapter_bundle or {}).get("weights", {}).items():
        extra_columns[f"v24_adapter_weight_{fam}"] = np.full(len(result), float(w))
    extra_columns["v24_stable_training_cutoff"] = np.full(
        len(result), bundle["stable_training_cutoff"], dtype=object
    )
    extra_columns["v24_adaptive_calibration_radius_mean"] = np.full(
        len(result), calibration_meta.get("adaptive_calibration_radius_mean")
    )
    result = pd.concat(
        [result.reset_index(drop=True), pd.DataFrame(extra_columns)],
        axis=1,
        copy=False,
    )
    return result


def _connect_live_read(db: str) -> sqlite3.Connection:
    """Open a collector-friendly read connection for live inference."""
    conn = sqlite3.connect(db, timeout=LIVE_SQLITE_BUSY_TIMEOUT_MS / 1000.0)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={LIVE_SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA query_only=ON")
    return conn


def _latest_observation_timestamp(db: str) -> pd.Timestamp | None:
    """Return the newest usable raw observation without materializing the table."""
    with closing(_connect_live_read(db)) as conn:
        source = peak.discover_observation_source(conn)
        quote = lambda value: '"' + str(value).replace('"', '""') + '"'
        row = conn.execute(
            f"SELECT MAX({quote(source['time'])}) FROM {quote(source['table'])} "
            f"WHERE {quote(source['token'])} IS NOT NULL "
            f"AND {quote(source['mc'])} IS NOT NULL AND {quote(source['mc'])}>0"
        ).fetchone()
    return _utc(row[0]) if row and row[0] else None


def _current_sequence_fingerprints(
    observations: pd.DataFrame,
    latest: pd.Timestamp,
    cfg: V24Config,
) -> pd.DataFrame:
    """Calculate only the latest causal sequence rows, without cache writes."""
    current_tokens = set(
        observations.loc[observations.snapshot_at == latest, "token_key"].astype(str)
    )
    rows: list[dict[str, Any]] = []
    bases = [c for c in _SEQUENCE_BASES if c in observations.columns]
    active = observations[observations.token_key.astype(str).isin(current_tokens)]
    for token, group in active.groupby("token_key", sort=False):
        group = group.sort_values("snapshot_at").reset_index(drop=True)
        positions = np.flatnonzero(
            (
                pd.to_datetime(group.snapshot_at, format="ISO8601", utc=True)
                == latest
            ).to_numpy()
        )
        if not len(positions):
            continue
        times = pd.to_datetime(group.snapshot_at, format="ISO8601", utc=True).astype("int64").to_numpy()
        arrays = {c: _obs_numeric(group, c) for c in bases}
        rows.append(
            _fingerprint_for_index(str(token), group, times, arrays, int(positions[-1]), cfg)
        )
    return pd.DataFrame(rows, columns=None)


def _current_inference_inputs(
    conn: sqlite3.Connection,
    cfg: V24Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Build label-free features for the newest durable raw capture.

    Live inference must never inner-join against the training-label table.  New
    observations are expected to be unlabeled, so using ``load_v24_frame`` here
    silently selected an older labeled snapshot and produced stale predictions.
    """
    observations, _ = peak.load_observations(conn)
    if observations.empty:
        raise RuntimeError("No usable Axiom observations are available for V24 prediction.")
    latest = _utc(observations.snapshot_at.max())
    current_observations = observations[observations.snapshot_at == latest].copy()
    expected_tokens = set(current_observations.token_key.astype(str))

    features = peak.build_fallback_features(observations, emit_at=latest)
    if features.empty:
        raise RuntimeError(f"No causal features were produced for latest capture {latest.isoformat()}.")

    # Preserve any safe external enrichment already cached for this exact capture,
    # but never require the durable cache to be current for live prediction.
    try:
        cached = peak._load_cached_feature_frame(conn, current_at=latest)
    except sqlite3.DatabaseError:
        cached = pd.DataFrame()
    if not cached.empty:
        extra = [c for c in cached.columns if c not in features.columns]
        if extra:
            features = features.merge(
                cached[["token_key", "snapshot_at", *extra]],
                on=["token_key", "snapshot_at"],
                how="left",
            )

    produced_tokens = set(features.token_key.astype(str))
    missing = sorted(expected_tokens - produced_tokens)
    if missing:
        raise RuntimeError(
            "Current-feature construction omitted latest-capture tokens; refusing "
            f"partial prediction: missing={missing[:10]}"
        )
    features = features[features.token_key.astype(str).isin(expected_tokens)].copy()
    if set(pd.to_datetime(features.snapshot_at, format="ISO8601", utc=True)) != {latest}:
        raise RuntimeError("Current-feature construction returned a non-current snapshot.")

    sequence = _current_sequence_fingerprints(observations, latest, cfg)
    sequence_tokens = set(sequence.token_key.astype(str)) if not sequence.empty else set()
    missing_sequence = sorted(expected_tokens - sequence_tokens)
    if missing_sequence:
        raise RuntimeError(
            "Current sequence construction omitted latest-capture tokens; refusing "
            f"partial prediction: missing={missing_sequence[:10]}"
        )
    return features.reset_index(drop=True), sequence.reset_index(drop=True), latest


def _current_feature_rows(
    conn: sqlite3.Connection,
    cfg: V24Config,
) -> tuple[pd.DataFrame, SequenceFingerprintSource]:
    frame, sequence, _ = _current_inference_inputs(conn, cfg)
    return frame, sequence


def _is_sqlite_locked(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and any(
        marker in str(exc).lower()
        for marker in ("database is locked", "database table is locked", "database is busy")
    )


def _run_live_write_with_retry(db: str, operation: Any) -> Any:
    """Run one short write transaction while giving the collector priority."""
    last_error: BaseException | None = None
    for attempt in range(LIVE_SQLITE_WRITE_RETRIES):
        conn = sqlite3.connect(db, timeout=LIVE_SQLITE_BUSY_TIMEOUT_MS / 1000.0)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={LIVE_SQLITE_BUSY_TIMEOUT_MS}")
        try:
            conn.execute("BEGIN IMMEDIATE")
            result = operation(conn)
            conn.commit()
            return result
        except BaseException as exc:
            try:
                conn.rollback()
            except sqlite3.DatabaseError:
                pass
            if not _is_sqlite_locked(exc) or attempt + 1 >= LIVE_SQLITE_WRITE_RETRIES:
                raise
            last_error = exc
        finally:
            conn.close()
        time.sleep(min(2.0, LIVE_SQLITE_RETRY_BASE_SECONDS * (2 ** attempt)))
    if last_error is not None:  # pragma: no cover - loop always returns or raises
        raise last_error
    raise RuntimeError("Live SQLite write retry loop exited unexpectedly.")


def _atomic_write_prediction_csv(frame: pd.DataFrame, out_path: str) -> None:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def predict_current(
    db: str,
    model_path: str,
    out_path: str,
    cfg: V24Config,
    *,
    persist_source: bool = True,
) -> pd.DataFrame:
    """Predict the exact latest capture and atomically publish its CSV.

    ``persist_source=False`` is reserved for isolated paper benchmarks. Those
    runs deliberately have training feedback disabled, so writing their
    prediction ledger, cohort assignments or heartbeat into the collector DB is
    both unnecessary and a source of writer-lock contention. Ordinary V24
    prediction commands retain source-provenance persistence by default.
    """
    bundle=joblib.load(model_path)
    if bundle.get("schema_version")!=SCHEMA_VERSION: raise RuntimeError("V24 model schema mismatch")

    for attempt in range(LIVE_PREDICTION_SNAPSHOT_RETRIES):
        with closing(_connect_live_read(db)) as conn:
            current, sequence, latest = _current_inference_inputs(conn, cfg)
            out = predict_frame(conn, bundle, current, sequence, cfg)

        observed_snapshots = set(pd.to_datetime(out.snapshot_at, format="ISO8601", utc=True))
        if observed_snapshots != {latest}:
            raise RuntimeError(
                "V24 prediction output did not preserve the exact latest-capture timestamp."
            )
        newest = _latest_observation_timestamp(db)
        if newest != latest:
            if attempt + 1 < LIVE_PREDICTION_SNAPSHOT_RETRIES:
                continue
            raise RuntimeError(
                "Raw collection advanced while V24 prediction was being calculated; "
                "refusing to publish a stale prediction CSV. Retry on the next cycle."
            )

        metadata = pd.DataFrame({
            "v24_model_hash": np.full(len(out), _hash_file(model_path), dtype=object),
            "v24_model_training_cutoff": np.full(
                len(out), _model_training_cutoff_from_bundle(bundle).isoformat(), dtype=object
            ),
        })
        out = pd.concat([out.reset_index(drop=True), metadata], axis=1, copy=False)

        if persist_source:
            def persist(conn: sqlite3.Connection) -> int:
                _ensure_live_token_assignments(conn, current.token_key.astype(str), cfg)
                _ensure_live_cohorts_through(conn, latest, cfg)
                _upsert_capture_heartbeat(
                    conn,
                    latest,
                    valid_capture=True,
                    row_count=len(current),
                    source="live_prediction",
                    details={"label_free_current_inference": True},
                )
                return record_live_prediction_frame(conn, out, model_path, bundle, cfg)

            _run_live_write_with_retry(db, persist)
        newest = _latest_observation_timestamp(db)
        if newest != latest:
            if attempt + 1 < LIVE_PREDICTION_SNAPSHOT_RETRIES:
                continue
            raise RuntimeError(
                "Raw collection advanced before V24 prediction publication; "
                "refusing to replace the last known-good prediction CSV."
            )
        _atomic_write_prediction_csv(out, out_path)
        return out

    raise RuntimeError("V24 prediction could not stabilize on one raw capture snapshot.")


# ---------------------------------------------------------------------------
# Prediction provenance and OOS-only policy training
# ---------------------------------------------------------------------------

def _token_assignment_status(conn:sqlite3.Connection,token_key:str)->dict[str,Any]:
    r=conn.execute(f"SELECT forecast_cohort_id,forecast_role,policy_cohort_id,policy_role FROM {TOKEN_ASSIGNMENT_TABLE} WHERE token_key=?",(str(token_key),)).fetchone()
    if not r: return {}
    fs=conn.execute(f"SELECT status FROM {COHORT_TABLE} WHERE cohort_id=?",(r[0],)).fetchone(); ps=conn.execute(f"SELECT status FROM {POLICY_COHORT_TABLE} WHERE cohort_id=?",(r[2],)).fetchone()
    return {"forecast_cohort_id":r[0],"forecast_role":r[1],"forecast_status":fs[0] if fs else None,"policy_cohort_id":r[2],"policy_role":r[3],"policy_status":ps[0] if ps else None}


def _development_eligible_assignment(info:dict[str,Any])->tuple[bool,str|None]:
    if not info: return False,"missing_token_lifetime_assignment"
    if info.get("forecast_role")=="audit" or info.get("policy_role")=="audit": return False,"sealed_audit_token"
    if info.get("forecast_role")=="promotion" and info.get("forecast_status")!="consumed": return False,"unused_forecast_promotion_token"
    if info.get("policy_role")=="promotion" and info.get("policy_status")!="consumed": return False,"unused_policy_promotion_token"
    return True,None


def record_prediction(
    conn: sqlite3.Connection,
    token_key: str,
    decision_at: Any,
    generated_at: Any,
    model_hash: str | None,
    training_cutoff: Any,
    prediction: dict[str, Any],
    provenance: str,
    cfg: V24Config,
    fold_id: str | None = None,
) -> bool:
    decision = _utc(decision_at)
    generated = _utc(generated_at)
    cutoff = _utc(training_cutoff)
    reason = None
    eligible = True
    if cutoff >= decision:
        eligible, reason = False, "forecaster_training_cutoff_not_before_decision"
    elif provenance == "live":
        lag = abs((generated - decision).total_seconds()) / 60.0
        if lag > cfg.max_live_prediction_lag_minutes:
            eligible, reason = False, "live_prediction_not_contemporaneous"
    elif provenance != "crossfit":
        eligible, reason = False, "unsupported_provenance"
    oos_valid=eligible
    info=_token_assignment_status(conn,str(token_key)); dev_ok,dev_reason=_development_eligible_assignment(info)
    if not dev_ok:
        eligible=False; reason=dev_reason
    pid = str(uuid.uuid4())
    try:
        conn.execute(
            f"""INSERT INTO {PREDICTION_LEDGER}
                (prediction_id,token_key,decision_at,generated_at,forecaster_model_hash,
                 forecaster_training_cutoff,provenance,fold_id,prediction_json,
                 policy_training_eligible,ineligibility_reason,target_definition_hash,feature_definition_hash,execution_definition_hash,data_vintage_hash,oos_valid)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                pid, str(token_key), decision.isoformat(), generated.isoformat(), model_hash,
                cutoff.isoformat(), provenance, fold_id, _json(prediction), int(eligible), reason,
                prediction.get("__target_definition_hash"),prediction.get("__feature_definition_hash"),prediction.get("__execution_definition_hash"),prediction.get("__data_vintage_hash"),int(oos_valid),
            ),
        )
        return eligible
    except sqlite3.IntegrityError:
        return False


def record_live_prediction_frame(conn: sqlite3.Connection, pred: pd.DataFrame, model_path: str, bundle: dict[str, Any], cfg: V24Config) -> int:
    model_hash = _hash_file(model_path)
    now = pd.Timestamp.now(tz="UTC")
    stored = 0
    for r in pred.to_dict("records"):
        token = r.pop("token_key")
        decision = r.pop("snapshot_at")
        r["__target_definition_hash"]=bundle.get("target_definition_hash"); r["__feature_definition_hash"]=bundle.get("feature_definition_hash"); r["__execution_definition_hash"]=bundle.get("execution_definition_hash"); r["__data_vintage_hash"]=bundle.get("training_data_hash")
        if record_prediction(
            conn, token, decision, now, model_hash, _model_training_cutoff_from_bundle(bundle),
            r, "live", cfg,
        ):
            stored += 1
    return stored


def eligible_oos_predictions(conn:sqlite3.Connection)->pd.DataFrame:
    """Dynamically unlock OOS predictions only after both one-use reserve roles are consumed."""
    df=pd.read_sql_query(f"SELECT * FROM {PREDICTION_LEDGER} WHERE oos_valid=1 ORDER BY decision_at",conn)
    if df.empty: return df
    assign=_token_assignments(conn); fc=_cohort_rows(conn)[["cohort_id","status"]].rename(columns={"cohort_id":"forecast_cohort_id","status":"forecast_status"}); pc=pd.read_sql_query(f"SELECT cohort_id,status FROM {POLICY_COHORT_TABLE}",conn).rename(columns={"cohort_id":"policy_cohort_id","status":"policy_status"})
    x=df.merge(assign,on="token_key",how="left").merge(fc,on="forecast_cohort_id",how="left").merge(pc,on="policy_cohort_id",how="left")
    ok=(x.forecast_role.ne("audit")&x.policy_role.ne("audit")&(~x.forecast_role.eq("promotion")|x.forecast_status.eq("consumed"))&(~x.policy_role.eq("promotion")|x.policy_status.eq("consumed")))
    x=x[ok].copy(); x["decision_at"]=pd.to_datetime(x.decision_at,format="ISO8601", utc=True); x["generated_at"]=pd.to_datetime(x.generated_at,format="ISO8601", utc=True)
    return x


def _model_training_cutoff_from_bundle(bundle: dict[str, Any]) -> pd.Timestamp:
    cutoffs = [_utc(bundle["stable_training_cutoff"])]
    adapter = bundle.get("adapter") or {}
    if adapter.get("training_cutoff"):
        cutoffs.append(_utc(adapter["training_cutoff"]))
    return max(cutoffs)


# ---------------------------------------------------------------------------
# Evaluation and one-use promotion
# ---------------------------------------------------------------------------

def _cohort_frame(frame: pd.DataFrame, cohort: sqlite3.Row) -> pd.DataFrame:
    # Membership is token-birth based, not decision-time based.  This returns the
    # entire observed lifetime of tokens assigned to the one-use cohort.
    if "forecast_cohort_id" in frame.columns:
        return frame[frame.forecast_cohort_id.astype(str)==str(cohort["cohort_id"])].copy()
    return frame.iloc[0:0].copy()


def _required_promotion_components(cfg:V24Config)->list[str]:
    comps=[]
    for h in cfg.promotion_required_horizons_minutes:
        comps += [f"peak_brier_{h}",f"death_brier_{h}"]
    for thr in cfg.promotion_required_thresholds:
        for h in cfg.promotion_required_horizons_minutes:
            comps.append(f"barrier_brier_{_threshold_tag(thr)}_{h}")
    return comps


def _token_promotion_losses(conn:sqlite3.Connection,bundle:dict[str,Any],frame:pd.DataFrame,seqraw:SequenceFingerprintSource,cfg:V24Config)->pd.DataFrame:
    if frame.empty:return pd.DataFrame()
    model_frame,_=_prepare_model_frame(conn,frame,seqraw,cfg,encoder=bundle["sequence_encoder"],sequence_challenger=bundle.get("sequence_challenger"))
    truth=add_barrier_targets(conn,model_frame,cfg); pred=predict_frame(conn,bundle,frame,seqraw,cfg)
    rows=[]
    for pos,(idx,r) in enumerate(model_frame.iterrows()):
        token=str(r.token_key); decision=_utc(r.decision_at); event,event_min,known=first_competing_event(r,cfg)
        rec={"token_key":token}
        for h in cfg.promotion_required_horizons_minutes:
            # If the row is censored before h without an event, this component is unknown.
            if event!=EVENT_NONE and event_min<=h:
                yp=1.0 if event==EVENT_PEAK else 0.0; yd=1.0 if event==EVENT_DEATH else 0.0
            elif known and event_min>=h:
                yp=0.0; yd=0.0
            else:
                yp=yd=np.nan
            for name,y,col in ((f"peak_brier_{h}",yp,f"p_first_peak_by_{h}m"),(f"death_brier_{h}",yd,f"p_death_by_{h}m")):
                pv=_finite(pred.iloc[pos].get(col)) if pos < len(pred) else None
                rec[name]=(pv-y)**2 if pv is not None and np.isfinite(y) else np.nan
        for thr in cfg.promotion_required_thresholds:
            for h in cfg.promotion_required_horizons_minutes:
                target=f"hit_plus{_threshold_tag(thr)}_by_{h}m"; col=f"p_{target}"; y=_finite(truth.iloc[idx].get(target)); pv=_finite(pred.iloc[pos].get(col)) if pos < len(pred) else None
                rec[f"barrier_brier_{_threshold_tag(thr)}_{h}"]=(pv-y)**2 if pv is not None and y is not None else np.nan
        rows.append(rec)
    d=pd.DataFrame(rows)
    if d.empty:return d
    # First collapse correlated minute decisions inside each token.
    return d.groupby("token_key",as_index=False).mean(numeric_only=True)


def evaluate_bundle(conn:sqlite3.Connection,bundle:dict[str,Any],frame:pd.DataFrame,seqraw:SequenceFingerprintSource,cfg:V24Config)->dict[str,Any]:
    losses=_token_promotion_losses(conn,bundle,frame,seqraw,cfg)
    if losses.empty:return {"available":False,"reason":"empty evaluation frame"}
    required=_required_promotion_components(cfg); missing=[c for c in required if c not in losses.columns or not np.isfinite(pd.to_numeric(losses[c],errors="coerce")).any()]
    if missing:return {"available":False,"reason":"missing required promotion coverage","missing_components":missing,"tokens":int(len(losses))}
    # Fixed metric coverage: a model cannot improve by omitting a difficult head.
    valid=losses[["token_key"]+required].copy()
    valid[required]=valid[required].apply(pd.to_numeric,errors="coerce")
    token_score=valid[required].mean(axis=1,skipna=False)
    valid["composite_error"]=token_score
    valid=valid[np.isfinite(valid.composite_error)].copy()
    if valid.empty:return {"available":False,"reason":"no tokens have complete fixed metric coverage","tokens":0}
    return {"available":True,"composite_error":float(valid.composite_error.mean()),"tokens":int(len(valid)),"rows":int(len(frame)),
            "required_components":required,"token_losses":dict(zip(valid.token_key.astype(str),valid.composite_error.astype(float))),
            "component_means":{c:float(valid[c].mean()) for c in required}}


def compare_promotion(candidate:dict[str,Any],champion:dict[str,Any],cfg:V24Config)->tuple[bool,str]:
    if not candidate.get("available"):return False,f"candidate unavailable: {candidate.get('reason')}"
    if not champion.get("available"):return False,"champion unavailable on fixed required promotion contract; refuse automatic promotion"
    c=candidate.get("token_losses",{}); h=champion.get("token_losses",{}); common=sorted(set(c)&set(h))
    if len(common)<cfg.promotion_min_tokens:return False,f"only {len(common)} independent evaluation tokens; need {cfg.promotion_min_tokens}"
    # Positive diff means candidate has lower error / is better.
    diff=pd.Series({k:float(h[k])-float(c[k]) for k in common},dtype=float)
    boot=_paired_token_bootstrap(diff,cfg.promotion_bootstrap_samples,cfg.promotion_confidence,seed=9137)
    rel=float(boot["mean"])/max(abs(float(np.mean([h[k] for k in common]))),1e-12)
    # Material-degradation guard is computed on the same fixed components.
    degraded=[]
    for comp in candidate.get("required_components",[]):
        cv=candidate.get("component_means",{}).get(comp); hv=champion.get("component_means",{}).get(comp)
        if cv is not None and hv not in (None,0) and float(cv)/float(hv)>1.0+cfg.max_material_degradation: degraded.append(comp)
    promoted=bool(boot["ci_low"]>0.0 and rel>=cfg.promotion_margin and not degraded)
    reason=f"paired_token_error_gain={boot['mean']:.7f}; relative_gain={rel:.5f}; ci=[{boot['ci_low']:.7f},{boot['ci_high']:.7f}]; p_better={boot['p_better']:.4f}; n={boot['n_tokens']}; degraded={','.join(degraded) or 'none'}"
    return promoted,reason


def _save_bundle(bundle: dict[str, Any], root: str, name: str) -> str:
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return str(path)


def _register_model(conn:sqlite3.Connection,path:str,bundle:dict[str,Any],status:str,metrics:dict[str,Any],notes:str="")->str:
    version=str(uuid.uuid4())
    conn.execute(
        f"""INSERT INTO {MODEL_REGISTRY}
        (version_id,created_at,model_path,model_hash,status,stable_training_cutoff,stable_generation,adapter_round,
         adapter_training_start,adapter_training_cutoff,metrics_json,notes,stable_created_at,last_compaction_at,
         target_definition_hash,feature_definition_hash,execution_definition_hash,training_data_hash)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (version,_now_iso(),path,_hash_file(path),status,bundle["stable_training_cutoff"],int(bundle.get("stable_generation",1)),
         int(bundle.get("adapter_round",0)),(bundle.get("adapter") or {}).get("training_start"),(bundle.get("adapter") or {}).get("training_cutoff"),
         _json(metrics),notes,bundle.get("stable_created_at"),bundle.get("last_compaction_at"),bundle.get("target_definition_hash"),
         bundle.get("feature_definition_hash"),bundle.get("execution_definition_hash"),bundle.get("training_data_hash")),
    ); return version


def bootstrap_v24(db: str, model_root: str, cfg: V24Config, *, allow_small: bool = False) -> dict[str, Any]:
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.row_factory = sqlite3.Row
        peak_cfg = peak.PeakStructureConfig(
            horizon_minutes=cfg.horizon_minutes,
            death_gap_minutes=cfg.operational_gap_minutes,
            death_missed_cycles=int(cfg.operational_gap_minutes),
            age_out_minutes=cfg.age_out_minutes,
        )
        peak.refresh_labels(db, peak_cfg)
        frame, seqraw, source = load_v24_frame(conn, cfg)
        cohort = next_one_use_promotion_cohort(conn, cfg)
        if cohort is None:
            raise RuntimeError("No fully matured one-use promotion cohort is available for V24 bootstrap.")
        cutoff = _utc(cohort["start_at"])
        eval_frame = _cohort_frame(frame, cohort)
        eval_tokens = set(eval_frame.token_key.astype(str))
        bundle = fit_batch_bundle(conn, frame, seqraw, cutoff, cfg, allow_small=allow_small, generation=1, exclude_tokens=eval_tokens)
        cand_eval = evaluate_bundle(conn, bundle, eval_frame, seqraw, cfg)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = _save_bundle(bundle, model_root, f"challengers/v24_bootstrap_{stamp}.joblib")
        champion = Path(model_root) / "champion.joblib"
        champion.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, champion)
        promotion_id = str(uuid.uuid4())
        conn.execute(
            f"""INSERT INTO {PROMOTION_TABLE}
                (promotion_id,created_at,cohort_id,candidate_path,candidate_hash,
                 champion_before_path,champion_before_hash,promoted,metrics_json,reason)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (promotion_id, _now_iso(), cohort["cohort_id"], candidate, _hash_file(candidate), None, None, 1, _json({"candidate": cand_eval}), "first V24 bootstrap"),
        )
        conn.execute(
            f"UPDATE {COHORT_TABLE} SET status='consumed',consumed_at=?,promotion_id=? WHERE cohort_id=?",
            (_now_iso(), promotion_id, cohort["cohort_id"]),
        )
        _register_model(conn, str(champion), bundle, "champion", cand_eval, "V24 bootstrap")
        conn.commit()
        return {
            "bootstrapped": True, "champion": str(champion), "cohort": cohort["cohort_id"],
            "evaluation": cand_eval, "sources": source,
        }


def maintain_v24(db: str, model_root: str, cfg: V24Config, *, allow_small: bool = False, force_compaction: bool = False) -> dict[str, Any]:
    champion_path = Path(model_root) / "champion.joblib"
    if not champion_path.exists():
        return bootstrap_v24(db, model_root, cfg, allow_small=allow_small)
    champion = joblib.load(champion_path)
    if champion.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Existing champion is not V24-compatible; run bootstrap-v24 into a clean V24 model directory.")
    current_target_hash=target_definition_hash(cfg); current_execution_hash=execution_definition_hash(cfg)
    if champion.get("target_definition_hash") != current_target_hash:
        raise RuntimeError("V24 target-definition hash changed. Refusing warm adapter/promotion comparison; bootstrap a clean model generation.")
    if champion.get("execution_definition_hash") != current_execution_hash:
        raise RuntimeError("V24 execution-definition hash changed. Refusing warm adapter/promotion comparison; bootstrap a clean model generation.")
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.row_factory = sqlite3.Row
        peak_cfg = peak.PeakStructureConfig(
            horizon_minutes=cfg.horizon_minutes,
            death_gap_minutes=cfg.operational_gap_minutes,
            death_missed_cycles=int(cfg.operational_gap_minutes),
            age_out_minutes=cfg.age_out_minutes,
        )
        peak.refresh_labels(db, peak_cfg)
        frame, seqraw, _ = load_v24_frame(conn, cfg)
        # Calibration learning is label-dependent maintenance work.  Keep it out
        # of the minute-sensitive live prediction path so collection retains
        # priority and current inference never depends on label freshness.
        update_adaptive_calibration(conn, frame, cfg)
        cohort = next_one_use_promotion_cohort(conn, cfg)
        if cohort is None:
            return {"trained": False, "reason": "no fully matured unused promotion cohort"}
        cutoff = _utc(cohort["start_at"])
        if _model_training_cutoff_from_bundle(champion) >= cutoff:
            raise RuntimeError("Champion training cutoff is not earlier than the next promotion cohort; refusing adaptive holdout leakage.")

        eval_frame = _cohort_frame(frame, cohort)
        eval_tokens = set(eval_frame.token_key.astype(str))
        compact = force_compaction or should_compact(champion, cfg, cutoff)
        if compact:
            candidate_bundle = fit_batch_bundle(
                conn, frame, seqraw, cutoff, cfg, allow_small=allow_small,
                generation=int(champion.get("stable_generation", 1)) + 1,
                exclude_tokens=eval_tokens,
            )
            mode = "periodic_compaction_full_refit"
        else:
            candidate_bundle = dict(champion)
            candidate_bundle["adapter"] = fit_online_adapter(
                conn, champion, frame, seqraw, cutoff, cfg, allow_small=allow_small, exclude_tokens=eval_tokens
            )
            candidate_bundle["adapter_round"] = int(champion.get("adapter_round", 0)) + 1
            candidate_bundle["created_at"] = _now_iso()
            mode = "stable_plus_online_adapter"

        cand_eval = evaluate_bundle(conn, candidate_bundle, eval_frame, seqraw, cfg)
        champ_eval = evaluate_bundle(conn, champion, eval_frame, seqraw, cfg)
        promoted, reason = compare_promotion(cand_eval, champ_eval, cfg)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate_path = _save_bundle(candidate_bundle, model_root, f"challengers/v24_{stamp}.joblib")
        before_hash = _hash_file(champion_path)
        if promoted:
            shutil.copy2(candidate_path, champion_path)
        promotion_id = str(uuid.uuid4())
        metrics = {"candidate": cand_eval, "champion_before": champ_eval, "mode": mode}
        conn.execute(
            f"""INSERT INTO {PROMOTION_TABLE}
                (promotion_id,created_at,cohort_id,candidate_path,candidate_hash,
                 champion_before_path,champion_before_hash,promoted,metrics_json,reason)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                promotion_id, _now_iso(), cohort["cohort_id"], candidate_path,
                _hash_file(candidate_path), str(champion_path), before_hash,
                int(promoted), _json(metrics), reason,
            ),
        )
        # One-use means consumed whether candidate wins or loses.
        conn.execute(
            f"UPDATE {COHORT_TABLE} SET status='consumed',consumed_at=?,promotion_id=? WHERE cohort_id=?",
            (_now_iso(), promotion_id, cohort["cohort_id"]),
        )
        _register_model(
            conn, str(champion_path if promoted else candidate_path), candidate_bundle,
            "champion" if promoted else "rejected", metrics, reason,
        )
        conn.commit()
        return {
            "trained": True, "mode": mode, "promoted": promoted,
            "cohort": cohort["cohort_id"], "candidate": candidate_path,
            "champion": str(champion_path), "reason": reason, "metrics": metrics,
        }


# ---------------------------------------------------------------------------
# Cross-fitted historical predictions for policy training
# ---------------------------------------------------------------------------

def crossfit_policy_predictions(db:str,cfg:V24Config,*,max_folds:int=5,allow_small:bool=False)->dict[str,Any]:
    """Strict rolling-origin OOS forecasts with historical data-vintage enforcement."""
    stored=folds_used=0
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.row_factory=sqlite3.Row; frame,seqraw,_=load_v24_frame(conn,cfg); assignments=_token_assignments(conn)
        # Only blocks whose token roles are presently development-eligible may be replayed.
        blocks=[]
        for b in sorted(int(x) for x in frame.calendar_cohort_ordinal.unique() if int(x)>=cfg.warmup_blocks):
            toks=assignments[assignments.birth_ordinal==b].token_key.astype(str).tolist() if not assignments.empty else []
            if not toks: continue
            if any(_development_eligible_assignment(_token_assignment_status(conn,t))[0] for t in toks): blocks.append(b)
        for b in blocks[-max_folds:]:
            te=frame[frame.calendar_cohort_ordinal==b].copy(); te=te[te.token_key.astype(str).map(lambda t:_development_eligible_assignment(_token_assignment_status(conn,t))[0])]
            if te.empty: continue
            cohort=conn.execute(f"SELECT * FROM {COHORT_TABLE} WHERE ordinal=?",(b,)).fetchone(); test_start=_utc(cohort['start_at']) if cohort else te.snapshot_at.min(); test_tokens=set(te.token_key.astype(str))
            # Historical values repaired after the test start cannot be used to recreate a past prediction.
            te=te[_vintage_known_by(te,test_start+pd.Timedelta(minutes=cfg.max_live_prediction_lag_minutes))].copy()
            if te.empty: continue
            try:
                fold_bundle=fit_batch_bundle(conn,frame,seqraw,test_start,cfg,allow_small=allow_small,generation=0,exclude_tokens=test_tokens)
            except MemoryError:
                raise
            except Exception:
                continue
            pred=predict_frame(conn,fold_bundle,te,seqraw,cfg); model_hash=bundle_identity_hash(fold_bundle); fold_id=f"rolling_oos_birthblock_{b:06d}"
            for r in pred.to_dict('records'):
                token=r.pop('token_key'); decision=r.pop('snapshot_at'); r['__target_definition_hash']=fold_bundle.get('target_definition_hash'); r['__feature_definition_hash']=fold_bundle.get('feature_definition_hash'); r['__execution_definition_hash']=fold_bundle.get('execution_definition_hash'); r['__data_vintage_hash']=fold_bundle.get('training_data_hash')
                if record_prediction(conn,token,decision,decision,model_hash,_model_training_cutoff_from_bundle(fold_bundle),r,'crossfit',cfg,fold_id=fold_id): stored+=1
            folds_used+=1
        conn.commit()
    return {'stored_oos_predictions':stored,'folds':folds_used,'mode':'strict_rolling_origin_lifetime_vintage_locked'}


# ---------------------------------------------------------------------------
# Distributional policy objective using only OOS prediction provenance
# ---------------------------------------------------------------------------

def _prediction_state_frame(rows: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    states = [_loads(x) for x in rows.prediction_json]
    if features is None:
        features = sorted({k for d in states for k, v in d.items() if _finite(v) is not None})
    X = pd.DataFrame([{c: _finite(d.get(c)) for c in features} for d in states], columns=features, dtype=float)
    return X, features


def _fit_distribution_head(rows: pd.DataFrame, target: str, cfg: V24Config, allow_small: bool) -> dict[str, Any] | None:
    data = rows.copy()
    if data.empty or target not in data.columns:
        return None
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data[data[target].notna() & np.isfinite(data[target])].copy()
    if len(data) < (20 if allow_small else cfg.policy_min_rows):
        return None
    X, features = _prediction_state_frame(data)
    if not features:
        return None
    heads = {}
    n_estimators = cfg.small_estimators if allow_small else 350
    for q in cfg.policy_quantiles:
        fitted = []
        for name, m in _reg_components(n_estimators, quantile=q):
            try:
                m.fit(X, data[target].to_numpy(dtype=float), sample_weight=_token_weights(data.token_key))
                fitted.append((name, m))
            except MemoryError:
                raise
            except Exception:
                continue
        if fitted:
            heads[f"q{int(q*100):02d}"] = {"models": fitted, "quantile": q}
    if not heads:
        return None
    return {"kind": "distributional_return", "features": features, "quantile_heads": heads, "target": target, "tail_risk_weight": cfg.policy_tail_risk_weight, "upside_weight": cfg.policy_upside_weight}


def predict_return_distribution(head: dict[str, Any], states: list[dict[str, Any]], cfg: V24Config) -> dict[str, np.ndarray]:
    features = head["features"]
    X = pd.DataFrame([{c: _finite(s.get(c)) for c in features} for s in states], columns=features, dtype=float)
    out = {}
    ordered = []
    names = []
    for name, qh in sorted(head["quantile_heads"].items(), key=lambda kv: kv[1]["quantile"]):
        preds = []
        for _, m in qh["models"]:
            preds.append(np.asarray(m.predict(X), dtype=float))
        p = np.mean(preds, axis=0) if preds else np.full(len(X), np.nan)
        ordered.append(p)
        names.append(name)
    if ordered:
        mat = np.sort(np.column_stack(ordered), axis=1)
        for j, name in enumerate(names):
            out[name] = mat[:, j]
        # Lower-tail mean approximates CVaR; use q05/q10 where available.
        tail_cols = [out[x] for x in ("q05", "q10") if x in out]
        tail = np.mean(tail_cols, axis=0) if tail_cols else out[names[0]]
        median = out.get("q50", out[names[len(names)//2]])
        upper = out.get("q75", out[names[min(len(names)-1, len(names)//2 + 1)]])
        downside = np.maximum(0.0, -tail)
        score = median + cfg.policy_upside_weight * np.maximum(0.0, upper - median) - cfg.policy_tail_risk_weight * downside
        out["risk_score"] = score
        out["cvar_proxy"] = tail
    return out


def next_one_use_policy_cohort(conn: sqlite3.Connection, cfg: V24Config) -> sqlite3.Row | None:
    conn.row_factory=sqlite3.Row
    latest=_latest_capture(conn)
    if latest is None: return None
    deadline=latest-pd.Timedelta(minutes=cfg.horizon_minutes+max(cfg.counterfactual_horizons_minutes))
    return conn.execute(
        f"SELECT * FROM {POLICY_COHORT_TABLE} WHERE role='promotion' AND status='available' AND end_at<=? ORDER BY ordinal LIMIT 1",
        (deadline.isoformat(),),
    ).fetchone()


def refresh_counterfactual_policy_targets(conn:sqlite3.Connection,cfg:V24Config)->dict[str,int]:
    """Build behavior-independent ENTRY and HOLD targets from the observed future path.

    ENTRY uses the first observation strictly after the decision as the hypothetical
    executable fill. HOLD/EXIT decisions compare the future hold value against the
    next-observation EXIT fill. Targets are written only once their horizon is
    actually observed or an earlier V24 terminal event makes the path complete.
    """
    migrate(conn)
    from . import axiom_self_teach as selfteach
    selfteach.migrate(conn)
    obs,_=peak.load_observations(conn)
    if obs.empty: return {"written":0}
    obs=obs[["token_key","snapshot_at","market_cap_usd"]].dropna().copy()
    obs["token_key"]=obs.token_key.astype(str); obs["snapshot_at"]=pd.to_datetime(obs.snapshot_at,format="ISO8601", utc=True)
    by={k:g.sort_values("snapshot_at") for k,g in obs.groupby("token_key")}
    life=pd.read_sql_query(f"SELECT token_key,terminal_at FROM {LIFETIME_TABLE} WHERE terminal_at IS NOT NULL",conn)
    terminal_by={}
    if not life.empty:
        life["terminal_at"]=pd.to_datetime(life.terminal_at,format="ISO8601", utc=True)
        for k,g in life.groupby("token_key"): terminal_by[str(k)]=list(g.terminal_at.sort_values())
    sources=[]
    if _table_exists(conn,"axiom_paper_candidates_v20"):
        c=pd.read_sql_query("SELECT token_key,snapshot_at AS decision_at,market_cap_usd AS decision_mc FROM axiom_paper_candidates_v20",conn)
        c["action_kind"]="entry"; sources.append(c)
    if _table_exists(conn,"axiom_paper_marks_v20"):
        m=pd.read_sql_query("SELECT token_key,snapshot_at AS decision_at,market_cap_usd AS decision_mc FROM axiom_paper_marks_v20 WHERE COALESCE(price_available,1)=1",conn)
        m["action_kind"]="hold"; sources.append(m)
    if not sources: return {"written":0}
    src=pd.concat(sources,ignore_index=True).drop_duplicates(["token_key","decision_at","action_kind"])
    src["decision_at"]=pd.to_datetime(src.decision_at,format="ISO8601", utc=True); src["decision_mc"]=pd.to_numeric(src.decision_mc,errors="coerce")
    written=0; now=_now_iso()
    for r in src.itertuples(index=False):
        token=str(r.token_key); decision=_utc(r.decision_at); dmc=float(r.decision_mc) if np.isfinite(r.decision_mc) and r.decision_mc>0 else None
        if dmc is None or token not in by: continue
        g=by[token]; future=g[g.snapshot_at>decision]
        if future.empty: continue
        first=future.iloc[0]; fill_at=_utc(first.snapshot_at); fill_mc=float(first.market_cap_usd)
        for h in cfg.counterfactual_horizons_minutes:
            deadline=decision+pd.Timedelta(minutes=int(h))
            path=future[future.snapshot_at<=deadline]
            if path.empty: continue
            term_candidates=[t for t in terminal_by.get(token,[]) if decision<t<=deadline]
            terminal=min(term_candidates) if term_candidates else None
            complete=bool(g.snapshot_at.max()>=deadline or terminal is not None)
            if not complete: continue
            end_at=min(deadline,terminal) if terminal is not None else deadline
            usable=path[path.snapshot_at<=end_at]
            if usable.empty: continue
            last=usable.iloc[-1]; term_mc=float(last.market_cap_usd)
            prices=pd.to_numeric(usable.market_cap_usd,errors="coerce").dropna().to_numpy(dtype=float)
            if not len(prices): continue
            exit_now=fill_mc/dmc-1.0
            # ENTRY return is measured from the only executable entry fill.
            entry_ret=term_mc/fill_mc-1.0 if fill_mc>0 else np.nan
            # HOLD value is relative to current position value; advantage subtracts
            # the next-observation EXIT fill so it answers HOLD versus EXIT NOW.
            hold_ret=term_mc/dmc-1.0
            hold_adv=(1.0+hold_ret)/max(1e-9,1.0+exit_now)-1.0
            best=float(np.max(prices)/dmc-1.0); worst=float(np.min(prices)/dmc-1.0)
            fp=_stable_hash([token,decision.isoformat(),str(r.action_kind),int(h),fill_at.isoformat(),float(fill_mc),float(term_mc),end_at.isoformat()])
            conn.execute(
                f"""INSERT INTO {COUNTERFACTUAL_TABLE}
                (token_key,decision_at,action_kind,horizon_minutes,decision_mc,next_observed_at,next_observed_mc,
                 exit_now_return,hold_terminal_return,hold_best_return,hold_worst_return,target_ready_at,
                 source_fingerprint,created_at,hold_advantage_return,entry_execution_return)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(token_key,decision_at,action_kind,horizon_minutes) DO UPDATE SET
                 decision_mc=excluded.decision_mc,next_observed_at=excluded.next_observed_at,next_observed_mc=excluded.next_observed_mc,
                 exit_now_return=excluded.exit_now_return,hold_terminal_return=excluded.hold_terminal_return,
                 hold_best_return=excluded.hold_best_return,hold_worst_return=excluded.hold_worst_return,
                 target_ready_at=excluded.target_ready_at,source_fingerprint=excluded.source_fingerprint,
                 hold_advantage_return=excluded.hold_advantage_return,entry_execution_return=excluded.entry_execution_return""",
                (token,decision.isoformat(),str(r.action_kind),int(h),dmc,fill_at.isoformat(),fill_mc,exit_now,hold_ret,best,worst,
                 end_at.isoformat(),fp,now,hold_adv,entry_ret),
            ); written+=1
    conn.commit(); return {"written":written}


def _policy_ledger_for_role(conn:sqlite3.Connection,*,eval_cohort_id:str|None=None)->pd.DataFrame:
    if eval_cohort_id is None:
        return eligible_oos_predictions(conn)
    df=pd.read_sql_query(f"SELECT * FROM {PREDICTION_LEDGER} WHERE oos_valid=1 ORDER BY decision_at",conn)
    if df.empty: return df
    a=_token_assignments(conn)[["token_key","policy_cohort_id","policy_role","forecast_cohort_id","forecast_role"]]
    fc=_cohort_rows(conn)[["cohort_id","status"]].rename(columns={"cohort_id":"forecast_cohort_id","status":"forecast_status"})
    x=df.merge(a,on="token_key",how="left").merge(fc,on="forecast_cohort_id",how="left")
    forecast_dev=x.forecast_role.eq("train") | (x.forecast_role.eq("promotion") & x.forecast_status.eq("consumed"))
    return x[(x.policy_cohort_id.astype(str)==str(eval_cohort_id)) & x.policy_role.eq("promotion") & forecast_dev].copy()


def _entry_policy_training_rows(conn:sqlite3.Connection,*,eval_cohort_id:str|None=None,horizon:int=60)->pd.DataFrame:
    led=_policy_ledger_for_role(conn,eval_cohort_id=eval_cohort_id)
    if led.empty: return pd.DataFrame()
    cf=pd.read_sql_query(f"SELECT * FROM {COUNTERFACTUAL_TABLE} WHERE action_kind='entry' AND horizon_minutes=? AND entry_execution_return IS NOT NULL",conn,params=(int(horizon),))
    if cf.empty: return pd.DataFrame()
    cf["decision_at"]=pd.to_datetime(cf.decision_at,format="ISO8601", utc=True); led["decision_at"]=pd.to_datetime(led.decision_at,format="ISO8601", utc=True)
    out=led.merge(cf,on=["token_key","decision_at"],how="inner"); out["target"]=pd.to_numeric(out.entry_execution_return,errors="coerce")
    return out


def _hold_policy_training_rows(conn:sqlite3.Connection,*,eval_cohort_id:str|None=None,horizon:int=60)->pd.DataFrame:
    led=_policy_ledger_for_role(conn,eval_cohort_id=eval_cohort_id)
    if led.empty: return pd.DataFrame()
    cf=pd.read_sql_query(f"SELECT * FROM {COUNTERFACTUAL_TABLE} WHERE action_kind='hold' AND horizon_minutes=? AND hold_advantage_return IS NOT NULL",conn,params=(int(horizon),))
    if cf.empty: return pd.DataFrame()
    cf["decision_at"]=pd.to_datetime(cf.decision_at,format="ISO8601", utc=True); led["decision_at"]=pd.to_datetime(led.decision_at,format="ISO8601", utc=True)
    out=led.merge(cf,on=["token_key","decision_at"],how="inner"); out["target"]=pd.to_numeric(out.hold_advantage_return,errors="coerce")
    return out


def _policy_action_scores(head:dict[str,Any]|None,rows:pd.DataFrame,cfg:V24Config)->np.ndarray:
    if head is None or rows.empty: return np.zeros(len(rows),dtype=float)
    states=[_loads(x) for x in rows.prediction_json]
    dist=predict_return_distribution(head,states,cfg)
    return np.asarray(dist.get("risk_score",np.zeros(len(rows))),dtype=float)


def _paired_token_bootstrap(diff_by_token:pd.Series,samples:int,confidence:float,seed:int=1729)->dict[str,float]:
    vals=pd.to_numeric(diff_by_token,errors="coerce").dropna().to_numpy(dtype=float)
    if not len(vals): return {"n_tokens":0,"mean":float('nan'),"ci_low":float('nan'),"ci_high":float('nan'),"p_better":0.0}
    rng=np.random.default_rng(seed); means=np.empty(max(100,int(samples)),dtype=float)
    for i in range(len(means)): means[i]=float(np.mean(rng.choice(vals,size=len(vals),replace=True)))
    alpha=max(0.0,min(0.5,(1.0-float(confidence))/2.0))
    return {"n_tokens":int(len(vals)),"mean":float(np.mean(vals)),"ci_low":float(np.quantile(means,alpha)),
            "ci_high":float(np.quantile(means,1-alpha)),"p_better":float(np.mean(means>0.0))}


def evaluate_policy_bundle(bundle:dict[str,Any]|None,entry:pd.DataFrame,hold:pd.DataFrame,cfg:V24Config)->dict[str,Any]:
    token_parts=[]
    for rows,head_name in ((entry,"entry_head"),(hold,"hold_head")):
        if rows.empty: continue
        scores=_policy_action_scores(bundle.get(head_name) if bundle else None,rows,cfg)
        target=pd.to_numeric(rows.target,errors="coerce").to_numpy(dtype=float)
        # Value is incremental return versus PASS/EXIT baseline (zero).
        value=np.where(scores>0.0,target,0.0)
        tmp=pd.DataFrame({"token_key":rows.token_key.astype(str),"value":value})
        token_parts.append(tmp.groupby("token_key",as_index=False).value.mean())
    if not token_parts: return {"available":False,"reason":"no evaluable policy targets"}
    allv=pd.concat(token_parts,ignore_index=True).groupby("token_key").value.mean()
    return {"available":True,"tokens":int(len(allv)),"mean_value":float(allv.mean()),"token_values":allv.to_dict()}


def doubly_robust_entry_ope(conn:sqlite3.Connection,bundle:dict[str,Any],cfg:V24Config,horizon:int=60)->dict[str,Any]:
    """Behavior-propensity-aware diagnostic for the ENTRY policy.

    Counterfactual all-candidate targets drive supervised fitting. DR/OPE remains
    an independent check against the actual behavior-policy ledger.
    """
    if not _table_exists(conn,"axiom_paper_candidates_v20"): return {"available":False}
    c=pd.read_sql_query("SELECT * FROM axiom_paper_candidates_v20 WHERE action_probability IS NOT NULL",conn)
    if c.empty: return {"available":False}
    c["decision_at"]=pd.to_datetime(c.snapshot_at,format="ISO8601", utc=True)
    cf=pd.read_sql_query(f"SELECT token_key,decision_at,entry_execution_return FROM {COUNTERFACTUAL_TABLE} WHERE action_kind='entry' AND horizon_minutes=?",conn,params=(int(horizon),))
    if cf.empty:return {"available":False}
    cf["decision_at"]=pd.to_datetime(cf.decision_at,format="ISO8601", utc=True); x=c.merge(cf,on=["token_key","decision_at"],how="inner")
    led=eligible_oos_predictions(conn)[["token_key","decision_at","prediction_json"]]
    x=x.merge(led,on=["token_key","decision_at"],how="inner")
    if x.empty:return {"available":False}
    score=_policy_action_scores(bundle.get("entry_head"),x,cfg); pi=(score>0).astype(int); a=pd.to_numeric(x.chosen,errors="coerce").fillna(0).astype(int).to_numpy()
    p=np.clip(pd.to_numeric(x.action_probability,errors="coerce").fillna(cfg.propensity_floor).to_numpy(dtype=float),cfg.propensity_floor,1-cfg.propensity_floor)
    mu=np.where(a==1,p,1-p); q1=score; q0=np.zeros(len(x)); qpi=np.where(pi==1,q1,q0); qa=np.where(a==1,q1,q0)
    # Actual behavior reward uses conservative closed-trade return when available; PASS=0.
    pos=pd.read_sql_query("SELECT token_key,entry_decision_at,COALESCE(execution_net_return_pct,net_return_pct) reward FROM axiom_paper_positions_v20 WHERE status='closed' AND entry_decision_at IS NOT NULL",conn)
    reward=np.zeros(len(x),dtype=float)
    if not pos.empty:
        pos["decision_at"]=pd.to_datetime(pos.entry_decision_at,format="ISO8601", utc=True); rm={(str(r.token_key),_utc(r.decision_at)):float(r.reward) for r in pos.itertuples(index=False) if _finite(r.reward) is not None}
        for i,r in enumerate(x.itertuples(index=False)):
            if a[i]==1: reward[i]=rm.get((str(r.token_key),_utc(r.decision_at)),float(r.entry_execution_return))
    w=np.minimum(1.0/np.maximum(mu,cfg.propensity_floor),cfg.dr_clip_weight)
    dr=qpi+(a==pi).astype(float)*w*(reward-qa)
    tok=pd.DataFrame({"token_key":x.token_key.astype(str),"dr":dr}).groupby("token_key").dr.mean()
    return {"available":True,"rows":int(len(x)),"tokens":int(len(tok)),"dr_value":float(tok.mean()),"effective_weight_mean":float(np.mean(w))}


def train_distributional_policy(db:str,policy_root:str,cfg:V24Config,*,allow_small:bool=False)->dict[str,Any]:
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.row_factory=sqlite3.Row; migrate(conn)
        from . import axiom_self_teach as selfteach
        selfteach.migrate(conn); refresh_counterfactual_policy_targets(conn,cfg); refresh_policy_cohorts(conn); refresh_token_assignments(conn)
        cohort=next_one_use_policy_cohort(conn,cfg)
        if cohort is None:
            return {"trained":False,"reason":"no mature unused policy-promotion cohort"}
        entry=_entry_policy_training_rows(conn,horizon=60); hold=_hold_policy_training_rows(conn,horizon=60)
        entry_head=_fit_distribution_head(entry,"target",cfg,allow_small); hold_head=_fit_distribution_head(hold,"target",cfg,allow_small)
        if entry_head is None and hold_head is None: raise RuntimeError("No V24 counterfactual policy head has enough development-eligible OOS forecasts.")
        candidate={"schema_version":"v21_self_teaching_incremental_72h_2_execution_accounting","v24_policy_schema":SCHEMA_VERSION,
                   "created_at":_now_iso(),"entry_head":entry_head,"hold_head":hold_head,"oos_only":True,"counterfactual_targets":True,
                   "config":asdict(cfg),"training_rows_entry":int(len(entry)),"training_rows_hold":int(len(hold)),
                   "target_definition_hash":target_definition_hash(cfg),"execution_definition_hash":execution_definition_hash(cfg)}
        eval_entry=_entry_policy_training_rows(conn,eval_cohort_id=str(cohort["cohort_id"]),horizon=60)
        eval_hold=_hold_policy_training_rows(conn,eval_cohort_id=str(cohort["cohort_id"]),horizon=60)
        cand_eval=evaluate_policy_bundle(candidate,eval_entry,eval_hold,cfg)
        champion_path=Path(policy_root)/"champion.joblib"; champion=joblib.load(champion_path) if champion_path.exists() else None
        champ_eval=evaluate_policy_bundle(champion,eval_entry,eval_hold,cfg) if champion else {"available":True,"tokens":cand_eval.get("tokens",0),"mean_value":0.0,"token_values":{k:0.0 for k in cand_eval.get("token_values",{})}}
        common=sorted(set(cand_eval.get("token_values",{}))&set(champ_eval.get("token_values",{})))
        diff=pd.Series({k:float(cand_eval["token_values"][k])-float(champ_eval["token_values"][k]) for k in common},dtype=float)
        boot=_paired_token_bootstrap(diff,cfg.policy_promotion_bootstrap_samples,cfg.policy_promotion_confidence)
        promoted=bool(boot["n_tokens"] >= (3 if allow_small else cfg.policy_min_promotion_tokens) and boot["ci_low"]>0.0)
        reason=f"paired_token_value_mean={boot['mean']:.6f}; ci=[{boot['ci_low']:.6f},{boot['ci_high']:.6f}]; n={boot['n_tokens']}"
        Path(policy_root).mkdir(parents=True,exist_ok=True); stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); cand_path=Path(policy_root)/f"policy_v24_{stamp}.joblib"; joblib.dump(candidate,cand_path)
        before_hash=_hash_file(champion_path) if champion_path.exists() else None
        if promoted: joblib.dump(candidate,champion_path)
        pid=str(uuid.uuid4()); metrics={"candidate":{k:v for k,v in cand_eval.items() if k!="token_values"},"champion":{k:v for k,v in champ_eval.items() if k!="token_values"},"bootstrap":boot}
        conn.execute(f"INSERT INTO {POLICY_PROMOTION_TABLE}(promotion_id,created_at,cohort_id,candidate_path,candidate_hash,champion_before_path,champion_before_hash,promoted,metrics_json,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (pid,_now_iso(),cohort["cohort_id"],str(cand_path),_hash_file(cand_path),str(champion_path) if champion_path.exists() else None,before_hash,int(promoted),_json(metrics),reason))
        conn.execute(f"UPDATE {POLICY_COHORT_TABLE} SET status='consumed',consumed_at=?,promotion_id=? WHERE cohort_id=?",(_now_iso(),pid,cohort["cohort_id"]))
        status="champion" if promoted else "rejected"; vid=str(uuid.uuid4())
        conn.execute(f"INSERT INTO {POLICY_REGISTRY}(version_id,created_at,model_path,model_hash,status,training_rows_entry,training_rows_hold,oos_only,metrics_json,notes,target_definition_hash,execution_definition_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                     (vid,_now_iso(),str(cand_path),_hash_file(cand_path),status,len(entry),len(hold),1,_json(metrics),reason,candidate["target_definition_hash"],candidate["execution_definition_hash"]))
        if promoted:
            conn.execute("UPDATE axiom_policy_versions_v20 SET status='retired' WHERE status='champion'")
            conn.execute("INSERT INTO axiom_policy_versions_v20(version_id,created_at,model_path,model_hash,status,metrics_json,training_closed_trades,training_hold_samples,notes) VALUES(?,?,?,?,?,?,?,?,?)",
                         (vid,_now_iso(),str(champion_path),_hash_file(champion_path),"champion",_json({"v24_oos_only":True,"counterfactual":True}),len(entry),len(hold),"V24 one-use promoted counterfactual distributional policy"))
        ope=doubly_robust_entry_ope(conn,candidate,cfg,60); conn.commit()
        return {"trained":True,"promoted":promoted,"policy_candidate":str(cand_path),"champion":str(champion_path) if champion_path.exists() else None,
                "cohort_id":cohort["cohort_id"],"entry_rows":len(entry),"hold_rows":len(hold),"promotion":metrics,"dr_ope":ope}


# ---------------------------------------------------------------------------
# Liquidity/slippage interface; inactive until executed data exist
# ---------------------------------------------------------------------------

def fit_liquidity_model(
    conn:sqlite3.Connection,cfg:V24Config,allow_small:bool=False,*,cutoff:pd.Timestamp|None=None
)->dict[str,Any]:
    """Fit only from execution observations available by the historical cutoff.

    Slippage is signed adverse execution: favorable price improvement is zero,
    not a positive cost. Activation additionally requires a later chronological
    validation slice rather than a row-count threshold alone.
    """
    migrate(conn); params=[]; where=[]
    if cutoff is not None:
        where.append("observed_at<=?"); params.append(_utc(cutoff).isoformat())
        # A late-ingested execution record cannot be used by an earlier model.
        where.append("(first_ingested_at IS NULL OR first_ingested_at<=?)"); params.append(_utc(cutoff).isoformat())
    q=f"SELECT * FROM {LIQUIDITY_TABLE}"+(" WHERE "+" AND ".join(where) if where else "")+" ORDER BY observed_at"
    data=pd.read_sql_query(q,conn,params=params)
    if data.empty:return {"active":False,"reason":"no cutoff-eligible executed-liquidity observations","fallback_round_trip_bps":cfg.fallback_round_trip_bps,"training_cutoff":_iso(cutoff) if cutoff is not None else None}
    data["observed_at"]=pd.to_datetime(data.observed_at,format="ISO8601", utc=True); qprice=pd.to_numeric(data.quoted_price,errors="coerce"); eprice=pd.to_numeric(data.executed_price,errors="coerce")
    side=data.get("side",pd.Series([None]*len(data))).astype(str).str.upper()
    slip=pd.to_numeric(data.slippage_bps,errors="coerce")
    valid=qprice.gt(0)&eprice.gt(0)
    buy=((eprice/qprice)-1.0)*10000.0; sell=((qprice/eprice)-1.0)*10000.0
    calc=np.where(side.eq("SELL"),sell,buy); calc=np.maximum(0.0,np.asarray(calc,dtype=float))
    slip=np.where(valid,np.where(np.isfinite(calc),calc,slip),slip); data["slippage_bps"]=pd.to_numeric(slip,errors="coerce")
    data=data[data.slippage_bps.notna()&np.isfinite(data.slippage_bps)&(data.slippage_bps>=0)].copy()
    min_rows=30 if allow_small else cfg.liquidity_min_rows
    min_val=8 if allow_small else cfg.liquidity_min_validation_rows
    if len(data)<min_rows:return {"active":False,"reason":f"only {len(data)} cutoff-eligible rows; need {min_rows}","fallback_round_trip_bps":cfg.fallback_round_trip_bps,"training_cutoff":_iso(cutoff) if cutoff is not None else None}
    feats=[c for c in ("trade_size_usd","liquidity_usd","volume_usd","market_cap_usd") if c in data]
    for c in feats:
        data[c]=pd.to_numeric(data[c],errors="coerce"); data[f"log__{c}"]=np.log1p(np.clip(data[c],0,None))
    if "side" in data: data["side_sell"]=side.eq("SELL").astype(float); feats2=[f"log__{c}" for c in feats]+["side_sell"]
    else: feats2=[f"log__{c}" for c in feats]
    if not feats2:return {"active":False,"reason":"no liquidity explanatory fields","fallback_round_trip_bps":cfg.fallback_round_trip_bps}
    cut=max(1,min(len(data)-1,int(len(data)*(1.0-cfg.liquidity_validation_fraction)))); tr=data.iloc[:cut].copy(); va=data.iloc[cut:].copy()
    if len(va)<min_val:return {"active":False,"reason":f"only {len(va)} chronological validation rows; need {min_val}","fallback_round_trip_bps":cfg.fallback_round_trip_bps}
    head=_fit_blended_regression(tr.assign(token_key=tr.token_key.astype(str)),feats2,"slippage_bps",cfg.small_estimators if allow_small else 250,quantile=.75)
    if head is None:return {"active":False,"reason":"slippage head failed","fallback_round_trip_bps":cfg.fallback_round_trip_bps}
    pv=_predict_blended(head,va); y=va.slippage_bps.to_numpy(dtype=float); mask=np.isfinite(pv)&np.isfinite(y)
    if not mask.any():return {"active":False,"reason":"no valid chronological validation predictions","fallback_round_trip_bps":cfg.fallback_round_trip_bps}
    pin=float(mean_pinball_loss(y[mask],pv[mask],alpha=.75)); coverage=float(np.mean(y[mask]<=pv[mask]))
    # A q75 model grossly under-covering its target is not deployment-ready.
    active=bool(coverage>=0.60)
    return {"active":active,"head":head if active else None,"features":feats2,"rows":len(data),"training_rows":len(tr),"validation_rows":len(va),
            "validation_pinball":pin,"validation_q75_coverage":coverage,"training_cutoff":_iso(cutoff) if cutoff is not None else None,
            "reason":"chronologically_validated" if active else "q75 chronological coverage below 0.60","fallback_round_trip_bps":cfg.fallback_round_trip_bps}


def estimate_slippage_bps(model: dict[str, Any], trade_size_usd: float, liquidity_usd: float | None = None, volume_usd: float | None = None, market_cap_usd: float | None = None) -> float:
    if not model.get("active") or not model.get("head"):
        return float(model.get("fallback_round_trip_bps", 100.0)) / 2.0
    row = pd.DataFrame([{
        "log__trade_size_usd": math.log1p(max(0.0, trade_size_usd)),
        "log__liquidity_usd": math.log1p(max(0.0, liquidity_usd or 0.0)),
        "log__volume_usd": math.log1p(max(0.0, volume_usd or 0.0)),
        "log__market_cap_usd": math.log1p(max(0.0, market_cap_usd or 0.0)),
        "side_sell": 0.0,
    }])
    pred = _predict_blended(model["head"], row)[0]
    return float(max(0.0, pred)) if np.isfinite(pred) else float(model.get("fallback_round_trip_bps", 100.0)) / 2.0


# ---------------------------------------------------------------------------
# Sealed audit diagnostics and status
# ---------------------------------------------------------------------------

def audit_manifest(conn: sqlite3.Connection, cfg: V24Config, reveal: bool = False) -> dict[str, Any]:
    migrate(conn)
    rows = pd.read_sql_query(
        f"SELECT cohort_id,ordinal,start_at,end_at,status FROM {COHORT_TABLE} WHERE role='audit' ORDER BY ordinal",
        conn,
    )
    if rows.empty:
        return {"sealed_audit_cohorts": 0, "revealed": False}
    latest = _latest_capture(conn)
    mature = 0
    if latest is not None:
        mature = int(sum(pd.to_datetime(rows.end_at, format="ISO8601", utc=True) + pd.Timedelta(minutes=cfg.horizon_minutes) <= latest))
    if not reveal:
        return {
            "sealed_audit_cohorts": int(len(rows)), "mature_sealed_audit_cohorts": mature,
            "revealed": False,
            "note": "Audit rows are excluded from training, CPCV, cross-fit policy generation, and promotion evaluation.",
        }
    # Explicit reveal is informational only; it does not alter role/status and thus
    # cannot make the rows training-eligible.
    return {"sealed_audit_cohorts": int(len(rows)), "mature_sealed_audit_cohorts": mature, "revealed": True, "cohorts": rows.to_dict("records")}


def evaluate_sealed_audit_stream(conn: sqlite3.Connection, cfg: V24Config) -> dict[str, Any]:
    """Score only predictions that were produced prospectively inside audit blocks.

    The audit is time-locked: outcomes are not reported until the cohort has been
    mature for audit_min_age_days. Audit data remain role='audit' forever.
    """
    migrate(conn)
    latest = _latest_capture(conn)
    if latest is None:
        return {"available": False, "reason": "no observations"}
    unlock_before = latest - pd.Timedelta(minutes=2*cfg.horizon_minutes) - pd.Timedelta(days=cfg.audit_min_age_days)
    cohorts = pd.read_sql_query(
        f"SELECT start_at,end_at FROM {COHORT_TABLE} WHERE role='audit' AND end_at <= ? ORDER BY ordinal",
        conn, params=(unlock_before.isoformat(),),
    )
    if cohorts.empty:
        return {"available": False, "reason": "no time-unlocked mature audit cohort"}
    led = pd.read_sql_query(
        f"""SELECT token_key,decision_at,prediction_json FROM {PREDICTION_LEDGER}
            WHERE provenance='live' AND ineligibility_reason='sealed_audit_cohort'
            ORDER BY decision_at""", conn
    )
    if led.empty:
        return {"available": False, "reason": "no prospective audit predictions recorded"}
    led["decision_at"] = pd.to_datetime(led.decision_at, format="ISO8601", utc=True)
    allowed = np.zeros(len(led), dtype=bool)
    for r in cohorts.itertuples(index=False):
        a, b = _utc(r.start_at), _utc(r.end_at)
        allowed |= ((led.decision_at >= a) & (led.decision_at < b)).to_numpy()
    led = led[allowed].copy()
    if led.empty:
        return {"available": False, "reason": "audit predictions are not yet time-unlocked"}
    labels = pd.read_sql_query(
        f"SELECT token_key,decision_at,next_substantial_peak_at,terminal_at,terminal_reason,path_end_at,label_finalized FROM {peak.LABEL_TABLE}", conn
    )
    labels["decision_at"] = pd.to_datetime(labels.decision_at, format="ISO8601", utc=True)
    data = led.merge(labels, on=["token_key","decision_at"], how="inner")
    if data.empty:
        return {"available": False, "reason": "audit labels unavailable"}
    horizon_key = f"p_first_peak_by_{max(cfg.survival_bins_minutes)}m"
    actual=[]; probs=[]
    for _, r in data.iterrows():
        ev, _, known = first_competing_event(r, cfg)
        if not known:
            continue
        p = _finite(_loads(r.prediction_json).get(horizon_key))
        if p is None:
            continue
        actual.append(1.0 if ev == EVENT_PEAK else 0.0)
        probs.append(p)
    if not probs:
        return {"available": False, "reason": f"no usable {horizon_key} audit predictions"}
    a=np.asarray(actual); p=np.clip(np.asarray(probs),0,1)
    metrics={
        "rows": int(len(p)),
        "peak_brier": float(np.mean((p-a)**2)),
        "peak_log_loss": float(log_loss(a.astype(int), np.clip(p,1e-6,1-1e-6), labels=[0,1])) if len(np.unique(a))>1 else None,
        "horizon_probability": horizon_key,
    }
    conn.execute(
        f"INSERT INTO {AUDIT_RESULTS_TABLE}(audit_result_id,created_at,model_family,prediction_rows,metric_json,note) VALUES(?,?,?,?,?,?)",
        (str(uuid.uuid4()),_now_iso(),SCHEMA_VERSION,len(p),_json(metrics),"Prospective sealed audit; never development-eligible"),
    )
    conn.commit()
    return {"available": True, **metrics}


def status(db: str, cfg: V24Config) -> dict[str, Any]:
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        refresh_calendar_cohorts(conn, cfg)
        refresh_lifetimes(conn, cfg)
        counts = {
            "train_cohorts": conn.execute(f"SELECT COUNT(*) FROM {COHORT_TABLE} WHERE role='train'").fetchone()[0],
            "available_promotion_cohorts": conn.execute(f"SELECT COUNT(*) FROM {COHORT_TABLE} WHERE role='promotion' AND status='available'").fetchone()[0],
            "consumed_promotion_cohorts": conn.execute(f"SELECT COUNT(*) FROM {COHORT_TABLE} WHERE role='promotion' AND status='consumed'").fetchone()[0],
            "sealed_audit_cohorts": conn.execute(f"SELECT COUNT(*) FROM {COHORT_TABLE} WHERE role='audit'").fetchone()[0],
            "oos_policy_predictions": conn.execute(f"SELECT COUNT(*) FROM {PREDICTION_LEDGER} WHERE policy_training_eligible=1").fetchone()[0],
            "rejected_prediction_provenance": conn.execute(f"SELECT COUNT(*) FROM {PREDICTION_LEDGER} WHERE policy_training_eligible=0").fetchone()[0],
            "lifetimes": conn.execute(f"SELECT COUNT(*) FROM {LIFETIME_TABLE}").fetchone()[0],
            "tokens": conn.execute(f"SELECT COUNT(DISTINCT token_key) FROM {LIFETIME_TABLE}").fetchone()[0],
        }
        next_c = next_one_use_promotion_cohort(conn, cfg)
        champ = Path(CHAMPION_DEFAULT)
        bundle = joblib.load(champ) if champ.exists() else None
        return {
            "schema_version": SCHEMA_VERSION,
            **{k: int(v) for k, v in counts.items()},
            "next_promotion_cohort": dict(next_c) if next_c else None,
            "champion_exists": champ.exists(),
            "champion_training_cutoff": bundle.get("stable_training_cutoff") if bundle else None,
            "stable_generation": bundle.get("stable_generation") if bundle else None,
            "adapter_round": bundle.get("adapter_round") if bundle else None,
            "adapter_active": bool(bundle and bundle.get("adapter")),
            "audit": audit_manifest(conn, cfg, reveal=False),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cfg_from_args(args: argparse.Namespace) -> V24Config:
    return V24Config(
        cohort_hours=getattr(args, "cohort_hours", 24),
        promotion_every_n_blocks=getattr(args, "promotion_every", 4),
        audit_every_n_blocks=getattr(args, "audit_every", 5),
        warmup_blocks=getattr(args, "warmup_blocks", 7),
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="V24 leakage-hardened 72h event/survival forecaster and OOS policy tools")
    sp = p.add_subparsers(dest="cmd", required=True)

    def common(x: argparse.ArgumentParser) -> None:
        x.add_argument("--db", default="data/live.sqlite")
        x.add_argument("--cohort-hours", type=int, default=24)
        x.add_argument("--promotion-every", type=int, default=4)
        x.add_argument("--audit-every", type=int, default=5)
        x.add_argument("--warmup-blocks", type=int, default=7)

    x = sp.add_parser("bootstrap")
    common(x); x.add_argument("--model-root", default=MODEL_ROOT_DEFAULT); x.add_argument("--allow-small", action="store_true")
    x = sp.add_parser("maintain")
    common(x); x.add_argument("--model-root", default=MODEL_ROOT_DEFAULT); x.add_argument("--allow-small", action="store_true"); x.add_argument("--force-compaction", action="store_true")
    x = sp.add_parser("predict")
    common(x); x.add_argument("--model", default=CHAMPION_DEFAULT); x.add_argument("--out", default=PREDICTIONS_DEFAULT)
    x = sp.add_parser("crossfit-policy-predictions")
    common(x); x.add_argument("--max-folds", type=int, default=5); x.add_argument("--allow-small", action="store_true")
    x = sp.add_parser("train-policy")
    common(x); x.add_argument("--policy-root", default=POLICY_ROOT_DEFAULT); x.add_argument("--allow-small", action="store_true")
    x = sp.add_parser("status")
    common(x)
    x = sp.add_parser("rebuild-sequence-cache")
    common(x)
    x = sp.add_parser("audit-manifest")
    common(x); x.add_argument("--reveal", action="store_true")
    x = sp.add_parser("audit-evaluate")
    common(x)

    args = p.parse_args(argv)
    cfg = _cfg_from_args(args)
    if args.cmd == "bootstrap":
        out = bootstrap_v24(args.db, args.model_root, cfg, allow_small=args.allow_small)
    elif args.cmd == "maintain":
        out = maintain_v24(args.db, args.model_root, cfg, allow_small=args.allow_small, force_compaction=args.force_compaction)
    elif args.cmd == "predict":
        out = {"rows": len(predict_current(args.db, args.model, args.out, cfg)), "out": args.out}
    elif args.cmd == "crossfit-policy-predictions":
        out = crossfit_policy_predictions(args.db, cfg, max_folds=args.max_folds, allow_small=args.allow_small)
    elif args.cmd == "train-policy":
        out = train_distributional_policy(args.db, args.policy_root, cfg, allow_small=args.allow_small)
    elif args.cmd == "status":
        out = status(args.db, cfg)
    elif args.cmd == "rebuild-sequence-cache":
        with closing(sqlite3.connect(args.db)) as conn, conn:
            obs,_=peak.load_observations(conn)
            out=refresh_sequence_fingerprint_cache(conn,obs,cfg,force=True)
    elif args.cmd == "audit-manifest":
        with closing(sqlite3.connect(args.db)) as conn, conn:
            refresh_calendar_cohorts(conn, cfg)
            out = audit_manifest(conn, cfg, reveal=args.reveal)
    elif args.cmd == "audit-evaluate":
        with closing(sqlite3.connect(args.db)) as conn, conn:
            refresh_calendar_cohorts(conn, cfg)
            out = evaluate_sealed_audit_stream(conn, cfg)
    else:  # pragma: no cover
        raise RuntimeError(args.cmd)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
