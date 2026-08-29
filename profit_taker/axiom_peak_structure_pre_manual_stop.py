"""V24 peak-structure facade with raw-source, identity, feature and 24h lifecycle safety guards."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import axiom_peak_structure_base as _base

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

_impl = _base._impl

PREDICTION_HORIZON_MINUTES = 24 * 60
AXIOM_VIEW_MINUTES = 25 * 60
AGE_OUT_GUARD_MINUTES = 24 * 60
SCHEMA_VERSION = "v24_peak_structure_24h_incremental_v2"
DEFAULT_OUT = "data/axiom_peak_structure_predictions_24h.csv"


@dataclass(frozen=True)
class PeakStructureConfig(_impl.PeakStructureConfig):
    """24h modeling contract with one additional visible Axiom buffer hour."""

    horizon_minutes: int = PREDICTION_HORIZON_MINUTES
    age_out_minutes: int = AGE_OUT_GUARD_MINUTES


_impl.PeakStructureConfig = PeakStructureConfig
_base.PeakStructureConfig = PeakStructureConfig
_impl.SCHEMA_VERSION = SCHEMA_VERSION
_base.SCHEMA_VERSION = SCHEMA_VERSION
_impl.DEFAULT_OUT = DEFAULT_OUT
_base.DEFAULT_OUT = DEFAULT_OUT

_OPERATIONAL_MODEL_FEATURES = {
    "observation_id", "cycle_id", "capture_id", "attempt_id", "session_id",
    "created_at", "first_ingested_at", "last_corrected_at", "value_version",
}
_OPERATIONAL_MODEL_FRAGMENTS = (
    "observation_id", "cycle_id", "capture_id", "attempt_id", "session_id",
    "raw_payload", "payload_sha", "collector_schema", "ingestion_provenance",
)
_original_safe_feature_name = _impl._safe_feature_name


def _safe_feature_name(name: str) -> bool:
    lname = str(name).lower()
    if lname in _OPERATIONAL_MODEL_FEATURES:
        return False
    if any(fragment in lname for fragment in _OPERATIONAL_MODEL_FRAGMENTS):
        return False
    return bool(_original_safe_feature_name(name))


def _load_cached_feature_frame(conn: sqlite3.Connection, current_at: pd.Timestamp | None = None) -> pd.DataFrame:
    if current_at is None:
        df = pd.read_sql_query(
            f"SELECT token_key,snapshot_at,features_json FROM {FEATURE_CACHE_TABLE} ORDER BY snapshot_at",
            conn,
        )
    else:
        df = pd.read_sql_query(
            f"SELECT token_key,snapshot_at,features_json FROM {FEATURE_CACHE_TABLE} WHERE snapshot_at=?",
            conn,
            params=(_utc_iso(current_at),),
        )
    if df.empty:
        return pd.DataFrame(columns=["token_key", "snapshot_at"])
    expanded = pd.DataFrame([_flatten_numeric_json(v) for v in df.features_json])
    keep = [c for c in expanded.columns if _safe_feature_name(c)]
    expanded = expanded.reindex(columns=keep)
    base = df[["token_key", "snapshot_at"]].copy()
    base["snapshot_at"] = _to_timestamp(base["snapshot_at"])
    return pd.concat([base.reset_index(drop=True), expanded.reset_index(drop=True)], axis=1)


_impl._safe_feature_name = _safe_feature_name
_impl._load_cached_feature_frame = _load_cached_feature_frame

discover_observation_source = _base.discover_observation_source
load_observations = _base.load_observations
_impl.discover_observation_source = discover_observation_source
_impl.load_observations = load_observations


def infer_token_terminal(
    token_df: pd.DataFrame,
    capture_times: list[pd.Timestamp],
    presence: dict[pd.Timestamp, set[str]],
    config: PeakStructureConfig,
) -> tuple[pd.Timestamp | None, str | None]:
    """Treat the 24h model boundary as natural age-out, never operational death."""
    if token_df.empty:
        return None, None
    last = token_df.sort_values("snapshot_at").iloc[-1]
    last_seen = pd.Timestamp(last.snapshot_at)
    age = None
    if "age_minutes" in token_df.columns:
        raw = pd.to_numeric(pd.Series([last.get("age_minutes")]), errors="coerce").iloc[0]
        if pd.notna(raw):
            age = float(raw)
    if age is not None and age >= float(config.age_out_minutes):
        return last_seen, "age_out_24h_model_window"

    run_start, run_end, valid_count = _impl._contiguous_absence_run(last_seen, capture_times, config)
    if run_start is None or run_end is None:
        return None, None
    absent_minutes = max(0.0, (run_end - run_start).total_seconds() / 60.0)
    if (
        absent_minutes >= config.death_gap_minutes
        and valid_count >= int(config.heartbeat_min_valid_captures)
    ):
        return run_start + pd.Timedelta(minutes=config.death_gap_minutes), "dead_after_valid_capture_absence"
    return None, None


_impl.infer_token_terminal = infer_token_terminal
_base.infer_token_terminal = infer_token_terminal

_original_refresh_labels = _impl.refresh_labels


def _stored_label_contract_mismatch(db: str, config: PeakStructureConfig) -> bool:
    """Return True when durable labels were finalized under a different target contract."""
    try:
        with sqlite3.connect(db) as conn:
            if LABEL_TABLE not in _impl._all_tables(conn):
                return False
            rows = conn.execute(
                f"SELECT schema_version,config_json FROM {LABEL_TABLE} "
                "WHERE label_finalized=1 ORDER BY decision_at DESC LIMIT 50"
            ).fetchall()
    except sqlite3.DatabaseError:
        return False
    expected = asdict(config)
    for schema, raw in rows:
        if str(schema) != SCHEMA_VERSION:
            return True
        try:
            saved = json.loads(raw) if raw else {}
        except Exception:
            return True
        if int(saved.get("horizon_minutes", -1)) != int(expected["horizon_minutes"]):
            return True
        if float(saved.get("age_out_minutes", -1)) != float(expected["age_out_minutes"]):
            return True
    return False


def refresh_labels(db: str, config: PeakStructureConfig, full_rebuild: bool = False) -> dict[str, object]:
    """Refresh labels, rebuilding all derived truth if the lifecycle contract changed.

    Raw observations are never deleted. Only derived peak/label state is recomputed,
    preventing finalized 72h truth from being silently reused by the 24h generation.
    """
    contract_rebuild = _stored_label_contract_mismatch(db, config)
    result = _original_refresh_labels(db, config, full_rebuild=bool(full_rebuild or contract_rebuild))
    if isinstance(result, dict):
        result["contract_rebuild"] = bool(contract_rebuild)
        result["prediction_horizon_minutes"] = int(config.horizon_minutes)
        result["axiom_view_minutes"] = AXIOM_VIEW_MINUTES
    return result


_impl.refresh_labels = refresh_labels
_base.refresh_labels = refresh_labels


def lifecycle_contract() -> dict[str, int | str]:
    return {
        "prediction_horizon_minutes": PREDICTION_HORIZON_MINUTES,
        "axiom_view_minutes": AXIOM_VIEW_MINUTES,
        "age_out_guard_minutes": AGE_OUT_GUARD_MINUTES,
        "buffer_minutes": AXIOM_VIEW_MINUTES - PREDICTION_HORIZON_MINUTES,
        "schema_version": SCHEMA_VERSION,
        "compatibility_note": "Legacy *_72h database column names are retained only as storage identifiers; active truth is bounded by config.horizon_minutes.",
    }


def __getattr__(name: str):
    return getattr(_base, name)


if __name__ == "__main__":  # pragma: no cover
    main()
