"""V24 peak-structure facade with raw-source, identity and feature safety guards."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from . import axiom_peak_structure_base as _base

for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)

_impl = _base._impl

# Durable collection/order/provenance metadata are useful for auditing but are not
# market state. Letting them into boosted trees can teach collection chronology or
# database repair history instead of token behavior.
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
    """Read cached features while stripping operational columns from old caches too."""
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


# Patch the preserved implementation so every internal feature builder/cache reader
# resolves these hardened functions too.
_impl._safe_feature_name = _safe_feature_name
_impl._load_cached_feature_frame = _load_cached_feature_frame

# Re-export the already-hardened canonical source/identity functions from the prior
# facade explicitly so callers cannot fall back to the legacy heuristic versions.
discover_observation_source = _base.discover_observation_source
load_observations = _base.load_observations
_impl.discover_observation_source = discover_observation_source
_impl.load_observations = load_observations


def __getattr__(name: str):
    return getattr(_base, name)


if __name__ == "__main__":  # pragma: no cover
    main()
