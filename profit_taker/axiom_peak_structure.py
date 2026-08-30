"""V24 peak-structure facade with raw-source, identity, feature and 24h lifecycle safety guards."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import axiom_manual_stop as manual_stop
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


def _manual_censor_relabel(db: str, config: PeakStructureConfig) -> dict[str, int]:
    """Rebuild affected token episodes without allowing information across a stop.

    A stop is a right-censor boundary, not a terminal failure. Confirmed positives
    before the boundary remain positive. Any unresolved future target stays NULL.
    Reappearing observations belong to the next episode and cannot retroactively
    confirm a peak from the pre-stop episode.
    """
    with sqlite3.connect(db) as conn:
        manual_stop.migrate(conn)
        censor_map = manual_stop.censors_by_token(conn)
        if not censor_map:
            return {"tokens": 0, "labels": 0, "peak_events": 0}
        observations, _ = load_observations(conn)
        if observations.empty:
            return {"tokens": 0, "labels": 0, "peak_events": 0}
        observations = observations.sort_values(["token_key", "snapshot_at"]).reset_index(drop=True)
        latest_capture = pd.Timestamp(observations.snapshot_at.max())
        capture_times, presence = _impl._capture_index(observations)
        cfg_json = json.dumps(asdict(config), sort_keys=True)
        now = datetime.now(timezone.utc).isoformat()
        labels_written = 0
        peak_events_written = 0
        tokens_written = 0

        for token, records in censor_map.items():
            g_all = observations[observations.token_key.astype(str) == str(token)].copy()
            if g_all.empty:
                continue
            g_all = g_all.sort_values("snapshot_at").reset_index(drop=True)
            boundaries = sorted(
                {pd.Timestamp(r["censor_at"]) for r in records if r.get("censor_at")}
            )
            if not boundaries:
                continue
            tokens_written += 1

            episodes: list[tuple[pd.Timestamp | None, pd.Timestamp | None, pd.DataFrame, list]] = []
            previous: pd.Timestamp | None = None
            for boundary in [*boundaries, None]:
                mask = pd.Series(True, index=g_all.index)
                if previous is not None:
                    mask &= g_all.snapshot_at > previous
                if boundary is not None:
                    mask &= g_all.snapshot_at <= boundary
                episode = g_all[mask].copy().sort_values("snapshot_at").reset_index(drop=True)
                if not episode.empty:
                    peaks = _impl.find_substantial_peaks(episode, config)
                    episodes.append((previous, boundary, episode, peaks))
                previous = boundary

            # The retained peak cache is token-wide. Rebuild it from independent
            # censor-bounded episodes so recurrent-event targets also cannot bridge.
            conn.execute(f"DELETE FROM {_impl.PEAK_EVENT_TABLE} WHERE token_key=?", (str(token),))
            for _, _, _, peaks in episodes:
                for p in peaks:
                    conn.execute(
                        f"""INSERT OR REPLACE INTO {_impl.PEAK_EVENT_TABLE}
                            (token_key,trough_at,trough_price,peak_at,peak_price,confirmed_at,
                             confirmation_price,runup_pct,confirmation_retrace_pct,config_json)
                            VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        (
                            str(token), p.trough_at, p.trough_price, p.peak_at, p.peak_price,
                            p.confirmed_at, p.confirmation_price, p.runup_pct,
                            p.confirmation_retrace_pct, cfg_json,
                        ),
                    )
                    peak_events_written += 1

            for _, boundary, episode, peaks in episodes:
                if boundary is None:
                    terminal_at, terminal_reason = infer_token_terminal(
                        episode, capture_times, presence, config
                    )
                    known_through = latest_capture
                else:
                    terminal_at, terminal_reason = None, None
                    known_through = boundary

                for i in range(len(episode)):
                    decision_at = pd.Timestamp(episode.iloc[i].snapshot_at)
                    rec = _impl.label_decision_from_events(
                        episode,
                        i,
                        known_through,
                        terminal_at,
                        terminal_reason,
                        config,
                        peaks,
                    )
                    if boundary is not None:
                        horizon_end = decision_at + pd.Timedelta(minutes=config.horizon_minutes)
                        if boundary < horizon_end:
                            rec["terminal_at"] = None
                            rec["terminal_reason"] = "manual_stop_censored"
                            rec["path_end_at"] = _impl._utc_iso(boundary)
                            rec["label_finalized"] = 0
                            if rec.get("has_next_substantial_peak_before_terminal_72h") is None:
                                rec["label_status_next_peak"] = "censored_collection_stop"
                                rec["label_ready_at"] = None
                            rec["target_fingerprint"] = _impl._target_fingerprint(rec)

                    old = conn.execute(
                        f"SELECT target_fingerprint FROM {LABEL_TABLE} WHERE token_key=? AND decision_at=?",
                        (str(token), rec["decision_at"]),
                    ).fetchone()
                    if old is None or str(old[0] or "") != str(rec.get("target_fingerprint") or ""):
                        rec["learning_updated_at"] = now
                    cols = list(rec)
                    placeholders = ",".join("?" for _ in cols)
                    col_sql = ",".join(f'"{c}"' for c in cols)
                    updates = ",".join(
                        f'"{c}"=excluded."{c}"' for c in cols if c not in ("token_key", "decision_at")
                    )
                    conn.execute(
                        f"INSERT INTO {LABEL_TABLE} ({col_sql}) VALUES ({placeholders}) "
                        f"ON CONFLICT(token_key,decision_at) DO UPDATE SET {updates}",
                        [rec[c] for c in cols],
                    )
                    labels_written += 1

        conn.commit()
        return {
            "tokens": tokens_written,
            "labels": labels_written,
            "peak_events": peak_events_written,
        }


def refresh_labels(db: str, config: PeakStructureConfig, full_rebuild: bool = False) -> dict[str, object]:
    """Refresh labels, rebuilding all derived truth if the lifecycle contract changed.

    Raw observations are never deleted. Only derived peak/label state is recomputed,
    preventing finalized 72h truth from being silently reused by the 24h generation.
    Manual collection stops are then applied as explicit right-censor boundaries.
    """
    contract_rebuild = _stored_label_contract_mismatch(db, config)
    result = _original_refresh_labels(db, config, full_rebuild=bool(full_rebuild or contract_rebuild))
    censor_result = _manual_censor_relabel(db, config)
    if isinstance(result, dict):
        result["contract_rebuild"] = bool(contract_rebuild)
        result["prediction_horizon_minutes"] = int(config.horizon_minutes)
        result["axiom_view_minutes"] = AXIOM_VIEW_MINUTES
        result["manual_stop_censoring"] = censor_result
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
