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


def _censor_boundaries(records: list[dict[str, str]]) -> list[pd.Timestamp]:
    values = {
        pd.Timestamp(r["censor_at"])
        for r in records
        if r.get("censor_at")
    }
    return sorted(values)


def _split_censor_episodes(
    token_df: pd.DataFrame,
    boundaries: list[pd.Timestamp],
    config: PeakStructureConfig,
) -> list[tuple[pd.Timestamp | None, pd.DataFrame, list]]:
    """Split one token path so no swing can be confirmed across a collection stop."""
    episodes: list[tuple[pd.Timestamp | None, pd.DataFrame, list]] = []
    previous: pd.Timestamp | None = None
    for boundary in [*boundaries, None]:
        mask = pd.Series(True, index=token_df.index)
        if previous is not None:
            mask &= token_df.snapshot_at > previous
        if boundary is not None:
            mask &= token_df.snapshot_at <= boundary
        episode = token_df[mask].copy().sort_values("snapshot_at").reset_index(drop=True)
        if not episode.empty:
            peaks = _impl.find_substantial_peaks(episode, config)
            episodes.append((boundary, episode, peaks))
        previous = boundary
    return episodes


def _replace_censored_peak_cache(
    conn: sqlite3.Connection,
    token: str,
    episodes: list[tuple[pd.Timestamp | None, pd.DataFrame, list]],
    config: PeakStructureConfig,
) -> int:
    conn.execute(f"DELETE FROM {_impl.PEAK_EVENT_TABLE} WHERE token_key=?", (token,))
    cfg_json = json.dumps(asdict(config), sort_keys=True)
    written = 0
    for _, _, peaks in episodes:
        for p in peaks:
            conn.execute(
                f"""INSERT OR REPLACE INTO {_impl.PEAK_EVENT_TABLE}
                    (token_key,trough_at,trough_price,peak_at,peak_price,confirmed_at,
                     confirmation_price,runup_pct,confirmation_retrace_pct,config_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    token, p.trough_at, p.trough_price, p.peak_at, p.peak_price,
                    p.confirmed_at, p.confirmation_price, p.runup_pct,
                    p.confirmation_retrace_pct, cfg_json,
                ),
            )
            written += 1
    last_observation = max(pd.Timestamp(ep.snapshot_at.max()) for _, ep, _ in episodes)
    conn.execute(
        f"""INSERT INTO {_impl.STATE_TABLE}(token_key,last_observation_at,last_peak_refresh_at,config_json)
            VALUES(?,?,?,?)
            ON CONFLICT(token_key) DO UPDATE SET
              last_observation_at=excluded.last_observation_at,
              last_peak_refresh_at=excluded.last_peak_refresh_at,
              config_json=excluded.config_json""",
        (token, _impl._utc_iso(last_observation), datetime.now(timezone.utc).isoformat(), cfg_json),
    )
    return written


def _upsert_label(
    conn: sqlite3.Connection,
    rec: dict,
    old: tuple[int, str | None, str | None] | None,
    now: str,
) -> tuple[bool, bool]:
    old_fp = old[1] if old else None
    learning_changed = old is None or old_fp != rec.get("target_fingerprint")
    rec["learning_updated_at"] = now if learning_changed else None
    cols = list(rec)
    placeholders = ",".join("?" for _ in cols)
    col_sql = ",".join(f'"{c}"' for c in cols)
    updates = ",".join(
        (
            f'"{c}"=CASE WHEN excluded."{c}" IS NULL AND "{LABEL_TABLE}"."{c}" IS NOT NULL '
            f'THEN "{LABEL_TABLE}"."{c}" ELSE excluded."{c}" END'
        )
        if c == "learning_updated_at"
        else f'"{c}"=excluded."{c}"'
        for c in cols
        if c not in ("token_key", "decision_at")
    )
    conn.execute(
        f"INSERT INTO {LABEL_TABLE} ({col_sql}) VALUES ({placeholders}) "
        f"ON CONFLICT(token_key,decision_at) DO UPDATE SET {updates}",
        [rec[c] for c in cols],
    )
    return old is None, learning_changed


def _censored_episode_terminal(
    episode: pd.DataFrame,
    boundary: pd.Timestamp,
    capture_times: list[pd.Timestamp],
    presence: dict[pd.Timestamp, set[str]],
    config: PeakStructureConfig,
) -> tuple[pd.Timestamp | None, str | None]:
    """Preserve a real terminal already known before the manual stop, if any."""
    known_captures = [t for t in capture_times if t <= boundary]
    terminal_at, terminal_reason = infer_token_terminal(
        episode, known_captures, presence, config
    )
    if terminal_at is not None and terminal_at <= boundary:
        return terminal_at, terminal_reason
    return None, None


def refresh_labels(db: str, config: PeakStructureConfig, full_rebuild: bool = False) -> dict[str, object]:
    """Incrementally refresh labels with manual stops treated as right-censoring.

    A stop never becomes a failure. Confirmed outcomes that occurred before the
    stop remain known, but unresolved targets are censored at the last durable
    successful capture. Observations after a restart form a new episode and cannot
    retrospectively confirm a peak or policy outcome in the pre-stop episode.
    """
    contract_rebuild = _stored_label_contract_mismatch(db, config)
    rebuild = bool(full_rebuild or contract_rebuild)
    with sqlite3.connect(db) as conn:
        _impl.migrate(conn)
        manual_stop.migrate(conn)
        observations, source = load_observations(conn)
        if observations.empty:
            return {
                "stored": 0,
                "source": source,
                "mode": "full" if rebuild else "incremental",
                "contract_rebuild": contract_rebuild,
            }
        observations = observations.sort_values(["token_key", "snapshot_at"]).reset_index(drop=True)
        latest_capture = pd.Timestamp(observations.snapshot_at.max())
        capture_times, presence = _impl._capture_index(observations)
        cfg_json = json.dumps(asdict(config), sort_keys=True)
        censor_map = manual_stop.censors_by_token(conn)

        state = {
            str(r[0]): {"last_observation_at": r[1], "config_json": r[2]}
            for r in conn.execute(
                f"SELECT token_key,last_observation_at,config_json FROM {_impl.STATE_TABLE}"
            ).fetchall()
        }
        changed_tokens: set[str] = set()
        for token, g in observations.groupby("token_key", sort=False):
            token = str(token)
            last_obs = _impl._utc_iso(g.snapshot_at.max())
            prior = state.get(token)
            if (
                rebuild
                or prior is None
                or prior.get("last_observation_at") != last_obs
                or prior.get("config_json") != cfg_json
            ):
                changed_tokens.add(token)
        # Censor metadata can be written without a new market observation, so these
        # token caches must be reconsidered even when last_observation_at is unchanged.
        changed_tokens.update(str(t) for t in censor_map)

        peak_events_before = int(
            conn.execute(f"SELECT COUNT(*) FROM {_impl.PEAK_EVENT_TABLE}").fetchone()[0]
        )
        peak_tokens_refreshed = 0
        manual_peak_events = 0
        episode_cache: dict[str, list[tuple[pd.Timestamp | None, pd.DataFrame, list]]] = {}
        for token in changed_tokens:
            g = observations[observations.token_key.astype(str) == token].sort_values("snapshot_at").reset_index(drop=True)
            if g.empty:
                continue
            records = censor_map.get(token, [])
            if records:
                episodes = _split_censor_episodes(g, _censor_boundaries(records), config)
                episode_cache[token] = episodes
                manual_peak_events += _replace_censored_peak_cache(conn, token, episodes, config)
            else:
                _impl._refresh_token_peaks(conn, token, g, config)
            peak_tokens_refreshed += 1

        existing = pd.read_sql_query(
            f"SELECT token_key,decision_at,label_finalized,target_fingerprint,config_json FROM {LABEL_TABLE}",
            conn,
        )
        existing_map: dict[tuple[str, str], tuple[int, str | None, str | None]] = {}
        unfinalized_tokens: set[str] = set()
        if not existing.empty:
            for r in existing.itertuples(index=False):
                key = (str(r.token_key), str(r.decision_at))
                existing_map[key] = (int(r.label_finalized or 0), r.target_fingerprint, r.config_json)
                if not int(r.label_finalized or 0):
                    unfinalized_tokens.add(str(r.token_key))

        tokens_to_label = set(changed_tokens) | unfinalized_tokens
        if rebuild:
            tokens_to_label = set(map(str, observations.token_key.unique()))

        now = datetime.now(timezone.utc).isoformat()
        inserted = updated = unchanged = 0
        terminal_counts: dict[str, int] = {}
        censored_decisions = 0

        def write_record(rec: dict, old: tuple[int, str | None, str | None] | None) -> None:
            nonlocal inserted, updated, unchanged
            was_inserted, learning_changed = _upsert_label(conn, rec, old, now)
            if was_inserted:
                inserted += 1
            elif learning_changed:
                updated += 1
            else:
                unchanged += 1

        for token in tokens_to_label:
            g = observations[observations.token_key.astype(str) == token].sort_values("snapshot_at").reset_index(drop=True)
            if g.empty:
                continue
            records = censor_map.get(token, [])
            if not records:
                terminal_at, terminal_reason = infer_token_terminal(g, capture_times, presence, config)
                terminal_counts[terminal_reason or "open"] = terminal_counts.get(terminal_reason or "open", 0) + 1
                peaks = _impl._peaks_for_token(conn, token)
                for i in range(len(g)):
                    decision_iso = _impl._utc_iso(g.iloc[i].snapshot_at)
                    key = (token, decision_iso)
                    old = existing_map.get(key)
                    if old and old[0] and old[2] == cfg_json and not rebuild:
                        continue
                    rec = _impl.label_decision_from_events(
                        g, i, latest_capture, terminal_at, terminal_reason, config, peaks
                    )
                    write_record(rec, old)
                continue

            episodes = episode_cache.get(token)
            if episodes is None:
                episodes = _split_censor_episodes(g, _censor_boundaries(records), config)
                episode_cache[token] = episodes
            for boundary, episode, peaks in episodes:
                if boundary is None:
                    terminal_at, terminal_reason = infer_token_terminal(
                        episode, capture_times, presence, config
                    )
                    known_through = latest_capture
                else:
                    terminal_at, terminal_reason = _censored_episode_terminal(
                        episode, boundary, capture_times, presence, config
                    )
                    known_through = boundary
                terminal_counts[terminal_reason or ("manual_stop_censored" if boundary is not None else "open")] = (
                    terminal_counts.get(terminal_reason or ("manual_stop_censored" if boundary is not None else "open"), 0) + 1
                )

                for i in range(len(episode)):
                    decision_at = pd.Timestamp(episode.iloc[i].snapshot_at)
                    decision_iso = _impl._utc_iso(decision_at)
                    key = (token, decision_iso)
                    old = existing_map.get(key)
                    if (
                        boundary is None
                        and old
                        and old[0]
                        and old[2] == cfg_json
                        and not rebuild
                    ):
                        continue
                    rec = _impl.label_decision_from_events(
                        episode,
                        i,
                        known_through,
                        terminal_at,
                        terminal_reason,
                        config,
                        peaks,
                    )
                    horizon_end = decision_at + pd.Timedelta(minutes=config.horizon_minutes)
                    if boundary is not None and terminal_at is None and boundary < horizon_end:
                        rec["terminal_at"] = None
                        rec["terminal_reason"] = "manual_stop_censored"
                        rec["path_end_at"] = _impl._utc_iso(boundary)
                        rec["label_finalized"] = 0
                        if rec.get("has_next_substantial_peak_before_terminal_72h") is None:
                            rec["label_status_next_peak"] = "censored_collection_stop"
                            rec["label_ready_at"] = None
                        rec["target_fingerprint"] = _impl._target_fingerprint(rec)
                        censored_decisions += 1
                    write_record(rec, old)

        conn.commit()

        def scalar(sql: str) -> int:
            return int(conn.execute(sql).fetchone()[0])

        result: dict[str, object] = {
            "mode": "full" if rebuild else "incremental",
            "source": source,
            "latest_capture": _impl._utc_iso(latest_capture),
            "changed_tokens": len(changed_tokens),
            "peak_tokens_refreshed": peak_tokens_refreshed,
            "peak_events_before": peak_events_before,
            "peak_events_after": scalar(f"SELECT COUNT(*) FROM {_impl.PEAK_EVENT_TABLE}"),
            "labels_inserted": inserted,
            "labels_learning_updated": updated,
            "labels_rechecked_unchanged": unchanged,
            "total_labels": scalar(f"SELECT COUNT(*) FROM {LABEL_TABLE}"),
            "next_peak_positive": scalar(
                f"SELECT COUNT(*) FROM {LABEL_TABLE} WHERE has_next_substantial_peak_before_terminal_72h=1"
            ),
            "next_peak_negative": scalar(
                f"SELECT COUNT(*) FROM {LABEL_TABLE} WHERE has_next_substantial_peak_before_terminal_72h=0"
            ),
            "next_peak_immature": scalar(
                f"SELECT COUNT(*) FROM {LABEL_TABLE} WHERE has_next_substantial_peak_before_terminal_72h IS NULL"
            ),
            "later_higher_positive": scalar(
                f"SELECT COUNT(*) FROM {LABEL_TABLE} WHERE later_higher_peak_before_terminal_72h=1"
            ),
            "terminal_tokens_touched": terminal_counts,
            "config": asdict(config),
            "contract_rebuild": contract_rebuild,
            "prediction_horizon_minutes": int(config.horizon_minutes),
            "axiom_view_minutes": AXIOM_VIEW_MINUTES,
            "manual_stop_censoring": {
                "tokens": len(censor_map),
                "censored_decisions": censored_decisions,
                "peak_events_rebuilt": manual_peak_events,
            },
        }
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
