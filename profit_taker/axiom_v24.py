"""Active V24 facade with explicit neutral censor boundaries for collector stops."""
from __future__ import annotations

import hashlib
import sqlite3

import numpy as np
import pandas as pd

from . import axiom_v24_pre_manual_stop as _current
from . import axiom_manual_stop as manual_stop

for _name in dir(_current):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_current, _name)

_impl = _current._impl
_base = _current._base
_original_contiguous_capture_absence = _impl._contiguous_capture_absence
_original_add_barrier_targets = _impl.add_barrier_targets
_original_add_recurrent_targets = _current.add_recurrent_targets


def _contiguous_capture_absence(conn: sqlite3.Connection, last_seen: pd.Timestamp, cfg: V24Config, *, upto: pd.Timestamp | None = None):
    end = _utc(upto or pd.Timestamp.now(tz="UTC"))
    # No successful captures after a manual/unclean stop may be used to prove a
    # token absent from the monitoring session that ended before that boundary.
    if manual_stop.first_collection_boundary_between(conn, last_seen, end) is not None:
        return None, None, 0
    return _original_contiguous_capture_absence(conn, last_seen, cfg, upto=end)


def refresh_lifetimes(conn: sqlite3.Connection, cfg: V24Config) -> dict[str, int]:
    """Split token episodes at collection-stop boundaries and right-censor them."""
    migrate(conn)
    manual_stop.migrate(conn)
    obs, _ = peak.load_observations(conn)
    if obs.empty:
        return {"tokens": 0, "lifetimes": 0, "new_lifetimes": 0}
    refresh_capture_heartbeats(conn, obs)
    latest_capture = _utc(obs.snapshot_at.max())
    censors = manual_stop.censors_by_token(conn)
    old_total = int(conn.execute(f"SELECT COUNT(*) FROM {LIFETIME_TABLE}").fetchone()[0])

    for token, g in obs.groupby("token_key", sort=False):
        token = str(token); g = g.sort_values("snapshot_at").reset_index(drop=True)
        conn.execute(f"DELETE FROM {LIFETIME_TABLE} WHERE token_key=?", (token,))
        start = last = _utc(g.iloc[0].snapshot_at); count = 1
        segments: list[tuple[pd.Timestamp,pd.Timestamp,int,str | None,pd.Timestamp | None]] = []
        for j in range(1, len(g)):
            t = _utc(g.iloc[j].snapshot_at)
            censor = manual_stop.censor_from_map(censors, token, last, t)
            if censor is not None:
                reason = "natural_" + str(censor["reason"])
                segments.append((start, last, count, reason, _utc(censor["censor_at"])))
                start = last = t; count = 1
                continue
            run_start, run_end, ncap = _contiguous_capture_absence(conn, last, cfg, upto=t)
            terminal = None
            if run_start is not None and run_end is not None and (run_end-run_start).total_seconds()/60.0 >= cfg.operational_gap_minutes and ncap >= cfg.heartbeat_min_valid_captures_for_death:
                terminal = run_start + pd.Timedelta(minutes=cfg.operational_gap_minutes)
            if terminal is not None and terminal < t:
                segments.append((start, last, count, "operational_gap_then_reappearance", terminal))
                start = t; count = 1
            else:
                count += 1
            last = t

        reason = None; terminal_at = None
        censor = manual_stop.censor_from_map(censors, token, last, pd.Timestamp.now(tz="UTC"))
        if censor is not None:
            reason = "natural_" + str(censor["reason"]); terminal_at = _utc(censor["censor_at"])
        else:
            age = pd.to_numeric(pd.Series([g.iloc[-1].get("age_minutes")]), errors="coerce").iloc[0] if "age_minutes" in g.columns else np.nan
            if pd.notna(age) and float(age) >= cfg.age_out_minutes:
                reason = "natural_axiom_age_out"; terminal_at = last
            else:
                run_start, run_end, ncap = _contiguous_capture_absence(conn, last, cfg, upto=latest_capture)
                if run_start is not None and run_end is not None and (run_end-run_start).total_seconds()/60.0 >= cfg.operational_gap_minutes and ncap >= cfg.heartbeat_min_valid_captures_for_death:
                    reason = "dead_after_valid_capture_absence"; terminal_at = run_start + pd.Timedelta(minutes=cfg.operational_gap_minutes)
        segments.append((start, last, count, reason, terminal_at))

        for ep, (first_seen, last_seen, n, terminal_reason, terminal) in enumerate(segments):
            lid = hashlib.sha256(f"{token}|{ep}|{first_seen.isoformat()}".encode()).hexdigest()[:24]
            conn.execute(f"INSERT INTO {LIFETIME_TABLE}(lifetime_id,token_key,episode_index,first_seen_at,last_seen_at,terminal_at,terminal_reason,observation_count) VALUES(?,?,?,?,?,?,?,?)", (lid, token, ep, first_seen.isoformat(), last_seen.isoformat(), terminal.isoformat() if terminal is not None else None, terminal_reason, int(n)))
    conn.commit()
    total = int(conn.execute(f"SELECT COUNT(*) FROM {LIFETIME_TABLE}").fetchone()[0])
    return {"tokens": int(obs.token_key.nunique()), "lifetimes": total, "new_lifetimes": max(0, total-old_total)}


def add_barrier_targets(conn: sqlite3.Connection, frame: pd.DataFrame, cfg: V24Config, *, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    out = _original_add_barrier_targets(conn, frame, cfg, as_of=as_of)
    censors = manual_stop.censors_by_token(conn)
    for i, row in out.iterrows():
        decision = _utc(row.decision_at)
        for h in cfg.probability_horizons_minutes:
            censor = manual_stop.censor_from_map(censors, str(row.token_key), decision, decision + pd.Timedelta(minutes=int(h)))
            if censor is not None:
                for thr in cfg.upside_thresholds:
                    col = f"hit_plus{_threshold_tag(thr)}_by_{h}m"
                    if col in out.columns: out.at[i, col] = np.nan
    return out


def add_recurrent_targets(conn: sqlite3.Connection, frame: pd.DataFrame, cfg: V24Config, *, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    out = _original_add_recurrent_targets(conn, frame, cfg, as_of=as_of)
    censors = manual_stop.censors_by_token(conn)
    for i, row in out.iterrows():
        decision = _utc(row.decision_at)
        for h in RECURRENT_HORIZONS_MINUTES:
            censor = manual_stop.censor_from_map(censors, str(row.token_key), decision, decision + pd.Timedelta(minutes=int(h)))
            if censor is None: continue
            for col in list(out.columns):
                if col.endswith(f"_{h}m") and (col.startswith("recurrent_peak_count_") or col.startswith("second_peak_by_") or col.startswith("later_higher_")):
                    out.at[i, col] = np.nan
    return out


for module in (_impl, _base):
    module._contiguous_capture_absence = _contiguous_capture_absence
    module.refresh_lifetimes = refresh_lifetimes
    module.add_barrier_targets = add_barrier_targets
    module.add_recurrent_targets = add_recurrent_targets


def __getattr__(name: str):
    return getattr(_current, name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
