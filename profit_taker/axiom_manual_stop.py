from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SESSION_TABLE = "axiom_v24_collection_run_sessions"
CENSOR_TABLE = "axiom_v24_collection_censors"
PEAK_SNAPSHOT_TABLE = "axiom_v24_collection_censor_peak_snapshots"
DEFAULT_ACTIVE_LOOKBACK_MINUTES = 50.0


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()} if _table_exists(conn, table) else set()


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(f"""
    CREATE TABLE IF NOT EXISTS {SESSION_TABLE}(
      session_id TEXT PRIMARY KEY, source TEXT NOT NULL, started_at TEXT NOT NULL,
      last_capture_at TEXT, stopped_at TEXT, censor_at TEXT, stop_reason TEXT,
      status TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS idx_{SESSION_TABLE}_status ON {SESSION_TABLE}(status,started_at);
    CREATE TABLE IF NOT EXISTS {CENSOR_TABLE}(
      session_id TEXT NOT NULL, token_key TEXT NOT NULL, last_seen_at TEXT NOT NULL,
      censor_at TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
      PRIMARY KEY(session_id,token_key));
    CREATE INDEX IF NOT EXISTS idx_{CENSOR_TABLE}_token_time ON {CENSOR_TABLE}(token_key,censor_at);
    CREATE TABLE IF NOT EXISTS {PEAK_SNAPSHOT_TABLE}(
      session_id TEXT NOT NULL, token_key TEXT NOT NULL, decision_at TEXT NOT NULL,
      row_json TEXT NOT NULL, censor_at TEXT NOT NULL, reason TEXT NOT NULL,
      created_at TEXT NOT NULL, PRIMARY KEY(session_id,token_key,decision_at));
    """)
    conn.commit()


def _observations(conn: sqlite3.Connection) -> pd.DataFrame:
    from . import axiom_peak_structure as peak
    obs, _ = peak.load_observations(conn)
    if obs.empty:
        return obs
    obs = obs.copy()
    obs["token_key"] = obs.token_key.astype(str)
    obs["snapshot_at"] = pd.to_datetime(obs.snapshot_at, utc=True, errors="coerce")
    return obs.dropna(subset=["snapshot_at"])


def _active_tokens(conn: sqlite3.Connection, censor_at: pd.Timestamp, lookback: float) -> dict[str, pd.Timestamp]:
    obs = _observations(conn)
    if obs.empty:
        return {}
    lo = censor_at - pd.Timedelta(minutes=float(lookback))
    recent = obs[(obs.snapshot_at <= censor_at) & (obs.snapshot_at >= lo)]
    if recent.empty:
        return {}
    s = recent.groupby("token_key", sort=False).snapshot_at.max()
    return {str(k): _utc(v) for k, v in s.items()}


def _snapshot_labels(conn: sqlite3.Connection, session_id: str, active: dict[str, pd.Timestamp], censor_at: pd.Timestamp, reason: str) -> int:
    table = "axiom_peak_structure_labels_v21"
    if not active or not _table_exists(conn, table):
        return 0
    cols = _columns(conn, table)
    if not {"token_key", "decision_at"}.issubset(cols):
        return 0
    names = [str(r[1]) for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
    unfinished = "AND COALESCE(label_finalized,0)=0" if "label_finalized" in cols else ""
    n = 0
    for token in active:
        for row in conn.execute(f'SELECT * FROM "{table}" WHERE token_key=? AND decision_at<=? {unfinished}', (token, censor_at.isoformat())).fetchall():
            rec = dict(zip(names, row))
            conn.execute(f"INSERT OR IGNORE INTO {PEAK_SNAPSHOT_TABLE}(session_id,token_key,decision_at,row_json,censor_at,reason,created_at) VALUES(?,?,?,?,?,?,?)", (session_id, token, str(rec["decision_at"]), json.dumps(rec, sort_keys=True, default=str), censor_at.isoformat(), reason, _now_iso()))
            n += int(conn.execute("SELECT changes()").fetchone()[0] > 0)
    return n


def _censor_paper_state(conn: sqlite3.Connection, censor_at: pd.Timestamp, reason: str) -> dict[str, int]:
    out = {"paper_positions_censored": 0, "pending_entries_cancelled": 0}
    if _table_exists(conn, "axiom_paper_positions_v20"):
        cols = _columns(conn, "axiom_paper_positions_v20")
        sets = ["status='censored'"]; vals: list[Any] = []
        if "closed_at" in cols: sets.append("closed_at=?"); vals.append(censor_at.isoformat())
        if "close_reason" in cols: sets.append("close_reason=?"); vals.append(reason)
        if "exit_kind" in cols: sets.append("exit_kind='collection_censored'")
        conn.execute(f"UPDATE axiom_paper_positions_v20 SET {','.join(sets)} WHERE status='open'", vals)
        out["paper_positions_censored"] = int(conn.execute("SELECT changes()").fetchone()[0])
    if _table_exists(conn, "axiom_paper_pending_entries_v24"):
        cols = _columns(conn, "axiom_paper_pending_entries_v24")
        sets = ["status='cancelled'"]; vals = []
        if "cancelled_at" in cols: sets.append("cancelled_at=?"); vals.append(censor_at.isoformat())
        if "cancel_reason" in cols: sets.append("cancel_reason=?"); vals.append(reason)
        conn.execute(f"UPDATE axiom_paper_pending_entries_v24 SET {','.join(sets)} WHERE status='pending'", vals)
        out["pending_entries_cancelled"] = int(conn.execute("SELECT changes()").fetchone()[0])
    return out


def _stop_conn(conn: sqlite3.Connection, session_id: str, *, reason: str, stopped_at: Any | None = None, active_lookback_minutes: float = DEFAULT_ACTIVE_LOOKBACK_MINUTES) -> dict[str, Any]:
    migrate(conn)
    row = conn.execute(f"SELECT last_capture_at,status FROM {SESSION_TABLE} WHERE session_id=?", (session_id,)).fetchone()
    if not row:
        return {"stopped": False, "reason": "session_not_found", "session_id": session_id}
    if str(row[1]) != "active":
        return {"stopped": False, "reason": "already_stopped", "session_id": session_id}
    stop_ts = _utc(stopped_at or pd.Timestamp.now(tz="UTC"))
    censor_at = _utc(row[0]) if row[0] else stop_ts
    active = _active_tokens(conn, censor_at, active_lookback_minutes)
    for token, last_seen in active.items():
        conn.execute(f"INSERT OR REPLACE INTO {CENSOR_TABLE}(session_id,token_key,last_seen_at,censor_at,reason,created_at) VALUES(?,?,?,?,?,?)", (session_id, token, last_seen.isoformat(), censor_at.isoformat(), reason, _now_iso()))
    snap = _snapshot_labels(conn, session_id, active, censor_at, reason)
    paper = _censor_paper_state(conn, censor_at, reason)
    conn.execute(f"UPDATE {SESSION_TABLE} SET stopped_at=?,censor_at=?,stop_reason=?,status='stopped' WHERE session_id=?", (stop_ts.isoformat(), censor_at.isoformat(), reason, session_id))
    conn.commit()
    return {"stopped": True, "session_id": session_id, "stopped_at": stop_ts.isoformat(), "censor_at": censor_at.isoformat(), "reason": reason, "active_tokens_censored": len(active), "peak_rows_snapshotted": snap, **paper}


def start_collection_session(db: str, *, source: str = "axiom_migrated_runner", started_at: Any | None = None) -> str:
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        migrate(conn)
        stale = conn.execute(f"SELECT session_id FROM {SESSION_TABLE} WHERE status='active' ORDER BY started_at").fetchall()
        for (old_id,) in stale:
            _stop_conn(conn, str(old_id), reason="unclean_restart_censored")
        sid = str(uuid.uuid4()); ts = _utc(started_at or pd.Timestamp.now(tz="UTC"))
        conn.execute(f"INSERT INTO {SESSION_TABLE}(session_id,source,started_at,status,created_at) VALUES(?,?,?,'active',?)", (sid, source, ts.isoformat(), _now_iso()))
        conn.commit(); return sid


def note_successful_capture(db: str, session_id: str, capture_at: Any | None = None) -> None:
    with sqlite3.connect(db) as conn:
        migrate(conn)
        if capture_at is None:
            row = conn.execute("SELECT MAX(captured_at) FROM capture_cycles").fetchone()
            capture_at = row[0] if row and row[0] else None
        if capture_at is not None:
            conn.execute(f"UPDATE {SESSION_TABLE} SET last_capture_at=? WHERE session_id=? AND status='active'", (_utc(capture_at).isoformat(), session_id)); conn.commit()


def stop_collection_session(db: str, session_id: str | None = None, *, reason: str = "manual_stop_censored", stopped_at: Any | None = None) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        migrate(conn)
        if session_id is None:
            row = conn.execute(f"SELECT session_id FROM {SESSION_TABLE} WHERE status='active' ORDER BY started_at DESC LIMIT 1").fetchone()
            if not row: return {"stopped": False, "reason": "no_active_session"}
            session_id = str(row[0])
        return _stop_conn(conn, session_id, reason=reason, stopped_at=stopped_at)


def first_collection_boundary_between(conn: sqlite3.Connection, start: Any, end: Any) -> pd.Timestamp | None:
    if not _table_exists(conn, SESSION_TABLE): return None
    a, b = _utc(start), _utc(end)
    row = conn.execute(f"SELECT censor_at FROM {SESSION_TABLE} WHERE status='stopped' AND censor_at IS NOT NULL AND censor_at>=? AND censor_at<=? ORDER BY censor_at LIMIT 1", (a.isoformat(), b.isoformat())).fetchone()
    return _utc(row[0]) if row else None


def censors_by_token(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    if not _table_exists(conn, CENSOR_TABLE): return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for token, censor_at, reason, last_seen_at, session_id in conn.execute(f"SELECT token_key,censor_at,reason,last_seen_at,session_id FROM {CENSOR_TABLE} ORDER BY token_key,censor_at").fetchall():
        out.setdefault(str(token), []).append({"censor_at": _utc(censor_at), "reason": str(reason), "last_seen_at": _utc(last_seen_at), "session_id": str(session_id)})
    return out


def censor_from_map(censor_map: dict[str, list[dict[str, Any]]], token_key: str, start: Any, end: Any) -> dict[str, Any] | None:
    a, b = _utc(start), _utc(end)
    for rec in censor_map.get(str(token_key), []):
        t = _utc(rec["censor_at"])
        if a <= t <= b: return rec
    return None


def apply_peak_label_censors(conn: sqlite3.Connection) -> dict[str, int]:
    migrate(conn)
    table = "axiom_peak_structure_labels_v21"
    if not _table_exists(conn, table): return {"restored": 0, "neutralized": 0}
    cols = _columns(conn, table); restored = neutralized = 0
    snapshots = conn.execute(f"SELECT row_json,censor_at,reason FROM {PEAK_SNAPSHOT_TABLE} ORDER BY censor_at").fetchall()
    snap_keys: set[tuple[str, str]] = set()
    for row_json, censor_at, reason in snapshots:
        try: original = json.loads(row_json)
        except Exception: continue
        token, decision = str(original.get("token_key") or ""), str(original.get("decision_at") or "")
        if not token or not decision: continue
        snap_keys.add((token, decision))
        updates = {k: v for k, v in original.items() if k in cols and k not in {"token_key", "decision_at", "config_json", "schema_version", "target_fingerprint"}}
        if "terminal_at" in cols: updates["terminal_at"] = censor_at
        if "terminal_reason" in cols: updates["terminal_reason"] = reason
        if "path_end_at" in cols: updates["path_end_at"] = censor_at
        if "label_finalized" in cols: updates["label_finalized"] = 0
        if "label_status_next_peak" in cols and not original.get("has_next_substantial_peak_before_terminal_72h"): updates["label_status_next_peak"] = "censored_collection_stop"
        if updates:
            conn.execute(f'UPDATE "{table}" SET ' + ",".join(f'"{k}"=?' for k in updates) + " WHERE token_key=? AND decision_at=?", [*updates.values(), token, decision]); restored += int(conn.execute("SELECT changes()").fetchone()[0] > 0)
    # If a decision label did not exist yet when Ctrl+C occurred, fail closed: do not
    # allow post-restart observations to manufacture a completed negative or positive.
    target_cols = [c for c in cols if c.startswith(("has_next_", "next_substantial_peak_", "later_higher_peak_", "post_next_peak_", "time_to_"))]
    for token, recs in censors_by_token(conn).items():
        for rec in recs:
            cutoff = _utc(rec["censor_at"]); reason = str(rec["reason"])
            rows = conn.execute(f'SELECT decision_at FROM "{table}" WHERE token_key=? AND decision_at<=?', (token, cutoff.isoformat())).fetchall()
            for (decision,) in rows:
                if (token, str(decision)) in snap_keys: continue
                try: d = _utc(decision)
                except Exception: continue
                if d + pd.Timedelta(hours=24) <= cutoff: continue
                sets = [f'"{c}"=NULL' for c in target_cols]
                if "label_finalized" in cols: sets.append('"label_finalized"=0')
                if "label_status_next_peak" in cols: sets.append('"label_status_next_peak"="censored_collection_stop"')
                if "terminal_at" in cols: sets.append('"terminal_at"=?')
                if "terminal_reason" in cols: sets.append('"terminal_reason"=?')
                if "path_end_at" in cols: sets.append('"path_end_at"=?')
                params: list[Any] = []
                if "terminal_at" in cols: params.append(cutoff.isoformat())
                if "terminal_reason" in cols: params.append(reason)
                if "path_end_at" in cols: params.append(cutoff.isoformat())
                if sets:
                    conn.execute(f'UPDATE "{table}" SET ' + ",".join(sets) + " WHERE token_key=? AND decision_at=?", [*params, token, decision]); neutralized += int(conn.execute("SELECT changes()").fetchone()[0] > 0)
    conn.commit(); return {"restored": restored, "neutralized": neutralized}


def status(db: str) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        migrate(conn)
        active = conn.execute(f"SELECT COUNT(*) FROM {SESSION_TABLE} WHERE status='active'").fetchone()[0]
        stopped = conn.execute(f"SELECT COUNT(*) FROM {SESSION_TABLE} WHERE status='stopped'").fetchone()[0]
        censors = conn.execute(f"SELECT COUNT(*) FROM {CENSOR_TABLE}").fetchone()[0]
        latest = conn.execute(f"SELECT session_id,source,started_at,last_capture_at,stopped_at,censor_at,stop_reason,status FROM {SESSION_TABLE} ORDER BY started_at DESC LIMIT 1").fetchone()
        keys = ["session_id","source","started_at","last_capture_at","stopped_at","censor_at","stop_reason","status"]
        return {"active_sessions": int(active), "stopped_sessions": int(stopped), "neutral_token_censors": int(censors), "latest_session": dict(zip(keys, latest)) if latest else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="V24 collection-stop neutral censor control"); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stop", "status"):
        p = sub.add_parser(name); p.add_argument("--db", default="data/live.sqlite")
    args = ap.parse_args(); out = stop_collection_session(args.db) if args.cmd == "stop" else status(args.db)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__": main()
