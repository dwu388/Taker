from __future__ import annotations

"""Durable neutral-censor boundaries for interrupted Axiom collection runs.

This module intentionally uses only the Python standard library so the lightweight
collector environment does not need the modeling stack (pandas/numpy/sklearn).
"""

import argparse
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .db import RAW_DB_DEFAULT, migrate as migrate_raw

RUN_TABLE = "axiom_v24_collection_run_sessions"
CENSOR_TABLE = "axiom_v24_collection_censors"
DEFAULT_ACTIVE_LOOKBACK_MINUTES = 50.0


def _utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(value: Any) -> str:
    return _utc(value).isoformat(timespec="milliseconds")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, name: str) -> set[str]:
    if not _table_exists(conn, name):
        return set()
    return {str(r[1]) for r in conn.execute(f'PRAGMA table_info("{name}")').fetchall()}


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
            run_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            started_at TEXT NOT NULL,
            last_cycle_id INTEGER,
            last_capture_at TEXT,
            stopped_at TEXT,
            censor_at TEXT,
            stop_reason TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_{RUN_TABLE}_status
            ON {RUN_TABLE}(status, started_at);

        CREATE TABLE IF NOT EXISTS {CENSOR_TABLE} (
            run_id TEXT NOT NULL,
            token_key TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            censor_at TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(run_id, token_key)
        );
        CREATE INDEX IF NOT EXISTS idx_{CENSOR_TABLE}_token_time
            ON {CENSOR_TABLE}(token_key, censor_at);
        """
    )
    conn.commit()


def _latest_successful_cycle(
    conn: sqlite3.Connection, started_at: Any
) -> tuple[int, datetime] | None:
    if not _table_exists(conn, "capture_cycles"):
        return None
    row = conn.execute(
        """
        SELECT cycle_id, captured_at
        FROM capture_cycles
        WHERE completed=1 AND clipboard_valid=1
          AND julianday(captured_at) >= julianday(?)
        ORDER BY julianday(captured_at) DESC, cycle_id DESC
        LIMIT 1
        """,
        (_iso(started_at),),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), _utc(row[1])


def _active_tokens(
    conn: sqlite3.Connection,
    censor_at: datetime,
    lookback_minutes: float,
) -> dict[str, datetime]:
    if not _table_exists(conn, "axiom_observations"):
        return {}
    rows = conn.execute(
        """
        SELECT token_key, MAX(snapshot_at) AS last_seen_at
        FROM axiom_observations
        WHERE julianday(snapshot_at) <= julianday(?)
          AND julianday(snapshot_at) > julianday(?) - (? / 1440.0)
        GROUP BY token_key
        """,
        (_iso(censor_at), _iso(censor_at), float(lookback_minutes)),
    ).fetchall()
    out: dict[str, datetime] = {}
    for token, last_seen in rows:
        if token is None or last_seen is None:
            continue
        out[str(token)] = _utc(last_seen)
    return out


def _censor_paper_state(
    conn: sqlite3.Connection, censor_at: datetime, reason: str
) -> dict[str, int]:
    result = {"paper_positions_censored": 0, "pending_entries_cancelled": 0}

    if _table_exists(conn, "axiom_paper_positions_v20"):
        cols = _columns(conn, "axiom_paper_positions_v20")
        sets = ["status='censored'"]
        params: list[Any] = []
        if "closed_at" in cols:
            sets.append("closed_at=?")
            params.append(_iso(censor_at))
        if "close_reason" in cols:
            sets.append("close_reason=?")
            params.append(reason)
        if "exit_kind" in cols:
            sets.append("exit_kind='collection_censored'")
        if "pending_exit_at" in cols:
            sets.append("pending_exit_at=NULL")
        if "pending_exit_reason" in cols:
            sets.append("pending_exit_reason=NULL")
        for name in (
            "exit_mc", "gross_return_pct", "net_return_pct", "peak_capture_ratio", "reward",
            "exit_mc_observed", "exit_mc_execution_proxy", "observed_gross_return_pct",
            "execution_gross_return_pct", "observed_net_return_pct", "execution_net_return_pct",
            "observed_peak_capture_ratio", "execution_peak_capture_ratio",
            "observed_reward", "execution_reward",
        ):
            if name in cols:
                sets.append(f'"{name}"=NULL')
        conn.execute(
            f"UPDATE axiom_paper_positions_v20 SET {','.join(sets)} WHERE status='open'",
            params,
        )
        result["paper_positions_censored"] = int(conn.execute("SELECT changes()").fetchone()[0])

    if _table_exists(conn, "axiom_paper_pending_entries_v24"):
        cols = _columns(conn, "axiom_paper_pending_entries_v24")
        sets = ["status='cancelled'"]
        params = []
        if "cancelled_at" in cols:
            sets.append("cancelled_at=?")
            params.append(_iso(censor_at))
        if "cancel_reason" in cols:
            sets.append("cancel_reason=?")
            params.append(reason)
        conn.execute(
            f"UPDATE axiom_paper_pending_entries_v24 SET {','.join(sets)} WHERE status='pending'",
            params,
        )
        result["pending_entries_cancelled"] = int(conn.execute("SELECT changes()").fetchone()[0])

    return result


def _stop_conn(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    reason: str,
    stopped_at: Any | None = None,
    active_lookback_minutes: float = DEFAULT_ACTIVE_LOOKBACK_MINUTES,
) -> dict[str, Any]:
    migrate(conn)
    row = conn.execute(
        f"SELECT started_at,last_cycle_id,last_capture_at,status FROM {RUN_TABLE} WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if row is None:
        return {"stopped": False, "reason": "run_not_found", "run_id": run_id}
    if str(row[3]) != "active":
        return {"stopped": False, "reason": "already_stopped", "run_id": run_id}

    stop_time = _utc(stopped_at or _now())
    last_cycle_id = int(row[1]) if row[1] is not None else None
    last_capture = _utc(row[2]) if row[2] else None

    # Always reconcile against durable SQLite truth. Ctrl+C can arrive after a
    # capture transaction commits but before the runner records its convenience
    # heartbeat; the committed cycle must still be the censor boundary.
    latest = _latest_successful_cycle(conn, row[0])
    if latest is not None:
        latest_cycle_id, latest_capture = latest
        if last_capture is None or latest_capture > last_capture:
            last_cycle_id, last_capture = latest_cycle_id, latest_capture

    censor_at = last_capture or stop_time
    active = _active_tokens(conn, censor_at, active_lookback_minutes) if last_capture else {}
    created = _iso(_now())
    for token, last_seen in active.items():
        conn.execute(
            f"""INSERT OR REPLACE INTO {CENSOR_TABLE}
                (run_id,token_key,last_seen_at,censor_at,reason,created_at)
                VALUES(?,?,?,?,?,?)""",
            (run_id, token, _iso(last_seen), _iso(censor_at), reason, created),
        )

    paper = _censor_paper_state(conn, censor_at, reason)
    conn.execute(
        f"""UPDATE {RUN_TABLE}
            SET last_cycle_id=?,last_capture_at=?,stopped_at=?,censor_at=?,stop_reason=?,status='stopped'
            WHERE run_id=?""",
        (
            last_cycle_id,
            _iso(last_capture) if last_capture else None,
            _iso(stop_time),
            _iso(censor_at),
            reason,
            run_id,
        ),
    )
    conn.commit()
    return {
        "stopped": True,
        "run_id": run_id,
        "stopped_at": _iso(stop_time),
        "censor_at": _iso(censor_at),
        "reason": reason,
        "active_tokens_censored": len(active),
        **paper,
    }


def start_collection_session(
    db_path: str,
    *,
    source: str = "axiom_migrated_runner",
    started_at: Any | None = None,
) -> str:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    migrate_raw(db_path)
    with sqlite3.connect(db_path) as conn:
        migrate(conn)
        active = [
            str(r[0])
            for r in conn.execute(
                f"SELECT run_id FROM {RUN_TABLE} WHERE status='active' ORDER BY started_at"
            ).fetchall()
        ]
        for old_run in active:
            _stop_conn(conn, old_run, reason="unclean_restart_censored")
        run_id = str(uuid.uuid4())
        start = _utc(started_at or _now())
        conn.execute(
            f"""INSERT INTO {RUN_TABLE}
                (run_id,source,started_at,status,created_at) VALUES(?,?,?,'active',?)""",
            (run_id, source, _iso(start), _iso(_now())),
        )
        conn.commit()
        return run_id


def note_successful_capture(
    db_path: str,
    run_id: str,
    *,
    capture_at: Any | None = None,
    cycle_id: int | None = None,
) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        migrate(conn)
        row = conn.execute(
            f"SELECT started_at,status FROM {RUN_TABLE} WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None or str(row[1]) != "active":
            return {"updated": False, "reason": "run_not_active"}
        if capture_at is None or cycle_id is None:
            latest = _latest_successful_cycle(conn, row[0])
            if latest is None:
                return {"updated": False, "reason": "no_successful_capture"}
            found_cycle, found_at = latest
            if cycle_id is None:
                cycle_id = found_cycle
            if capture_at is None:
                capture_at = found_at
        conn.execute(
            f"UPDATE {RUN_TABLE} SET last_cycle_id=?,last_capture_at=? WHERE run_id=? AND status='active'",
            (int(cycle_id), _iso(capture_at), run_id),
        )
        conn.commit()
        return {"updated": True, "run_id": run_id, "capture_at": _iso(capture_at), "cycle_id": int(cycle_id)}


def stop_collection_session(
    db_path: str,
    run_id: str | None = None,
    *,
    reason: str = "manual_stop_censored",
    stopped_at: Any | None = None,
) -> dict[str, Any]:
    migrate_raw(db_path)
    with sqlite3.connect(db_path) as conn:
        migrate(conn)
        if run_id is None:
            row = conn.execute(
                f"SELECT run_id FROM {RUN_TABLE} WHERE status='active' ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return {"stopped": False, "reason": "no_active_run"}
            run_id = str(row[0])
        return _stop_conn(conn, run_id, reason=reason, stopped_at=stopped_at)


def censors_by_token(conn: sqlite3.Connection) -> dict[str, list[dict[str, str]]]:
    if not _table_exists(conn, CENSOR_TABLE):
        return {}
    out: dict[str, list[dict[str, str]]] = {}
    for run_id, token, last_seen, censor_at, reason in conn.execute(
        f"SELECT run_id,token_key,last_seen_at,censor_at,reason FROM {CENSOR_TABLE} ORDER BY token_key,censor_at"
    ):
        out.setdefault(str(token), []).append(
            {
                "run_id": str(run_id),
                "last_seen_at": _iso(last_seen),
                "censor_at": _iso(censor_at),
                "reason": str(reason),
            }
        )
    return out


def prune_counterfactual_targets(
    conn: sqlite3.Connection,
    table_name: str = "axiom_v24_counterfactual_policy_targets",
) -> int:
    """Remove policy targets whose required future window crosses a stop censor."""
    if not _table_exists(conn, table_name) or not _table_exists(conn, CENSOR_TABLE):
        return 0
    cols = _columns(conn, table_name)
    if not {"token_key", "decision_at", "horizon_minutes"}.issubset(cols):
        return 0
    censors = {
        token: [_utc(r["censor_at"]) for r in rows]
        for token, rows in censors_by_token(conn).items()
    }
    delete_ids: list[int] = []
    for rowid, token, decision_at, horizon in conn.execute(
        f'SELECT rowid,token_key,decision_at,horizon_minutes FROM "{table_name}"'
    ).fetchall():
        decision = _utc(decision_at)
        deadline = decision + timedelta(minutes=float(horizon))
        if any(decision <= c <= deadline for c in censors.get(str(token), [])):
            delete_ids.append(int(rowid))
    if delete_ids:
        conn.executemany(
            f'DELETE FROM "{table_name}" WHERE rowid=?', [(x,) for x in delete_ids]
        )
        conn.commit()
    return len(delete_ids)


def status(db_path: str) -> dict[str, Any]:
    migrate_raw(db_path)
    with sqlite3.connect(db_path) as conn:
        migrate(conn)
        active = int(conn.execute(f"SELECT COUNT(*) FROM {RUN_TABLE} WHERE status='active'").fetchone()[0])
        stopped = int(conn.execute(f"SELECT COUNT(*) FROM {RUN_TABLE} WHERE status='stopped'").fetchone()[0])
        censored = int(conn.execute(f"SELECT COUNT(*) FROM {CENSOR_TABLE}").fetchone()[0])
        last = conn.execute(
            f"SELECT run_id,censor_at,stop_reason FROM {RUN_TABLE} WHERE status='stopped' ORDER BY stopped_at DESC LIMIT 1"
        ).fetchone()
        return {
            "active_runs": active,
            "stopped_runs": stopped,
            "token_censors": censored,
            "last_stop": {"run_id": last[0], "censor_at": last[1], "reason": last[2]} if last else None,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record or inspect neutral V24 collection-stop censors")
    sub = parser.add_subparsers(dest="cmd", required=True)
    stop = sub.add_parser("stop")
    stop.add_argument("--db", default=RAW_DB_DEFAULT)
    stat = sub.add_parser("status")
    stat.add_argument("--db", default=RAW_DB_DEFAULT)
    args = parser.parse_args(argv)
    out = stop_collection_session(args.db) if args.cmd == "stop" else status(args.db)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
