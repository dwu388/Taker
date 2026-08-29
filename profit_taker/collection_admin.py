from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .db import COLLECTOR_SCHEMA_VERSION, RAW_DB_DEFAULT, connect, migrate


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _scalar(con: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = con.execute(sql, params).fetchone()
    return row[0] if row else None


def initialize_collection(db_path: str, purpose: str = "v24_production_raw_collection") -> dict[str, Any]:
    migrate(db_path)
    con = connect(db_path)
    try:
        sessions = con.execute("SELECT session_id,started_at,purpose,collector_schema FROM collection_sessions ORDER BY started_at").fetchall()
        if len(sessions) > 1:
            raise RuntimeError("Raw database has multiple collection sessions; refusing automatic production collection")
        if sessions:
            row = sessions[0]
            return {"initialized": False, "resuming": True, "session_id": row["session_id"], "started_at": row["started_at"], "purpose": row["purpose"], "collector_schema": row["collector_schema"], "db": str(db_path)}
        raw_counts = {
            "capture_cycles": int(_scalar(con, "SELECT COUNT(*) FROM capture_cycles") or 0),
            "observations": int(_scalar(con, "SELECT COUNT(*) FROM axiom_observations") or 0),
            "attempts": int(_scalar(con, "SELECT COUNT(*) FROM capture_attempts") or 0),
        }
        if any(raw_counts.values()):
            raise RuntimeError("Database contains unmarked collection data; refusing to call it a fresh production dataset. " f"Counts={raw_counts}. Use a new database path or explicitly migrate/adopt it outside the production launcher.")
        session_id = str(uuid.uuid4())
        started_at = _now_iso()
        con.execute("INSERT INTO collection_sessions(session_id,started_at,purpose,collector_schema) VALUES(?,?,?,?)", (session_id, started_at, purpose, COLLECTOR_SCHEMA_VERSION))
        con.commit()
        return {"initialized": True, "resuming": False, "session_id": session_id, "started_at": started_at, "purpose": purpose, "collector_schema": COLLECTOR_SCHEMA_VERSION, "db": str(db_path)}
    except Exception:
        con.rollback(); raise
    finally:
        con.close()


def _identity_conflicts(con: sqlite3.Connection) -> dict[str, int]:
    token_key_multi_mint = int(_scalar(con, """SELECT COUNT(*) FROM (SELECT token_key FROM axiom_observations WHERE token_address IS NOT NULL AND token_address<>'' GROUP BY token_key HAVING COUNT(DISTINCT token_address)>1)""") or 0)
    mint_multi_key = int(_scalar(con, """SELECT COUNT(*) FROM (SELECT token_address FROM axiom_observations WHERE token_address IS NOT NULL AND token_address<>'' GROUP BY token_address HAVING COUNT(DISTINCT token_key)>1)""") or 0)
    hint_multi_mint = int(_scalar(con, """SELECT COUNT(*) FROM (SELECT short_address_hint FROM axiom_observations WHERE short_address_hint IS NOT NULL AND short_address_hint<>'' AND token_address IS NOT NULL AND token_address<>'' GROUP BY short_address_hint HAVING COUNT(DISTINCT token_address)>1)""") or 0)
    return {"token_key_to_multiple_mints": token_key_multi_mint, "mint_to_multiple_token_keys": mint_multi_key, "short_hint_to_multiple_mints": hint_multi_mint}


def collection_status(db_path: str) -> dict[str, Any]:
    migrate(db_path)
    con = connect(db_path)
    try:
        integrity_row = con.execute("PRAGMA integrity_check").fetchone()
        integrity = str(integrity_row[0]) if integrity_row else "unknown"
        sessions = [dict(r) for r in con.execute("SELECT session_id,started_at,purpose,collector_schema,created_at FROM collection_sessions ORDER BY started_at").fetchall()]
        cycles = int(_scalar(con, "SELECT COUNT(*) FROM capture_cycles") or 0)
        observations = int(_scalar(con, "SELECT COUNT(*) FROM axiom_observations") or 0)
        attempts = int(_scalar(con, "SELECT COUNT(*) FROM capture_attempts") or 0)
        successful_attempts = int(_scalar(con, "SELECT COUNT(*) FROM capture_attempts WHERE success=1") or 0)
        failed_attempts = int(_scalar(con, "SELECT COUNT(*) FROM capture_attempts WHERE success=0") or 0)
        payload_cycles = int(_scalar(con, "SELECT COUNT(*) FROM capture_payloads") or 0)
        unique_tokens = int(_scalar(con, "SELECT COUNT(DISTINCT token_key) FROM axiom_observations") or 0)
        full_mint_rows = int(_scalar(con, "SELECT COUNT(*) FROM axiom_observations WHERE token_address IS NOT NULL AND token_address<>''") or 0)
        conflicts = _identity_conflicts(con)
        session_marked = len(sessions) == 1
        payload_complete = cycles == payload_cycles
        identity_ok = not any(conflicts.values())
        return {
            "db": str(Path(db_path)), "collector_schema": COLLECTOR_SCHEMA_VERSION,
            "session_marked": session_marked, "sessions": sessions,
            "sqlite_integrity": integrity, "sqlite_integrity_ok": integrity.lower() == "ok",
            "capture_cycles": cycles, "capture_attempts": attempts,
            "successful_attempts": successful_attempts, "failed_attempts": failed_attempts,
            "attempt_success_rate": (successful_attempts / attempts) if attempts else None,
            "first_capture_at": _scalar(con, "SELECT MIN(captured_at) FROM capture_cycles"),
            "last_capture_at": _scalar(con, "SELECT MAX(captured_at) FROM capture_cycles"),
            "rows_detected_min": _scalar(con, "SELECT MIN(rows_detected) FROM capture_cycles"),
            "rows_detected_avg": _scalar(con, "SELECT AVG(rows_detected) FROM capture_cycles"),
            "rows_detected_max": _scalar(con, "SELECT MAX(rows_detected) FROM capture_cycles"),
            "observations": observations, "unique_tokens": unique_tokens,
            "full_mint_rows": full_mint_rows, "short_only_rows": observations - full_mint_rows,
            "full_mint_row_fraction": (full_mint_rows / observations) if observations else None,
            "raw_payload_cycles": payload_cycles, "raw_payload_complete": payload_complete,
            "identity_conflicts": conflicts, "identity_ok": identity_ok,
            "ready_to_collect": bool(session_marked and integrity.lower() == "ok" and identity_ok and payload_complete),
            "duration_gate": None,
            "note": "Collection readiness is integrity-based; no 17-day or 20-day stop/training trigger is hardcoded.",
        }
    finally:
        con.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize and audit the canonical V24 raw collection database")
    sub = parser.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init"); init.add_argument("--db", default=RAW_DB_DEFAULT); init.add_argument("--purpose", default="v24_production_raw_collection")
    status = sub.add_parser("status"); status.add_argument("--db", default=RAW_DB_DEFAULT)
    args = parser.parse_args(argv)
    out = initialize_collection(args.db, args.purpose) if args.cmd == "init" else collection_status(args.db)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
