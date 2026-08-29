"""Production collection administration with end-to-end integrity checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from typing import Any, Sequence

from . import collection_admin_base as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


def _payload_integrity(con) -> dict[str, int]:
    checked = 0
    errors = 0
    for row in con.execute("SELECT sha256,byte_count,compression,payload FROM capture_payloads"):
        checked += 1
        try:
            if str(row["compression"]) != "zlib":
                raise ValueError(f"unsupported compression {row['compression']}")
            raw = zlib.decompress(row["payload"])
            if len(raw) != int(row["byte_count"]):
                raise ValueError("byte_count mismatch")
            if hashlib.sha256(raw).hexdigest() != str(row["sha256"]):
                raise ValueError("sha256 mismatch")
        except Exception:
            errors += 1
    return {"checked": checked, "errors": errors}


def collection_status(db_path: str) -> dict[str, Any]:
    base = _impl.collection_status(db_path)
    con = connect(db_path)
    try:
        sessions = con.execute(
            "SELECT session_id FROM collection_sessions ORDER BY started_at, created_at"
        ).fetchall()
        active_session = str(sessions[0][0]) if len(sessions) == 1 else None
        cycle_row_mismatches = int(_scalar(con, """
            SELECT COUNT(*) FROM (
              SELECT c.cycle_id,c.rows_detected,COUNT(o.observation_id) AS stored_rows
              FROM capture_cycles c
              LEFT JOIN axiom_observations o ON o.cycle_id=c.cycle_id
              GROUP BY c.cycle_id,c.rows_detected
              HAVING COUNT(o.observation_id)<>c.rows_detected
            )
        """) or 0)
        if active_session is None:
            cycles_wrong_session = int(_scalar(con, "SELECT COUNT(*) FROM capture_cycles") or 0)
            attempts_wrong_session = int(_scalar(con, "SELECT COUNT(*) FROM capture_attempts") or 0)
        else:
            cycles_wrong_session = int(_scalar(
                con,
                "SELECT COUNT(*) FROM capture_cycles WHERE session_id IS NULL OR session_id<>?",
                (active_session,),
            ) or 0)
            attempts_wrong_session = int(_scalar(
                con,
                "SELECT COUNT(*) FROM capture_attempts WHERE session_id IS NULL OR session_id<>?",
                (active_session,),
            ) or 0)
        successful_attempt_cycle_mismatch = abs(
            int(base.get("successful_attempts") or 0) - int(base.get("capture_cycles") or 0)
        )
        payload = _payload_integrity(con)
    finally:
        con.close()

    base.update({
        "cycle_row_count_mismatches": cycle_row_mismatches,
        "cycles_outside_active_session": cycles_wrong_session,
        "attempts_outside_active_session": attempts_wrong_session,
        "successful_attempt_cycle_count_difference": successful_attempt_cycle_mismatch,
        "raw_payload_integrity_checked": payload["checked"],
        "raw_payload_integrity_errors": payload["errors"],
    })
    base["ready_to_collect"] = bool(
        base.get("ready_to_collect")
        and cycle_row_mismatches == 0
        and cycles_wrong_session == 0
        and attempts_wrong_session == 0
        and successful_attempt_cycle_mismatch == 0
        and payload["errors"] == 0
    )
    base["note"] = (
        "Collection readiness requires one marked session, SQLite integrity, identity consistency, "
        "one intact raw payload per successful cycle, exact detected/stored row counts, and matching "
        "successful-attempt/cycle accounting. No elapsed-day training gate is hardcoded."
    )
    return base


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize and audit the canonical V24 raw collection database")
    sub = parser.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init")
    init.add_argument("--db", default=RAW_DB_DEFAULT)
    init.add_argument("--purpose", default="v24_production_raw_collection")
    status = sub.add_parser("status")
    status.add_argument("--db", default=RAW_DB_DEFAULT)
    args = parser.parse_args(argv)
    out = initialize_collection(args.db, args.purpose) if args.cmd == "init" else collection_status(args.db)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
