"""Production collector with explicit neutral-censor semantics for manual stops."""
from __future__ import annotations

import json
import sys
from typing import Any

from . import axiom_migrated_runner_pre_manual_stop as _current
from . import axiom_manual_stop as manual_stop
from .db import RAW_DB_DEFAULT

for _name in dir(_current):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_current, _name)

_active_manual_session: str | None = None
_active_db: str | None = None
_original_run_once = _current.run_once


def _db_from_argv(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--db" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--db="):
            return arg.split("=", 1)[1]
    return RAW_DB_DEFAULT


def _flag(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in argv)


def run_once(*args: Any, **kwargs: Any) -> dict:
    result = _original_run_once(*args, **kwargs)
    if _active_manual_session and _active_db:
        # Query the durable capture cycle written by process_rows so the censor
        # boundary is always the last actually persisted successful capture.
        manual_stop.note_successful_capture(_active_db, _active_manual_session)
    return result


def main() -> None:
    global _active_manual_session, _active_db
    argv = list(sys.argv[1:])
    replay = _flag(argv, "--clipboard-file")
    once = _flag(argv, "--once")
    db = _db_from_argv(argv)

    # Preserve all existing production initialization and sentinel/report behavior.
    purpose = "v24_replay_experiment" if replay else "v24_production_raw_collection"
    _current.initialize_collection(db, purpose=purpose)
    _current._sync_test_and_extension_hooks()

    # A one-shot/replay is a bounded capture, not a manually interrupted monitoring
    # session. Only the continuous production loop gets a durable censor session.
    if not replay and not once:
        _active_db = db
        _active_manual_session = manual_stop.start_collection_session(db)

    _current._impl.run_once = run_once
    try:
        _current._impl.main()
    except KeyboardInterrupt:
        if _active_manual_session and _active_db:
            outcome = manual_stop.stop_collection_session(
                _active_db,
                _active_manual_session,
                reason="manual_stop_censored",
            )
            print(json.dumps({"collection_stop": outcome}, indent=2, default=str), flush=True)
        # Ctrl+C is an intentional normal shutdown after the neutral boundary is durable.
        return
    finally:
        _current._impl.run_once = _current.run_once


if __name__ == "__main__":
    main()
