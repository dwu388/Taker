"""Production-hardened public collector facade.

The previous collector is preserved byte-for-byte in
:mod:`profit_taker.axiom_migrated_runner_base`. This facade adds production
invariants without changing parsing/cadence behavior:

1. every browser copy is preceded by a unique verified clipboard sentinel, so a
   failed clipboard clear/copy can never replay the previous valid Axiom snapshot;
2. direct invocation initializes/resumes a session whose purpose matches capture
   provenance, keeping replay files out of the authoritative production session;
3. after successful production cycles, a human-readable model-performance report
   is regenerated only when its configured interval has elapsed;
4. Ctrl+C closes the current collector run at its last durable successful capture,
   creating a neutral censor boundary rather than manufacturing token deaths.
"""
from __future__ import annotations

import json
import sys
import uuid
from typing import Any

from . import axiom_manual_stop as manual_stop
from . import axiom_migrated_runner_base as _impl
from .collection_admin import initialize_collection
from .common import load_json
from .db import RAW_DB_DEFAULT
from .performance_report import (
    BENCHMARK_DB_DEFAULT,
    RECENT_EVENTS_DEFAULT,
    REPORT_INTERVAL_MINUTES_DEFAULT,
    REPORT_OUTPUT_DEFAULT,
    maybe_generate_report,
)

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

_original_capture_clipboard = _impl._capture_clipboard
_original_run_once = _impl.run_once
_active_manual_run_id: str | None = None
_active_manual_db: str | None = None


def _prime_clipboard_with_sentinel(pyperclip: Any) -> str:
    sentinel = f"__V24_CLIPBOARD_SENTINEL_{uuid.uuid4().hex}__"
    try:
        pyperclip.copy(sentinel)
        observed = pyperclip.paste()
    except Exception as exc:
        raise RuntimeError(
            "Could not prime/verify the clipboard before Axiom capture; refusing a capture that could reuse stale text"
        ) from exc
    if observed != sentinel:
        raise RuntimeError(
            "Clipboard sentinel verification failed before Axiom capture; refusing a capture that could reuse stale text"
        )
    return sentinel


def _capture_clipboard(cfg: dict, cycle_count: int) -> tuple[str, str]:
    try:
        import pyperclip
    except Exception as exc:
        raise RuntimeError("Clipboard collection requires pyperclip in an interactive desktop session") from exc
    _prime_clipboard_with_sentinel(pyperclip)
    return _original_capture_clipboard(cfg, cycle_count)


_impl._capture_clipboard = _capture_clipboard


def _sync_test_and_extension_hooks() -> None:
    """Keep monkeypatch/extension points on the public module effective."""
    for name in (
        "clipboard_looks_like_axiom",
        "rows_from_clipboard",
        "clipboard_diagnostics",
        "diagnostics_json",
        "_raw_mc_card_count",
        "process_rows",
        "record_failed_capture_attempt",
    ):
        if name in globals():
            setattr(_impl, name, globals()[name])
    _impl._capture_clipboard = _capture_clipboard


def _report_settings(args: Any) -> dict[str, Any]:
    config_path = str(getattr(args, "config", "axiom_migrated_config.json"))
    cfg = load_json(config_path, {}) or {}
    raw = cfg.get("performance_report", {}) if isinstance(cfg, dict) else {}
    section = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(section.get("enabled", True)),
        "interval_minutes": max(1, int(section.get("interval_minutes", REPORT_INTERVAL_MINUTES_DEFAULT))),
        "output_path": str(section.get("output_path", REPORT_OUTPUT_DEFAULT)),
        "benchmark_db": str(section.get("benchmark_db", BENCHMARK_DB_DEFAULT)),
        "recent_events": max(1, int(section.get("recent_events", RECENT_EVENTS_DEFAULT))),
    }


def _maybe_refresh_performance_report(args: Any) -> dict[str, Any]:
    settings = _report_settings(args)
    if not settings["enabled"]:
        return {"generated": False, "reason": "disabled"}
    return maybe_generate_report(
        str(getattr(args, "db", RAW_DB_DEFAULT)),
        output_path=settings["output_path"],
        benchmark_db=settings["benchmark_db"],
        interval_minutes=settings["interval_minutes"],
        recent_events=settings["recent_events"],
    )


def run_once(*args: Any, **kwargs: Any) -> dict:
    _sync_test_and_extension_hooks()
    result = _original_run_once(*args, **kwargs)
    collector_args = args[0] if args else kwargs.get("args")
    if collector_args is not None:
        try:
            result["performance_report"] = _maybe_refresh_performance_report(collector_args)
        except Exception as exc:
            # Reporting is diagnostic. It must never turn a durable successful raw
            # capture into a failed collection cycle.
            result["performance_report"] = {
                "generated": False,
                "reason": "report_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    if _active_manual_run_id and _active_manual_db:
        # This executes only after process_rows has durably committed the capture.
        # The stop boundary therefore cannot advance past uncommitted/failed data.
        manual_stop.note_successful_capture(_active_manual_db, _active_manual_run_id)
    return result


# Base ``main`` resolves ``run_once`` in its own module globals. Bind the wrapper
# there so the actual one-minute CLI loop receives the same scheduled reporting
# behavior that direct callers/tests receive.
_impl.run_once = run_once


def _db_from_argv(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--db" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--db="):
            return arg.split("=", 1)[1]
    return RAW_DB_DEFAULT


def _has_clipboard_file(argv: list[str]) -> bool:
    return any(arg == "--clipboard-file" or arg.startswith("--clipboard-file=") for arg in argv)


def _has_once(argv: list[str]) -> bool:
    return any(arg == "--once" or arg.startswith("--once=") for arg in argv)


def main() -> None:
    from .collector_lock import collector_lock
    with collector_lock(_db_from_argv(list(sys.argv[1:]))):
        _main_locked()


def _main_locked() -> None:
    global _active_manual_run_id, _active_manual_db
    argv = list(sys.argv[1:])
    replay = _has_clipboard_file(argv)
    once = _has_once(argv)
    db = _db_from_argv(argv)
    purpose = "v24_replay_experiment" if replay else "v24_production_raw_collection"
    # Replays get their own explicitly non-production session and therefore cannot
    # be mixed into an existing authoritative production raw database.
    initialize_collection(db, purpose=purpose)
    _sync_test_and_extension_hooks()
    _impl.run_once = run_once

    run_id: str | None = None
    if not replay and not once:
        run_id = manual_stop.start_collection_session(db)
        _active_manual_run_id = run_id
        _active_manual_db = db

    try:
        _impl.main()
    except KeyboardInterrupt:
        if run_id is not None:
            outcome = manual_stop.stop_collection_session(
                db, run_id, reason="manual_stop_censored"
            )
            print(json.dumps({"collection_stop": outcome}, indent=2, default=str), flush=True)
        # Ctrl+C is a requested clean stop, so do not convert it into an error exit.
        return
    finally:
        _active_manual_run_id = None
        _active_manual_db = None


if __name__ == "__main__":
    main()
