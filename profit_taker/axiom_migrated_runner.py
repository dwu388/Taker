"""Production-hardened public collector facade.

The previous collector is preserved byte-for-byte in
:mod:`profit_taker.axiom_migrated_runner_base`.  This facade adds two production
invariants without changing parsing/cadence behavior:

1. every browser copy is preceded by a unique verified clipboard sentinel, so a
   failed clipboard clear/copy can never replay the previous valid Axiom snapshot;
2. direct interactive invocation initializes/resumes the marked production
   collection session just like the BAT launchers do.
"""
from __future__ import annotations

import sys
import uuid
from typing import Any

from . import axiom_migrated_runner_base as _impl
from .collection_admin import initialize_collection
from .db import RAW_DB_DEFAULT

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

_original_capture_clipboard = _impl._capture_clipboard


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
    # The preserved implementation subsequently tries to clear the clipboard. If
    # that clear fails, the sentinel remains. A failed Ctrl+C therefore yields a
    # non-Axiom sentinel rather than the previous minute's valid Axiom payload.
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
        "process_rows",
        "record_failed_capture_attempt",
    ):
        if name in globals():
            setattr(_impl, name, globals()[name])
    _impl._capture_clipboard = _capture_clipboard


def run_once(*args: Any, **kwargs: Any) -> dict:
    _sync_test_and_extension_hooks()
    return _impl.run_once(*args, **kwargs)


def _db_from_argv(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--db" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--db="):
            return arg.split("=", 1)[1]
    return RAW_DB_DEFAULT


def main() -> None:
    argv = list(sys.argv[1:])
    # Replay files are still explicit research captures and get a marked session;
    # this prevents accidental writes to an unmarked database regardless of entry
    # point. initialize_collection is idempotent when resuming the one session.
    initialize_collection(_db_from_argv(argv))
    _sync_test_and_extension_hooks()
    _impl.main()


if __name__ == "__main__":
    main()
