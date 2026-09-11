from __future__ import annotations

import argparse
import inspect
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .axiom_clipboard import (
    clipboard_diagnostics,
    clipboard_looks_like_axiom,
    diagnostics_json,
    rows_from_clipboard,
)
from .axiom_migrated_process import process_rows, record_failed_capture_attempt
from .common import load_json
from .db import RAW_DB_DEFAULT

CAPTURE_CLICK = (662, 114)
REFRESH_EVERY_CYCLES = 601

# Kept as a public diagnostic/compatibility description of the normal capture
# sequence. Runtime timing is configurable below and no longer burns seconds in
# fixed waits when the clipboard is already ready.
MACRO_EVENTS = [
    (0.000, "mouse_down", CAPTURE_CLICK),
    (0.030, "mouse_up", CAPTURE_CLICK),
    (0.350, "hotkey", ("ctrl", "a")),
    (0.200, "hotkey", ("ctrl", "c")),
    (0.050, "read_clipboard", ()),
    (0.000, "mouse_down", CAPTURE_CLICK),
    (0.030, "mouse_up", CAPTURE_CLICK),
]


def _attach_capture_context(exc: Exception, *, clipboard_text: str | None = None, clipboard_valid: bool | None = None, rows_detected: int | None = None, candidate_cards: int | None = None) -> Exception:
    for name, value in (("clipboard_text", clipboard_text), ("clipboard_valid", clipboard_valid), ("rows_detected", rows_detected), ("candidate_cards", candidate_cards)):
        if value is not None:
            try:
                setattr(exc, name, value)
            except Exception:
                pass
    return exc


def _capture_error(message: str, *, text: str = "", valid: bool = False, rows: int = 0, candidates: int | None = None) -> RuntimeError:
    return _attach_capture_context(RuntimeError(message), clipboard_text=text, clipboard_valid=valid, rows_detected=rows, candidate_cards=candidates)


def _raw_mc_card_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().upper() == "MC")


def _capture_seconds(cap: dict, key: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(cap.get(key, default))
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


def _capture_int(cap: dict, key: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(cap.get(key, default))
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


def _click_capture_anchor(pyautogui, hold_seconds: float) -> None:
    x, y = CAPTURE_CLICK
    pyautogui.moveTo(x, y)
    pyautogui.mouseDown(button="left")
    if hold_seconds > 0:
        time.sleep(hold_seconds)
    pyautogui.mouseUp(button="left")


def _ensure_capture_marker(pyperclip) -> str:
    """Guarantee that a failed Ctrl+C cannot be mistaken for a fresh snapshot.

    The public facade normally primes a fresh verified V24 sentinel first. Direct
    calls to this base module are also protected: if no public sentinel is present,
    install and verify a local marker before touching the browser.
    """
    try:
        current = pyperclip.paste() or ""
    except Exception:
        current = ""
    if current.startswith("__V24_CLIPBOARD_SENTINEL_"):
        return current
    marker = f"__V24_CAPTURE_SENTINEL_{uuid.uuid4().hex}__"
    try:
        pyperclip.copy(marker)
        observed = pyperclip.paste()
    except Exception as exc:
        raise RuntimeError("Could not prime/verify the clipboard before Axiom capture") from exc
    if observed != marker:
        raise RuntimeError("Clipboard capture marker verification failed before Axiom capture")
    return marker


def _capture_clipboard(cfg: dict, cycle_count: int) -> tuple[str, str]:
    try:
        import pyautogui
        import pyperclip
    except Exception as exc:
        raise RuntimeError("Clipboard collection requires pyautogui and pyperclip in an interactive desktop session") from exc

    cap = cfg.get("capture", {}) if isinstance(cfg, dict) else {}
    if not isinstance(cap, dict):
        cap = {}
    read_timeout = _capture_seconds(cap, "clipboard_read_timeout_seconds", 6.0, minimum=1.0, maximum=30.0)
    poll_seconds = _capture_seconds(cap, "clipboard_poll_seconds", 0.05, minimum=0.01, maximum=1.0)
    click_hold = _capture_seconds(cap, "click_hold_seconds", 0.03, minimum=0.01, maximum=0.25)
    focus_settle = _capture_seconds(cap, "focus_settle_seconds", 0.35, minimum=0.10, maximum=2.0)
    selection_settle = _capture_seconds(cap, "selection_settle_seconds", 0.20, minimum=0.10, maximum=2.0)
    post_copy_settle = _capture_seconds(cap, "post_copy_settle_seconds", 0.05, minimum=0.0, maximum=1.0)
    refresh_settle = _capture_seconds(cap, "refresh_settle_seconds", 2.0, minimum=1.0, maximum=10.0)
    retry_after = _capture_seconds(cap, "copy_retry_after_seconds", 0.75, minimum=0.25, maximum=3.0)
    max_retries = _capture_int(cap, "copy_retries", 2, minimum=0, maximum=5)

    marker = _ensure_capture_marker(pyperclip)
    clipboard_text = ""
    captured_at: str | None = None
    last_nonempty = ""

    _click_capture_anchor(pyautogui, click_hold)
    time.sleep(focus_settle)
    try:
        if cycle_count % REFRESH_EVERY_CYCLES == 0:
            pyautogui.hotkey("ctrl", "shift", "r")
            # Refresh is rare and remains deliberately conservative. Normal cycles
            # do not pay this cost.
            time.sleep(refresh_settle)

        pyautogui.hotkey("ctrl", "a")
        time.sleep(selection_settle)
        pyautogui.hotkey("ctrl", "c")
        if post_copy_settle > 0:
            time.sleep(post_copy_settle)

        started_wait = time.monotonic()
        deadline = started_wait + read_timeout
        next_retry = started_wait + retry_after
        retries = 0
        while time.monotonic() < deadline:
            try:
                candidate = pyperclip.paste() or ""
            except Exception:
                candidate = ""
            # The marker check is independent of structural validation. Even a
            # structurally valid old Axiom payload cannot be accepted unless the
            # clipboard changed after this capture started.
            if candidate != marker:
                if candidate.strip():
                    last_nonempty = candidate
                if clipboard_looks_like_axiom(candidate):
                    clipboard_text = candidate
                    # Timestamp the actual successful copy, not the later UI
                    # cleanup, improving minute-level temporal accuracy.
                    captured_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                    break
            now = time.monotonic()
            if retries < max_retries and now >= next_retry:
                # A browser can occasionally miss a key chord while busy. Retrying
                # selection+copy is safer than accepting stale data and usually
                # recovers without waiting for the full timeout.
                pyautogui.hotkey("ctrl", "a")
                time.sleep(selection_settle)
                pyautogui.hotkey("ctrl", "c")
                retries += 1
                next_retry = now + retry_after
            time.sleep(poll_seconds)
        else:
            if last_nonempty:
                raise _capture_error("Clipboard text did not pass the Axiom structural validator; nothing was stored", text=last_nonempty)
            raise _capture_error("Ctrl+C did not produce fresh clipboard text before timeout; nothing was stored")
    finally:
        # Always clear the page selection/focus state, even if validation fails.
        # This replaces the old ~2 second cleanup delay with the same physical
        # click performed immediately after the clipboard has been captured.
        try:
            _click_capture_anchor(pyautogui, click_hold)
        except Exception:
            pass

    if not clipboard_text or captured_at is None:
        raise _capture_error("Clipboard capture completed without a valid fresh Axiom payload")
    return clipboard_text, captured_at


def _safe_capture_stem(snapshot_at: str) -> str:
    dt = datetime.fromisoformat(snapshot_at.replace("Z", "+00:00"))
    return "Axiom-Clipboard-" + dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


ARTIFACT_RETENTION_CYCLES = 10


def _prune_cycle_artifacts(output_dir: str | Path, keep_cycles: int = ARTIFACT_RETENTION_CYCLES) -> dict:
    out_dir = Path(output_dir)
    result = {"cycles_retained": 0, "cycles_deleted": 0, "files_deleted": 0, "errors": []}
    if not out_dir.exists():
        return result
    cycles: dict[str, list[Path]] = {}
    try:
        paths = list(out_dir.glob("Axiom-Clipboard-*.selection*"))
    except OSError as exc:
        result["errors"].append(f"glob: {type(exc).__name__}: {exc}")
        return result
    for path in paths:
        try:
            if not path.is_file():
                continue
        except OSError as exc:
            result["errors"].append(f"stat {path}: {type(exc).__name__}: {exc}")
            continue
        cycle_stem = path.name.split(".selection", 1)[0]
        cycles.setdefault(cycle_stem, []).append(path)
    ordered = sorted(cycles.items(), key=lambda item: item[0], reverse=True)
    result["cycles_retained"] = min(len(ordered), max(0, int(keep_cycles)))
    for _, files in ordered[max(0, int(keep_cycles)):]:
        cycle_had_file = False
        for path in files:
            try:
                path.unlink(); result["files_deleted"] += 1; cycle_had_file = True
            except FileNotFoundError:
                continue
            except OSError as exc:
                result["errors"].append(f"delete {path}: {type(exc).__name__}: {exc}")
        if cycle_had_file:
            result["cycles_deleted"] += 1
    return result


def _best_effort_selection_artifacts(output_dir: str | Path, stem: str, clipboard_text: str, diag: dict) -> list[str]:
    out_dir = Path(output_dir)
    selection_path = out_dir / f"{stem}.selection.txt"
    diag_path = out_dir / f"{stem}.selection.json"
    errors: list[str] = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return [f"mkdir {out_dir}: {type(exc).__name__}: {exc}"]
    try:
        selection_path.write_text(clipboard_text, encoding="utf-8")
    except OSError as exc:
        errors.append(f"write {selection_path}: {type(exc).__name__}: {exc}")
    try:
        diag_path.write_text(diagnostics_json(diag), encoding="utf-8")
    except OSError as exc:
        errors.append(f"write {diag_path}: {type(exc).__name__}: {exc}")
    return errors


def run_once(args, cycle_count: int, *, attempt_started_at: str | None = None) -> dict:
    cfg = load_json(args.config, {})
    if args.clipboard_file:
        clipboard_text = Path(args.clipboard_file).read_text(encoding="utf-8")
        snapshot_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        attempt_source = "replay_file"
    else:
        clipboard_text, snapshot_at = _capture_clipboard(cfg, cycle_count)
        attempt_source = "interactive_clipboard"
    if not clipboard_looks_like_axiom(clipboard_text):
        raise _capture_error("Clipboard selection is not a valid Axiom capture; no observation was written", text=clipboard_text)
    rows = rows_from_clipboard(clipboard_text)
    candidate_cards = _raw_mc_card_count(clipboard_text)
    if not rows:
        raise _capture_error("Valid Axiom clipboard text produced no token cards; no observation was written", text=clipboard_text, valid=True, candidates=candidate_cards)
    if candidate_cards > 0 and candidate_cards != len(rows):
        raise _capture_error(f"Clipboard parse completeness check failed: {candidate_cards} MC card blocks but {len(rows)} parsed rows; partial capture was rejected", text=clipboard_text, valid=True, rows=len(rows), candidates=candidate_cards)
    invalid_identity = [i for i, row in enumerate(rows) if not row.get("token_key")]
    if invalid_identity:
        raise _capture_error(f"Clipboard rows lack stable token identity at indexes {invalid_identity}; nothing was stored", text=clipboard_text, valid=True, rows=len(rows), candidates=candidate_cards)
    invalid_market_cap = [i for i, row in enumerate(rows) if row.get("market_cap_usd") is None or float(row.get("market_cap_usd") or 0.0) <= 0.0]
    if invalid_market_cap:
        raise _capture_error(f"Clipboard rows lack positive market cap at indexes {invalid_market_cap}; partial capture was rejected", text=clipboard_text, valid=True, rows=len(rows), candidates=candidate_cards)
    out_dir = Path(args.output_dir)
    stem = _safe_capture_stem(snapshot_at)
    selection_path = out_dir / f"{stem}.selection.txt"
    diag = clipboard_diagnostics(clipboard_text, rows)
    diag["snapshot_at"] = snapshot_at
    diag["selection_path"] = str(selection_path)
    diag["candidate_mc_blocks"] = candidate_cards
    diag["parse_complete"] = True
    try:
        process_kwargs = {"screenshot_rows_detected": len(rows)}
        process_parameters = inspect.signature(process_rows).parameters
        if "raw_clipboard_text" in process_parameters:
            process_kwargs["raw_clipboard_text"] = clipboard_text
        if "attempt_started_at" in process_parameters:
            process_kwargs["attempt_started_at"] = attempt_started_at or snapshot_at
        if "attempt_source" in process_parameters:
            process_kwargs["attempt_source"] = attempt_source
        result = process_rows(args.db, snapshot_at, str(selection_path), rows, True, args.output_dir, **process_kwargs)
    except Exception as exc:
        _attach_capture_context(exc, clipboard_text=clipboard_text, clipboard_valid=True, rows_detected=len(rows), candidate_cards=candidate_cards)
        raise
    artifact_errors = list(result.get("artifact_write_errors") or [])
    artifact_errors.extend(_best_effort_selection_artifacts(args.output_dir, stem, clipboard_text, diag))
    result["artifact_write_errors"] = artifact_errors
    result["clipboard_report"] = diag
    result["artifact_cleanup"] = _prune_cycle_artifacts(args.output_dir, keep_cycles=ARTIFACT_RETENTION_CYCLES)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Axiom one-minute clipboard-only collector; screenshots and OCR are not used")
    ap.add_argument("--db", default=RAW_DB_DEFAULT)
    ap.add_argument("--config", default="axiom_migrated_config.json")
    ap.add_argument("--output-dir", default="data/axiom_migrated")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--clipboard-file", help="Replay a saved Axiom .selection.txt instead of controlling the browser")
    args = ap.parse_args()
    cfg = load_json(args.config, {})
    interval = max(1, int(cfg.get("capture", {}).get("cycle_seconds", 60)))
    cycle_count = 0
    while True:
        cycle_count += 1
        started = time.monotonic()
        attempt_started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        try:
            print(json.dumps(run_once(args, cycle_count, attempt_started_at=attempt_started_at), indent=2, default=str), flush=True)
        except Exception as exc:
            attempt_id = None
            attempt_log_error = None
            try:
                attempt_id = record_failed_capture_attempt(args.db, started_at=attempt_started_at, completed_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"), clipboard_valid=bool(getattr(exc, "clipboard_valid", False)), rows_detected=int(getattr(exc, "rows_detected", 0) or 0), source="replay_file" if args.clipboard_file else "interactive_clipboard", error_type=type(exc).__name__, error_message=str(exc), raw_clipboard_text=getattr(exc, "clipboard_text", None), details={"candidate_cards": getattr(exc, "candidate_cards", None)})
            except Exception as log_exc:
                attempt_log_error = f"{type(log_exc).__name__}: {log_exc}"
            print(json.dumps({"error": type(exc).__name__, "message": str(exc), "collection_mode": "clipboard_only", "stored": 0, "failed_attempt_id": attempt_id, "attempt_log_error": attempt_log_error}, indent=2), flush=True)
            if args.once or args.clipboard_file:
                raise
        if args.once or args.clipboard_file:
            break
        sleep_seconds = max(1, interval - (time.monotonic() - started))
        next_run = datetime.now().astimezone() + timedelta(seconds=sleep_seconds)
        print(f"Next run: {next_run.strftime('%Y-%m-%d %I:%M:%S %p %Z')}", flush=True)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
