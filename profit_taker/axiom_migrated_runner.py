from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .axiom_clipboard import (
    clipboard_diagnostics,
    clipboard_looks_like_axiom,
    diagnostics_json,
    rows_from_clipboard,
)
from .axiom_migrated_process import process_rows
from .common import load_json

MACRO_EVENTS = [
    (1.857, "mouse_down", (785, 116)),
    (0.124, "mouse_up", (785, 116)),
    (2.000, "hotkey", ("ctrl", "a")),
    (2.000, "hotkey", ("ctrl", "c")),
    (2.500, "read_clipboard", ()),
    (1.857, "mouse_down", (785, 116)),
    (0.124, "mouse_up", (785, 116)),
]


def _capture_clipboard(cfg: dict, cycle_count: int) -> tuple[str, str]:
    try:
        import pyautogui
        import pyperclip
    except Exception as exc:
        raise RuntimeError("Clipboard collection requires pyautogui and pyperclip in an interactive desktop session") from exc

    cap = cfg.get("capture", {})
    read_timeout = float(cap.get("clipboard_read_timeout_seconds", 6.0))
    poll_seconds = float(cap.get("clipboard_poll_seconds", 0.25))
    try:
        pyperclip.copy("")
    except Exception:
        pass

    clipboard_text = ""
    macro_events = MACRO_EVENTS.copy()

    if cycle_count % 601 == 0:
        macro_events.insert(
            2,
            (2.000, "hotkey", ("ctrl", "shift", "r"))
        )

    for delay_seconds, action, payload in macro_events:
        time.sleep(delay_seconds)
        if action == "mouse_down":
            x, y = payload; pyautogui.moveTo(x, y); pyautogui.mouseDown(button="left")
        elif action == "mouse_up":
            x, y = payload; pyautogui.moveTo(x, y); pyautogui.mouseUp(button="left")
        elif action == "hotkey":
            pyautogui.hotkey(*payload)
        elif action == "read_clipboard":
            deadline = time.time() + read_timeout
            last_nonempty = ""
            while time.time() < deadline:
                try:
                    candidate = pyperclip.paste() or ""
                except Exception:
                    candidate = ""
                if candidate.strip():
                    last_nonempty = candidate
                if clipboard_looks_like_axiom(candidate):
                    clipboard_text = candidate
                    break
                time.sleep(poll_seconds)
            else:
                if last_nonempty:
                    raise RuntimeError("Clipboard text did not pass the Axiom structural validator; nothing was stored")
                raise RuntimeError("Ctrl+C did not produce clipboard text before timeout; nothing was stored")
        else:
            raise ValueError(f"Unknown MACRO_EVENTS action: {action}")

    if not clipboard_text:
        raise RuntimeError("MACRO_EVENTS completed without a valid Axiom clipboard capture")
    return clipboard_text, datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_capture_stem(snapshot_at: str) -> str:
    dt = datetime.fromisoformat(snapshot_at.replace("Z", "+00:00"))
    return "Axiom-Clipboard-" + dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


ARTIFACT_RETENTION_CYCLES = 10


def _prune_cycle_artifacts(
    output_dir: str | Path,
    keep_cycles: int = ARTIFACT_RETENTION_CYCLES,
) -> dict:
    """Best-effort cleanup of disposable per-capture diagnostics.

    Database persistence is authoritative. Cleanup errors are reported but never
    allowed to make a successfully committed observation cycle look failed.
    """
    out_dir = Path(output_dir)
    result = {
        "cycles_retained": 0,
        "cycles_deleted": 0,
        "files_deleted": 0,
        "errors": [],
    }
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
                path.unlink()
                result["files_deleted"] += 1
                cycle_had_file = True
            except FileNotFoundError:
                continue
            except OSError as exc:
                result["errors"].append(f"delete {path}: {type(exc).__name__}: {exc}")
        if cycle_had_file:
            result["cycles_deleted"] += 1

    return result


def run_once(args, cycle_count: int) -> dict:
    cfg = load_json(args.config, {})
    if args.clipboard_file:
        clipboard_text = Path(args.clipboard_file).read_text(encoding="utf-8")
        snapshot_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        clipboard_text, snapshot_at = _capture_clipboard(cfg, cycle_count)

    if not clipboard_looks_like_axiom(clipboard_text):
        raise RuntimeError("Clipboard selection is not a valid Axiom capture; no observation was written")
    rows = rows_from_clipboard(clipboard_text)
    if not rows:
        raise RuntimeError("Valid Axiom clipboard text produced no token cards; no observation was written")
    invalid_identity = [i for i, row in enumerate(rows) if not row.get("token_key")]
    if invalid_identity:
        raise RuntimeError(f"Clipboard rows lack stable token identity at indexes {invalid_identity}; nothing was stored")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_capture_stem(snapshot_at)
    selection_path = out_dir / f"{stem}.selection.txt"
    selection_path.write_text(clipboard_text, encoding="utf-8")
    diag = clipboard_diagnostics(clipboard_text, rows)
    diag["snapshot_at"] = snapshot_at
    diag["selection_path"] = str(selection_path)
    (out_dir / f"{stem}.selection.json").write_text(diagnostics_json(diag), encoding="utf-8")

    result = process_rows(
        args.db,
        snapshot_at,
        str(selection_path),
        rows,
        True,
        args.output_dir,
        screenshot_rows_detected=len(rows),
    )

    result["clipboard_report"] = diag
    result["artifact_cleanup"] = _prune_cycle_artifacts(
        args.output_dir,
        keep_cycles=ARTIFACT_RETENTION_CYCLES,
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Axiom one-minute clipboard-only collector; screenshots and OCR are not used")
    ap.add_argument("--db", default="data/live.sqlite")
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

        try:
            print(
                json.dumps(
                    run_once(args, cycle_count),
                    indent=2,
                    default=str
                ),
                flush=True
            )
        except Exception as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc), "collection_mode": "clipboard_only", "stored": 0}, indent=2), flush=True)
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
