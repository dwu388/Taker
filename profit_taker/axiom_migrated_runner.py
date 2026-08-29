from __future__ import annotations

import argparse
import inspect
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


# ---------------------------------------------------------------------------
# Clipboard-only capture path using the older/simple MACRO_EVENTS syntax.
#
# There is intentionally NO screenshot capture, NO OCR import, NO OCR matching,
# and NO OCR fallback.
#
# Each tuple is:
#     (delay_before_action_seconds, action, payload)
# ---------------------------------------------------------------------------

MACRO_EVENTS = [
    # Focus the Axiom page.
    (1.857, "mouse_down", (785, 116)),
    (0.124, "mouse_up", (785, 116)),

    # Select and copy the Axiom page.
    (2.000, "hotkey", ("ctrl", "a")),
    (2.000, "hotkey", ("ctrl", "c")),

    # Read/validate the copied page text.
    (2.500, "read_clipboard", ()),

    # Focus the Axiom page.
    (1.857, "mouse_down", (785, 116)),
    (0.124, "mouse_up", (785, 116)),

]


def _capture_clipboard(cfg: dict) -> tuple[str, str]:
    try:
        import pyautogui
        import pyperclip
    except Exception as exc:
        raise RuntimeError(
            "Clipboard collection requires pyautogui and pyperclip in an "
            "interactive Windows desktop session"
        ) from exc

    cap = cfg.get("capture", {})
    read_timeout = float(cap.get("clipboard_read_timeout_seconds", 6.0))
    poll_seconds = float(cap.get("clipboard_poll_seconds", 0.25))

    # Clear stale clipboard contents so a failed Ctrl+C cannot reuse the
    # previous one-minute capture.
    try:
        pyperclip.copy("")
    except Exception:
        pass

    clipboard_text = ""

    for delay_seconds, action, payload in MACRO_EVENTS:
        time.sleep(delay_seconds)

        if action == "mouse_down":
            x, y = payload
            pyautogui.moveTo(x, y)
            pyautogui.mouseDown(button="left")

        elif action == "mouse_up":
            x, y = payload
            pyautogui.moveTo(x, y)
            pyautogui.mouseUp(button="left")

        elif action == "hotkey":
            pyautogui.hotkey(*payload)

        elif action == "press":
            pyautogui.press(*payload)

        elif action == "read_clipboard":
            # Ctrl+C may finish asynchronously on a heavy browser page, so
            # keep the simple macro syntax while still waiting for a valid
            # Axiom selection instead of accepting stale/partial text.
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
                    raise RuntimeError(
                        "Clipboard copy completed, but the copied text did not "
                        "pass the Axiom Migrated structural validator. Nothing "
                        "was stored."
                    )

                raise RuntimeError(
                    "Ctrl+C did not produce clipboard text before timeout. "
                    "Nothing was stored."
                )

        else:
            raise ValueError(f"Unknown MACRO_EVENTS action: {action}")

    if not clipboard_text:
        raise RuntimeError(
            "MACRO_EVENTS completed without a valid Axiom clipboard capture."
        )

    snapshot_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return clipboard_text, snapshot_at


def _safe_capture_stem(snapshot_at: str) -> str:
    dt = datetime.fromisoformat(snapshot_at.replace("Z", "+00:00"))
    return "Axiom-Clipboard-" + dt.astimezone(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def _call_process_rows(
    db: str,
    snapshot_at: str,
    source_path: str,
    rows: list[dict],
    output_dir: str,
) -> dict:
    """Call both the older and current process_rows signatures safely."""
    kwargs = {}

    try:
        signature = inspect.signature(process_rows)
        if "screenshot_rows_detected" in signature.parameters:
            kwargs["screenshot_rows_detected"] = 0
    except Exception:
        pass

    return process_rows(
        db,
        snapshot_at,
        source_path,
        rows,
        True,
        output_dir,
        **kwargs,
    )


def run_once(args) -> dict:
    cfg = load_json(args.config, {})

    if args.clipboard_file:
        selection_path = Path(args.clipboard_file)
        clipboard_text = selection_path.read_text(encoding="utf-8")
        snapshot_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        clipboard_text, snapshot_at = _capture_clipboard(cfg)
        selection_path = None

    if not clipboard_looks_like_axiom(clipboard_text):
        raise RuntimeError(
            "Clipboard selection is not a valid Axiom Migrated capture. "
            "No observation was written."
        )

    rows = rows_from_clipboard(clipboard_text)
    if not rows:
        raise RuntimeError(
            "Axiom clipboard text passed structural validation but no token "
            "cards were parsed. No observation was written."
        )

    # The short address is the stable identity. Never manufacture an identity
    # from token name, row order, screenshot geometry, or OCR.
    invalid_identity = [
        i for i, row in enumerate(rows) if not row.get("token_key")
    ]
    if invalid_identity:
        raise RuntimeError(
            "Clipboard parser produced rows without stable token identity: "
            f"{invalid_identity}. Nothing was stored."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_capture_stem(snapshot_at)

    # Always save a canonical local copy of the selected Axiom text.
    copied_selection_path = out_dir / f"{stem}.selection.txt"
    copied_selection_path.write_text(clipboard_text, encoding="utf-8")
    selection_path = copied_selection_path

    diag = clipboard_diagnostics(clipboard_text, rows)
    diag["snapshot_at"] = snapshot_at
    diag["selection_path"] = str(selection_path)

    (out_dir / f"{stem}.selection.json").write_text(
        diagnostics_json(diag),
        encoding="utf-8",
    )

    result = _call_process_rows(
        args.db,
        snapshot_at,
        str(selection_path),
        rows,
        args.output_dir,
    )

    result["collection_mode"] = "clipboard_only"
    result["clipboard_report"] = diag
    result["extract_report"] = {
        "mode": "disabled",
        "ocr_enabled": False,
        "screenshot_enabled": False,
        "screenshot_rows": 0,
    }

    if args.with_model:
        model = Path(args.model)

        if model.exists():
            from .axiom_predict_24h import (
                export_handoff,
                plan_handoff,
                predict_rows,
                write_csv,
            )

            preds = predict_rows(args.db, str(model), False)
            write_csv(preds, args.predictions, args.rank_by)
            result["handoff"] = plan_handoff(args.db, preds)
            export_handoff(args.db)
            result["predictions"] = len(preds)
        else:
            result["prediction_warning"] = f"Model not found: {model}"

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Axiom Migrated 1-minute collector - clipboard selection only. "
            "Screenshots and OCR are not used."
        )
    )

    ap.add_argument("--db", default="data/live.sqlite")
    ap.add_argument("--config", default="axiom_migrated_config.json")
    ap.add_argument("--output-dir", default="data/axiom_migrated")
    ap.add_argument("--once", action="store_true")

    ap.add_argument(
        "--clipboard-file",
        help=(
            "Replay a saved Axiom .selection.txt file instead of controlling "
            "the browser. Useful for parser validation."
        ),
    )

    ap.add_argument("--with-model", action="store_true")
    ap.add_argument("--model", default="models/axiom24/latest.joblib")
    ap.add_argument(
        "--predictions",
        default="data/axiom_predictions_24h.csv",
    )
    ap.add_argument("--rank-by", default="p_hit_plus100_by_12h")

    args = ap.parse_args()

    cfg = load_json(args.config, {})
    interval = 60

    while True:
        started = time.monotonic()

        try:
            print(
                json.dumps(run_once(args), indent=2, default=str),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "collection_mode": "clipboard_only",
                        "stored": 0,
                    },
                    indent=2,
                ),
                flush=True,
            )

            if args.once or args.clipboard_file:
                raise

        if args.once or args.clipboard_file:
            break

        sleep_seconds = max(
            1,
            interval - (time.monotonic() - started),
        )

        next_run = (
            datetime.now().astimezone()
            + timedelta(seconds=sleep_seconds)
        )

        print(
            f"Next run: {next_run.strftime('%Y-%m-%d %I:%M:%S %p %Z')}",
            flush=True,
        )

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
