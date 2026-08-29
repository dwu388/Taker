from __future__ import annotations

import argparse
import json
from pathlib import Path

from profit_taker.axiom_clipboard import (
    clipboard_diagnostics,
    clipboard_looks_like_axiom,
    rows_from_clipboard,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a saved Axiom clipboard selection without OCR")
    ap.add_argument("selection", help="Path to a .selection.txt file")
    ap.add_argument("--rows", action="store_true", help="Print parsed rows as JSON")
    args = ap.parse_args()

    text = Path(args.selection).read_text(encoding="utf-8")
    valid = clipboard_looks_like_axiom(text)
    rows = rows_from_clipboard(text) if valid else []

    report = clipboard_diagnostics(text, rows)
    print(json.dumps(report, indent=2))

    if args.rows:
        print(json.dumps(rows, indent=2, default=str))

    if not valid or not rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
