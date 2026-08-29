from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


FILES = (
    "profit_taker/axiom_clipboard.py",
    "profit_taker/axiom_migrated_runner.py",
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Install the clipboard-only Axiom collector patch")
    ap.add_argument("project_dir", help="Existing Profit Taker project directory")
    args = ap.parse_args()

    patch_root = Path(__file__).resolve().parent
    project_root = Path(args.project_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not (project_root / "profit_taker").is_dir():
        raise SystemExit(f"Not a Profit Taker project: {project_root}")

    for rel in FILES:
        src = patch_root / rel
        dst = project_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            backup = dst.with_name(dst.name + f".pre_clipboard_only_{stamp}.bak")
            shutil.copy2(dst, backup)
            print(f"backup: {backup}")

        shutil.copy2(src, dst)
        print(f"installed: {dst}")

    print("\nCollector data path is now clipboard-only. Existing OCR modules are left on disk but are not imported or called by axiom_migrated_runner.")


if __name__ == "__main__":
    main()
