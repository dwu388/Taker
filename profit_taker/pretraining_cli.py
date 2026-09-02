from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Sequence

from .db import RAW_DB_DEFAULT
from . import pretraining_contract_v3 as contract


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="V24 pre-first-training contract tools")
    sp = p.add_subparsers(dest="cmd", required=True)
    for name in ("refresh-targets", "audit", "readiness", "baselines", "profile", "enrich-friction"):
        x = sp.add_parser(name)
        x.add_argument("--db", default=RAW_DB_DEFAULT)
    args = p.parse_args(argv)
    cfg = contract.PretrainingConfig()
    if args.cmd == "refresh-targets":
        out = contract.refresh_pretraining_targets(args.db, cfg)
    elif args.cmd == "audit":
        out = contract.collection_audit(args.db, cfg)
    elif args.cmd == "readiness":
        out = contract.training_readiness(args.db, cfg)
    elif args.cmd == "baselines":
        out = contract.evaluate_baselines(args.db, cfg)
    elif args.cmd == "profile":
        out = contract.first_model_profile(cfg)
    elif args.cmd == "enrich-friction":
        with sqlite3.connect(args.db) as conn:
            out = contract.enrich_counterfactual_friction(conn, cfg)
    else:
        raise RuntimeError(args.cmd)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
