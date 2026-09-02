from __future__ import annotations

import argparse
import json
from typing import Sequence

from . import axiom_v24 as v24
from .pretraining_contract import PretrainingConfig, assert_training_ready, evaluate_baselines, first_model_profile
from .db import RAW_DB_DEFAULT


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Leakage-safe first V24 bootstrap with mandatory readiness and baseline checks")
    p.add_argument("--db", default=RAW_DB_DEFAULT)
    p.add_argument("--model-root", default=v24.MODEL_ROOT_DEFAULT)
    p.add_argument("--profile", choices=("first_model", "full"), default="first_model")
    args = p.parse_args(argv)

    pcfg = PretrainingConfig()
    readiness = assert_training_ready(args.db, pcfg)
    baselines = evaluate_baselines(args.db, pcfg)
    if not baselines.get("available"):
        raise RuntimeError("V24 production bootstrap refused: preregistered baselines are not evaluable yet")

    cfg = v24.V24Config()
    profile = first_model_profile(pcfg)
    if args.profile == "first_model":
        # This is deliberately not --allow-small.  Only capacity is reduced; all
        # ordinary V24 holdout and sample-size safeguards remain in force.
        cfg.stable_estimators = int(profile["estimators"])
        cfg.adapter_estimators = min(int(cfg.adapter_estimators), 60)
        cfg.sequence_challenger_min_tokens = max(int(cfg.sequence_challenger_min_tokens), pcfg.ts2vec_min_tokens)

    out = v24.bootstrap_v24(args.db, args.model_root, cfg, allow_small=False)
    out["pretraining_readiness"] = readiness
    out["preregistered_baselines"] = baselines
    out["training_profile"] = profile if args.profile == "first_model" else {"name": "full"}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
