from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from typing import Any, Sequence

from . import axiom_v24 as v24
from .db import RAW_DB_DEFAULT
from .pretraining_contract import (
    PretrainingConfig,
    assert_training_ready,
    enrich_counterfactual_friction,
    evaluate_baselines,
    first_model_profile,
    refresh_pretraining_targets,
    target_contract_hash,
)


def _combined_hash(base_hash: str, pretraining_hash: str) -> str:
    payload = json.dumps(
        {"v24_target_definition_hash": base_hash, "pretraining_target_definition_hash": pretraining_hash},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def install_target_hash_contract(pcfg: PretrainingConfig | None = None) -> None:
    """Make the active V24 model hash include the pretraining target contract.

    The retained V24 code calls its module-global ``target_definition_hash`` from
    fit, maintenance, registry, and policy paths.  Patching all three facade layers
    keeps those paths consistent without editing the large retained implementation.
    """
    pcfg = pcfg or PretrainingConfig()
    if getattr(v24, "_pretraining_target_hash_installed", False):
        return
    original = v24.target_definition_hash
    phash = target_contract_hash(pcfg)

    def combined(cfg: Any) -> str:
        return _combined_hash(str(original(cfg)), phash)

    v24.target_definition_hash = combined
    v24._base.target_definition_hash = combined
    v24._impl.target_definition_hash = combined
    v24._pretraining_target_hash_installed = True
    v24._pretraining_target_definition_hash = phash


def _refresh_contract(db: str, pcfg: PretrainingConfig) -> dict[str, Any]:
    targets = refresh_pretraining_targets(db, pcfg)
    with sqlite3.connect(db) as conn:
        friction = enrich_counterfactual_friction(conn, pcfg)
    return {"targets": targets, "counterfactual_friction": friction}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="V24 runtime with frozen pre-first-training target contract")
    sp = p.add_subparsers(dest="cmd", required=True)

    def common(x: argparse.ArgumentParser) -> None:
        x.add_argument("--db", default=RAW_DB_DEFAULT)
        x.add_argument("--cohort-hours", type=int, default=24)
        x.add_argument("--promotion-every", type=int, default=4)
        x.add_argument("--audit-every", type=int, default=5)
        x.add_argument("--warmup-blocks", type=int, default=7)

    x = sp.add_parser("bootstrap"); common(x); x.add_argument("--model-root", default=v24.MODEL_ROOT_DEFAULT); x.add_argument("--profile", choices=("first_model","full"), default="first_model")
    x = sp.add_parser("maintain"); common(x); x.add_argument("--model-root", default=v24.MODEL_ROOT_DEFAULT); x.add_argument("--force-compaction", action="store_true")
    x = sp.add_parser("predict"); common(x); x.add_argument("--model", default=v24.CHAMPION_DEFAULT); x.add_argument("--out", default=v24.PREDICTIONS_DEFAULT)
    x = sp.add_parser("crossfit-policy-predictions"); common(x); x.add_argument("--max-folds", type=int, default=5)
    x = sp.add_parser("train-policy"); common(x); x.add_argument("--policy-root", default=v24.POLICY_ROOT_DEFAULT)
    x = sp.add_parser("status"); common(x)
    x = sp.add_parser("rebuild-sequence-cache"); common(x)
    x = sp.add_parser("audit-manifest"); common(x); x.add_argument("--reveal", action="store_true")
    x = sp.add_parser("audit-evaluate"); common(x)

    args = p.parse_args(argv)
    pcfg = PretrainingConfig()
    install_target_hash_contract(pcfg)
    cfg = v24.V24Config(
        cohort_hours=args.cohort_hours,
        promotion_every_n_blocks=args.promotion_every,
        audit_every_n_blocks=args.audit_every,
        warmup_blocks=args.warmup_blocks,
    )

    contract = _refresh_contract(args.db, pcfg)
    if args.cmd == "bootstrap":
        readiness = assert_training_ready(args.db, pcfg)
        baselines = evaluate_baselines(args.db, pcfg)
        if not baselines.get("available"):
            raise RuntimeError("V24 production bootstrap refused: preregistered baselines are not evaluable")
        profile = first_model_profile(pcfg)
        if args.profile == "first_model":
            cfg.stable_estimators = int(profile["estimators"])
            cfg.adapter_estimators = min(int(cfg.adapter_estimators), 60)
            cfg.sequence_challenger_min_tokens = max(int(cfg.sequence_challenger_min_tokens), pcfg.ts2vec_min_tokens)
        out = v24.bootstrap_v24(args.db, args.model_root, cfg, allow_small=False)
        out.update(pretraining_readiness=readiness, preregistered_baselines=baselines, training_profile=profile if args.profile == "first_model" else {"name":"full"})
    elif args.cmd == "maintain":
        out = v24.maintain_v24(args.db, args.model_root, cfg, allow_small=False, force_compaction=args.force_compaction)
    elif args.cmd == "predict":
        out = {"rows": len(v24.predict_current(args.db, args.model, args.out, cfg)), "out": args.out}
    elif args.cmd == "crossfit-policy-predictions":
        out = v24.crossfit_policy_predictions(args.db, cfg, max_folds=args.max_folds, allow_small=False)
    elif args.cmd == "train-policy":
        out = v24.train_distributional_policy(args.db, args.policy_root, cfg, allow_small=False)
    elif args.cmd == "status":
        out = v24.status(args.db, cfg)
    elif args.cmd == "rebuild-sequence-cache":
        with sqlite3.connect(args.db) as conn:
            obs, _ = v24.peak.load_observations(conn)
            out = v24.refresh_sequence_fingerprint_cache(conn, obs, cfg, force=True)
    elif args.cmd == "audit-manifest":
        with sqlite3.connect(args.db) as conn:
            v24.refresh_calendar_cohorts(conn, cfg)
            out = v24.audit_manifest(conn, cfg, reveal=args.reveal)
    elif args.cmd == "audit-evaluate":
        with sqlite3.connect(args.db) as conn:
            v24.refresh_calendar_cohorts(conn, cfg)
            out = v24.evaluate_sealed_audit_stream(conn, cfg)
    else:
        raise RuntimeError(args.cmd)

    if isinstance(out, dict):
        out.setdefault("pretraining_contract", contract)
        out.setdefault("pretraining_target_definition_hash", target_contract_hash(pcfg))
        out.setdefault("combined_v24_target_definition_hash", v24.target_definition_hash(cfg))
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
