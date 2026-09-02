from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Sequence

from . import axiom_v24 as v24
from . import pretraining_contract_v3 as contract
from . import v24_contract_runtime as shared
from .db import RAW_DB_DEFAULT


def _cfg(args: argparse.Namespace) -> v24.V24Config:
    return v24.V24Config(
        cohort_hours=args.cohort_hours,
        promotion_every_n_blocks=args.promotion_every,
        audit_every_n_blocks=args.audit_every,
        warmup_blocks=args.warmup_blocks,
    )


def _apply_first_model_profile(cfg: v24.V24Config, pcfg: contract.PretrainingConfig) -> dict:
    profile = contract.first_model_profile(pcfg)
    cfg.stable_estimators = int(profile["estimators"])
    cfg.adapter_estimators = min(int(cfg.adapter_estimators), 60)
    cfg.survival_bins_minutes = tuple(x for x in cfg.survival_bins_minutes if int(x) <= 240)
    cfg.probability_horizons_minutes = (60, 240)
    cfg.promotion_required_horizons_minutes = (240,)
    cfg.upside_thresholds = (0.30, 0.50)
    cfg.sequence_windows_minutes = (60, 240)
    # First-model TS2Vec is deliberately OFF even though the readiness report can
    # separately say whether enough tokens exist for a later challenger.
    cfg.sequence_challenger_min_tokens = 10**9
    # Restrict the active recurrent/marked-event facade to the 4h diagnostic range.
    v24.RECURRENT_HORIZONS_MINUTES = (240,)
    return profile


def _refresh(db: str, pcfg: contract.PretrainingConfig) -> dict:
    targets = contract.refresh_pretraining_targets(db, pcfg)
    with sqlite3.connect(db) as conn:
        friction = contract.enrich_counterfactual_friction(conn, pcfg)
    return {"targets": targets, "counterfactual_friction": friction}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Production V24 runtime with pretraining contract and reduced first-model profile")
    sp = p.add_subparsers(dest="cmd", required=True)

    def common(x):
        x.add_argument("--db", default=RAW_DB_DEFAULT)
        x.add_argument("--cohort-hours", type=int, default=24)
        x.add_argument("--promotion-every", type=int, default=4)
        x.add_argument("--audit-every", type=int, default=5)
        x.add_argument("--warmup-blocks", type=int, default=7)

    x=sp.add_parser("bootstrap"); common(x); x.add_argument("--model-root",default=v24.MODEL_ROOT_DEFAULT); x.add_argument("--profile",choices=("first_model","full"),default="first_model")
    x=sp.add_parser("maintain"); common(x); x.add_argument("--model-root",default=v24.MODEL_ROOT_DEFAULT); x.add_argument("--force-compaction",action="store_true")
    x=sp.add_parser("predict"); common(x); x.add_argument("--model",default=v24.CHAMPION_DEFAULT); x.add_argument("--out",default=v24.PREDICTIONS_DEFAULT)
    x=sp.add_parser("crossfit-policy-predictions"); common(x); x.add_argument("--max-folds",type=int,default=5)
    x=sp.add_parser("train-policy"); common(x); x.add_argument("--policy-root",default=v24.POLICY_ROOT_DEFAULT)
    x=sp.add_parser("status"); common(x)
    x=sp.add_parser("rebuild-sequence-cache"); common(x)
    x=sp.add_parser("audit-manifest"); common(x); x.add_argument("--reveal",action="store_true")
    x=sp.add_parser("audit-evaluate"); common(x)
    args=p.parse_args(argv)

    pcfg=contract.PretrainingConfig()
    # Rebind shared target hash patch to the tightened contract implementation.
    shared.target_contract_hash = contract.target_contract_hash
    shared.install_target_hash_contract(pcfg)
    cfg=_cfg(args)
    pretraining=_refresh(args.db,pcfg)

    if args.cmd=="bootstrap":
        readiness=contract.assert_training_ready(args.db,pcfg)
        baselines=contract.evaluate_baselines(args.db,pcfg)
        if not baselines.get("available"):
            raise RuntimeError("V24 production bootstrap refused: preregistered baselines are not evaluable")
        profile={"name":"full"}
        if args.profile=="first_model": profile=_apply_first_model_profile(cfg,pcfg)
        out=v24.bootstrap_v24(args.db,args.model_root,cfg,allow_small=False)
        out.update(pretraining_readiness=readiness,preregistered_baselines=baselines,training_profile=profile)
    elif args.cmd=="maintain": out=v24.maintain_v24(args.db,args.model_root,cfg,allow_small=False,force_compaction=args.force_compaction)
    elif args.cmd=="predict": out={"rows":len(v24.predict_current(args.db,args.model,args.out,cfg)),"out":args.out}
    elif args.cmd=="crossfit-policy-predictions": out=v24.crossfit_policy_predictions(args.db,cfg,max_folds=args.max_folds,allow_small=False)
    elif args.cmd=="train-policy": out=v24.train_distributional_policy(args.db,args.policy_root,cfg,allow_small=False)
    elif args.cmd=="status": out=v24.status(args.db,cfg)
    elif args.cmd=="rebuild-sequence-cache":
        with sqlite3.connect(args.db) as conn:
            obs,_=v24.peak.load_observations(conn); out=v24.refresh_sequence_fingerprint_cache(conn,obs,cfg,force=True)
    elif args.cmd=="audit-manifest":
        with sqlite3.connect(args.db) as conn:
            v24.refresh_calendar_cohorts(conn,cfg); out=v24.audit_manifest(conn,cfg,reveal=args.reveal)
    elif args.cmd=="audit-evaluate":
        with sqlite3.connect(args.db) as conn:
            v24.refresh_calendar_cohorts(conn,cfg); out=v24.evaluate_sealed_audit_stream(conn,cfg)
    else: raise RuntimeError(args.cmd)

    if isinstance(out,dict):
        out.setdefault("pretraining_contract",pretraining)
        out.setdefault("pretraining_target_definition_hash",contract.target_contract_hash(pcfg))
        out.setdefault("combined_v24_target_definition_hash",v24.target_definition_hash(cfg))
    print(json.dumps(out,indent=2,default=str)); return 0


if __name__=="__main__":
    raise SystemExit(main())
