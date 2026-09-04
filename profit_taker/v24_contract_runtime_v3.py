from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from . import axiom_v24 as v24
from . import pretraining_contract_v3 as contract
from . import v24_contract_runtime as shared
from .db import RAW_DB_DEFAULT


def _converged_monotonic_probability_projection(
    matrix: np.ndarray,
    *,
    max_iterations: int = 256,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Project a partially observed probability grid to both monotonic axes.

    Rows represent increasing time horizons and therefore must be nondecreasing.
    Columns represent increasing gain thresholds and therefore must be
    nonincreasing. Alternating isotonic projections are repeated to convergence
    instead of using a fixed pass count, which can leave a small constraint
    violation and make a second projection change the result.
    """
    x = np.asarray(matrix, dtype=float).copy()
    if x.ndim != 2:
        raise ValueError("probability matrix must be 2-dimensional")

    for _ in range(max(1, int(max_iterations))):
        previous = x.copy()
        for r in range(x.shape[0]):
            x[r, :] = v24._impl._isotonic_1d_observed(x[r, :], True)
        for c in range(x.shape[1]):
            x[:, c] = v24._impl._isotonic_1d_observed(x[:, c], False)

        delta = np.abs(x - previous)
        finite_delta = delta[np.isfinite(delta)]
        if finite_delta.size == 0 or float(np.max(finite_delta)) <= float(tolerance):
            break

    finite = np.isfinite(x)
    x[finite] = np.clip(x[finite], 0.0, 1.0)
    return x


# The retained implementation resolves this function from its own globals at
# prediction time. Patch all public facade layers so direct V24 callers and the
# official contract runtime use exactly the same converged projection semantics.
v24.monotonic_probability_projection = _converged_monotonic_probability_projection
v24._base.monotonic_probability_projection = _converged_monotonic_probability_projection
v24._impl.monotonic_probability_projection = _converged_monotonic_probability_projection


def _cfg(args: argparse.Namespace) -> v24.V24Config:
    return v24.V24Config(
        cohort_hours=args.cohort_hours,
        promotion_every_n_blocks=args.promotion_every,
        audit_every_n_blocks=args.audit_every,
        warmup_blocks=args.warmup_blocks,
    )


def _recurrent_grid_for_cfg(cfg: v24.V24Config) -> tuple[int, ...]:
    """Derive the active recurrent grid from the forecast contract.

    The retained implementation historically used a mutable module-level tuple.
    Deriving it from the persisted probability horizon contract makes a champion's
    serialized config sufficient to reconstruct its recurrent/marked-event heads.
    """
    return tuple(
        int(h)
        for h in cfg.probability_horizons_minutes
        if 240 <= int(h) <= int(cfg.horizon_minutes)
    )


def _activate_recurrent_grid(cfg: v24.V24Config) -> tuple[int, ...]:
    grid = _recurrent_grid_for_cfg(cfg)
    for module in (v24, getattr(v24, "_base", None), getattr(v24, "_impl", None)):
        if module is not None:
            setattr(module, "RECURRENT_HORIZONS_MINUTES", grid)
    return grid


def _hash_recurrent_contract(base_hash: str, cfg: v24.V24Config) -> str:
    payload = json.dumps(
        {
            "base_v24_target_definition_hash": str(base_hash),
            "recurrent_horizons_minutes": list(_recurrent_grid_for_cfg(cfg)),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _install_recurrent_target_hash() -> None:
    """Include the recurrent grid in the target identity before pretraining hash composition."""
    if getattr(v24, "_recurrent_grid_target_hash_installed", False):
        return
    original = v24.target_definition_hash

    def recurrent_aware(cfg) -> str:
        return _hash_recurrent_contract(str(original(cfg)), cfg)

    v24.target_definition_hash = recurrent_aware
    v24._base.target_definition_hash = recurrent_aware
    v24._impl.target_definition_hash = recurrent_aware
    v24._recurrent_grid_target_hash_installed = True


def _config_from_bundle(path: str | Path, fallback: v24.V24Config) -> tuple[v24.V24Config, str]:
    model_path = Path(path)
    if not model_path.exists():
        return fallback, "cli_defaults_no_champion"
    try:
        bundle = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load V24 champion configuration from {model_path}") from exc
    raw = bundle.get("config") if isinstance(bundle, dict) else None
    if not isinstance(raw, dict):
        raise RuntimeError(f"V24 champion {model_path} does not contain a persisted config")

    allowed = set(v24.V24Config.__dataclass_fields__)
    kwargs = {}
    for name, value in raw.items():
        if name not in allowed:
            continue
        default_value = getattr(fallback, name, None)
        if isinstance(default_value, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    cfg = v24.V24Config(**kwargs)
    return cfg, f"champion_bundle:{model_path}"


def _runtime_cfg(args: argparse.Namespace, command: str | None = None) -> tuple[v24.V24Config, str]:
    cmd = str(command or getattr(args, "cmd", ""))
    fallback = _cfg(args)
    if cmd == "bootstrap":
        cfg, source = fallback, "bootstrap_cli"
    elif cmd == "maintain":
        cfg, source = _config_from_bundle(Path(args.model_root) / "champion.joblib", fallback)
    elif cmd == "predict":
        cfg, source = _config_from_bundle(args.model, fallback)
    else:
        cfg, source = _config_from_bundle(v24.CHAMPION_DEFAULT, fallback)
    _activate_recurrent_grid(cfg)
    _validate_cfg(cfg)
    return cfg, source


def _validate_cfg(cfg: v24.V24Config) -> None:
    horizon = int(cfg.horizon_minutes)
    if horizon <= 0:
        raise RuntimeError("V24 config invalid: horizon_minutes must be positive")

    def checked(name: str, values) -> tuple[int, ...]:
        out = tuple(int(x) for x in values)
        if any(x <= 0 for x in out):
            raise RuntimeError(f"V24 config invalid: {name} contains a non-positive horizon")
        if any(x > horizon for x in out):
            raise RuntimeError(f"V24 config invalid: {name} exceeds horizon_minutes={horizon}")
        return out

    survival = checked("survival_bins_minutes", cfg.survival_bins_minutes)
    probability = checked("probability_horizons_minutes", cfg.probability_horizons_minutes)
    required = checked("promotion_required_horizons_minutes", cfg.promotion_required_horizons_minutes)
    checked("sequence_windows_minutes", cfg.sequence_windows_minutes)
    recurrent = _recurrent_grid_for_cfg(cfg)

    if not survival:
        raise RuntimeError("V24 config invalid: survival_bins_minutes is empty")
    if not probability:
        raise RuntimeError("V24 config invalid: probability_horizons_minutes is empty")
    if not set(required).issubset(set(probability)):
        raise RuntimeError(
            "V24 config invalid: promotion-required horizons are not all produced by probability heads"
        )
    if not set(recurrent).issubset(set(probability)):
        raise RuntimeError("V24 config invalid: recurrent horizons are outside the forecast contract")


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
    _activate_recurrent_grid(cfg)
    _validate_cfg(cfg)
    return profile


def _refresh(db: str, pcfg: contract.PretrainingConfig) -> dict:
    targets = contract.refresh_pretraining_targets(db, pcfg)
    with sqlite3.connect(db) as conn:
        friction = contract.enrich_counterfactual_friction(conn, pcfg)
    return {"targets": targets, "counterfactual_friction": friction}


def _pretraining_for_command(command: str, db: str, pcfg: contract.PretrainingConfig) -> dict:
    # Historical target materialization is intentionally excluded from latency-
    # sensitive forecast maintenance/prediction commands. Bootstrap and policy
    # training may pay the refresh cost because they consume the corresponding
    # readiness/economic evidence directly.
    if command in {"bootstrap", "train-policy"}:
        out = _refresh(db, pcfg)
        out["mode"] = "full_refresh"
        out["targets_refreshed"] = True
        return out
    return {
        "mode": "deferred_for_latency_sensitive_command",
        "targets_refreshed": False,
    }


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
    # Bind recurrent target identity first, then compose it with the latest
    # pretraining truth contract. Both hashes therefore describe the actual heads.
    _install_recurrent_target_hash()
    shared.target_contract_hash = contract.target_contract_hash
    shared.install_target_hash_contract(pcfg)
    cfg,cfg_source=_runtime_cfg(args,args.cmd)
    pretraining=_pretraining_for_command(args.cmd,args.db,pcfg)

    if args.cmd=="bootstrap":
        readiness=contract.assert_training_ready(args.db,pcfg)
        baselines=contract.evaluate_baselines(args.db,pcfg)
        if not baselines.get("available"):
            raise RuntimeError("V24 production bootstrap refused: preregistered baselines are not evaluable")
        profile={"name":"full"}
        if args.profile=="first_model":
            profile=_apply_first_model_profile(cfg,pcfg)
            cfg_source="bootstrap_first_model_profile"
        else:
            _activate_recurrent_grid(cfg); _validate_cfg(cfg)
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
        out.setdefault("runtime_config_source",cfg_source)
        out.setdefault("recurrent_horizons_minutes",list(_recurrent_grid_for_cfg(cfg)))
    print(json.dumps(out,indent=2,default=str)); return 0


if __name__=="__main__":
    raise SystemExit(main())
