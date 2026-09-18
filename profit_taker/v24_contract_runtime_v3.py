from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from . import axiom_v24 as v24
from . import pretraining_contract_v3 as contract
from . import v24_contract_runtime as shared
from .db import RAW_DB_DEFAULT


_FULL_GRADUATION_MAX_NEW_COMPONENT_BRIER = 0.25
_FIRST_MODEL_REDUCED_FIELDS = {
    "stable_estimators",
    "adapter_estimators",
    "survival_bins_minutes",
    "probability_horizons_minutes",
    "promotion_required_horizons_minutes",
    "promotion_required_thresholds",
    "upside_thresholds",
    "sequence_windows_minutes",
    "sequence_challenger_min_tokens",
}


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


def _cfg_values(cfg: v24.V24Config) -> dict[str, Any]:
    return {
        name: getattr(cfg, name)
        for name in v24.V24Config.__dataclass_fields__
    }


def _clone_cfg(cfg: v24.V24Config, **overrides: Any) -> v24.V24Config:
    values = _cfg_values(cfg)
    values.update(overrides)
    return v24.V24Config(**values)


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
    # Reduced champions created before the graduation fix can contain a stale
    # +100% promotion requirement even though they only fitted +30%/+50% heads.
    # Promotion requirements are evaluation metadata, not part of target identity,
    # so normalize this recognized legacy profile without mutating model semantics.
    if _is_first_model_profile(cfg):
        cfg = _first_model_evaluation_cfg(cfg)
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
    upside = tuple(float(x) for x in cfg.upside_thresholds)
    required_thresholds = tuple(float(x) for x in cfg.promotion_required_thresholds)

    if not survival:
        raise RuntimeError("V24 config invalid: survival_bins_minutes is empty")
    if not probability:
        raise RuntimeError("V24 config invalid: probability_horizons_minutes is empty")
    if not set(required).issubset(set(probability)):
        raise RuntimeError(
            "V24 config invalid: promotion-required horizons are not all produced by probability heads"
        )
    if not set(required_thresholds).issubset(set(upside)):
        raise RuntimeError(
            "V24 config invalid: promotion-required upside thresholds are not all produced by barrier heads"
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
    # The reduced bootstrap does not fit +100% heads, so it must not require one
    # for its one-use promotion evaluation.
    cfg.promotion_required_thresholds = (0.50,)
    cfg.sequence_windows_minutes = (60, 240)
    # First-model TS2Vec is deliberately OFF even though the readiness report can
    # separately say whether enough tokens exist for a later challenger.
    cfg.sequence_challenger_min_tokens = 10**9
    _activate_recurrent_grid(cfg)
    _validate_cfg(cfg)
    return profile


def _is_first_model_profile(cfg: v24.V24Config) -> bool:
    bins = tuple(int(x) for x in cfg.survival_bins_minutes)
    return bool(
        tuple(int(x) for x in cfg.probability_horizons_minutes) == (60, 240)
        and tuple(int(x) for x in cfg.promotion_required_horizons_minutes) == (240,)
        and tuple(float(x) for x in cfg.upside_thresholds) == (0.30, 0.50)
        and tuple(int(x) for x in cfg.sequence_windows_minutes) == (60, 240)
        and bins
        and max(bins) <= 240
        and int(cfg.sequence_challenger_min_tokens) >= 10**9
    )


def _full_graduation_cfg(champion_cfg: v24.V24Config) -> v24.V24Config:
    """Restore only fields deliberately reduced by the first-model profile.

    Holdout cadence, execution assumptions, calibration settings and every other
    operator/config choice are inherited from the champion. The target/model fields
    reduced for bootstrap are restored from the current full V24 defaults.
    """
    defaults = v24.V24Config()
    values = _cfg_values(champion_cfg)
    for name in _FIRST_MODEL_REDUCED_FIELDS:
        values[name] = getattr(defaults, name)
    cfg = v24.V24Config(**values)
    _activate_recurrent_grid(cfg)
    _validate_cfg(cfg)
    return cfg


def _first_model_evaluation_cfg(cfg: v24.V24Config) -> v24.V24Config:
    """Make pre-fix first champions evaluable without changing their model target hash."""
    if not _is_first_model_profile(cfg):
        return cfg
    out = _clone_cfg(cfg, promotion_required_thresholds=(0.50,))
    _validate_cfg(out)
    return out


def _shared_graduation_components(
    champion_cfg: v24.V24Config,
    full_cfg: v24.V24Config,
) -> list[str]:
    champion_required = set(v24._required_promotion_components(champion_cfg))
    return [
        name
        for name in v24._required_promotion_components(full_cfg)
        if name in champion_required
    ]


def _evaluation_from_component_losses(
    losses: pd.DataFrame,
    required_components: Sequence[str],
) -> dict[str, Any]:
    required = list(required_components)
    if losses is None or losses.empty:
        return {"available": False, "reason": "empty evaluation frame", "tokens": 0}
    missing = [c for c in required if c not in losses.columns]
    if missing:
        return {
            "available": False,
            "reason": "missing required promotion coverage",
            "missing_components": missing,
            "tokens": int(len(losses)),
        }
    valid = losses[["token_key"] + required].copy()
    valid[required] = valid[required].apply(pd.to_numeric, errors="coerce")
    valid["composite_error"] = valid[required].mean(axis=1, skipna=False)
    valid = valid[np.isfinite(valid.composite_error)].copy()
    if valid.empty:
        return {
            "available": False,
            "reason": "no tokens have complete fixed metric coverage",
            "tokens": 0,
        }
    return {
        "available": True,
        "composite_error": float(valid.composite_error.mean()),
        "tokens": int(len(valid)),
        "required_components": required,
        "token_losses": dict(zip(valid.token_key.astype(str), valid.composite_error.astype(float))),
        "component_means": {c: float(valid[c].mean()) for c in required},
    }


def _graduation_promotion_decision(
    candidate_shared: dict[str, Any],
    champion_shared: dict[str, Any],
    candidate_full: dict[str, Any],
    champion_cfg: v24.V24Config,
    full_cfg: v24.V24Config,
) -> tuple[bool, dict[str, Any]]:
    """Require both shared-head improvement and credible newly introduced heads.

    A reduced champion cannot be compared on horizons it does not produce. The
    paired promotion test therefore uses only fixed components shared by both
    models. The full-only binary Brier components must additionally have complete
    fixed coverage, enough independent tokens, and beat the 0.25 error of an
    uninformative p=0.5 predictor before the model may graduate.
    """
    shared_ok, shared_reason = v24.compare_promotion(candidate_shared, champion_shared, full_cfg)
    shared = _shared_graduation_components(champion_cfg, full_cfg)
    full_required = v24._required_promotion_components(full_cfg)
    full_only = [name for name in full_required if name not in set(shared)]
    full_tokens = int(candidate_full.get("tokens", 0) or 0)
    component_means = candidate_full.get("component_means") or {}
    missing_new = [
        name
        for name in full_only
        if component_means.get(name) is None or not np.isfinite(float(component_means[name]))
    ]
    weak_new = [
        name
        for name in full_only
        if name not in missing_new
        and float(component_means[name]) > _FULL_GRADUATION_MAX_NEW_COMPONENT_BRIER
    ]
    full_ok = bool(
        candidate_full.get("available")
        and full_tokens >= int(full_cfg.promotion_min_tokens)
        and full_only
        and not missing_new
        and not weak_new
    )
    if not candidate_full.get("available"):
        full_reason = f"full candidate unavailable: {candidate_full.get('reason')}"
    elif full_tokens < int(full_cfg.promotion_min_tokens):
        full_reason = f"only {full_tokens} full-contract evaluation tokens; need {full_cfg.promotion_min_tokens}"
    elif not full_only:
        full_reason = "no newly introduced full-contract promotion components"
    elif missing_new:
        full_reason = "missing full-only components: " + ",".join(missing_new)
    elif weak_new:
        full_reason = (
            "full-only Brier exceeds uninformative 0.25 ceiling: " + ",".join(weak_new)
        )
    else:
        full_reason = "full-only fixed coverage passes 0.25 Brier ceiling"
    details = {
        "shared_contract_pass": bool(shared_ok),
        "shared_contract_reason": shared_reason,
        "full_contract_pass": full_ok,
        "full_contract_reason": full_reason,
        "shared_required_components": shared,
        "full_only_required_components": full_only,
        "full_only_component_means": {
            name: component_means.get(name) for name in full_only
        },
        "max_new_component_brier": _FULL_GRADUATION_MAX_NEW_COMPONENT_BRIER,
    }
    return bool(shared_ok and full_ok), details


def _graduate_first_model_champion(
    db: str,
    model_root: str,
    champion_cfg: v24.V24Config,
    full_cfg: v24.V24Config,
    *,
    allow_small: bool = False,
) -> dict[str, Any]:
    """Train a clean full-contract challenger on a fresh one-use promotion cohort."""
    champion_path = Path(model_root) / "champion.joblib"
    if not champion_path.exists():
        raise RuntimeError("Full graduation requires an existing reduced V24 champion")
    champion = joblib.load(champion_path)
    if champion.get("schema_version") != v24.SCHEMA_VERSION:
        raise RuntimeError("Existing champion is not compatible with the active V24 schema")
    if champion.get("target_definition_hash") != v24.target_definition_hash(champion_cfg):
        raise RuntimeError(
            "Reduced champion target-definition hash does not match its persisted config; refusing graduation"
        )
    if champion.get("execution_definition_hash") != v24.execution_definition_hash(champion_cfg):
        raise RuntimeError(
            "Reduced champion execution-definition hash does not match its persisted config; refusing graduation"
        )

    with closing(sqlite3.connect(db)) as conn, conn:
        conn.row_factory = sqlite3.Row
        v24.migrate(conn)
        peak_cfg = v24.peak.PeakStructureConfig(
            horizon_minutes=full_cfg.horizon_minutes,
            death_gap_minutes=full_cfg.operational_gap_minutes,
            death_missed_cycles=int(full_cfg.operational_gap_minutes),
            age_out_minutes=full_cfg.age_out_minutes,
        )
        v24.peak.refresh_labels(db, peak_cfg)

        _activate_recurrent_grid(full_cfg)
        full_frame, full_seq, _ = v24.load_v24_frame(conn, full_cfg)
        cohort = v24.next_one_use_promotion_cohort(conn, full_cfg)
        if cohort is None:
            _activate_recurrent_grid(champion_cfg)
            return {
                "trained": False,
                "mode": "first_model_to_full_graduation",
                "promoted": False,
                "reason": "no fully matured unused promotion cohort for full graduation",
                "graduation_pending": True,
            }
        cutoff = v24._utc(cohort["start_at"])
        if v24._model_training_cutoff_from_bundle(champion) >= cutoff:
            raise RuntimeError(
                "Champion training cutoff is not earlier than the graduation promotion cohort; refusing holdout leakage"
            )

        full_eval_frame = v24._cohort_frame(full_frame, cohort)
        eval_tokens = set(full_eval_frame.token_key.astype(str))
        candidate_bundle = v24.fit_batch_bundle(
            conn,
            full_frame,
            full_seq,
            cutoff,
            full_cfg,
            allow_small=allow_small,
            generation=int(champion.get("stable_generation", 1)) + 1,
            exclude_tokens=eval_tokens,
        )
        candidate_bundle["graduation_from_target_definition_hash"] = champion.get("target_definition_hash")
        candidate_bundle["graduation_contract"] = "first_model_to_full_v1"
        candidate_full_eval = v24.evaluate_bundle(
            conn, candidate_bundle, full_eval_frame, full_seq, full_cfg
        )
        candidate_losses = v24._token_promotion_losses(
            conn, candidate_bundle, full_eval_frame, full_seq, full_cfg
        )

        champion_eval_cfg = _first_model_evaluation_cfg(champion_cfg)
        _activate_recurrent_grid(champion_eval_cfg)
        champion_frame, champion_seq, _ = v24.load_v24_frame(conn, champion_eval_cfg)
        champion_eval_frame = v24._cohort_frame(champion_frame, cohort)
        champion_losses = v24._token_promotion_losses(
            conn, champion, champion_eval_frame, champion_seq, champion_eval_cfg
        )
        shared = _shared_graduation_components(champion_eval_cfg, full_cfg)
        candidate_shared_eval = _evaluation_from_component_losses(candidate_losses, shared)
        champion_shared_eval = _evaluation_from_component_losses(champion_losses, shared)
        promoted, graduation = _graduation_promotion_decision(
            candidate_shared_eval,
            champion_shared_eval,
            candidate_full_eval,
            champion_eval_cfg,
            full_cfg,
        )
        reason = (
            "first_model_to_full_graduation; "
            f"shared={graduation['shared_contract_reason']}; "
            f"full={graduation['full_contract_reason']}"
        )

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate_path = v24._save_bundle(
            candidate_bundle,
            model_root,
            f"challengers/v24_full_graduation_{stamp}.joblib",
        )
        before_hash = v24._hash_file(champion_path)
        if promoted:
            shutil.copy2(candidate_path, champion_path)
        promotion_id = str(uuid.uuid4())
        metrics = {
            "mode": "first_model_to_full_graduation",
            "candidate_full": candidate_full_eval,
            "candidate_shared": candidate_shared_eval,
            "champion_shared": champion_shared_eval,
            "graduation": graduation,
            "from_target_definition_hash": champion.get("target_definition_hash"),
            "to_target_definition_hash": candidate_bundle.get("target_definition_hash"),
        }
        conn.execute(
            f"""INSERT INTO {v24.PROMOTION_TABLE}
                (promotion_id,created_at,cohort_id,candidate_path,candidate_hash,
                 champion_before_path,champion_before_hash,promoted,metrics_json,reason)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                promotion_id,
                v24._now_iso(),
                cohort["cohort_id"],
                candidate_path,
                v24._hash_file(candidate_path),
                str(champion_path),
                before_hash,
                int(promoted),
                v24._json(metrics),
                reason,
            ),
        )
        # The cohort has now been inspected and is consumed whether graduation wins
        # or loses. Reusing it would turn a one-use holdout into development data.
        conn.execute(
            f"UPDATE {v24.COHORT_TABLE} SET status='consumed',consumed_at=?,promotion_id=? WHERE cohort_id=?",
            (v24._now_iso(), promotion_id, cohort["cohort_id"]),
        )
        v24._register_model(
            conn,
            str(champion_path if promoted else candidate_path),
            candidate_bundle,
            "champion" if promoted else "rejected",
            metrics,
            reason,
        )
        conn.commit()

    active_cfg = full_cfg if promoted else champion_cfg
    _activate_recurrent_grid(active_cfg)
    return {
        "trained": True,
        "mode": "first_model_to_full_graduation",
        "promoted": promoted,
        "cohort": cohort["cohort_id"],
        "candidate": candidate_path,
        "champion": str(champion_path),
        "reason": reason,
        "metrics": metrics,
        "graduation_pending": not promoted,
    }


def _maintain_with_auto_graduation(
    db: str,
    model_root: str,
    champion_cfg: v24.V24Config,
    *,
    allow_small: bool = False,
    force_compaction: bool = False,
) -> tuple[dict[str, Any], v24.V24Config, str]:
    if not _is_first_model_profile(champion_cfg):
        out = v24.maintain_v24(
            db,
            model_root,
            champion_cfg,
            allow_small=allow_small,
            force_compaction=force_compaction,
        )
        return out, champion_cfg, "champion_standard_maintenance"

    full_cfg = _full_graduation_cfg(champion_cfg)
    out = _graduate_first_model_champion(
        db,
        model_root,
        champion_cfg,
        full_cfg,
        allow_small=allow_small,
    )
    if out.get("promoted"):
        return out, full_cfg, "auto_full_graduation_promoted"
    return out, champion_cfg, "first_model_champion_pending_full_graduation"


def _refresh(db: str, pcfg: contract.PretrainingConfig) -> dict:
    targets = contract.refresh_pretraining_targets(db, pcfg)
    with closing(sqlite3.connect(db)) as conn, conn:
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
    if args.cmd == "bootstrap":
        # Readiness reads derived peak events and calendar assignments. Build
        # those from raw observations before checking them on a fresh database.
        # Use the same label settings as bootstrap_v24 and the requested cohort
        # config; assignments remain immutable and no evaluation is consumed.
        peak_cfg = v24.peak.PeakStructureConfig(
            horizon_minutes=cfg.horizon_minutes,
            death_gap_minutes=cfg.operational_gap_minutes,
            death_missed_cycles=int(cfg.operational_gap_minutes),
            age_out_minutes=cfg.age_out_minutes,
        )
        v24.peak.refresh_labels(args.db, peak_cfg)
        with closing(sqlite3.connect(args.db)) as conn, conn:
            v24.refresh_token_assignments(conn, cfg)
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
    elif args.cmd=="maintain":
        out,cfg,cfg_source=_maintain_with_auto_graduation(
            args.db,
            args.model_root,
            cfg,
            allow_small=False,
            force_compaction=args.force_compaction,
        )
    elif args.cmd=="predict": out={"rows":len(v24.predict_current(args.db,args.model,args.out,cfg)),"out":args.out}
    elif args.cmd=="crossfit-policy-predictions": out=v24.crossfit_policy_predictions(args.db,cfg,max_folds=args.max_folds,allow_small=False)
    elif args.cmd=="train-policy": out=v24.train_distributional_policy(args.db,args.policy_root,cfg,allow_small=False)
    elif args.cmd=="status": out=v24.status(args.db,cfg)
    elif args.cmd=="rebuild-sequence-cache":
        with closing(sqlite3.connect(args.db)) as conn, conn:
            obs,_=v24.peak.load_observations(conn); out=v24.refresh_sequence_fingerprint_cache(conn,obs,cfg,force=True)
    elif args.cmd=="audit-manifest":
        with closing(sqlite3.connect(args.db)) as conn, conn:
            v24.refresh_calendar_cohorts(conn,cfg); out=v24.audit_manifest(conn,cfg,reveal=args.reveal)
    elif args.cmd=="audit-evaluate":
        with closing(sqlite3.connect(args.db)) as conn, conn:
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
