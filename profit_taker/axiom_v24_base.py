"""Public V24 compatibility facade.

The full V24 implementation is kept in :mod:`profit_taker.axiom_v24_impl`.
This facade preserves the historical import path while keeping small production
hardening fixes reviewable. The active outer facade specializes the lifecycle to
24 hours; this layer also prevents missing retired horizon heads and incompatible
policy champions from crossing that target-definition boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import datetime as _datetime
import joblib
import sqlite3
import uuid

from . import axiom_manual_stop as manual_stop
from . import axiom_v24_impl as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


# Retained implementation fit loops contain historical horizon literals. The active
# facade may deliberately omit a retired target (for example 4320m in the 24h line).
# A missing target is therefore an unavailable head, not an exception or an invented
# all-zero target.
_original_fit_blended_regression = _impl._fit_blended_regression


def _fit_blended_regression(data, features, target, n_estimators, *, quantile=None, poisson=False):
    if target not in data.columns:
        return None
    return _original_fit_blended_regression(
        data, features, target, n_estimators, quantile=quantile, poisson=poisson
    )


_impl._fit_blended_regression = _fit_blended_regression


# Counterfactual ENTRY/HOLD targets must never bridge a manual collection stop.
# The retained builder may see observations collected after a restart, so prune any
# target whose required future window crosses a durable neutral-censor boundary.
_original_refresh_counterfactual_policy_targets = _impl.refresh_counterfactual_policy_targets


def refresh_counterfactual_policy_targets(conn, cfg):
    result = _original_refresh_counterfactual_policy_targets(conn, cfg)
    pruned = manual_stop.prune_counterfactual_targets(conn, COUNTERFACTUAL_TABLE)
    if isinstance(result, dict):
        result["manual_stop_censored_targets_pruned"] = int(pruned)
    return result


_impl.refresh_counterfactual_policy_targets = refresh_counterfactual_policy_targets


def train_distributional_policy(
    db: str,
    policy_root: str,
    cfg: V24Config,
    *,
    allow_small: bool = False,
) -> dict[str, Any]:
    """Train/promote the V24 distributional policy using the supplied config.

    Policy cohorts and token assignments receive the active V24Config. A champion
    from a different target definition (notably the former 72h lifecycle) is never
    compared against or warm-promoted into the active generation.
    """
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        from . import axiom_self_teach as selfteach

        selfteach.migrate(conn)
        refresh_counterfactual_policy_targets(conn, cfg)
        refresh_policy_cohorts(conn, cfg)
        refresh_token_assignments(conn, cfg)
        cohort = next_one_use_policy_cohort(conn, cfg)
        if cohort is None:
            return {"trained": False, "reason": "no mature unused policy-promotion cohort"}

        entry = _entry_policy_training_rows(conn, horizon=60)
        hold = _hold_policy_training_rows(conn, horizon=60)
        entry_head = _fit_distribution_head(entry, "target", cfg, allow_small)
        hold_head = _fit_distribution_head(hold, "target", cfg, allow_small)
        if entry_head is None and hold_head is None:
            raise RuntimeError("No V24 counterfactual policy head has enough development-eligible OOS forecasts.")

        active_target_hash = target_definition_hash(cfg)
        active_execution_hash = execution_definition_hash(cfg)
        candidate = {
            "schema_version": selfteach.SCHEMA_VERSION,
            "v24_policy_schema": SCHEMA_VERSION,
            "created_at": _now_iso(),
            "entry_head": entry_head,
            "hold_head": hold_head,
            "oos_only": True,
            "counterfactual_targets": True,
            "config": asdict(cfg),
            "training_rows_entry": int(len(entry)),
            "training_rows_hold": int(len(hold)),
            "target_definition_hash": active_target_hash,
            "execution_definition_hash": active_execution_hash,
        }

        eval_entry = _entry_policy_training_rows(conn, eval_cohort_id=str(cohort["cohort_id"]), horizon=60)
        eval_hold = _hold_policy_training_rows(conn, eval_cohort_id=str(cohort["cohort_id"]), horizon=60)
        cand_eval = evaluate_policy_bundle(candidate, eval_entry, eval_hold, cfg)
        champion_path = Path(policy_root) / "champion.joblib"
        champion = joblib.load(champion_path) if champion_path.exists() else None
        champion_compatible = bool(
            champion
            and champion.get("target_definition_hash") == active_target_hash
            and champion.get("execution_definition_hash") == active_execution_hash
            and champion.get("v24_policy_schema") == SCHEMA_VERSION
            and bool(champion.get("oos_only", False))
        )
        if champion and not champion_compatible:
            champion = None

        champ_eval = (
            evaluate_policy_bundle(champion, eval_entry, eval_hold, cfg)
            if champion
            else {
                "available": True,
                "tokens": cand_eval.get("tokens", 0),
                "mean_value": 0.0,
                "token_values": {k: 0.0 for k in cand_eval.get("token_values", {})},
            }
        )
        common = sorted(set(cand_eval.get("token_values", {})) & set(champ_eval.get("token_values", {})))
        diff = pd.Series(
            {k: float(cand_eval["token_values"][k]) - float(champ_eval["token_values"][k]) for k in common},
            dtype=float,
        )
        boot = _paired_token_bootstrap(
            diff,
            cfg.policy_promotion_bootstrap_samples,
            cfg.policy_promotion_confidence,
        )
        promoted = bool(
            boot["n_tokens"] >= (3 if allow_small else cfg.policy_min_promotion_tokens)
            and boot["ci_low"] > 0.0
        )
        reason = (
            f"paired_token_value_mean={boot['mean']:.6f}; "
            f"ci=[{boot['ci_low']:.6f},{boot['ci_high']:.6f}]; n={boot['n_tokens']}"
        )
        if not champion_compatible and champion_path.exists():
            reason += "; previous champion excluded because target/execution/schema definition changed"

        Path(policy_root).mkdir(parents=True, exist_ok=True)
        stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cand_path = Path(policy_root) / f"policy_v24_{stamp}.joblib"
        joblib.dump(candidate, cand_path)
        before_hash = _hash_file(champion_path) if champion_path.exists() else None
        if promoted:
            joblib.dump(candidate, champion_path)

        pid = str(uuid.uuid4())
        metrics = {
            "candidate": {k: v for k, v in cand_eval.items() if k != "token_values"},
            "champion": {k: v for k, v in champ_eval.items() if k != "token_values"},
            "bootstrap": boot,
            "previous_champion_compatible": champion_compatible,
        }
        conn.execute(
            f"INSERT INTO {POLICY_PROMOTION_TABLE}(promotion_id,created_at,cohort_id,candidate_path,candidate_hash,champion_before_path,champion_before_hash,promoted,metrics_json,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                pid, _now_iso(), cohort["cohort_id"], str(cand_path), _hash_file(cand_path),
                str(champion_path) if champion_path.exists() else None, before_hash,
                int(promoted), _json(metrics), reason,
            ),
        )
        conn.execute(
            f"UPDATE {POLICY_COHORT_TABLE} SET status='consumed',consumed_at=?,promotion_id=? WHERE cohort_id=?",
            (_now_iso(), pid, cohort["cohort_id"]),
        )
        status_value = "champion" if promoted else "rejected"
        vid = str(uuid.uuid4())
        conn.execute(
            f"INSERT INTO {POLICY_REGISTRY}(version_id,created_at,model_path,model_hash,status,training_rows_entry,training_rows_hold,oos_only,metrics_json,notes,target_definition_hash,execution_definition_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                vid, _now_iso(), str(cand_path), _hash_file(cand_path), status_value,
                len(entry), len(hold), 1, _json(metrics), reason,
                candidate["target_definition_hash"], candidate["execution_definition_hash"],
            ),
        )
        if promoted:
            conn.execute("UPDATE axiom_policy_versions_v20 SET status='retired' WHERE status='champion'")
            conn.execute(
                "INSERT INTO axiom_policy_versions_v20(version_id,created_at,model_path,model_hash,status,metrics_json,training_closed_trades,training_hold_samples,notes) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    vid, _now_iso(), str(champion_path), _hash_file(champion_path), "champion",
                    _json({"v24_oos_only": True, "counterfactual": True, "target_definition_hash": active_target_hash}),
                    len(entry), len(hold), "V24 one-use promoted counterfactual distributional policy",
                ),
            )
        ope = doubly_robust_entry_ope(conn, candidate, cfg, 60)
        conn.commit()
        return {
            "trained": True,
            "promoted": promoted,
            "policy_candidate": str(cand_path),
            "champion": str(champion_path) if champion_path.exists() else None,
            "cohort_id": cohort["cohort_id"],
            "entry_rows": len(entry),
            "hold_rows": len(hold),
            "promotion": metrics,
            "dr_ope": ope,
        }


_impl.train_distributional_policy = train_distributional_policy
main = _impl.main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
