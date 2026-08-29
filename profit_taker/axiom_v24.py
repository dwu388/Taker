"""Public V24 compatibility facade.

The full V24 implementation is kept in :mod:`profit_taker.axiom_v24_impl`.
This facade preserves the historical ``profit_taker.axiom_v24`` import path while
allowing small, reviewable fixes without rewriting the large implementation file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import datetime as _datetime
import joblib
import sqlite3
import uuid

from . import axiom_v24_impl as _impl

# Preserve the complete public/private module surface expected by existing callers.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


def train_distributional_policy(
    db: str,
    policy_root: str,
    cfg: V24Config,
    *,
    allow_small: bool = False,
) -> dict[str, Any]:
    """Train/promote the V24 distributional policy using the supplied config.

    ``refresh_policy_cohorts`` and ``refresh_token_assignments`` both require the
    active :class:`V24Config`.  Forwarding ``cfg`` here is essential because policy
    cohort scheduling and immutable token assignment depend on those settings.
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

        candidate = {
            "schema_version": "v21_self_teaching_incremental_72h_2_execution_accounting",
            "v24_policy_schema": SCHEMA_VERSION,
            "created_at": _now_iso(),
            "entry_head": entry_head,
            "hold_head": hold_head,
            "oos_only": True,
            "counterfactual_targets": True,
            "config": asdict(cfg),
            "training_rows_entry": int(len(entry)),
            "training_rows_hold": int(len(hold)),
            "target_definition_hash": target_definition_hash(cfg),
            "execution_definition_hash": execution_definition_hash(cfg),
        }

        eval_entry = _entry_policy_training_rows(conn, eval_cohort_id=str(cohort["cohort_id"]), horizon=60)
        eval_hold = _hold_policy_training_rows(conn, eval_cohort_id=str(cohort["cohort_id"]), horizon=60)
        cand_eval = evaluate_policy_bundle(candidate, eval_entry, eval_hold, cfg)
        champion_path = Path(policy_root) / "champion.joblib"
        champion = joblib.load(champion_path) if champion_path.exists() else None
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
        }
        conn.execute(
            f"INSERT INTO {POLICY_PROMOTION_TABLE}(promotion_id,created_at,cohort_id,candidate_path,candidate_hash,champion_before_path,champion_before_hash,promoted,metrics_json,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                pid,
                _now_iso(),
                cohort["cohort_id"],
                str(cand_path),
                _hash_file(cand_path),
                str(champion_path) if champion_path.exists() else None,
                before_hash,
                int(promoted),
                _json(metrics),
                reason,
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
                vid,
                _now_iso(),
                str(cand_path),
                _hash_file(cand_path),
                status_value,
                len(entry),
                len(hold),
                1,
                _json(metrics),
                reason,
                candidate["target_definition_hash"],
                candidate["execution_definition_hash"],
            ),
        )
        if promoted:
            conn.execute("UPDATE axiom_policy_versions_v20 SET status='retired' WHERE status='champion'")
            conn.execute(
                "INSERT INTO axiom_policy_versions_v20(version_id,created_at,model_path,model_hash,status,metrics_json,training_closed_trades,training_hold_samples,notes) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    vid,
                    _now_iso(),
                    str(champion_path),
                    _hash_file(champion_path),
                    "champion",
                    _json({"v24_oos_only": True, "counterfactual": True}),
                    len(entry),
                    len(hold),
                    "V24 one-use promoted counterfactual distributional policy",
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


# The implementation CLI resolves globals in axiom_v24_impl; replace its buggy
# function binding so ``python -m profit_taker.axiom_v24 train-policy`` also uses
# the corrected function above.
_impl.train_distributional_policy = train_distributional_policy
main = _impl.main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
