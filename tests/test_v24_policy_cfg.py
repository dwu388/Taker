from __future__ import annotations

from profit_taker import axiom_self_teach as selfteach
from profit_taker import axiom_v24 as v24


def test_train_distributional_policy_forwards_cfg_to_policy_refresh_helpers(tmp_path, monkeypatch):
    """Regression: policy bootstrap helpers must receive the active V24Config."""
    cfg = v24.V24Config(promotion_every_n_blocks=7, warmup_blocks=3)
    seen = {}

    monkeypatch.setattr(v24, "migrate", lambda conn: None)
    monkeypatch.setattr(selfteach, "migrate", lambda conn: None)
    monkeypatch.setattr(v24, "refresh_counterfactual_policy_targets", lambda conn, passed: seen.setdefault("counterfactual", passed))
    monkeypatch.setattr(v24, "refresh_policy_cohorts", lambda conn, passed: seen.setdefault("policy_cohorts", passed))
    monkeypatch.setattr(v24, "refresh_token_assignments", lambda conn, passed: seen.setdefault("token_assignments", passed))
    monkeypatch.setattr(v24, "next_one_use_policy_cohort", lambda conn, passed: (seen.setdefault("next_cohort", passed), None)[1])

    result = v24.train_distributional_policy(
        str(tmp_path / "policy.sqlite"),
        str(tmp_path / "models"),
        cfg,
        allow_small=True,
    )

    assert result == {"trained": False, "reason": "no mature unused policy-promotion cohort"}
    assert seen == {
        "counterfactual": cfg,
        "policy_cohorts": cfg,
        "token_assignments": cfg,
        "next_cohort": cfg,
    }


def test_train_policy_cli_binding_uses_fixed_function():
    """The implementation CLI must resolve the corrected facade function too."""
    assert v24._impl.train_distributional_policy is v24.train_distributional_policy
