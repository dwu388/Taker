from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from profit_taker import axiom_budget_benchmark as benchmark
from profit_taker import axiom_v24 as v24
from profit_taker import performance_report
from profit_taker import v24_contract_runtime_v3 as runtime


def _args(**overrides):
    values = {
        "cmd": "predict",
        "cohort_hours": 24,
        "promotion_every": 4,
        "audit_every": 5,
        "warmup_blocks": 7,
        "model": v24.CHAMPION_DEFAULT,
        "model_root": v24.MODEL_ROOT_DEFAULT,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_first_model_profile_reconstructs_its_recurrent_contract():
    cfg = v24.V24Config()
    try:
        runtime._apply_first_model_profile(cfg, runtime.contract.PretrainingConfig())
        assert cfg.probability_horizons_minutes == (60, 240)
        assert cfg.promotion_required_horizons_minutes == (240,)
        assert cfg.promotion_required_thresholds == (0.50,)
        assert set(cfg.promotion_required_thresholds).issubset(set(cfg.upside_thresholds))
        assert runtime._recurrent_grid_for_cfg(cfg) == (240,)
        assert tuple(v24.RECURRENT_HORIZONS_MINUTES) == (240,)
        assert runtime._is_first_model_profile(cfg)
        runtime._validate_cfg(cfg)
    finally:
        runtime._activate_recurrent_grid(v24.V24Config())


def test_runtime_cfg_uses_persisted_champion_config_and_normalizes_legacy_first_profile(tmp_path):
    cfg = v24.V24Config()
    cfg.stable_estimators = 150
    cfg.survival_bins_minutes = (5, 15, 30, 60, 120, 240)
    cfg.probability_horizons_minutes = (60, 240)
    cfg.promotion_required_horizons_minutes = (240,)
    cfg.upside_thresholds = (0.30, 0.50)
    cfg.sequence_windows_minutes = (60, 240)
    cfg.sequence_challenger_min_tokens = 10**9
    # Simulate a reduced champion written before this fix. It inherited +100% as
    # a promotion requirement even though that head was never fitted.
    cfg.promotion_required_thresholds = (0.50, 1.00)
    model = tmp_path / "champion.joblib"
    joblib.dump({"config": asdict(cfg)}, model)

    try:
        loaded, source = runtime._runtime_cfg(_args(model=str(model)), "predict")
        assert loaded.stable_estimators == 150
        assert loaded.probability_horizons_minutes == (60, 240)
        assert loaded.promotion_required_horizons_minutes == (240,)
        assert loaded.promotion_required_thresholds == (0.50,)
        assert runtime._recurrent_grid_for_cfg(loaded) == (240,)
        assert runtime._is_first_model_profile(loaded)
        assert source.startswith("champion_bundle:")
    finally:
        runtime._activate_recurrent_grid(v24.V24Config())


def test_recurrent_grid_is_part_of_target_identity():
    short = v24.V24Config()
    short.probability_horizons_minutes = (60, 240)
    full = v24.V24Config()
    assert runtime._hash_recurrent_contract("same-base", short) != runtime._hash_recurrent_contract(
        "same-base", full
    )


def test_config_validation_rejects_unproducible_promotion_head():
    cfg = v24.V24Config()
    cfg.probability_horizons_minutes = (60, 240)
    cfg.promotion_required_horizons_minutes = (240, 480)
    with pytest.raises(RuntimeError, match="promotion-required horizons"):
        runtime._validate_cfg(cfg)


def test_config_validation_rejects_unproducible_promotion_threshold():
    cfg = v24.V24Config()
    cfg.upside_thresholds = (0.30, 0.50)
    cfg.promotion_required_thresholds = (0.50, 1.00)
    with pytest.raises(RuntimeError, match="promotion-required upside thresholds"):
        runtime._validate_cfg(cfg)


def test_full_graduation_restores_only_reduced_model_fields():
    cfg = v24.V24Config()
    cfg.cohort_hours = 12
    cfg.promotion_every_n_blocks = 3
    cfg.audit_every_n_blocks = 7
    cfg.warmup_blocks = 5
    cfg.fallback_round_trip_bps = 175.0
    try:
        runtime._apply_first_model_profile(cfg, runtime.contract.PretrainingConfig())
        assert runtime._is_first_model_profile(cfg)
        full = runtime._full_graduation_cfg(cfg)
        defaults = v24.V24Config()

        assert full.probability_horizons_minutes == defaults.probability_horizons_minutes
        assert full.promotion_required_horizons_minutes == defaults.promotion_required_horizons_minutes
        assert full.promotion_required_thresholds == defaults.promotion_required_thresholds
        assert full.upside_thresholds == defaults.upside_thresholds
        assert full.sequence_windows_minutes == defaults.sequence_windows_minutes
        assert full.survival_bins_minutes == defaults.survival_bins_minutes
        assert full.sequence_challenger_min_tokens == defaults.sequence_challenger_min_tokens
        assert full.stable_estimators == defaults.stable_estimators
        assert full.adapter_estimators == defaults.adapter_estimators

        # Holdout/execution settings are inherited from the already-registered
        # champion instead of being silently reset during graduation.
        assert full.cohort_hours == 12
        assert full.promotion_every_n_blocks == 3
        assert full.audit_every_n_blocks == 7
        assert full.warmup_blocks == 5
        assert full.fallback_round_trip_bps == 175.0
        assert not runtime._is_first_model_profile(full)
        runtime._validate_cfg(full)
    finally:
        runtime._activate_recurrent_grid(v24.V24Config())


def test_graduation_compares_only_heads_shared_with_reduced_champion():
    first = v24.V24Config()
    try:
        runtime._apply_first_model_profile(first, runtime.contract.PretrainingConfig())
        full = runtime._full_graduation_cfg(first)
        first_eval = runtime._first_model_evaluation_cfg(first)
        shared = runtime._shared_graduation_components(first_eval, full)
        assert shared == [
            "peak_brier_240",
            "death_brier_240",
            "barrier_brier_50_240",
        ]
        assert "barrier_brier_100_240" not in shared
        assert "peak_brier_1440" not in shared
    finally:
        runtime._activate_recurrent_grid(v24.V24Config())


def test_graduation_requires_new_heads_to_clear_absolute_brier_guard(monkeypatch):
    champion_cfg = v24.V24Config()
    full_cfg = v24.V24Config()
    monkeypatch.setattr(runtime, "_shared_graduation_components", lambda *_: ["shared"])
    monkeypatch.setattr(v24, "_required_promotion_components", lambda _cfg: ["shared", "new"])
    monkeypatch.setattr(v24, "compare_promotion", lambda *_args, **_kwargs: (True, "shared pass"))

    candidate_shared = {"available": True}
    champion_shared = {"available": True}
    full_eval = {
        "available": True,
        "tokens": int(full_cfg.promotion_min_tokens),
        "component_means": {"shared": 0.10, "new": 0.251},
    }
    promoted, details = runtime._graduation_promotion_decision(
        candidate_shared, champion_shared, full_eval, champion_cfg, full_cfg
    )
    assert promoted is False
    assert details["shared_contract_pass"] is True
    assert details["full_contract_pass"] is False
    assert "new" in details["full_contract_reason"]

    full_eval["component_means"]["new"] = 0.249
    promoted, details = runtime._graduation_promotion_decision(
        candidate_shared, champion_shared, full_eval, champion_cfg, full_cfg
    )
    assert promoted is True
    assert details["full_contract_pass"] is True


def test_maintain_routes_reduced_champion_to_graduation_not_warm_adapter(monkeypatch):
    first = v24.V24Config()
    runtime._apply_first_model_profile(first, runtime.contract.PretrainingConfig())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reduced champion entered ordinary warm maintenance")

    monkeypatch.setattr(v24, "maintain_v24", forbidden)
    monkeypatch.setattr(
        runtime,
        "_graduate_first_model_champion",
        lambda *_args, **_kwargs: {"trained": False, "promoted": False, "graduation_pending": True},
    )
    try:
        out, active, source = runtime._maintain_with_auto_graduation("db", "models", first)
        assert out["graduation_pending"] is True
        assert runtime._is_first_model_profile(active)
        assert source == "first_model_champion_pending_full_graduation"
    finally:
        runtime._activate_recurrent_grid(v24.V24Config())


def test_live_prediction_commands_do_not_trigger_full_pretraining_refresh(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("full historical refresh entered latency-sensitive command")

    monkeypatch.setattr(runtime, "_refresh", forbidden)
    result = runtime._pretraining_for_command("predict", "unused.sqlite", runtime.contract.PretrainingConfig())
    assert result["targets_refreshed"] is False
    assert result["mode"] == "deferred_for_latency_sensitive_command"


def test_benchmark_default_matches_performance_report_default():
    assert benchmark.DEFAULT_BENCHMARK_DB == performance_report.BENCHMARK_DB_DEFAULT
    assert benchmark.DEFAULT_BENCHMARK_DB == "data/axiom_v24_1000_benchmark.sqlite"


def test_monotonic_projection_converges_preserves_missing_cells_and_is_idempotent():
    raw = np.asarray(
        [
            [0.75, np.nan, 0.40, 0.55],
            [0.82, 0.68, 0.52, 0.60],
            [0.66, 0.51, 0.35, 0.44],
        ],
        dtype=float,
    )
    missing = np.isnan(raw)
    once = v24.monotonic_probability_projection(raw)
    twice = v24.monotonic_probability_projection(once)

    np.testing.assert_array_equal(np.isnan(once), missing)
    np.testing.assert_allclose(once, twice, rtol=0.0, atol=1e-10, equal_nan=True)

    finite = once[np.isfinite(once)]
    assert np.all(finite >= 0.0)
    assert np.all(finite <= 1.0)

    # Across time horizons a fixed threshold cannot become less likely.
    for row in once:
        observed = row[np.isfinite(row)]
        if observed.size > 1:
            assert np.all(np.diff(observed) >= -1e-10)

    # Across increasingly difficult gain thresholds a fixed horizon cannot become
    # more likely.
    for col in once.T:
        observed = col[np.isfinite(col)]
        if observed.size > 1:
            assert np.all(np.diff(observed) <= 1e-10)
