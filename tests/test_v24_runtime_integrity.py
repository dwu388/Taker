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
        assert runtime._recurrent_grid_for_cfg(cfg) == (240,)
        assert tuple(v24.RECURRENT_HORIZONS_MINUTES) == (240,)
        runtime._validate_cfg(cfg)
    finally:
        runtime._activate_recurrent_grid(v24.V24Config())


def test_runtime_cfg_uses_persisted_champion_config(tmp_path):
    cfg = v24.V24Config()
    cfg.stable_estimators = 150
    cfg.survival_bins_minutes = (5, 15, 30, 60, 120, 240)
    cfg.probability_horizons_minutes = (60, 240)
    cfg.promotion_required_horizons_minutes = (240,)
    cfg.upside_thresholds = (0.30, 0.50)
    cfg.sequence_windows_minutes = (60, 240)
    model = tmp_path / "champion.joblib"
    joblib.dump({"config": asdict(cfg)}, model)

    try:
        loaded, source = runtime._runtime_cfg(_args(model=str(model)), "predict")
        assert loaded.stable_estimators == 150
        assert loaded.probability_horizons_minutes == (60, 240)
        assert loaded.promotion_required_horizons_minutes == (240,)
        assert runtime._recurrent_grid_for_cfg(loaded) == (240,)
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
