from __future__ import annotations

import sqlite3

import pandas as pd

from profit_taker.pretraining_contract import (
    PretrainingConfig,
    _barrier_outcome,
    _economic_collapse,
    first_model_profile,
    target_contract_hash,
    target_contract_payload,
)


def _path(values: list[float], start: str = "2026-09-01T00:00:00Z") -> pd.DataFrame:
    t0 = pd.Timestamp(start)
    return pd.DataFrame({
        "snapshot_at": [t0 + pd.Timedelta(minutes=i) for i in range(len(values))],
        "market_cap_usd": values,
    })


def test_triple_barrier_uses_next_observation_not_decision_price():
    g = _path([100, 120, 155, 80])
    out = _barrier_outcome(g, g.snapshot_at.iloc[0], 15, 0.30, -0.20, [])
    assert out["reference_mc"] == 120
    # +30% from executable 120 is 156, so the 155 observation must not count.
    assert out["outcome"] == "down_first"


def test_triple_barrier_tie_is_censored_not_invented_order():
    # Deliberately use a zero-width synthetic pair so both barriers are touched in
    # the same one-minute observation.  The labeler must refuse within-minute order.
    g = _path([100, 100, 100])
    out = _barrier_outcome(g, g.snapshot_at.iloc[0], 15, 0.0, 0.0, [])
    assert out["outcome"] == "censored"


def test_manual_stop_censors_barrier_window():
    g = _path([100, 100, 100, 140, 150])
    stop = g.snapshot_at.iloc[2]
    out = _barrier_outcome(g, g.snapshot_at.iloc[0], 15, 0.30, -0.20, [stop])
    assert out["outcome"] == "censored"


def test_economic_collapse_is_separate_observed_event_not_minus_100_settlement():
    cfg = PretrainingConfig(economic_collapse_drawdown_pct=0.85, economic_collapse_sustain_minutes=10)
    values = [100] + [120] * 2 + [15] * 11
    g = _path(values)
    out = _economic_collapse(g, g.snapshot_at.iloc[0], cfg, [])
    assert out["outcome"] == "economic_collapse"
    assert out["terminal_mc"] == 15
    assert out["gross_return"] > -1.0


def test_target_hash_tracks_barrier_collapse_and_friction_contract():
    a = PretrainingConfig()
    b = PretrainingConfig(default_round_trip_bps=300.0)
    c = PretrainingConfig(down_barriers=(-0.25, -0.35))
    assert target_contract_hash(a) != target_contract_hash(b)
    assert target_contract_hash(a) != target_contract_hash(c)
    payload = target_contract_payload(a)
    assert payload["economic_collapse"]["settlement"] == "retain_observed_return; total_loss_is_stress_only"


def test_first_model_is_reduced_without_allow_small_semantics():
    profile = first_model_profile()
    assert profile["production_readiness_required"] is True
    assert profile["estimators"] <= 150
    assert profile["ts2vec_enabled"] is False
    assert profile["long_horizon_heads_enabled"] is False
