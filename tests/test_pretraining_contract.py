from __future__ import annotations

import pandas as pd

from profit_taker import pretraining_contract_v4 as contract
from profit_taker import pretraining_capture_index

# Exercise the exact helper installed by the official V24 runtime.
pretraining_capture_index.install(contract)

PretrainingConfig = contract.PretrainingConfig
_barrier_outcome = contract._barrier_outcome
_economic_collapse = contract._economic_collapse
first_model_profile = contract.first_model_profile
target_contract_hash = contract.target_contract_hash
target_contract_payload = contract.target_contract_payload


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
    assert out["outcome"] == "down_first"


def test_triple_barrier_tie_is_censored_not_invented_order():
    g = _path([100, 100, 100])
    out = _barrier_outcome(g, g.snapshot_at.iloc[0], 15, 0.0, 0.0, [])
    assert out["outcome"] == "censored"


def test_manual_stop_censors_barrier_window():
    g = _path([100, 100, 100, 140, 150])
    stop = g.snapshot_at.iloc[2]
    out = _barrier_outcome(g, g.snapshot_at.iloc[0], 15, 0.30, -0.20, [stop])
    assert out["outcome"] == "censored"
    assert out["censor_reason"] == "manual_stop"


def test_triple_barrier_censors_missing_price_path_instead_of_resolving_neither():
    t0 = pd.Timestamp("2026-09-01T00:00:00Z")
    g = pd.DataFrame({
        "snapshot_at": [t0, t0 + pd.Timedelta(minutes=1), t0 + pd.Timedelta(minutes=20)],
        "market_cap_usd": [100.0, 105.0, 110.0],
    })
    out = _barrier_outcome(g, t0, 15, 0.30, -0.20, [], max_gap_minutes=5.0)
    assert out["outcome"] == "censored"
    assert "gap" in out["censor_reason"]


def test_indexed_capture_gap_lookup_preserves_v4_gap_semantics():
    t0 = pd.Timestamp("2026-09-01T00:00:00Z")
    captures = (
        t0,
        t0 + pd.Timedelta(minutes=1),
        t0 + pd.Timedelta(minutes=10),
        t0 + pd.Timedelta(minutes=11),
    )
    # The first gap wholly inside the query is still detected at its last-valid
    # capture, exactly like the original linear V4 implementation.
    assert contract._capture_gap_break(t0, t0 + pd.Timedelta(minutes=11), captures, 5.0) == t0 + pd.Timedelta(minutes=1)
    # A query that begins inside an older gap does not retroactively inspect time
    # before its decision boundary; this also matches the original semantics.
    assert contract._capture_gap_break(t0 + pd.Timedelta(minutes=8), t0 + pd.Timedelta(minutes=11), captures, 5.0) is None


def test_indexed_capture_gap_lookup_does_not_rescan_history(monkeypatch):
    t0 = pd.Timestamp("2026-09-01T00:00:00Z")
    captures = tuple(pd.date_range(t0, periods=20_000, freq="min"))
    original_utc = contract._utc
    calls = {"n": 0}

    def counting_utc(value):
        calls["n"] += 1
        return original_utc(value)

    monkeypatch.setattr(contract, "_utc", counting_utc)
    start = captures[-3]
    end = captures[-1]
    assert contract._capture_gap_break(start, end, captures, 5.0) is None
    # Only the query boundaries need normalization.  The old implementation
    # reconverted almost all 20k historical captures before reaching this window.
    assert calls["n"] <= 2


def test_economic_collapse_is_separate_observed_event_not_minus_100_settlement():
    cfg = PretrainingConfig(economic_collapse_drawdown_pct=0.85, economic_collapse_sustain_minutes=10)
    values = [100] + [120] * 2 + [15] * 11
    g = _path(values)
    out = _economic_collapse(g, g.snapshot_at.iloc[0], cfg, [])
    assert out["outcome"] == "economic_collapse"
    assert out["terminal_mc"] == 15
    assert out["gross_return"] > -1.0


def test_economic_collapse_cannot_bridge_observation_gap_to_fake_persistence():
    cfg = PretrainingConfig(economic_collapse_drawdown_pct=0.85, economic_collapse_sustain_minutes=10)
    t0 = pd.Timestamp("2026-09-01T00:00:00Z")
    g = pd.DataFrame({
        "snapshot_at": [t0, t0 + pd.Timedelta(minutes=1), t0 + pd.Timedelta(minutes=2), t0 + pd.Timedelta(minutes=20)],
        "market_cap_usd": [100.0, 120.0, 10.0, 10.0],
    })
    out = _economic_collapse(g, t0, cfg, [])
    assert out["outcome"] == "censored"
    assert "gap" in out["censor_reason"]


def test_target_hash_tracks_barrier_collapse_friction_and_continuity_contract():
    a = PretrainingConfig()
    b = PretrainingConfig(default_round_trip_bps=300.0)
    c = PretrainingConfig(down_barriers=(-0.25, -0.35))
    d = PretrainingConfig(max_price_observation_gap_minutes=10.0)
    assert target_contract_hash(a) != target_contract_hash(b)
    assert target_contract_hash(a) != target_contract_hash(c)
    assert target_contract_hash(a) != target_contract_hash(d)
    payload = target_contract_payload(a)
    assert payload["economic_collapse"]["settlement"] == "retain_observed_return; total_loss_is_stress_only"
    assert payload["triple_barrier"]["missing_price_semantics"] == "censor_on_token_disappearance_or_capture_gap"


def test_first_model_is_reduced_without_allow_small_semantics():
    profile = first_model_profile()
    assert profile["production_readiness_required"] is True
    assert profile["estimators"] <= 150
    assert profile["ts2vec_enabled"] is False
    assert profile["long_horizon_heads_enabled"] is False
