from __future__ import annotations

import math

import pandas as pd

from profit_taker import pretraining_contract_v4 as contract
from profit_taker import pretraining_path_index


def _path(values, minutes=None):
    t0 = pd.Timestamp("2026-09-01T00:00:00Z")
    offsets = list(range(len(values))) if minutes is None else list(minutes)
    return pd.DataFrame({
        "snapshot_at": [t0 + pd.Timedelta(minutes=i) for i in offsets],
        "market_cap_usd": values,
    })


def _assert_same(a, b):
    keys = {
        "outcome", "censor_reason", "event_at", "reference_at", "reference_mc",
        "terminal_mc", "gross_return", "target_ready_at",
    }
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if isinstance(av, float) or isinstance(bv, float):
            if av is None or bv is None:
                assert av is bv
            else:
                assert math.isclose(float(av), float(bv), rel_tol=1e-12, abs_tol=1e-12)
        else:
            assert av == bv, (key, av, bv, a, b)


def _fast(g, decision, horizon, up, down, censors, captures=(), max_gap=5.0):
    return pretraining_path_index.barrier_outcome(
        contract, g, decision, horizon, up, down, censors,
        capture_times=captures, max_gap_minutes=max_gap,
    )


def test_fast_barrier_matches_v4_for_resolved_and_neither_paths():
    g = _path([100, 100, 120, 135, 115, 90, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    decision = g.snapshot_at.iloc[0]
    for up in (0.30, 0.50):
        for down in (-0.20, -0.35):
            legacy = contract._barrier_outcome(g, decision, 15, up, down, [], max_gap_minutes=5.0)
            fast = _fast(g, decision, 15, up, down, [])
            _assert_same(legacy, fast)


def test_fast_barrier_matches_v4_for_manual_and_token_gap_censoring():
    g = _path([100, 105, 110, 112], minutes=[0, 1, 2, 20])
    decision = g.snapshot_at.iloc[0]
    stop = decision + pd.Timedelta(minutes=2)
    for censors in ([], [stop]):
        legacy = contract._barrier_outcome(g, decision, 15, 0.30, -0.20, censors, max_gap_minutes=5.0)
        fast = _fast(g, decision, 15, 0.30, -0.20, censors)
        _assert_same(legacy, fast)


def test_fast_barrier_matches_v4_for_capture_heartbeat_gap():
    g = _path([100, 101, 102, 103, 104, 105, 106], minutes=[0, 1, 2, 3, 4, 5, 6])
    decision = g.snapshot_at.iloc[0]
    captures = tuple([
        decision,
        decision + pd.Timedelta(minutes=1),
        decision + pd.Timedelta(minutes=2),
        decision + pd.Timedelta(minutes=6),
    ])
    legacy = contract._barrier_outcome(
        g, decision, 15, 0.30, -0.20, [], capture_times=captures, max_gap_minutes=2.0
    )
    fast = _fast(g, decision, 15, 0.30, -0.20, [], captures=captures, max_gap=2.0)
    _assert_same(legacy, fast)


def test_fast_barrier_does_not_use_pandas_row_iteration(monkeypatch):
    g = _path([100.0] * 241)

    def forbidden(*args, **kwargs):
        raise AssertionError("fast barrier evaluator must not call DataFrame.itertuples")

    monkeypatch.setattr(pd.DataFrame, "itertuples", forbidden)
    out = _fast(g, g.snapshot_at.iloc[0], 240, 0.30, -0.20, [])
    assert out["outcome"] == "neither"
