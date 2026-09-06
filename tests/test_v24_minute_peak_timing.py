from __future__ import annotations

import numpy as np
import pandas as pd

from profit_taker import axiom_v24 as v24


def test_minute_contract_keeps_primary_grid_and_adds_10m_confirmation_hazard():
    cfg = v24.V24Config()
    assert cfg.probability_horizons_minutes == (60, 240, 480, 720, 1440)
    assert cfg.survival_bins_minutes[:5] == (5, 10, 15, 30, 60)
    contract = v24.lifecycle_contract()
    assert contract["first_peak_confirmation_horizons_minutes"] == [5, 10, 15, 30, 60]
    assert contract["peak_timing_semantics"]["occurrence"] == "peak_at - decision_at"


def test_recurrent_targets_separate_occurrence_from_confirmation(monkeypatch):
    decision = pd.Timestamp("2026-09-01T00:00:00Z")
    events = {
        "T": [
            {
                "peak_at": decision + pd.Timedelta(minutes=10),
                "confirmed_at": decision + pd.Timedelta(minutes=15),
                "peak_price": 150.0,
                "confirmation_price": 126.0,
                "runup_pct": 0.50,
                "confirmation_retrace_pct": 0.16,
            },
            {
                "peak_at": decision + pd.Timedelta(minutes=43),
                "confirmed_at": decision + pd.Timedelta(minutes=50),
                "peak_price": 205.0,
                "confirmation_price": 170.0,
                "runup_pct": 0.62,
                "confirmation_retrace_pct": 0.17,
            },
        ]
    }
    monkeypatch.setattr(v24._impl, "_future_peak_lists", lambda conn: events)
    monkeypatch.setattr(
        v24._impl,
        "_known_followup_end",
        lambda row, cfg, as_of: decision + pd.Timedelta(minutes=60),
    )
    frame = pd.DataFrame(
        [
            {
                "token_key": "T",
                "decision_at": decision,
                "decision_market_cap_usd": 100.0,
                "label_finalized": 0,
                "path_end_at": decision + pd.Timedelta(minutes=60),
            }
        ]
    )
    out = v24.add_recurrent_targets(None, frame, v24.V24Config())
    row = out.iloc[0]
    assert row.recurrent_next_occurrence_gap_minutes == 10.0
    assert row.recurrent_next_gap_minutes == 15.0
    assert row.recurrent_next_confirmation_lag_minutes == 5.0
    assert row.recurrent_second_occurrence_gap_minutes == 33.0
    assert row.recurrent_second_gap_minutes == 35.0
    assert row.recurrent_second_confirmation_lag_minutes == 7.0


def test_minute_timing_fitter_uses_occurrence_targets(monkeypatch):
    calls = []

    def fake_fit(data, features, target, n_estimators, *, quantile=None, poisson=False):
        calls.append((target, quantile))
        return {"target": target, "quantile": quantile}

    monkeypatch.setattr(v24._impl, "_fit_blended_regression", fake_fit)
    frame = pd.DataFrame(
        {
            "token_key": ["A", "B"],
            "f": [1.0, 2.0],
            "recurrent_next_occurrence_gap_minutes": [10.0, 20.0],
            "recurrent_next_confirmation_lag_minutes": [5.0, 6.0],
            "recurrent_second_occurrence_gap_minutes": [30.0, 40.0],
            "recurrent_second_confirmation_lag_minutes": [7.0, 8.0],
            "recurrent_next_gap_minutes": [15.0, 26.0],
            "recurrent_second_gap_minutes": [35.0, 46.0],
            "recurrent_second_peak_relative_to_first": [0.30, -0.10],
        }
    )
    fitted = v24._fit_minute_timing_heads(v24._impl, frame, ["f"], {}, 20)
    assert "next_occurrence_q10" in fitted
    assert "next_occurrence_q50" in fitted
    assert "next_occurrence_q90" in fitted
    assert "next_confirmation_lag_q50" in fitted
    assert "second_occurrence_gap_q50" in fitted
    assert "second_confirmation_lag_q50" in fitted
    assert ("recurrent_next_occurrence_gap_minutes", 0.5) in calls


def test_minute_quantile_projection_is_ordered_and_nan_safe():
    pred = {
        "next_occurrence_q10": np.array([30.0, np.nan]),
        "next_occurrence_q25": np.array([20.0, 10.0]),
        "next_occurrence_q50": np.array([10.0, 20.0]),
        "next_occurrence_q75": np.array([40.0, 30.0]),
        "next_occurrence_q90": np.array([35.0, 40.0]),
    }
    v24.project_recurrent_outputs(pred)
    first = [pred[k][0] for k in (
        "next_occurrence_q10", "next_occurrence_q25", "next_occurrence_q50",
        "next_occurrence_q75", "next_occurrence_q90",
    )]
    assert first == sorted(first)
    assert np.isnan(pred["next_occurrence_q10"][1])
    observed_second = [pred[k][1] for k in (
        "next_occurrence_q25", "next_occurrence_q50", "next_occurrence_q75", "next_occurrence_q90",
    )]
    assert observed_second == sorted(observed_second)
