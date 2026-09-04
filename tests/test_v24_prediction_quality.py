from __future__ import annotations

from collections import Counter

import pandas as pd

from profit_taker import axiom_v24 as v24
from profit_taker import axiom_v24_base as v24base


def test_recurrent_count_heads_follow_active_grid_and_use_poisson(monkeypatch):
    cfg = v24.V24Config()
    assert v24base._active_recurrent_count_horizons(cfg) == (240, 480, 720, 1440)

    calls: list[tuple[str, bool]] = []

    def fake_fit(data, features, target, n_estimators, *, quantile=None, poisson=False):
        calls.append((target, bool(poisson)))
        return {"target": target, "poisson": bool(poisson)}

    monkeypatch.setattr(v24base._impl, "_fit_blended_regression", fake_fit)
    frame = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
            "recurrent_peak_count_240m": [0.0, 1.0],
            "recurrent_peak_count_480m": [1.0, 1.0],
            "recurrent_peak_count_720m": [1.0, 2.0],
            "recurrent_peak_count_1440m": [2.0, 3.0],
        }
    )
    old = {
        "next_gap_q50": {"target": "recurrent_next_gap_minutes"},
        "recurrent_peak_count_4320m": {"target": "retired"},
    }
    fitted = v24base._refit_recurrent_count_heads(
        frame, ["feature"], old, 20, cfg
    )

    assert "next_gap_q50" in fitted
    assert "recurrent_peak_count_4320m" not in fitted
    expected = {
        "recurrent_peak_count_240m",
        "recurrent_peak_count_480m",
        "recurrent_peak_count_720m",
        "recurrent_peak_count_1440m",
    }
    assert expected.issubset(fitted)
    assert {name for name, _ in calls} == expected
    assert all(poisson for _, poisson in calls)


def test_cpcv_subset_balances_calendar_block_coverage():
    base = pd.Timestamp("2026-08-01T00:00:00Z")
    rows = []
    for block in range(6):
        ts = base + pd.Timedelta(days=block * 3)
        rows.append(
            {
                "token_key": f"token-{block}",
                "calendar_cohort_ordinal": block,
                "label_interval_start": ts,
                "label_interval_end": ts + pd.Timedelta(hours=1),
            }
        )
    frame = pd.DataFrame(rows)
    cfg = v24.V24Config(
        cpcv_blocks=6,
        cpcv_test_blocks=2,
        cpcv_max_splits=6,
        promotion_purge_hours=0,
        promotion_embargo_hours=0,
    )

    splits = v24base.purged_cpcv_splits(frame, cfg)
    assert len(splits) == 6
    assert all(split.get("balanced_subset") for split in splits)

    counts = Counter(
        block
        for split in splits
        for block in split["test_blocks"]
    )
    assert set(counts) == set(range(6))
    assert max(counts.values()) - min(counts.values()) <= 1
    assert sum(counts.values()) == 12

    for split in splits:
        v24base.validate_no_overlap(frame, split)
