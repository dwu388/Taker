from __future__ import annotations

import sqlite3

import numpy as np

from profit_taker import axiom_peak_structure as peak
from profit_taker import axiom_v24 as v24
from profit_taker.db import migrate


def test_v24_defaults_and_schema():
    assert v24.SCHEMA_VERSION.startswith("v24_")
    assert v24.MODEL_ROOT_DEFAULT == "models/axiom_v24"
    assert v24.POLICY_ROOT_DEFAULT == "models/axiom_policy_v24"
    assert v24.PREDICTIONS_DEFAULT == "data/axiom_predictions_v24.csv"


def test_peak_confirmation_is_explicit():
    fields = peak.SwingPeak.__dataclass_fields__
    assert "peak_at" in fields
    assert "confirmed_at" in fields
    assert "confirmation_price" in fields


def test_monotonic_projection_preserves_missing_heads():
    raw = np.array([[0.60, np.nan, 0.30], [0.70, 0.55, 0.40]], dtype=float)
    projected = v24.monotonic_probability_projection(raw)
    assert np.isnan(projected[0, 1])
    assert projected.shape == raw.shape


def test_fresh_capture_schema_has_no_v18_feature_or_provider_tables(tmp_path):
    db = tmp_path / "fresh.sqlite"
    migrate(db)
    with sqlite3.connect(db) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "capture_cycles" in tables
    assert "axiom_observations" in tables
    assert "axiom_features_v18" not in tables
    assert "provider_usage" not in tables
