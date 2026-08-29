from __future__ import annotations

import json
import sqlite3

import pandas as pd

from profit_taker import axiom_peak_structure as peak
from profit_taker import axiom_v24 as v24
from profit_taker import axiom_v24_base as v24_base


def test_active_lifecycle_is_24h_with_one_hour_axiom_buffer():
    cfg = v24.V24Config()
    contract = v24.lifecycle_contract()

    assert cfg.horizon_minutes == 1440
    assert cfg.age_out_minutes == 1440
    assert contract["prediction_horizon_minutes"] == 1440
    assert contract["axiom_view_minutes"] == 1500
    assert contract["buffer_minutes"] == 60
    assert max(cfg.survival_bins_minutes) == 1440
    assert max(cfg.probability_horizons_minutes) == 1440
    assert max(cfg.sequence_windows_minutes) == 1440
    assert 2880 not in cfg.probability_horizons_minutes
    assert 4320 not in cfg.probability_horizons_minutes


def test_shared_probability_and_marked_peak_heads_never_exceed_24h():
    cfg = v24.V24Config()
    barrier = v24._impl._barrier_specs(cfg)
    higher = v24._impl._higher_specs(cfg)

    assert barrier
    assert higher
    assert max(int(h) for _, _, h in barrier) == 1440
    assert max(int(h) for _, _, h in higher) == 1440
    assert all(int(h) <= cfg.horizon_minutes for _, _, h in barrier)
    assert all(int(h) <= cfg.horizon_minutes for _, _, h in higher)


def test_peak_age_out_boundary_is_natural_not_operational_death():
    cfg = peak.PeakStructureConfig()
    token = pd.DataFrame(
        [{
            "token_key": "mint-a",
            "snapshot_at": pd.Timestamp("2026-08-29T12:00:00Z"),
            "market_cap_usd": 100000.0,
            "age_minutes": 1440.0,
        }]
    )

    terminal_at, reason = peak.infer_token_terminal(token, [], {}, cfg)

    assert terminal_at == pd.Timestamp("2026-08-29T12:00:00Z")
    assert reason == "age_out_24h_model_window"
    assert "dead" not in reason


def test_old_72h_finalized_labels_force_derived_contract_rebuild(tmp_path):
    db = tmp_path / "labels.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            f"CREATE TABLE {peak.LABEL_TABLE}("
            "schema_version TEXT, config_json TEXT, label_finalized INTEGER, decision_at TEXT)"
        )
        conn.execute(
            f"INSERT INTO {peak.LABEL_TABLE}(schema_version,config_json,label_finalized,decision_at) "
            "VALUES(?,?,1,?)",
            (
                "v21_peak_structure_72h_incremental_v1",
                json.dumps({"horizon_minutes": 4320, "age_out_minutes": 4260}),
                "2026-08-29T00:00:00+00:00",
            ),
        )
        conn.commit()

    assert peak._stored_label_contract_mismatch(str(db), peak.PeakStructureConfig()) is True


def test_target_hash_separates_24h_generation_from_legacy_72h_generation():
    active = v24.V24Config()
    legacy = v24.V24Config(
        horizon_minutes=4320,
        age_out_minutes=4260.0,
        survival_bins_minutes=(5, 15, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320),
        probability_horizons_minutes=(60, 240, 720, 1440, 2880, 4320),
        sequence_windows_minutes=(360, 720, 1440, 2880, 4320),
        promotion_required_horizons_minutes=(240, 720, 1440, 4320),
    )

    assert v24.target_definition_hash(active) != v24.target_definition_hash(legacy)
    assert v24.SCHEMA_VERSION.endswith("24h_v2")


def test_retired_recurrent_horizon_is_missing_head_not_training_error():
    data = pd.DataFrame(
        {
            "token_key": ["a", "b"],
            "decision_at": pd.to_datetime(["2026-08-29T00:00:00Z", "2026-08-29T00:01:00Z"]),
        }
    )
    assert v24_base._fit_blended_regression(
        data, [], "recurrent_peak_count_4320m", 10
    ) is None
