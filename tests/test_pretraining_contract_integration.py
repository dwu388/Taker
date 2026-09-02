from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from profit_taker import axiom_v24 as v24
from profit_taker.db import migrate as migrate_raw
from profit_taker import pretraining_contract_v3 as contract
from profit_taker import v24_contract_runtime as shared_runtime
from profit_taker import v24_contract_runtime_v3 as runtime


def test_economic_collapse_materialization_is_idempotent(tmp_path):
    db = tmp_path / "raw.sqlite"
    migrate_raw(db)
    with sqlite3.connect(db) as conn:
        for i, mc in enumerate((100.0, 120.0, 110.0)):
            conn.execute(
                """INSERT INTO axiom_observations
                (token_key,short_address_hint,snapshot_at,market_cap_usd,field_confidence_json,raw_ocr_json,source_json)
                VALUES('tok','tok...x',?,?,'{}','{}','{}')""",
                (f"2026-08-01T00:0{i}:00+00:00", mc),
            )
        conn.commit()
    contract.refresh_pretraining_targets(str(db))
    contract.refresh_pretraining_targets(str(db))
    with sqlite3.connect(db) as conn:
        n = conn.execute(
            f"SELECT COUNT(*) FROM {contract.TARGET_TABLE} WHERE target_kind='economic_collapse'"
        ).fetchone()[0]
    assert n == 3


def test_policy_compatibility_aliases_are_net_and_provenance_is_stable(tmp_path):
    db = tmp_path / "policy.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE axiom_v24_counterfactual_policy_targets(
            token_key TEXT, decision_at TEXT, action_kind TEXT, horizon_minutes INTEGER,
            entry_execution_return REAL, exit_now_return REAL,
            hold_terminal_return REAL, hold_advantage_return REAL,
            source_fingerprint TEXT)"""
        )
        conn.execute(
            "INSERT INTO axiom_v24_counterfactual_policy_targets VALUES('t','2026-08-01T00:00:00Z','entry',60,0.20,0.05,0.10,0.04,'old')"
        )
        conn.commit()
        out = contract.enrich_counterfactual_friction(conn, contract.PretrainingConfig(default_round_trip_bps=100.0))
        row1 = conn.execute(
            """SELECT entry_return_gross,entry_return_net,entry_execution_return,
                      exit_now_return_gross,exit_now_return_net,exit_now_return,
                      hold_terminal_return_gross,hold_terminal_return_net,hold_terminal_return,
                      hold_advantage_gross,hold_advantage_net,hold_advantage_return,
                      friction_definition_hash,source_fingerprint,pre_friction_source_fingerprint
               FROM axiom_v24_counterfactual_policy_targets"""
        ).fetchone()
        contract.enrich_counterfactual_friction(conn, contract.PretrainingConfig(default_round_trip_bps=100.0))
        row2 = conn.execute(
            """SELECT entry_return_gross,entry_return_net,entry_execution_return,
                      exit_now_return_gross,exit_now_return_net,exit_now_return,
                      hold_terminal_return_gross,hold_terminal_return_net,hold_terminal_return,
                      hold_advantage_gross,hold_advantage_net,hold_advantage_return,
                      friction_definition_hash,source_fingerprint,pre_friction_source_fingerprint
               FROM axiom_v24_counterfactual_policy_targets"""
        ).fetchone()
    assert out["stable_friction_provenance_rows"] == 1
    assert row1[0] == 0.20
    assert abs(row1[1] - 0.19) < 1e-12
    assert abs(row1[2] - 0.19) < 1e-12
    assert row1[3] == 0.05
    assert abs(row1[4] - 0.045) < 1e-12
    assert abs(row1[5] - 0.045) < 1e-12
    assert row1[6] == 0.10
    assert abs(row1[7] - 0.095) < 1e-12
    assert abs(row1[8] - 0.095) < 1e-12
    assert row1[9] == row1[10] == row1[11] == 0.04
    assert row1[12]
    assert row1[13] != "old"
    assert row1[14] == "old"
    # Repeated enrichment must be mathematically and provenance-idempotent.
    assert row2 == row1


def test_combined_target_hash_includes_pretraining_contract():
    cfg = v24.V24Config()
    base = v24.target_definition_hash(cfg)
    shared_runtime.install_target_hash_contract(contract.PretrainingConfig())
    combined = v24.target_definition_hash(cfg)
    assert combined != base
    assert len(combined) == 64


def test_runtime_uses_latest_contract_layer():
    assert runtime.contract is contract


def test_first_model_runtime_reduces_capacity_without_allow_small():
    args = SimpleNamespace(cohort_hours=24, promotion_every=4, audit_every=5, warmup_blocks=7)
    cfg = runtime._cfg(args)
    profile = runtime._apply_first_model_profile(cfg, contract.PretrainingConfig())
    assert profile["production_readiness_required"] is True
    assert cfg.stable_estimators <= 150
    assert cfg.probability_horizons_minutes == (60, 240)
    assert max(cfg.survival_bins_minutes) == 240
    assert cfg.upside_thresholds == (0.30, 0.50)
    assert cfg.sequence_challenger_min_tokens >= 10**9
    assert v24.RECURRENT_HORIZONS_MINUTES == (240,)
