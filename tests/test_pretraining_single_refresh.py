from __future__ import annotations

import sqlite3

from profit_taker import pretraining_contract_v4 as continuity_contract
from profit_taker import pretraining_contract_v5 as contract
from profit_taker import v24_contract_runtime_v4 as runtime
from profit_taker.db import migrate as migrate_raw


def _seed_observations(db) -> None:
    migrate_raw(db)
    with sqlite3.connect(db) as conn:
        for i, mc in enumerate((100.0, 103.0, 101.0, 106.0)):
            conn.execute(
                """INSERT INTO axiom_observations
                (token_key,short_address_hint,snapshot_at,market_cap_usd,
                 field_confidence_json,raw_ocr_json,source_json)
                VALUES('tok','tok...x',?,?,'{}','{}','{}')""",
                (f"2026-09-01T00:0{i}:00+00:00", mc),
            )
        conn.commit()


def _count_materializations(monkeypatch):
    calls = {"n": 0}
    original = contract._v1_materializer

    def counted(db, cfg=None):
        calls["n"] += 1
        return original(db, cfg)

    monkeypatch.setattr(contract, "_v1_materializer", counted)
    return calls


def test_direct_readiness_materializes_history_once(tmp_path, monkeypatch):
    db = tmp_path / "readiness.sqlite"
    _seed_observations(db)
    calls = _count_materializations(monkeypatch)

    report = contract.training_readiness(str(db))

    assert calls["n"] == 1
    assert report["pretraining_targets_refreshed_here"] is True
    assert report["pretraining_targets_reused"] is False


def test_bootstrap_scope_reuses_one_materialization_for_readiness_and_baselines(tmp_path, monkeypatch):
    db = tmp_path / "bootstrap.sqlite"
    _seed_observations(db)
    calls = _count_materializations(monkeypatch)

    with contract.command_refresh_scope():
        contract.refresh_pretraining_targets(str(db))
        readiness = contract.training_readiness(str(db))
        baselines = contract.evaluate_baselines(str(db))

    assert calls["n"] == 1
    assert readiness["pretraining_targets_refreshed_here"] is False
    assert readiness["pretraining_targets_reused"] is True
    assert baselines["pretraining_targets_refreshed_here"] is False
    assert baselines["pretraining_targets_reused"] is True


def test_single_refresh_scope_never_leaks_into_later_command(tmp_path, monkeypatch):
    db = tmp_path / "fresh-command.sqlite"
    _seed_observations(db)
    calls = _count_materializations(monkeypatch)

    with contract.command_refresh_scope():
        contract.refresh_pretraining_targets(str(db))
        contract.training_readiness(str(db))
    assert calls["n"] == 1

    # A separate command has no in-memory freshness marker and must rebuild once.
    contract.training_readiness(str(db))
    assert calls["n"] == 2


def test_official_runtime_preserves_v4_contract_identity_with_single_refresh_installed():
    assert runtime.contract is continuity_contract
    assert continuity_contract.refresh_pretraining_targets is contract.refresh_pretraining_targets
    assert continuity_contract.training_readiness is contract.training_readiness
    assert continuity_contract.evaluate_baselines is contract.evaluate_baselines
    assert continuity_contract.command_refresh_scope is contract.command_refresh_scope
