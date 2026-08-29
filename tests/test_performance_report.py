from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from profit_taker import axiom_migrated_runner as runner
from profit_taker.axiom_migrated_process import process_rows
from profit_taker.collection_admin import initialize_collection
from profit_taker.performance_report import maybe_generate_report


def _row() -> dict:
    return {
        "token_key": "abc...pump",
        "token_address": None,
        "short_address_hint": "abc...pump",
        "name": "Synthetic",
        "market_cap_usd": 12345.0,
        "volume_usd": 2000.0,
        "fees_sol": 1.0,
        "txns": 20,
        "training_eligible": True,
        "field_confidence": {},
        "source": {"data_origin": "clipboard_only"},
    }


def _ready_raw_db(tmp_path):
    db = tmp_path / "raw.sqlite"
    initialize_collection(str(db))
    process_rows(
        str(db),
        "2026-08-29T12:00:00.000+00:00",
        None,
        [_row()],
        True,
        tmp_path / "artifacts",
        screenshot_rows_detected=1,
        raw_clipboard_text="authoritative raw payload",
        attempt_started_at="2026-08-29T11:59:59.000+00:00",
        attempt_source="interactive_clipboard",
    )
    return db


def test_report_summarizes_recorded_successes_and_failures(tmp_path):
    db = _ready_raw_db(tmp_path)
    with sqlite3.connect(db) as con:
        con.executescript(
            """
            CREATE TABLE axiom_v24_model_registry(
                version_id TEXT, created_at TEXT, model_path TEXT, model_hash TEXT,
                status TEXT, stable_training_cutoff TEXT, stable_generation INTEGER,
                adapter_round INTEGER, metrics_json TEXT
            );
            INSERT INTO axiom_v24_model_registry VALUES(
                'v1','2026-08-29T13:00:00+00:00','m','hash','champion',
                '2026-08-29T12:00:00+00:00',1,0,'{"development_brier":0.20}'
            );
            CREATE TABLE axiom_v24_promotions(
                promotion_id TEXT, created_at TEXT, cohort_id TEXT, promoted INTEGER,
                metrics_json TEXT, reason TEXT
            );
            INSERT INTO axiom_v24_promotions VALUES(
                'p1','2026-08-29T14:00:00+00:00','c1',1,
                '{"candidate":{"score":0.10}}','candidate beat champion'
            );
            INSERT INTO axiom_v24_promotions VALUES(
                'p2','2026-08-29T15:00:00+00:00','c2',0,
                '{"candidate":{"score":0.30}}','candidate failed confidence gate'
            );
            CREATE TABLE axiom_v24_prediction_ledger(
                prediction_id TEXT, token_key TEXT, provenance TEXT, oos_valid INTEGER,
                policy_training_eligible INTEGER, ineligibility_reason TEXT
            );
            INSERT INTO axiom_v24_prediction_ledger VALUES('x','t1','live',1,1,NULL);
            CREATE TABLE axiom_v24_audit_results(
                audit_result_id TEXT, created_at TEXT, model_family TEXT,
                prediction_rows INTEGER, metric_json TEXT, note TEXT
            );
            INSERT INTO axiom_v24_audit_results VALUES(
                'a1','2026-08-29T16:00:00+00:00','v24',10,
                '{"token_balanced_peak_brier":0.12,"token_balanced_peak_log_loss":0.33,"tokens":5}',
                'sealed audit'
            );
            CREATE TABLE axiom_v24_policy_registry(
                version_id TEXT, created_at TEXT, status TEXT, training_rows_entry INTEGER,
                training_rows_hold INTEGER, oos_only INTEGER, metrics_json TEXT
            );
            INSERT INTO axiom_v24_policy_registry VALUES(
                'pv1','2026-08-29T16:00:00+00:00','champion',100,90,1,
                '{"bootstrap":{"ci_low":0.05}}'
            );
            CREATE TABLE axiom_v24_policy_promotions(
                promotion_id TEXT, created_at TEXT, cohort_id TEXT, promoted INTEGER,
                metrics_json TEXT, reason TEXT
            );
            INSERT INTO axiom_v24_policy_promotions VALUES(
                'pp1','2026-08-29T16:30:00+00:00','pc1',1,
                '{"bootstrap":{"ci_low":0.05}}','positive token-balanced value'
            );
            CREATE TABLE axiom_paper_positions_v20(
                position_id TEXT, token_key TEXT, opened_at TEXT, status TEXT, closed_at TEXT,
                execution_net_return_pct REAL, observed_net_return_pct REAL, net_return_pct REAL,
                execution_reward REAL, reward REAL, mfe_pct REAL, mae_pct REAL,
                execution_peak_capture_ratio REAL, close_reason TEXT
            );
            INSERT INTO axiom_paper_positions_v20 VALUES(
                '1','t1','2026-08-29T13:00:00+00:00','closed','2026-08-29T14:00:00+00:00',
                0.20,0.25,0.25,0.20,0.20,0.40,-0.10,0.50,'policy_exit'
            );
            INSERT INTO axiom_paper_positions_v20 VALUES(
                '2','t2','2026-08-29T13:00:00+00:00','closed','2026-08-29T14:30:00+00:00',
                -0.10,-0.08,-0.08,-0.10,-0.10,0.10,-0.20,0.20,'policy_exit'
            );
            """
        )
        con.commit()

    out = tmp_path / "CURRENT_MODEL_PERFORMANCE.txt"
    result = maybe_generate_report(str(db), output_path=str(out), benchmark_db=str(tmp_path / "missing-benchmark.sqlite"), force=True)
    text = out.read_text(encoding="utf-8")

    assert result["generated"] is True
    assert "Evidence maturity: PROSPECTIVE_AUDIT_EVIDENCE_AVAILABLE" in text
    assert "Forecast promotions: 2 total | 1 promoted | 1 rejected" in text
    assert "Token-balanced peak Brier: 0.1200" in text
    assert "Policy promotions: 1 total | 1 promoted | 0 rejected" in text
    assert "Closed-trade execution win rate: 50.0%" in text
    assert "candidate failed confidence gate" in text
    assert "No sealed prospective audit result is mature yet" not in text


def test_report_generation_respects_configured_interval(tmp_path):
    db = _ready_raw_db(tmp_path)
    out = tmp_path / "report.txt"
    start = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)

    first = maybe_generate_report(str(db), output_path=str(out), benchmark_db=str(tmp_path / "missing.sqlite"), interval_minutes=60, force=True, now=start)
    os.utime(out, (start.timestamp(), start.timestamp()))
    early = maybe_generate_report(str(db), output_path=str(out), benchmark_db=str(tmp_path / "missing.sqlite"), interval_minutes=60, now=start + timedelta(minutes=30))
    due = maybe_generate_report(str(db), output_path=str(out), benchmark_db=str(tmp_path / "missing.sqlite"), interval_minutes=60, now=start + timedelta(minutes=61))

    assert first["generated"] is True
    assert early == {
        "generated": False,
        "reason": "interval_not_elapsed",
        "output_path": str(out),
        "next_due_at": "2026-08-29T13:00:00+00:00",
        "interval_minutes": 60,
    }
    assert due["generated"] is True


def test_collector_calls_reporter_after_success_without_making_it_fatal(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "performance_report": {
            "enabled": True,
            "interval_minutes": 90,
            "output_path": str(tmp_path / "perf.txt"),
            "benchmark_db": str(tmp_path / "bench.sqlite"),
            "recent_events": 4,
        }
    }), encoding="utf-8")
    args = SimpleNamespace(config=str(cfg), db=str(tmp_path / "raw.sqlite"))
    seen = {}

    monkeypatch.setattr(runner, "_original_run_once", lambda *a, **k: {"rows_stored": 3})

    def fake_report(db_path, **kwargs):
        seen["db"] = db_path
        seen.update(kwargs)
        return {"generated": True, "output_path": kwargs["output_path"]}

    monkeypatch.setattr(runner, "maybe_generate_report", fake_report)
    result = runner.run_once(args, 1)

    assert result["rows_stored"] == 3
    assert result["performance_report"]["generated"] is True
    assert seen["interval_minutes"] == 90
    assert seen["recent_events"] == 4

    monkeypatch.setattr(runner, "maybe_generate_report", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    result = runner.run_once(args, 2)
    assert result["rows_stored"] == 3
    assert result["performance_report"]["reason"] == "report_error"
    assert "disk full" in result["performance_report"]["error"]
