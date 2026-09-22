from __future__ import annotations

import hashlib
import sqlite3
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from profit_taker import axiom_migrated_runner as runner
from profit_taker.axiom_migrated_process import process_rows, record_failed_capture_attempt
from profit_taker.collection_admin import collection_status, initialize_collection
from profit_taker.db import migrate


def _row(token: str = "abc...pump") -> dict:
    return {"token_key": token, "token_address": None, "short_address_hint": token, "name": "Synthetic", "market_cap_usd": 12345.0, "volume_usd": 2000.0, "fees_sol": 1.0, "txns": 20, "training_eligible": True, "field_confidence": {}, "source": {"data_origin": "clipboard_only"}}


def test_fresh_collection_session_is_idempotent_and_duration_agnostic(tmp_path):
    db = tmp_path / "raw.sqlite"
    first = initialize_collection(str(db)); second = initialize_collection(str(db)); status = collection_status(str(db))
    assert first["initialized"] is True
    assert second["resuming"] is True
    assert second["session_id"] == first["session_id"]
    assert status["ready_to_collect"] is True
    assert status["duration_gate"] is None


def test_fresh_collection_refuses_unmarked_existing_raw_data(tmp_path):
    db = tmp_path / "old.sqlite"; migrate(db)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO capture_cycles(captured_at,clipboard_valid,rows_detected,completed) VALUES(?,1,1,1)", ("2026-08-29T12:00:00+00:00",)); con.commit()
    with pytest.raises(RuntimeError, match="unmarked collection data"):
        initialize_collection(str(db))


def test_exact_raw_clipboard_payload_is_durable_and_hashed(tmp_path):
    db = tmp_path / "raw.sqlite"; initialize_collection(str(db)); raw = "EXACT UTF-8 RAW PAYLOAD\nMC\n$12.3K\n"
    out = process_rows(str(db), "2026-08-29T12:00:00.000+00:00", None, [_row()], True, tmp_path / "artifacts", screenshot_rows_detected=1, raw_clipboard_text=raw, attempt_started_at="2026-08-29T11:59:59.000+00:00")
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        payload = con.execute("SELECT sha256,byte_count,compression,payload FROM capture_payloads").fetchone()
        attempt = con.execute("SELECT success,cycle_id,raw_payload_sha256 FROM capture_attempts").fetchone()
    expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert out["raw_payload_sha256"] == expected
    assert payload["sha256"] == expected
    assert zlib.decompress(payload["payload"]).decode("utf-8") == raw
    assert attempt["success"] == 1
    assert attempt["cycle_id"] == out["cycle_id"]


def test_post_commit_artifact_failure_is_nonfatal(tmp_path):
    db = tmp_path / "raw.sqlite"; initialize_collection(str(db)); blocked = tmp_path / "not_a_directory"; blocked.write_text("file", encoding="utf-8")
    out = process_rows(str(db), "2026-08-29T12:00:00.000+00:00", None, [_row()], True, blocked, screenshot_rows_detected=1, raw_clipboard_text="raw")
    assert out["rows_inserted"] == 1
    assert out["artifact_write_errors"]
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM capture_cycles").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM axiom_observations").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM capture_payloads").fetchone()[0] == 1


def test_database_only_capture_writes_no_manual_review_files(tmp_path):
    db = tmp_path / "raw.sqlite"
    artifacts = tmp_path / "artifacts"
    initialize_collection(str(db))
    out = process_rows(
        str(db),
        "2026-08-29T12:00:00.000+00:00",
        None,
        [_row()],
        True,
        artifacts,
        screenshot_rows_detected=1,
        raw_clipboard_text="raw",
        write_review_artifacts=False,
    )

    assert db.is_file()
    assert not artifacts.exists()
    assert out["database"] == str(db)
    assert out["artifact_mode"] == "database_only"
    assert out["artifact_write_errors"] == []
    assert "json" not in out
    assert "csv" not in out


def test_failed_attempt_keeps_debug_payload_out_of_observations(tmp_path):
    db = tmp_path / "raw.sqlite"; initialize_collection(str(db))
    attempt_id = record_failed_capture_attempt(str(db), started_at="2026-08-29T12:00:00+00:00", clipboard_valid=True, rows_detected=2, error_type="RuntimeError", error_message="parse incomplete", raw_clipboard_text="failed raw text", details={"candidate_cards": 3})
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM axiom_observations").fetchone()[0] == 0
        assert con.execute("SELECT success FROM capture_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()[0] == 0
        payload = con.execute("SELECT payload FROM capture_attempt_payloads WHERE attempt_id=?", (attempt_id,)).fetchone()[0]
    assert zlib.decompress(payload).decode("utf-8") == "failed raw text"


def test_runner_rejects_partial_clipboard_parse_before_persistence(tmp_path, monkeypatch):
    selection = tmp_path / "capture.txt"; selection.write_text("synthetic", encoding="utf-8"); rows = [_row("a...pump"), _row("b...pump")]
    monkeypatch.setattr(runner, "clipboard_looks_like_axiom", lambda text: True)
    monkeypatch.setattr(runner, "rows_from_clipboard", lambda text: list(rows))
    monkeypatch.setattr(runner, "_raw_mc_card_count", lambda text: 3)
    args = SimpleNamespace(config=str(tmp_path / "missing.json"), clipboard_file=str(selection), output_dir=str(tmp_path / "artifacts"), db=str(tmp_path / "raw.sqlite"))
    with pytest.raises(RuntimeError, match="parse completeness check failed"):
        runner.run_once(args, 1)
    assert not Path(args.db).exists()


def _clipboard_card(address: str, market_cap: str) -> str:
    return "\n".join((
        "MC", market_cap, "V", "$10K", "TX 12", address,
        "TICK", "Token", "1m", "10", "2", "1", "0/1", "3", "12%",
    ))


def test_standalone_mc_ui_label_is_not_mistaken_for_a_missing_card(tmp_path):
    text = "\n".join((
        "Pulse", "Migrated", "Axiom dashboard controls " * 5,
        "MC",  # Sort/header label, not a token card.
        _clipboard_card("abc...pump", "$100K"),
        _clipboard_card("def...pump", "$90K"),
    ))
    rows = runner.rows_from_clipboard(text)

    assert len(rows) == 2
    assert runner._raw_mc_card_count(text) == 2

    db = tmp_path / "raw.sqlite"
    initialize_collection(str(db))
    args = SimpleNamespace(
        config=str(tmp_path / "missing.json"), clipboard_file=None,
        output_dir=str(tmp_path / "artifacts"), db=str(db), database_only=True,
    )
    runner._sync_test_and_extension_hooks()
    result = runner.process_capture(args, text, "2026-09-22T12:00:00.000+00:00")
    assert result["rows_inserted"] == 2


def test_structural_but_malformed_mc_card_still_fails_closed(tmp_path):
    malformed = "\n".join((
        "MC", "$80K", "V", "$5K", "TX 4", "missing", "token", "identity",
    ))
    text = "\n".join((
        "Pulse", "Migrated", "Axiom dashboard controls " * 5,
        _clipboard_card("abc...pump", "$100K"),
        _clipboard_card("def...pump", "$90K"),
        malformed,
    ))
    args = SimpleNamespace(
        config=str(tmp_path / "missing.json"), clipboard_file=None,
        output_dir=str(tmp_path / "artifacts"), db=str(tmp_path / "raw.sqlite"),
        database_only=True,
    )

    assert runner._raw_mc_card_count(text) == 3
    assert len(runner.rows_from_clipboard(text)) == 2
    with pytest.raises(RuntimeError, match="3 MC card blocks but 2 parsed rows"):
        runner.process_capture(args, text, "2026-09-22T12:00:00.000+00:00")


def test_collection_status_detects_payload_coverage_gap(tmp_path):
    db = tmp_path / "raw.sqlite"; initialize_collection(str(db))
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO capture_cycles(captured_at,clipboard_valid,rows_detected,completed) VALUES(?,1,1,1)", ("2026-08-29T12:00:00+00:00",)); con.commit()
    status = collection_status(str(db))
    assert status["raw_payload_complete"] is False
    assert status["ready_to_collect"] is False
