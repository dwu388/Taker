from __future__ import annotations

import sqlite3
import sys

from profit_taker import axiom_manual_stop as manual_stop
from profit_taker import axiom_migrated_runner as runner
from profit_taker.db import migrate


def _capture(db, timestamp: str, tokens: list[str]) -> int:
    migrate(db)
    with sqlite3.connect(db) as conn:
        cur = conn.execute(
            "INSERT INTO capture_cycles(captured_at,clipboard_valid,rows_detected,completed) VALUES(?,1,?,1)",
            (timestamp, len(tokens)),
        )
        cycle_id = int(cur.lastrowid)
        for token in tokens:
            conn.execute(
                """INSERT INTO axiom_observations
                (cycle_id,token_key,snapshot_at,market_cap_usd,field_confidence_json,raw_ocr_json,source_json)
                VALUES(?,?,?,?, '{}','{}','{}')""",
                (cycle_id, token, timestamp, 10000.0),
            )
        conn.commit()
        return cycle_id


def test_manual_stop_censors_only_still_unresolved_recent_tokens(tmp_path):
    db = str(tmp_path / "live.sqlite")
    run_id = manual_stop.start_collection_session(
        db, started_at="2026-08-29T11:55:00+00:00"
    )
    _capture(db, "2026-08-29T12:00:00+00:00", ["stale-token"])
    _capture(db, "2026-08-29T12:20:00+00:00", ["recent-missing"])
    cycle_id = _capture(db, "2026-08-29T13:00:00+00:00", ["present-token"])
    manual_stop.note_successful_capture(
        db, run_id, capture_at="2026-08-29T13:00:00+00:00", cycle_id=cycle_id
    )
    out = manual_stop.stop_collection_session(
        db, run_id, stopped_at="2026-08-29T13:00:05+00:00"
    )

    assert out["censor_at"].startswith("2026-08-29T13:00:00")
    assert out["active_tokens_censored"] == 2
    with sqlite3.connect(db) as conn:
        tokens = {
            r[0]
            for r in conn.execute(
                f"SELECT token_key FROM {manual_stop.CENSOR_TABLE} ORDER BY token_key"
            ).fetchall()
        }
    assert tokens == {"present-token", "recent-missing"}


def test_stop_reconciles_to_newer_durable_capture_even_if_runner_heartbeat_lags(tmp_path):
    db = str(tmp_path / "durable.sqlite")
    run_id = manual_stop.start_collection_session(
        db, started_at="2026-08-29T11:55:00+00:00"
    )
    first = _capture(db, "2026-08-29T12:00:00+00:00", ["A"])
    manual_stop.note_successful_capture(
        db, run_id, capture_at="2026-08-29T12:00:00+00:00", cycle_id=first
    )
    # Simulate Ctrl+C after the next capture transaction committed but before the
    # wrapper had time to update its convenience heartbeat.
    second = _capture(db, "2026-08-29T12:01:00+00:00", ["B"])
    out = manual_stop.stop_collection_session(
        db, run_id, stopped_at="2026-08-29T12:01:02+00:00"
    )

    assert out["censor_at"].startswith("2026-08-29T12:01:00")
    with sqlite3.connect(db) as conn:
        stored = conn.execute(
            f"SELECT last_cycle_id,last_capture_at FROM {manual_stop.RUN_TABLE} WHERE run_id=?",
            (run_id,),
        ).fetchone()
    assert stored[0] == second
    assert str(stored[1]).startswith("2026-08-29T12:01:00")


def test_manual_stop_censors_open_paper_state_without_return_target(tmp_path):
    db = str(tmp_path / "paper.sqlite")
    run_id = manual_stop.start_collection_session(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE axiom_paper_positions_v20(
            position_id TEXT PRIMARY KEY,status TEXT,closed_at TEXT,close_reason TEXT,exit_kind TEXT,
            reward REAL,net_return_pct REAL,observed_reward REAL,execution_reward REAL)"""
        )
        conn.execute(
            "INSERT INTO axiom_paper_positions_v20 VALUES('p','open',NULL,NULL,NULL,0.5,0.4,0.5,0.4)"
        )
        conn.execute(
            """CREATE TABLE axiom_paper_pending_entries_v24(
            pending_id TEXT PRIMARY KEY,status TEXT,cancelled_at TEXT,cancel_reason TEXT)"""
        )
        conn.execute("INSERT INTO axiom_paper_pending_entries_v24 VALUES('e','pending',NULL,NULL)")
        conn.commit()

    out = manual_stop.stop_collection_session(
        db, run_id, stopped_at="2026-08-29T13:00:00+00:00"
    )
    assert out["paper_positions_censored"] == 1
    assert out["pending_entries_cancelled"] == 1
    with sqlite3.connect(db) as conn:
        pos = conn.execute(
            "SELECT status,close_reason,exit_kind,reward,net_return_pct,observed_reward,execution_reward FROM axiom_paper_positions_v20"
        ).fetchone()
        pending = conn.execute(
            "SELECT status,cancel_reason FROM axiom_paper_pending_entries_v24"
        ).fetchone()
    assert pos[:3] == ("censored", "manual_stop_censored", "collection_censored")
    assert pos[3:] == (None, None, None, None)
    assert pending == ("cancelled", "manual_stop_censored")


def test_ctrl_c_records_manual_stop_in_continuous_runner(tmp_path, monkeypatch):
    db = str(tmp_path / "ctrlc.sqlite")
    monkeypatch.setattr(sys, "argv", ["axiom_migrated_runner", "--db", db])

    def interrupt_loop():
        raise KeyboardInterrupt

    monkeypatch.setattr(runner._impl, "main", interrupt_loop)
    runner.main()

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            f"SELECT status,stop_reason FROM {manual_stop.RUN_TABLE} ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    assert row == ("stopped", "manual_stop_censored")
