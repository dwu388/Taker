from __future__ import annotations

import sqlite3

import pytest

from profit_taker.axiom_migrated_process import process_rows
from profit_taker.collection_admin import collection_status, initialize_collection
from profit_taker.db import COLLECTOR_SCHEMA_VERSION, migrate


def _insert_session(db, *, purpose: str, schema: str) -> None:
    migrate(db)
    with sqlite3.connect(db) as con:
        con.execute(
            "INSERT INTO collection_sessions(session_id,started_at,purpose,collector_schema) VALUES('s','2026-08-29T12:00:00+00:00',?,?)",
            (purpose, schema),
        )
        con.commit()


def test_production_resume_rejects_mismatched_collector_schema(tmp_path):
    db = tmp_path / "wrong-schema.sqlite"
    _insert_session(db, purpose="v24_production_raw_collection", schema="v23_old_schema")
    with pytest.raises(RuntimeError, match="different collector schema"):
        initialize_collection(str(db))
    status = collection_status(str(db))
    assert status["session_schema_matches"] is False
    assert status["ready_to_collect"] is False


def test_interactive_persistence_rejects_mismatched_session_purpose(tmp_path):
    db = tmp_path / "wrong-purpose.sqlite"
    _insert_session(db, purpose="manual_replay_experiment", schema=COLLECTOR_SCHEMA_VERSION)
    with pytest.raises(RuntimeError, match="provenance does not match"):
        process_rows(
            str(db),
            "2026-08-29T12:01:00+00:00",
            None,
            [{"token_key": "abc...pump", "market_cap_usd": 10000.0}],
            True,
            tmp_path / "artifacts",
            screenshot_rows_detected=1,
            raw_clipboard_text="synthetic raw payload",
        )
    status = collection_status(str(db))
    assert status["session_purpose_matches"] is False
    assert status["ready_to_collect"] is False
