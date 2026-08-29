from __future__ import annotations

import pytest

from profit_taker.axiom_migrated_process import process_rows
from profit_taker.collection_admin import collection_status, initialize_collection


def _row():
    return {"token_key": "abc...pump", "market_cap_usd": 10000.0, "field_confidence": {}, "source": {}}


def test_replay_capture_cannot_enter_production_session(tmp_path):
    db = tmp_path / "production.sqlite"
    initialize_collection(str(db))
    with pytest.raises(RuntimeError, match="provenance does not match"):
        process_rows(
            str(db), "2026-08-29T12:00:00+00:00", None, [_row()], True,
            tmp_path / "artifacts", screenshot_rows_detected=1,
            raw_clipboard_text="replayed Axiom payload", attempt_source="replay_file",
        )
    status = collection_status(str(db))
    assert status["observations"] == 0
    assert status["ready_to_collect"] is True


def test_replay_session_is_explicitly_nonproduction(tmp_path):
    db = tmp_path / "replay.sqlite"
    initialize_collection(str(db), purpose="v24_replay_experiment")
    result = process_rows(
        str(db), "2026-08-29T12:00:00+00:00", None, [_row()], True,
        tmp_path / "artifacts", screenshot_rows_detected=1,
        raw_clipboard_text="replayed Axiom payload", attempt_source="replay_file",
    )
    assert result["rows_inserted"] == 1
    status = collection_status(str(db))
    assert status["session_purpose_matches"] is False
    assert status["ready_to_collect"] is False
