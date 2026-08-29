from __future__ import annotations

import hashlib
import inspect
import sqlite3
import zlib

import pandas as pd
import pytest

from profit_taker import axiom_migrated_process as process
from profit_taker import axiom_migrated_runner as runner
from profit_taker import axiom_peak_structure as peak
from profit_taker import axiom_v24 as v24
from profit_taker.collection_admin import collection_status, initialize_collection
from profit_taker.db import RAW_DB_DEFAULT


def _row(token: str = "abc...pump") -> dict:
    return {
        "token_key": token,
        "token_address": None,
        "short_address_hint": token,
        "name": "Synthetic",
        "market_cap_usd": 12345.0,
        "volume_usd": 2000.0,
        "fees_sol": 1.0,
        "txns": 20,
        "training_eligible": True,
        "field_confidence": {},
        "source": {"data_origin": "clipboard_only"},
    }


class _Clipboard:
    def __init__(self, *, writable: bool = True):
        self.value = "old valid Axiom payload"
        self.writable = writable

    def copy(self, value):
        if self.writable:
            self.value = value

    def paste(self):
        return self.value


def test_clipboard_sentinel_must_be_written_and_verified():
    cb = _Clipboard(writable=True)
    sentinel = runner._prime_clipboard_with_sentinel(cb)
    assert cb.paste() == sentinel
    assert sentinel.startswith("__V24_CLIPBOARD_SENTINEL_")

    stale = _Clipboard(writable=False)
    with pytest.raises(RuntimeError, match="sentinel verification failed"):
        runner._prime_clipboard_with_sentinel(stale)
    assert stale.paste() == "old valid Axiom payload"


def test_strict_persistence_rolls_back_duplicate_canonical_rows(tmp_path):
    db = tmp_path / "raw.sqlite"
    initialize_collection(str(db))
    rows = [_row("same...pump"), _row("same...pump")]
    with pytest.raises(sqlite3.IntegrityError):
        process.process_rows(
            str(db),
            "2026-08-29T12:00:00.000+00:00",
            None,
            rows,
            True,
            tmp_path / "artifacts",
            screenshot_rows_detected=2,
            raw_clipboard_text="raw duplicate payload",
        )
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM capture_cycles").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM axiom_observations").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM capture_attempts WHERE success=1").fetchone()[0] == 0


def test_persistence_requires_marked_collection_session(tmp_path):
    db = tmp_path / "unmarked.sqlite"
    with pytest.raises(RuntimeError, match="exactly one initialized collection session"):
        process.process_rows(
            str(db),
            "2026-08-29T12:00:00.000+00:00",
            None,
            [_row()],
            True,
            tmp_path / "artifacts",
            screenshot_rows_detected=1,
            raw_clipboard_text="raw",
        )


def test_collection_status_detects_cycle_row_accounting_mismatch(tmp_path):
    db = tmp_path / "raw.sqlite"
    initialized = initialize_collection(str(db))
    raw = b"raw"
    with sqlite3.connect(db) as con:
        cur = con.execute(
            "INSERT INTO capture_cycles(session_id,captured_at,clipboard_valid,rows_detected,completed) VALUES(?,?,1,2,1)",
            (initialized["session_id"], "2026-08-29T12:00:00+00:00"),
        )
        cycle = cur.lastrowid
        con.execute(
            "INSERT INTO capture_payloads(cycle_id,sha256,byte_count,compression,payload) VALUES(?,?,?,?,?)",
            (cycle, hashlib.sha256(raw).hexdigest(), len(raw), "zlib", zlib.compress(raw)),
        )
        con.execute(
            "INSERT INTO capture_attempts(session_id,started_at,completed_at,success,clipboard_valid,rows_detected,cycle_id) VALUES(?,?,?,1,1,2,?)",
            (initialized["session_id"], "2026-08-29T12:00:00+00:00", "2026-08-29T12:00:01+00:00", cycle),
        )
        con.execute(
            "INSERT INTO axiom_observations(cycle_id,token_key,snapshot_at,market_cap_usd) VALUES(?,?,?,?)",
            (cycle, "a...pump", "2026-08-29T12:00:00+00:00", 10000.0),
        )
        con.commit()
    status = collection_status(str(db))
    assert status["cycle_row_count_mismatches"] == 1
    assert status["raw_payload_integrity_errors"] == 0
    assert status["ready_to_collect"] is False


def test_operational_database_fields_are_never_market_features():
    for name in ("observation_id", "cycle_id", "raw__cycle_id", "raw__observation_id"):
        assert peak._safe_feature_name(name) is False

    frame = pd.DataFrame({
        "token_key": [f"t{i}" for i in range(6)],
        "snapshot_at": pd.date_range("2026-08-29", periods=6, freq="min", tz="UTC"),
        "market_signal": [1, 2, 3, 4, 5, 6],
        "raw__cycle_id": [10, 11, 12, 13, 14, 15],
        "value_version": [1, 1, 1, 2, 2, 3],
        "first_ingested_at": [1, 2, 3, 4, 5, 6],
    })
    features = v24._safe_feature_columns(frame)
    assert "market_signal" in features
    assert "raw__cycle_id" not in features
    assert "value_version" not in features
    assert "first_ingested_at" not in features


def test_late_historical_insert_invalidates_later_sequence_cache(tmp_path):
    db = tmp_path / "v24.sqlite"
    with sqlite3.connect(db) as con:
        v24.migrate(con)
        for ts in ("2026-08-29T12:00:00+00:00", "2026-08-29T12:02:00+00:00"):
            con.execute(
                f"INSERT INTO {v24.SEQUENCE_CACHE_TABLE}(token_key,snapshot_at,fingerprint_json,config_hash,created_at) VALUES(?,?,?,?,?)",
                ("tok", ts, "{}", "cfg", "2026-08-29T13:00:00+00:00"),
            )
        con.commit()
        observations = pd.DataFrame({
            "token_key": ["tok", "tok", "tok"],
            "snapshot_at": pd.to_datetime([
                "2026-08-29T12:00:00Z",
                "2026-08-29T12:01:00Z",
                "2026-08-29T12:02:00Z",
            ]),
        })
        out = v24._invalidate_sequence_cache_for_late_insertions(con, observations)
        remaining = [r[0] for r in con.execute(
            f"SELECT snapshot_at FROM {v24.SEQUENCE_CACHE_TABLE} WHERE token_key='tok' ORDER BY snapshot_at"
        )]
    assert out == {"tokens": 1, "rows": 1}
    assert remaining == ["2026-08-29T12:00:00+00:00"]


def test_direct_v24_cli_defaults_to_canonical_raw_database():
    assert v24._argv_with_canonical_db(["status"]) == ["status", "--db", RAW_DB_DEFAULT]
    assert v24._argv_with_canonical_db(["status", "--db", "other.sqlite"]) == ["status", "--db", "other.sqlite"]


def test_sealed_audit_evaluator_uses_token_reason_and_confirmation_safe_truth():
    source = inspect.getsource(v24.evaluate_sealed_audit_stream)
    assert "sealed_audit_token" in source
    assert "forecast_cohort_id" in source
    assert "next_substantial_peak_confirmed_at" in source
    assert "token_balanced_peak_brier" in source
