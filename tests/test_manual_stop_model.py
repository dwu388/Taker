from __future__ import annotations

import sqlite3

from profit_taker import axiom_manual_stop as manual_stop
from profit_taker import axiom_peak_structure as peak
from profit_taker import axiom_v24_base as v24base
from profit_taker import v24_contract_runtime_v4 as runtime_v4
from profit_taker.db import migrate


def _capture(db, timestamp: str, token: str, mc: float) -> int:
    migrate(db)
    with sqlite3.connect(db) as conn:
        cur = conn.execute(
            "INSERT INTO capture_cycles(captured_at,clipboard_valid,rows_detected,completed) VALUES(?,1,1,1)",
            (timestamp,),
        )
        cycle_id = int(cur.lastrowid)
        conn.execute(
            """INSERT INTO axiom_observations
            (cycle_id,token_key,snapshot_at,market_cap_usd,field_confidence_json,raw_ocr_json,source_json)
            VALUES(?,?,?,?, '{}','{}','{}')""",
            (cycle_id, token, timestamp, mc),
        )
        conn.commit()
        return cycle_id


def _manual_boundary(db, run_id: str, timestamp: str, cycle_id: int) -> None:
    manual_stop.note_successful_capture(db, run_id, capture_at=timestamp, cycle_id=cycle_id)
    manual_stop.stop_collection_session(db, run_id, stopped_at=timestamp)


def test_restart_cannot_retroactively_turn_pre_stop_path_into_peak(tmp_path):
    db = str(tmp_path / "peak.sqlite")
    run_id = manual_stop.start_collection_session(
        db, started_at="2026-08-29T11:59:00+00:00"
    )
    _capture(db, "2026-08-29T12:00:00+00:00", "A", 100.0)
    cycle = _capture(db, "2026-08-29T12:01:00+00:00", "A", 120.0)
    _manual_boundary(db, run_id, "2026-08-29T12:01:00+00:00", cycle)

    # If the two monitoring runs were incorrectly joined, 100 -> 150 -> 120 would
    # create a confirmed substantial peak. Each side of the stop separately has none.
    _capture(db, "2026-08-29T13:00:00+00:00", "A", 150.0)
    _capture(db, "2026-08-29T13:01:00+00:00", "A", 120.0)

    result = peak.refresh_labels(db, peak.PeakStructureConfig())
    assert result["manual_stop_censoring"]["tokens"] == 1
    with sqlite3.connect(db) as conn:
        pre = conn.execute(
            f"""SELECT label_status_next_peak,has_next_substantial_peak_before_terminal_72h,
                label_finalized,terminal_reason,path_end_at
                FROM {peak.LABEL_TABLE} WHERE token_key='A' AND decision_at LIKE '2026-08-29T12:00:%'"""
        ).fetchone()
        events = conn.execute(
            f"SELECT COUNT(*) FROM {peak.PEAK_EVENT_TABLE} WHERE token_key='A'"
        ).fetchone()[0]
    assert pre[0] == "censored_collection_stop"
    assert pre[1] is None
    assert pre[2] == 0
    assert pre[3] == "manual_stop_censored"
    assert str(pre[4]).startswith("2026-08-29T12:01:00")
    assert events == 0


def test_confirmed_success_before_stop_is_preserved(tmp_path):
    db = str(tmp_path / "positive.sqlite")
    run_id = manual_stop.start_collection_session(
        db, started_at="2026-08-29T11:59:00+00:00"
    )
    _capture(db, "2026-08-29T12:00:00+00:00", "B", 100.0)
    _capture(db, "2026-08-29T12:01:00+00:00", "B", 130.0)
    _capture(db, "2026-08-29T12:02:00+00:00", "B", 105.0)
    cycle = _capture(db, "2026-08-29T12:03:00+00:00", "B", 106.0)
    _manual_boundary(db, run_id, "2026-08-29T12:03:00+00:00", cycle)

    peak.refresh_labels(db, peak.PeakStructureConfig())
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            f"""SELECT label_status_next_peak,has_next_substantial_peak_before_terminal_72h,
                next_substantial_peak_confirmed_at,label_finalized
                FROM {peak.LABEL_TABLE} WHERE token_key='B' AND decision_at LIKE '2026-08-29T12:00:%'"""
        ).fetchone()
    assert row[0] == "positive_confirmed"
    assert row[1] == 1
    assert str(row[2]).startswith("2026-08-29T12:02:00")
    # The positive is known, but the rest of the 24h path remains right-censored.
    assert row[3] == 0


def test_repeated_refresh_keeps_censored_truth_and_learning_watermark_stable(tmp_path):
    db = str(tmp_path / "stable.sqlite")
    run_id = manual_stop.start_collection_session(
        db, started_at="2026-08-29T11:59:00+00:00"
    )
    _capture(db, "2026-08-29T12:00:00+00:00", "C", 100.0)
    cycle = _capture(db, "2026-08-29T12:01:00+00:00", "C", 110.0)
    _manual_boundary(db, run_id, "2026-08-29T12:01:00+00:00", cycle)
    _capture(db, "2026-08-29T13:00:00+00:00", "C", 160.0)
    _capture(db, "2026-08-29T13:01:00+00:00", "C", 120.0)

    first = peak.refresh_labels(db, peak.PeakStructureConfig())
    with sqlite3.connect(db) as conn:
        before = conn.execute(
            f"""SELECT target_fingerprint,learning_updated_at,label_status_next_peak
                FROM {peak.LABEL_TABLE}
                WHERE token_key='C' AND decision_at LIKE '2026-08-29T12:00:%'"""
        ).fetchone()
    second = peak.refresh_labels(db, peak.PeakStructureConfig())
    with sqlite3.connect(db) as conn:
        after = conn.execute(
            f"""SELECT target_fingerprint,learning_updated_at,label_status_next_peak
                FROM {peak.LABEL_TABLE}
                WHERE token_key='C' AND decision_at LIKE '2026-08-29T12:00:%'"""
        ).fetchone()

    assert first["labels_learning_updated"] >= 0
    assert second["labels_learning_updated"] == 0
    assert before == after
    assert after[2] == "censored_collection_stop"


def test_v24_counterfactual_refresh_prunes_windows_crossing_stop(tmp_path, monkeypatch):
    db = str(tmp_path / "counterfactual.sqlite")
    with sqlite3.connect(db) as conn:
        manual_stop.migrate(conn)
        conn.execute(
            f"""INSERT INTO {manual_stop.CENSOR_TABLE}
            (run_id,token_key,last_seen_at,censor_at,reason,created_at)
            VALUES('r','A','2026-08-29T12:10:00+00:00','2026-08-29T12:10:00+00:00',
                   'manual_stop_censored','2026-08-29T12:10:00+00:00')"""
        )
        conn.execute(
            f"""CREATE TABLE {v24base.COUNTERFACTUAL_TABLE}(
            token_key TEXT,decision_at TEXT,action_kind TEXT,horizon_minutes INTEGER)"""
        )
        conn.execute(
            f"INSERT INTO {v24base.COUNTERFACTUAL_TABLE} VALUES('A','2026-08-29T12:00:00+00:00','hold',60)"
        )
        conn.execute(
            f"INSERT INTO {v24base.COUNTERFACTUAL_TABLE} VALUES('A','2026-08-29T11:00:00+00:00','hold',30)"
        )
        conn.commit()

        monkeypatch.setattr(
            v24base,
            "_original_refresh_counterfactual_policy_targets",
            lambda _conn, _cfg: {"written": 0},
        )
        monkeypatch.setattr(
            runtime_v4.contract,
            "enrich_counterfactual_friction",
            lambda _conn, _cfg: {"updated": 0},
        )
        result = v24base.refresh_counterfactual_policy_targets(conn, None)
        remaining = conn.execute(
            f"SELECT decision_at FROM {v24base.COUNTERFACTUAL_TABLE} ORDER BY decision_at"
        ).fetchall()

    assert result["manual_stop_censored_targets_pruned"] == 1
    assert remaining == [("2026-08-29T11:00:00+00:00",)]
