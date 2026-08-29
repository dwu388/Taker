from __future__ import annotations

import sqlite3

import pandas as pd

from profit_taker import axiom_manual_stop as stop


def test_manual_stop_is_neutral_boundary(tmp_path):
    db = tmp_path / "live.sqlite"
    with sqlite3.connect(db) as conn:
        stop.migrate(conn)
        conn.execute(
            f"INSERT INTO {stop.SESSION_TABLE}(session_id,source,started_at,last_capture_at,stopped_at,censor_at,stop_reason,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            ("s", "test", "2026-08-29T10:00:00+00:00", "2026-08-29T10:10:00+00:00", "2026-08-29T10:11:00+00:00", "2026-08-29T10:10:00+00:00", "manual_stop_censored", "stopped", "2026-08-29T10:00:00+00:00"),
        )
        conn.commit()
        boundary = stop.first_collection_boundary_between(conn, pd.Timestamp("2026-08-29T10:09:00Z"), pd.Timestamp("2026-08-29T11:00:00Z"))
        assert boundary == pd.Timestamp("2026-08-29T10:10:00Z")


def test_censor_reason_is_not_death(tmp_path):
    db = tmp_path / "live.sqlite"
    with sqlite3.connect(db) as conn:
        stop.migrate(conn)
        conn.execute(
            f"INSERT INTO {stop.CENSOR_TABLE}(session_id,token_key,last_seen_at,censor_at,reason,created_at) VALUES(?,?,?,?,?,?)",
            ("s", "abc", "2026-08-29T10:10:00+00:00", "2026-08-29T10:10:00+00:00", "manual_stop_censored", "2026-08-29T10:11:00+00:00"),
        )
        conn.commit()
        mapping = stop.censors_by_token(conn)
        rec = stop.censor_from_map(mapping, "abc", pd.Timestamp("2026-08-29T10:00:00Z"), pd.Timestamp("2026-08-29T11:00:00Z"))
        assert rec is not None
        assert rec["reason"] == "manual_stop_censored"
        assert "dead" not in rec["reason"]


def test_open_paper_positions_are_censored_without_reward(tmp_path):
    db = tmp_path / "live.sqlite"
    with sqlite3.connect(db) as conn:
        stop.migrate(conn)
        conn.execute("CREATE TABLE axiom_paper_positions_v20(position_id TEXT PRIMARY KEY,status TEXT,closed_at TEXT,close_reason TEXT,exit_kind TEXT,reward REAL,net_return_pct REAL)")
        conn.execute("INSERT INTO axiom_paper_positions_v20(position_id,status,reward,net_return_pct) VALUES('p','open',NULL,NULL)")
        conn.commit()
        result = stop._censor_paper_state(conn, pd.Timestamp("2026-08-29T10:10:00Z"), "manual_stop_censored")
        row = conn.execute("SELECT status,close_reason,exit_kind,reward,net_return_pct FROM axiom_paper_positions_v20 WHERE position_id='p'").fetchone()
        assert result["paper_positions_censored"] == 1
        assert row == ("censored", "manual_stop_censored", "collection_censored", None, None)
