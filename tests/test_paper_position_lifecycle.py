from __future__ import annotations

import sqlite3

import joblib
import pandas as pd

from profit_taker import axiom_budget_benchmark as paper
from profit_taker import axiom_v24 as v24
from profit_taker.absence_utils import collector_absence
from profit_taker.db import migrate as migrate_raw


def capture(conn, at, tokens, *, valid=True):
    cursor = conn.execute(
        "INSERT INTO capture_cycles(captured_at,completed,clipboard_valid,rows_detected) VALUES(?,?,?,?)",
        (at.isoformat(), 1, int(valid), len(tokens)),
    )
    for token in tokens:
        conn.execute(
            "INSERT INTO axiom_observations(cycle_id,token_key,snapshot_at,market_cap_usd) VALUES(?,?,?,?)",
            (cursor.lastrowid, token, at.isoformat(), 100.0),
        )


def evidence(conn, start, end):
    return collector_absence(conn, "HELD", start, end, max_gap_minutes=5,
                             minimum_captures=10, close_minutes=50)


def test_raw_capture_absence_resets_on_reappearance_and_downtime(tmp_path):
    db = tmp_path / "raw.sqlite"
    migrate_raw(db)
    start = pd.Timestamp("2026-10-20T12:00:00Z")
    with sqlite3.connect(db) as conn:
        capture(conn, start, ["HELD"])
        for minute in range(1, 30):
            capture(conn, start + pd.Timedelta(minutes=minute), ["OTHER"])
        capture(conn, start + pd.Timedelta(minutes=30), ["HELD"])
        for minute in range(31, 80):
            capture(conn, start + pd.Timedelta(minutes=minute), ["OTHER"])
        assert not evidence(conn, start, start + pd.Timedelta(minutes=79))["terminal"]
        capture(conn, start + pd.Timedelta(minutes=80), ["OTHER"])
        assert evidence(conn, start, start + pd.Timedelta(minutes=80))["terminal"]
        capture(conn, start + pd.Timedelta(minutes=200), ["OTHER"])
        assert not evidence(conn, start, start + pd.Timedelta(minutes=200))["terminal"]


def test_current_board_keeps_visible_unpriced_identity(tmp_path):
    raw, predictions = tmp_path / "raw.sqlite", tmp_path / "pred.csv"
    migrate_raw(raw)
    now = pd.Timestamp("2026-10-20T12:00:00Z")
    with sqlite3.connect(raw) as conn:
        conn.execute(
            "INSERT INTO axiom_observations(token_key,snapshot_at,market_cap_usd) VALUES(?,?,NULL)",
            ("HELD", now.isoformat()),
        )
    pd.DataFrame([{"token_key": "HELD", "snapshot_at": now.isoformat(),
                   "p_first_peak_by_720m": .8}]).to_csv(predictions, index=False)
    snapshot, current = paper._read_current(str(raw), str(predictions))
    assert snapshot == now
    assert current["token_key"].tolist() == ["HELD"]
    assert pd.isna(current.iloc[0]["market_cap_usd"])


def test_status_counts_only_priced_profitable_sales_as_observable_wins(tmp_path):
    wallet = tmp_path / "wallet.sqlite"
    paper.init_benchmark(str(wallet), paper.BenchmarkConfig(recurrent_swing_enabled=False))
    at = "2026-10-20T12:00:00+00:00"
    with sqlite3.connect(wallet) as conn:
        bid = conn.execute("SELECT benchmark_id FROM benchmark_account_v22").fetchone()[0]
        for pid, available, kind, pnl in (
            ("SOLD_WIN", 1, "policy_exit", 12.0),
            ("SOLD_LOSS", 1, "policy_exit", -4.0),
            ("PROXY", 0, "disappearance_terminal", 9.0),
        ):
            conn.execute(
                """INSERT INTO benchmark_positions_v22
                (position_id,benchmark_id,token_key,opened_at,entry_mc,entry_notional_usd,
                 entry_fee_usd,entry_cash_spent_usd,exposure_units,entry_state_json,status,
                 last_seen_at,last_mc,last_mark_value_usd,mfe_pct,mae_pct,closed_at,
                 realized_pnl_usd,realized_return_pct,execution_realized_pnl_usd,
                 execution_realized_return_pct,price_available_at_exit,exit_kind)
                 VALUES(?,?,?,?,?,?,?,?,?,?,'closed',?,?,?,?,?,?,?,?,?,?,?,?)""",
                (pid, bid, pid, at, 100, 50, 0, 50, .5, '{}', at, 100, 50,
                 0, 0, at, pnl, pnl / 50, pnl, pnl / 50, available, kind),
            )
    report = paper.status(str(wallet))
    assert report["observable_exit_wins"]["profitable"] == 1
    assert report["observable_exit_wins"]["closed"] == 2
    assert report["observable_exit_wins"]["win_rate"] == .5
    assert report["unavailable_exit_estimates"]["closed"] == 1


def test_unpriced_board_and_collector_backed_terminal_close(tmp_path, monkeypatch):
    raw, wallet = tmp_path / "raw.sqlite", tmp_path / "wallet.sqlite"
    model, predictions = tmp_path / "model.joblib", tmp_path / "pred.csv"
    migrate_raw(raw)
    joblib.dump({"schema_version": v24.SCHEMA_VERSION}, model)
    predictions.write_text("token_key,v24_model_hash\n")
    paper.init_benchmark(str(wallet), paper.BenchmarkConfig(recurrent_swing_enabled=False))
    start = pd.Timestamp("2026-10-20T12:00:00Z")
    with sqlite3.connect(wallet) as conn:
        bid = conn.execute("SELECT benchmark_id FROM benchmark_account_v22").fetchone()[0]
        conn.execute(
            """INSERT INTO benchmark_positions_v22
            (position_id,benchmark_id,token_key,opened_at,entry_mc,entry_notional_usd,
             entry_fee_usd,entry_cash_spent_usd,exposure_units,entry_state_json,status,
             last_seen_at,last_mc,last_mark_value_usd,mfe_pct,mae_pct,pending_exit_at,pending_exit_reason)
            VALUES(?,?,?,?,?,?,?,?,?,?,'open',?,?,?,?,?,?,?)""",
            ("POSITION", bid, "HELD", start.isoformat(), 100, 50, 0, 50, 0.5,
             "{}", start.isoformat(), 120, 60, .2, 0,
             start.isoformat(), "policy_exit_next_observable"),
        )
        conn.execute("UPDATE benchmark_account_v22 SET cash_usd=950,execution_cash_usd=950")
    moment = start + pd.Timedelta(minutes=1)
    board = pd.DataFrame([{"token_key": "HELD", "market_cap_usd": float("nan"), "age_minutes": 20}])
    monkeypatch.setattr(paper, "_read_current", lambda *_args, **_kwargs: (moment, board.copy()))
    result = paper._cycle_v24(str(raw), str(wallet), str(predictions), str(model),
                              str(tmp_path / "absent-policy.joblib"), paper.BenchmarkConfig())
    assert result["exits"] == []
    assert result["position_lifecycle"][0]["lifecycle_state"] == "OBSERVED_UNPRICED"
    assert paper.status(str(wallet))["position_lifecycle"][0]["board_present"] is True
    with sqlite3.connect(raw) as conn:
        for minute in range(2, 53):
            capture(conn, start + pd.Timedelta(minutes=minute), ["OTHER"])
    moment = start + pd.Timedelta(minutes=52)
    board = pd.DataFrame([{"token_key": "OTHER", "market_cap_usd": 100.0, "age_minutes": 1}])
    result = paper._cycle_v24(str(raw), str(wallet), str(predictions), str(model),
                              str(tmp_path / "absent-policy.joblib"), paper.BenchmarkConfig())
    assert len(result["exits"]) == 1
    assert result["exits"][0]["price_available_at_exit"] is False
    assert result["exits"][0]["reason"] == "dead_after_valid_capture_absence"
    assert result["exits"][0]["execution_realized_pnl_usd"] <= 0
    assert result["open_positions"] == 0
    assert result["position_lifecycle"][0]["verified_missing_minutes"] >= 50
    assert paper.status(str(wallet))["open_positions"] == 0
    win_report = paper.status(str(wallet))
    assert win_report["observable_exit_wins"]["closed"] == 0
    assert win_report["observable_exit_wins"]["win_rate"] is None
    assert win_report["unavailable_exit_estimates"]["closed"] == 1
    assert paper._cycle_v24(str(raw), str(wallet), str(predictions), str(model),
                            str(tmp_path / "absent-policy.joblib"), paper.BenchmarkConfig())["processed"] is False
    with sqlite3.connect(wallet) as conn:
        assert conn.execute("SELECT COUNT(*) FROM benchmark_positions_v22 WHERE status='closed'").fetchone()[0] == 1
