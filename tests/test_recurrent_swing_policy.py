from __future__ import annotations

import sqlite3

import joblib
import pandas as pd
import pytest

from profit_taker import axiom_budget_benchmark as benchmark
from profit_taker import axiom_v24 as v24
from profit_taker import recurrent_swing_policy as swing
from profit_taker.db import migrate as migrate_raw


def short_setup(**overrides):
    state = {
        "p_first_peak_by_5m": 0.62,
        "p_first_peak_by_10m": 0.72,
        "p_first_peak_by_15m": 0.78,
        "p_first_peak_by_30m": 0.88,
        "p_first_peak_by_60m": 0.91,
        "next_occurrence_q25": 10.0,
        "next_occurrence_q50": 18.0,
        "next_occurrence_q75": 28.0,
        "next_occurrence_q90": 36.0,
        "next_confirmation_lag_q50": 5.0,
        "next_peak_multiple_q25": 1.25,
        "next_peak_multiple_q50": 1.43,
        "next_peak_multiple_q75": 1.70,
        "p_death_by_60m": 0.03,
        "p_hit_minus50_by_60m": 0.02,
        "p_later_higher_10pct_by_240m": 0.69,
        "second_occurrence_gap_q50": 95.0,
        "second_peak_relative_q50": 0.50,
    }
    state.update(overrides)
    return state


def test_short_horizon_heads_control_buy_now_instead_of_broad_lifecycle_probability():
    cfg = benchmark.BenchmarkConfig()
    good, good_kind = swing.short_term_setup(short_setup(), cfg, 0.40, "learned")
    distant, _ = swing.short_term_setup(
        short_setup(
            p_first_peak_by_5m=0.10,
            p_first_peak_by_10m=0.15,
            p_first_peak_by_15m=0.20,
            p_first_peak_by_30m=0.30,
            p_first_peak_by_60m=0.40,
            next_occurrence_q50=95.0,
            p_first_peak_by_720m=0.99,
        ),
        cfg,
        0.95,
        "learned",
    )

    assert good.available and good.qualifies
    assert good_kind == "recurrent_swing_learned"
    assert good.score > 0
    assert not distant.qualifies
    assert distant.reason == "short_peak_probability_too_low"

    uncertain, _ = swing.short_term_setup(
        short_setup(next_occurrence_q10=1.0, next_occurrence_q90=150.0),
        cfg,
        0.40,
        "learned",
    )
    assert uncertain.score < good.score
    assert uncertain.occurrence_spread_minutes == pytest.approx(149.0)


def test_peak_is_decision_boundary_not_an_unconditional_sale():
    cfg = benchmark.BenchmarkConfig()
    common = short_setup(
        p_first_peak_by_5m=0.80,
        next_occurrence_q50=5.0,
        next_peak_multiple_q25=1.01,
        next_peak_multiple_q50=1.02,
        pred_post_next_peak_retracement_pct_q50=0.30,
    )
    weak_continuation = dict(
        common,
        p_later_higher_10pct_by_240m=0.50,
        second_occurrence_gap_q50=95.0,
        second_peak_relative_q50=0.40,
    )
    strong_continuation = dict(
        common,
        p_later_higher_10pct_by_240m=0.80,
        second_occurrence_gap_q50=20.0,
        second_peak_relative_q50=0.40,
    )

    sell = swing.peak_boundary_decision(weak_continuation, 0.43, cfg)
    hold = swing.peak_boundary_decision(strong_continuation, 0.43, cfg)

    assert sell["at_peak_boundary"] and sell["sell"]
    assert sell["sell_reentry_value"] > sell["hold_through_value"]
    assert hold["at_peak_boundary"] and not hold["sell"]
    assert hold["reason"] == "hold_strong_near_second_peak"


def test_reentry_requires_wait_retracement_and_fresh_short_setup(tmp_path):
    db = tmp_path / "wallet.sqlite"
    benchmark.init_benchmark(str(db), benchmark.BenchmarkConfig())
    exit_at = pd.Timestamp("2026-09-22T12:00:00Z")
    cfg = benchmark.BenchmarkConfig()
    setup, _ = swing.short_term_setup(short_setup(), cfg, 0.5, "bootstrap")
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        benchmark.migrate(conn)
        swing.migrate(conn)
        bid = str(conn.execute(
            "SELECT benchmark_id FROM benchmark_account_v22 WHERE status='active'"
        ).fetchone()[0])
        conn.execute(
            """INSERT INTO benchmark_positions_v22
            (position_id,benchmark_id,token_key,opened_at,entry_mc,entry_notional_usd,
             entry_fee_usd,entry_cash_spent_usd,exposure_units,entry_state_json,status,
             last_seen_at,last_mc,last_mark_value_usd,mfe_pct,mae_pct,closed_at,exit_mc)
            VALUES('p',?,'T',?,100,49.75,.25,50,.4975,'{}','closed',?,140,69.3,.4,0,?,140)""",
            (bid, (exit_at - pd.Timedelta(minutes=15)).isoformat(), exit_at.isoformat(), exit_at.isoformat()),
        )
        pos = conn.execute("SELECT * FROM benchmark_positions_v22 WHERE position_id='p'").fetchone()
        swing.create_watch(conn, bid, pos, exit_at, 140.0, short_setup(), cfg)

        early = swing.reentry_eligibility(
            conn, bid, "T", exit_at + pd.Timedelta(minutes=2), 120.0, setup, cfg
        )
        no_reset = swing.reentry_eligibility(
            conn, bid, "T", exit_at + pd.Timedelta(minutes=4), 138.0, setup, cfg
        )
        allowed = swing.reentry_eligibility(
            conn, bid, "T", exit_at + pd.Timedelta(minutes=4), 120.0, setup, cfg
        )

    assert early[0] is False and early[1] == "swing_reentry_minimum_wait"
    assert no_reset[0] is False and no_reset[1] == "swing_reentry_reset_not_reached"
    assert allowed[0] is True and allowed[1] == "swing_reentry_after_reset"
    assert allowed[2]


def test_v24_wallet_can_sell_first_swing_and_reenter_same_token_lifecycle(
    tmp_path, monkeypatch
):
    source = tmp_path / "raw.sqlite"
    wallet = tmp_path / "wallet.sqlite"
    model = tmp_path / "champion.joblib"
    predictions = tmp_path / "predictions.csv"
    migrate_raw(source)
    joblib.dump({"schema_version": v24.SCHEMA_VERSION}, model)
    predictions.write_text("token_key,v24_model_hash\n", encoding="utf-8")
    benchmark.init_benchmark(str(wallet), benchmark.BenchmarkConfig())

    clock = {"at": pd.Timestamp("2026-09-22T12:00:00Z"), "mc": 100.0, "state": short_setup()}

    def current():
        return pd.DataFrame([{
            "token_key": "T",
            "market_cap_usd": clock["mc"],
            "v24_model_hash": "frozen",
            **clock["state"],
        }])

    monkeypatch.setattr(benchmark, "_read_current", lambda *_: (clock["at"], current()))
    no_policy = str(tmp_path / "no-policy.joblib")

    first = benchmark.cycle(
        str(source), str(wallet), str(predictions), str(model), no_policy,
        benchmark.BenchmarkConfig(),
    )
    assert len(first["pending_entries"]) == 1

    peak_state = short_setup(
        p_first_peak_by_5m=0.80,
        next_occurrence_q50=5.0,
        next_peak_multiple_q25=1.01,
        next_peak_multiple_q50=1.02,
        pred_post_next_peak_retracement_pct_q50=0.30,
        p_later_higher_10pct_by_240m=0.50,
        second_occurrence_gap_q50=95.0,
        second_peak_relative_q50=0.40,
    )
    clock.update(at=clock["at"] + pd.Timedelta(minutes=1), mc=100.0, state=peak_state)
    filled = benchmark.cycle(str(source), str(wallet), str(predictions), str(model), no_policy, benchmark.BenchmarkConfig())
    assert len(filled["entries"]) == 1
    assert filled["entries"][0]["swing_sequence"] == 1

    clock.update(at=clock["at"] + pd.Timedelta(minutes=2), mc=145.0)
    decision = benchmark.cycle(str(source), str(wallet), str(predictions), str(model), no_policy, benchmark.BenchmarkConfig())
    assert decision["exits"] == []

    clock.update(at=clock["at"] + pd.Timedelta(minutes=1), mc=140.0)
    sold = benchmark.cycle(str(source), str(wallet), str(predictions), str(model), no_policy, benchmark.BenchmarkConfig())
    assert sold["exits"][0]["reason"] == "recurrent_swing_peak_boundary"

    clock.update(at=clock["at"] + pd.Timedelta(minutes=3), mc=118.0, state=short_setup())
    second = benchmark.cycle(str(source), str(wallet), str(predictions), str(model), no_policy, benchmark.BenchmarkConfig())
    assert len(second["pending_entries"]) == 1
    assert second["pending_entries"][0]["reentry_watch_id"]

    clock.update(at=clock["at"] + pd.Timedelta(minutes=1), mc=120.0)
    refilled = benchmark.cycle(str(source), str(wallet), str(predictions), str(model), no_policy, benchmark.BenchmarkConfig())
    assert len(refilled["entries"]) == 1
    assert refilled["entries"][0]["swing_sequence"] == 2

    with sqlite3.connect(wallet) as conn:
        watches = conn.execute(
            "SELECT status,reentry_position_id FROM benchmark_swing_watch_v24"
        ).fetchall()
        sequences = conn.execute(
            "SELECT swing_sequence,status FROM benchmark_positions_v22 ORDER BY opened_at"
        ).fetchall()
    assert watches[0][0] == "reentered" and watches[0][1]
    assert sequences == [(1, "closed"), (2, "open")]
