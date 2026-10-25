from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import joblib
import pandas as pd
import pytest

from profit_taker import axiom_budget_benchmark as paper
from profit_taker import axiom_v24 as v24
from profit_taker import selective_paper_policy as selective
from profit_taker.db import migrate as migrate_raw


def forecast(probability=.8, **extra):
    # No minus-50 head: this is a supported shape of real V24 outputs.
    state = {f"p_first_peak_by_{h}m": probability for h in (5, 10, 15, 30, 60)}
    state.update(next_occurrence_q50=15.0, next_peak_multiple_q25=1.6,
                 next_peak_multiple_q50=1.8, p_death_by_60m=.01)
    state.update(extra)
    return state


def trial(index, prices=(100.0, 115.0, 115.0), state=None):
    at = pd.Timestamp("2026-09-01T00:00:00Z") + pd.Timedelta(minutes=index * 70)
    return selective.Trial(str(index), at, at + pd.Timedelta(seconds=20), 100.0,
        state or forecast(), tuple(at + pd.Timedelta(minutes=m) for m in range(1, len(prices) + 1)),
        tuple(prices))


def calibration(trials=None, **config):
    cfg = paper.BenchmarkConfig(selective_reference_tokens=20, **config)
    trials = trials if trials is not None else [trial(i) for i in range(45)]
    return selective.Calibration(trials, cfg, "model", trial(1000).at, "selective_validation")


def test_wins_require_later_token_evidence_and_positive_net_edge():
    good = calibration()
    setup, kind = good.assess(forecast(), 50.0)
    assert kind == selective.VERSION
    assert setup.qualifies
    assert setup.precision_evidence["validation_tokens"] == 25
    assert setup.precision_evidence["win_lower_bound"] >= .50
    assert setup.net_edge >= .01
    assert setup.precision_evidence["downside_head_available"] is False
    # A confident-looking forecast never substitutes for realized evidence.
    empty = calibration([]).assess(forecast(), 50.0)[0]
    assert empty.qualifies and empty.reason == "selective_forecast_warmup"
    thin = calibration([trial(i) for i in range(39)]).assess(forecast(), 50.0)[0]
    assert thin.qualifies and thin.reason == "selective_forecast_warmup"


def test_large_lucky_winners_cannot_mask_low_precision():
    samples = [trial(i) for i in range(20)]
    samples += [trial(i, (100.0, 300.0, 300.0) if i < 60 else (100.0, 80.0, 80.0)) for i in range(20, 120)]
    setup = calibration(samples).assess(forecast(), 50.0)[0]
    assert setup.net_edge > 0
    assert setup.precision_evidence["wins"] == 40
    assert setup.reason == "selective_early_outcomes_unprofitable"
    assert not setup.qualifies


def test_cost_positive_moderate_forecast_can_trade_during_warmup():
    empty = calibration([])
    state = forecast(.08, next_occurrence_q50=12,
                     next_peak_multiple_q25=1.10, next_peak_multiple_q50=1.20)
    setup = empty.assess(state, 30)[0]
    assert setup.qualifies and setup.reason == "selective_forecast_warmup"
    assert setup.precision_evidence["net_peak_q25"] >= empty.config.selective_min_net_edge
    assert not empty.assess({**state, "next_peak_multiple_q25": 1.05}, 30)[0].qualifies
    assert not empty.assess({**state, "next_occurrence_q50": 45}, 30)[0].qualifies
    assert not empty.assess({**state, "p_death_by_60m": .5}, 30)[0].qualifies


def test_five_bad_executable_outcomes_stop_warmup_entries():
    samples = [trial(i) for i in range(20)]
    samples += [trial(i, (100, 80, 80)) for i in range(20, 25)]
    setup = calibration(samples).assess(forecast(), 50)[0]
    assert setup.precision_evidence["validation_tokens"] == 5
    assert not setup.qualifies
    assert setup.reason == "selective_early_outcomes_unprofitable"


def test_fresh_wallet_warmup_uses_one_small_position(tmp_path, monkeypatch):
    source, wallet = tmp_path / "raw.sqlite", tmp_path / "wallet.sqlite"
    model = tmp_path / "model.joblib"
    migrate_raw(source)
    joblib.dump({"schema_version": v24.SCHEMA_VERSION}, model)
    paper.init_benchmark(str(wallet), paper.BenchmarkConfig())
    snapshot = pd.Timestamp("2026-09-01T00:00:00Z")
    board = pd.DataFrame([{"token_key": token, "market_cap_usd": 100.0, **forecast(.08)}
                          for token in ("A", "B")])
    monkeypatch.setattr(paper, "_read_current", lambda *_a, **_kw: (snapshot, board))
    result = paper._cycle_v24(str(source), str(wallet), "unused.csv", str(model), "missing.joblib",
                             paper.BenchmarkConfig())
    assert len(result["pending_entries"]) == 1
    assert result["pending_entries"][0]["reserved_cash_usd"] == pytest.approx(30.0)
    assert result["entry_evaluation"]["rejection_reasons"]["selective_warmup_slot_limit"] == 1


def test_replay_cannot_sell_at_hindsight_peak_or_same_decision_price():
    cfg = paper.BenchmarkConfig()
    spiked = trial(0, (100.0, 160.0, 80.0))
    net, reason = selective.replay(spiked, 50.0, cfg)
    assert reason == "selective_net_take_profit"
    assert net < 0  # Signal at 160, executable next price is 80.
    # The first observable price is the buy price, not a free profit.
    jumped = trial(0, (160.0, 160.0, 160.0))
    assert selective.replay(jumped, 50.0, cfg) == (None, "cancelled_at_fill")


def test_gaps_and_missing_sales_do_not_disappear_from_the_loss_denominator():
    cfg = paper.BenchmarkConfig()
    missing = trial(0, (100.0, 160.0))
    assert selective.replay(missing, 50.0, cfg) == (-1.0, "unresolved_exit")
    gap = trial(0)
    gap = replace(gap, times=(gap.times[0], gap.times[1], gap.times[2] + pd.Timedelta(minutes=10)))
    assert selective.replay(gap, 50.0, cfg) == (-1.0, "unresolved_gap")
    evidence = calibration([trial(i) if i < 20 else replace(trial(i), prices=(100.0,), times=(trial(i).times[0],))
                            for i in range(45)]).assess(forecast(), 50.0)[0]
    assert evidence.precision_evidence["validation_tokens"] == 25
    assert evidence.precision_evidence["unresolved"] == 25
    assert not evidence.qualifies


def test_expired_orders_are_unfilled_and_time_exit_uses_the_next_price():
    cfg = paper.BenchmarkConfig()
    late = trial(0)
    late = replace(late, times=tuple(t + pd.Timedelta(minutes=5) for t in late.times))
    assert selective.replay(late, 50.0, cfg) == (None, "unfilled")
    held = trial(0, (100.0,) * 63)
    net, reason = selective.replay(held, 50.0, cfg)
    assert reason == "selective_time_exit" and net < 0
    assert selective.exit_reason(-.20, .1, selective.exit_plan(cfg)) == "selective_net_stop_loss"


def test_forecast_support_and_actual_dollar_costs_are_required():
    cal = calibration()
    assert not cal.assess(forecast(.9), 50.0)[0].qualifies  # no extrapolation
    assert not cal.assess(forecast(), 2.0)[0].qualifies  # fixed SOL fees
    assert not cal.assess(forecast(p_death_by_60m=.6), 50.0)[0].qualifies
    assert not cal.assess(forecast(next_peak_multiple_q25=1.04), 50.0)[0].qualifies
    missing = forecast()
    del missing["next_peak_multiple_q25"]
    assert not cal.assess(missing, 50.0)[0].qualifies
    assert not cal.assess(forecast(p_first_peak_by_5m=.99), 50.0)[0].qualifies


def seed(source, wallet, cfg, model_hash):
    migrate_raw(source)
    paper.init_benchmark(str(wallet), cfg)
    records = [trial(i) for i in range(45)]
    with sqlite3.connect(wallet) as conn, sqlite3.connect(source) as raw:
        bid = conn.execute("SELECT benchmark_id FROM benchmark_account_v22").fetchone()[0]
        for record in records:
            state = {**record.state, "selective_policy_version": selective.VERSION,
                     "selective_available_at": record.available_at.isoformat()}
            conn.execute("""INSERT INTO benchmark_candidates_v22
                (benchmark_id,snapshot_at,token_key,market_cap_usd,entry_score,score_kind,chosen,state_json,forecast_model_hash)
                VALUES(?,?,?,?,0,?,0,?,?)""",
                (bid, record.at.isoformat(), record.token, 100, selective.VERSION, json.dumps(state), model_hash))
            raw.executemany("INSERT INTO axiom_observations(token_key,snapshot_at,market_cap_usd) VALUES(?,?,?)",
                [(record.token, t.isoformat(), p) for t, p in zip(record.times, record.prices)])
            conn.executemany("INSERT INTO benchmark_decision_clock_v24 VALUES(?,?,?)",
                [(bid, t.isoformat(), (t + pd.Timedelta(seconds=20)).isoformat()) for t in record.times])
    return bid, records


def test_evidence_loader_filters_versions_vintages_and_duplicate_token_minutes(tmp_path):
    source, wallet = tmp_path / "raw.sqlite", tmp_path / "wallet.sqlite"
    cfg = paper.BenchmarkConfig(selective_reference_tokens=20)
    bid, records = seed(source, wallet, cfg, "champion")
    asof = trial(46).at
    with sqlite3.connect(wallet) as conn:
        before = selective.build_calibration(conn, str(source), bid, records[0].at, "champion", cfg)
        assert not before.trials
        # Many minute copies of one token cannot inflate distinct validation.
        original = records[-1]
        for minute in range(1, 10):
            at = original.at + pd.Timedelta(minutes=minute)
            state = {**original.state, "selective_policy_version": selective.VERSION,
                     "selective_available_at": at.isoformat()}
            conn.execute("""INSERT INTO benchmark_candidates_v22
                (benchmark_id,snapshot_at,token_key,market_cap_usd,entry_score,score_kind,chosen,state_json,forecast_model_hash)
                VALUES(?,?,?,?,0,?,0,?,?)""",
                (bid, at.isoformat(), original.token, 100, selective.VERSION, json.dumps(state), "champion"))
        good = selective.build_calibration(conn, str(source), bid, asof, "champion", cfg)
        assert len(good.trials) == 45
        assert good.assess(forecast(), 50)[0].qualifies
        assert not selective.build_calibration(conn, str(source), bid, asof, "other-model", cfg).trials
        conn.execute("UPDATE benchmark_candidates_v22 SET state_json=?", (json.dumps(forecast()),))
        assert not selective.build_calibration(conn, str(source), bid, asof, "champion", cfg).trials


def test_shadow_exit_waits_for_measured_decision_availability():
    cfg = paper.BenchmarkConfig()
    sample = trial(0, (100, 160, 160, 80))
    # Exit signal at minute two finishes after minute three was captured.
    sample = replace(sample, decision_times=(sample.times[0], sample.times[2] + pd.Timedelta(seconds=10),
                                            sample.times[2], sample.times[3]))
    net, reason = selective.replay(sample, 50, cfg)
    assert reason == "selective_net_take_profit" and net < 0


def test_shadow_validation_does_not_use_boards_the_wallet_never_processed(tmp_path):
    source, wallet = tmp_path / "raw.sqlite", tmp_path / "wallet.sqlite"
    cfg = paper.BenchmarkConfig(selective_reference_tokens=20)
    bid, _ = seed(source, wallet, cfg, "champion")
    with sqlite3.connect(wallet) as conn:
        conn.execute("DELETE FROM benchmark_decision_clock_v24")
        cal = selective.build_calibration(conn, str(source), bid, trial(46).at, "champion", cfg)
    assert not cal.assess(forecast(), 50)[0].qualifies


def test_live_wallet_matches_replay_fees_and_later_sell_fill(tmp_path, monkeypatch):
    source, wallet = tmp_path / "raw.sqlite", tmp_path / "wallet.sqlite"
    model, predictions = tmp_path / "model.joblib", tmp_path / "pred.csv"
    joblib.dump({"schema_version": v24.SCHEMA_VERSION}, model)
    model_hash = paper._hash_file(model)
    cfg = paper.BenchmarkConfig(selective_reference_tokens=20)
    seed(source, wallet, cfg, model_hash)
    clock = {"at": trial(46).at, "mc": 100.0}
    monkeypatch.setattr(paper, "_read_current", lambda *_a, **_kw: (clock["at"], pd.DataFrame([
        {"token_key": "LIVE", "market_cap_usd": clock["mc"], "v24_model_hash": model_hash, **forecast()}])))
    # Learned/legacy scores must not bypass or contradict the selected policy.
    monkeypatch.setattr(paper, "_entry_score", lambda *_: pytest.fail("legacy entry score used"))
    monkeypatch.setattr(paper, "_hold_score", lambda *_: pytest.fail("legacy hold score used"))
    def cycle():
        return paper._cycle_v24(str(source), str(wallet), str(predictions), str(model),
                               str(tmp_path / "missing-policy"), cfg)
    first = cycle()
    assert len(first["pending_entries"]) == 1
    assert first["policy_version"] == selective.VERSION
    clock["at"] += pd.Timedelta(minutes=1)
    filled = cycle()
    assert len(filled["entries"]) == 1
    spend = filled["entries"][0]["cash_spent_usd"]
    clock.update(at=clock["at"] + pd.Timedelta(minutes=1), mc=115.0)
    assert cycle()["exits"] == []
    clock.update(at=clock["at"] + pd.Timedelta(minutes=1), mc=80.0)
    closed = cycle()["exits"][0]
    expected, _ = selective.replay(trial(46, (100.0, 115.0, 80.0)), spend, cfg)
    assert closed["execution_realized_return_pct"] == pytest.approx(expected)
    assert closed["price_available_at_exit"] and expected < 0
    status = paper.status(str(wallet))
    assert status["performance_by_entry_policy"][selective.VERSION]["observable_win_rate"] == 0.0


@pytest.mark.parametrize("delay,mc,reason", [(4, 100.0, "selective_entry_expired"),
    (1, 160.0, "selective_lower_upside_below_cost_buffer")])
def test_wallet_cancels_expired_or_chased_entries(tmp_path, monkeypatch, delay, mc, reason):
    source, wallet = tmp_path / "raw.sqlite", tmp_path / "wallet.sqlite"
    model = tmp_path / "model.joblib"
    joblib.dump({"schema_version": v24.SCHEMA_VERSION}, model)
    cfg = paper.BenchmarkConfig(selective_reference_tokens=20)
    seed(source, wallet, cfg, paper._hash_file(model))
    clock = {"at": trial(46).at, "mc": 100.0}
    monkeypatch.setattr(paper, "_read_current", lambda *_a, **_kw: (clock["at"], pd.DataFrame([
        {"token_key": "LIVE", "market_cap_usd": clock["mc"], **forecast()}])))
    def cycle():
        return paper._cycle_v24(str(source), str(wallet), "unused.csv", str(model), "missing.joblib", cfg)
    assert len(cycle()["pending_entries"]) == 1
    clock.update(at=clock["at"] + pd.Timedelta(minutes=delay), mc=mc)
    result = cycle()
    assert result["entries"] == []
    assert result["cancelled_pending"][0]["reason"] == reason


def test_existing_config_gets_selective_defaults_without_changing_saved_fees():
    cfg = paper._config_from_saved_json(json.dumps({"initial_cash_usd": 1000,
        "execution_cost_model": "axiom_cost_scenario", "sol_usd_reference": 103.0}))
    assert cfg.selective_entry_enabled
    assert cfg.sol_usd_reference == 103.0
    assert cfg.selective_min_win_lower_bound == .50
    assert cfg.selective_warmup_position_fraction == .03
    old = paper._config_from_saved_json(json.dumps({"execution_cost_model": "axiom_cost_scenario",
        "sol_usd_reference": 103.0, "selective_min_win_lower_bound": .75}))
    assert old.selective_min_win_lower_bound == .50
    assert old.sol_usd_reference == 103.0


def test_live_selective_decisions_record_real_availability_and_reject_incomplete_hashes(tmp_path, monkeypatch):
    source, wallet = tmp_path / "raw.sqlite", tmp_path / "wallet.sqlite"
    model = tmp_path / "model.joblib"
    joblib.dump({"schema_version": v24.SCHEMA_VERSION}, model)
    model_hash = paper._hash_file(model)
    cfg = paper.BenchmarkConfig(selective_reference_tokens=20)
    seed(source, wallet, cfg, model_hash)
    snapshot = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=2)
    board = pd.DataFrame([{"token_key": "LIVE", "market_cap_usd": 100.0,
                           "v24_model_hash": model_hash, **forecast()}])
    monkeypatch.setattr(paper, "_read_current", lambda *_a, **_kw: (snapshot, board.copy()))
    def cycle():
        return paper._cycle_v24(str(source), str(wallet), "unused.csv", str(model),
                                "missing.joblib", cfg, live_decisions=True)
    before = pd.Timestamp.now(tz="UTC")
    assert len(cycle()["pending_entries"]) == 1
    with sqlite3.connect(wallet) as conn:
        at, encoded = conn.execute("SELECT decision_at,entry_state_json FROM benchmark_pending_entries_v24 WHERE status='pending'").fetchone()
        assert pd.Timestamp(at) >= before
        assert json.loads(encoded)["selective_available_at"] == at
        observed, ready = conn.execute("SELECT snapshot_at,available_at FROM benchmark_decision_clock_v24 ORDER BY snapshot_at DESC LIMIT 1").fetchone()
        assert pd.Timestamp(observed) == snapshot and ready == at
    snapshot += pd.Timedelta(minutes=1)  # captured before order existed
    assert cycle()["entries"] == []
    snapshot = pd.Timestamp(at) + pd.Timedelta(seconds=1)
    assert len(cycle()["entries"]) == 1
    board.loc[1] = {"token_key": "BAD", "market_cap_usd": 100.0, "v24_model_hash": None, **forecast()}
    snapshot += pd.Timedelta(minutes=1)
    with pytest.raises(RuntimeError, match="do not match"):
        cycle()


def test_live_cli_cycle_never_falls_back_to_legacy_on_invalid_model(tmp_path):
    with pytest.raises(RuntimeError, match="legacy fallback is disabled"):
        paper.cycle("missing.sqlite", "missing-wallet.sqlite", "missing.csv", str(tmp_path / "absent.joblib"),
                    "missing-policy", paper.BenchmarkConfig(), live_decisions=True)


@pytest.mark.parametrize("changes", [{"selective_min_win_lower_bound": float("nan")},
    {"selective_stop_loss_net": 1.0}, {"selective_min_validation_tokens": 1}])
def test_invalid_selective_configuration_rejected(changes):
    with pytest.raises(ValueError):
        paper._validate_munger_config(paper.BenchmarkConfig(**changes))
