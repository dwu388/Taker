from __future__ import annotations

import math
import sqlite3

import joblib
import pandas as pd
import pytest

from profit_taker import axiom_budget_benchmark as benchmark
from profit_taker import axiom_v24 as v24
from profit_taker.db import migrate as migrate_raw


def test_munger_defaults_match_the_1000_account_plan():
    cfg = benchmark.BenchmarkConfig()
    benchmark._validate_munger_config(cfg)

    assert cfg.initial_cash_usd == 1000.0
    assert cfg.position_fraction == 0.05
    assert cfg.ordinary_position_fraction * cfg.initial_cash_usd == 50.0
    assert cfg.strong_position_fraction * cfg.initial_cash_usd == 75.0
    assert cfg.exceptional_position_fraction * cfg.initial_cash_usd == 100.0
    assert cfg.exceptional_position_fraction == 0.10
    assert cfg.max_total_exposure_fraction == 0.30
    assert cfg.max_correlated_exposure_fraction == 0.15
    assert cfg.min_cash_reserve_fraction == 0.70


def test_conviction_tiers_use_only_prior_same_kind_scores():
    cfg = benchmark.BenchmarkConfig(conviction_calibration_min_scores=4)
    snapshot = pd.Timestamp("2026-09-20T12:00:00Z")
    with sqlite3.connect(":memory:") as conn:
        benchmark.migrate(conn)
        rows = [0.10, 0.20, 0.30, 0.40]
        for i, score in enumerate(rows):
            conn.execute(
                """INSERT INTO benchmark_candidates_v22
                (benchmark_id,snapshot_at,token_key,market_cap_usd,entry_score,score_kind,chosen,
                 state_json,forecast_model_hash,policy_model_hash)
                VALUES('b',?,?,?,?, 'learned',0,'{}',NULL,NULL)""",
                (f"2026-09-20T11:0{i}:00+00:00", f"L{i}", 100.0, score),
            )
        # Different score scales must not contaminate learned-policy quantiles.
        for i in range(10):
            conn.execute(
                """INSERT INTO benchmark_candidates_v22
                (benchmark_id,snapshot_at,token_key,market_cap_usd,entry_score,score_kind,chosen,
                 state_json,forecast_model_hash,policy_model_hash)
                VALUES('b',?,?,?,?, 'bootstrap',0,'{}',NULL,NULL)""",
                (f"2026-09-20T10:{i:02d}:00+00:00", f"B{i}", 100.0, 100.0 + i),
            )
        conn.commit()
        thresholds = benchmark._conviction_thresholds(conn, "b", "learned", snapshot, cfg)

    assert thresholds == pytest.approx((0.325, 0.385))
    assert benchmark._conviction_tier(0.30, thresholds) == "ordinary"
    assert benchmark._conviction_tier(0.35, thresholds) == "strong"
    assert benchmark._conviction_tier(0.40, thresholds) == "exceptional"


def test_flat_score_history_does_not_promote_every_trade():
    assert benchmark._conviction_tier(0.25, (0.25, 0.25)) == "ordinary"


def test_allocation_enforces_total_bucket_and_cash_reserve_caps():
    cfg = benchmark.BenchmarkConfig()

    assert benchmark._allocation_amount(
        target_cash=100.0,
        available_cash=1000.0,
        current_cash=1000.0,
        execution_equity=1000.0,
        committed_total=225.0,
        committed_bucket=0.0,
        config=cfg,
    ) == pytest.approx(75.0)
    assert benchmark._allocation_amount(
        target_cash=100.0,
        available_cash=1000.0,
        current_cash=1000.0,
        execution_equity=1000.0,
        committed_total=100.0,
        committed_bucket=100.0,
        config=cfg,
    ) == pytest.approx(50.0)
    assert benchmark._allocation_amount(
        target_cash=100.0,
        available_cash=740.0,
        current_cash=740.0,
        execution_equity=1000.0,
        committed_total=0.0,
        committed_bucket=0.0,
        config=cfg,
    ) == pytest.approx(40.0)


def test_recent_return_correlation_creates_shared_risk_bucket(tmp_path):
    source = tmp_path / "raw.sqlite"
    migrate_raw(source)
    returns_a = [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, -0.01, 0.02, 0.01, -0.03, 0.02, 0.01]
    returns_c = [0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.04, -0.03, 0.02, 0.01, -0.01, 0.03]
    prices = {"A": [100.0], "B": [200.0], "C": [300.0]}
    for ra, rc in zip(returns_a, returns_c):
        prices["A"].append(prices["A"][-1] * math.exp(ra))
        prices["B"].append(prices["B"][-1] * math.exp(2.0 * ra))
        prices["C"].append(prices["C"][-1] * math.exp(rc))
    times = pd.date_range("2026-09-20T10:00:00Z", periods=len(prices["A"]), freq="min")
    with sqlite3.connect(source) as conn:
        for token, values in prices.items():
            for timestamp, value in zip(times, values):
                conn.execute(
                    """INSERT INTO axiom_observations
                    (cycle_id,token_key,snapshot_at,market_cap_usd,field_confidence_json,raw_ocr_json,source_json)
                    VALUES(NULL,?,?,?,'{}','{}','{}')""",
                    (token, timestamp.isoformat(), value),
                )
        conn.commit()

    buckets = benchmark._correlation_buckets(
        str(source), {"A", "B", "C"}, times[-1], benchmark.BenchmarkConfig()
    )
    assert buckets["A"] == buckets["B"]
    assert buckets["C"] != buckets["A"]


def test_v24_cycle_reserves_five_percent_without_retraining(tmp_path, monkeypatch):
    source = tmp_path / "raw.sqlite"
    benchmark_db = tmp_path / "benchmark.sqlite"
    model = tmp_path / "champion.joblib"
    predictions = tmp_path / "predictions.csv"
    migrate_raw(source)
    joblib.dump({"schema_version": v24.SCHEMA_VERSION}, model)
    predictions.write_text("token_key,v24_model_hash\n", encoding="utf-8")
    benchmark.init_benchmark(str(benchmark_db), benchmark.BenchmarkConfig())

    snapshot = pd.Timestamp("2026-09-20T12:00:00Z")

    def current_frame():
        return pd.DataFrame({
            "token_key": [f"T{i}" for i in range(5)],
            "market_cap_usd": [100.0] * 5,
            "p_first_peak_by_720m": [0.9] * 5,
            "pred_next_substantial_peak_multiple_q50": [2.0] * 5,
            "pred_time_to_next_substantial_peak_minutes_q50": [10.0] * 5,
            "p_death_by_720m": [0.0] * 5,
            "p_hit_minus50_by_720m": [0.0] * 5,
            "v24_model_hash": ["frozen"] * 5,
        })

    monkeypatch.setattr(benchmark, "_read_current", lambda *_args: (snapshot, current_frame()))
    result = benchmark.cycle(
        str(source), str(benchmark_db), str(predictions), str(model),
        str(tmp_path / "no-policy.joblib"), benchmark.BenchmarkConfig(),
    )

    assert result["training_feedback"] == "disabled"
    assert result["policy_version"] == "bootstrap"
    assert len(result["pending_entries"]) == 5
    assert sum(row["reserved_cash_usd"] for row in result["pending_entries"]) == pytest.approx(250.0)
    assert {row["conviction_tier"] for row in result["pending_entries"]} == {"ordinary"}
    assert {row["reserved_cash_usd"] for row in result["pending_entries"]} == {50.0}

    snapshot = snapshot + pd.Timedelta(minutes=1)
    filled = benchmark.cycle(
        str(source), str(benchmark_db), str(predictions), str(model),
        str(tmp_path / "no-policy.joblib"), benchmark.BenchmarkConfig(),
    )
    assert len(filled["entries"]) == 5
    assert sum(row["cash_spent_usd"] for row in filled["entries"]) == pytest.approx(250.0)
    assert filled["execution_cash_usd"] == pytest.approx(750.0)

    with sqlite3.connect(benchmark_db) as conn:
        tiers = conn.execute(
            "SELECT conviction_tier,target_position_fraction FROM benchmark_positions_v22"
        ).fetchall()
    assert tiers == [("ordinary", 0.05)] * 5
