from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from profit_taker import axiom_budget_benchmark as benchmark
from profit_taker import axiom_clipboard as clipboard
from profit_taker import axiom_migrated_runner as runner
from profit_taker import axiom_peak_structure as peak
from profit_taker import axiom_self_teach as selfteach
from profit_taker import axiom_v24 as v24
from profit_taker.absence_utils import verified_absence_minutes
from profit_taker.db import migrate


def test_v24_defaults_and_schema():
    assert v24.SCHEMA_VERSION.startswith("v24_")
    assert v24.MODEL_ROOT_DEFAULT == "models/axiom_v24"
    assert v24.POLICY_ROOT_DEFAULT == "models/axiom_policy_v24"
    assert v24.PREDICTIONS_DEFAULT == "data/axiom_predictions_v24.csv"


def test_peak_confirmation_is_explicit():
    fields = peak.SwingPeak.__dataclass_fields__
    assert "peak_at" in fields
    assert "confirmed_at" in fields
    assert "confirmation_price" in fields


def test_monotonic_projection_preserves_missing_heads():
    raw = np.array([[0.60, np.nan, 0.30], [0.70, 0.55, 0.40]], dtype=float)
    projected = v24.monotonic_probability_projection(raw)
    assert np.isnan(projected[0, 1])
    assert projected.shape == raw.shape


def test_fresh_capture_schema_has_no_v18_feature_or_provider_tables(tmp_path):
    db = tmp_path / "fresh.sqlite"
    migrate(db)
    with sqlite3.connect(db) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "capture_cycles" in tables
    assert "axiom_observations" in tables
    assert "axiom_features_v18" not in tables
    assert "provider_usage" not in tables


def test_observation_source_is_pinned_even_after_v24_liquidity_migration(tmp_path):
    db = tmp_path / "source.sqlite"
    migrate(db)
    with sqlite3.connect(db) as con:
        v24.migrate(con)
        con.execute(
            """INSERT INTO axiom_observations
            (cycle_id,token_key,snapshot_at,market_cap_usd,field_confidence_json,raw_ocr_json,source_json)
            VALUES(NULL,'raw-token','2026-08-28T12:00:00+00:00',12345,'{}','{}','{}')"""
        )
        con.execute(
            f"""INSERT INTO {v24.LIQUIDITY_TABLE}
            (observation_id,token_key,observed_at,trade_size_usd,market_cap_usd,extra_json)
            VALUES('liq-1','liquidity-only','2026-08-28T12:00:00+00:00',100,99999,'{{}}')"""
        )
        con.commit()
        source = peak.discover_observation_source(con)
        obs, loaded_source = peak.load_observations(con)

    assert source["table"] == "axiom_observations"
    assert loaded_source["table"] == "axiom_observations"
    assert obs.token_key.tolist() == ["raw-token"]


def test_known_full_mint_collision_is_rejected_before_training(tmp_path):
    db = tmp_path / "collision.sqlite"
    migrate(db)
    with sqlite3.connect(db) as con:
        rows = [
            ("same-key", "Mint111111111111111111111111111111111111", "Abc...pump", "2026-08-28T12:00:00+00:00"),
            ("same-key", "Mint222222222222222222222222222222222222", "Abc...pump", "2026-08-28T12:01:00+00:00"),
        ]
        for key, mint, hint, ts in rows:
            con.execute(
                """INSERT INTO axiom_observations
                (cycle_id,token_key,token_address,short_address_hint,snapshot_at,market_cap_usd,field_confidence_json,raw_ocr_json,source_json)
                VALUES(NULL,?,?,?,?,10000,'{}','{}','{}')""",
                (key, mint, hint, ts),
            )
        con.commit()
        with pytest.raises(RuntimeError, match="identity collision"):
            peak.load_observations(con)


def test_verified_absence_excludes_collector_outage():
    last_seen = pd.Timestamp("2026-08-28T14:00:00Z")
    run_start = pd.Timestamp("2026-08-28T15:00:00Z")
    run_end = pd.Timestamp("2026-08-28T15:10:00Z")

    assert (run_end - last_seen).total_seconds() / 60.0 == 70.0
    assert verified_absence_minutes(run_start, run_end) == 10.0


def test_paper_and_benchmark_use_contiguous_run_start_for_absence():
    paper_source = inspect.getsource(selfteach._paper_cycle_v24)
    benchmark_source = inspect.getsource(benchmark._cycle_v24)

    assert "verified_absence_minutes(run_start, run_end)" in paper_source
    assert "verified_absence_minutes(run_start, run_end)" in benchmark_source
    assert "run_end - last_seen" not in paper_source
    assert "run_end - last_seen" not in benchmark_source


def test_full_mint_is_used_only_when_it_matches_short_address_and_preserves_case():
    mint = "AbC123456789ABCDEFGHJKLMNPQRSTUVWXyzpump"
    lines = [f"https://axiom.trade/token/{mint}"]
    assert clipboard._matching_full_mint(lines, "AbC...pump") == mint
    assert clipboard._matching_full_mint(lines, "ZZZ...pump") is None
    card = clipboard.ClipboardCard(short_address_hint="AbC...pump", token_address=mint)
    row = clipboard.clipboard_card_to_row(card, 0)
    assert row["token_key"] == mint


def test_runner_passes_actual_rows_detected(tmp_path, monkeypatch):
    selection = tmp_path / "capture.txt"
    selection.write_text("synthetic", encoding="utf-8")
    rows = [
        {"token_key": "a", "market_cap_usd": 1.0},
        {"token_key": "b", "market_cap_usd": 2.0},
        {"token_key": "c", "market_cap_usd": 3.0},
    ]
    captured = {}

    monkeypatch.setattr(runner, "clipboard_looks_like_axiom", lambda text: True)
    monkeypatch.setattr(runner, "rows_from_clipboard", lambda text: list(rows))
    monkeypatch.setattr(runner, "clipboard_diagnostics", lambda text, parsed: {"observations": len(parsed)})
    monkeypatch.setattr(runner, "diagnostics_json", lambda diag: "{}")

    def fake_process(db, snapshot, source, parsed, valid, output_dir, *, screenshot_rows_detected=None):
        captured["rows_detected"] = screenshot_rows_detected
        return {"rows_stored": len(parsed)}

    monkeypatch.setattr(runner, "process_rows", fake_process)
    args = SimpleNamespace(
        config=str(tmp_path / "missing.json"),
        clipboard_file=str(selection),
        output_dir=str(tmp_path / "artifacts"),
        db=str(tmp_path / "live.sqlite"),
    )
    result = runner.run_once(args, 1)

    assert captured["rows_detected"] == len(rows)
    assert result["rows_stored"] == len(rows)


def test_artifact_pruning_is_best_effort(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    out.mkdir()
    for i in range(12):
        stem = f"Axiom-Clipboard-20260828T12{i:02d}00Z"
        (out / f"{stem}.selection.txt").write_text("x", encoding="utf-8")

    original_unlink = Path.unlink
    blocked = sorted(out.glob("*.selection.txt"))[0]

    def flaky_unlink(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("locked for regression test")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    result = runner._prune_cycle_artifacts(out, keep_cycles=10)

    assert result["cycles_retained"] == 10
    assert result["errors"]
    assert any("PermissionError" in err for err in result["errors"])


def test_benchmark_default_is_canonical_v24_path():
    assert benchmark.DEFAULT_BENCHMARK_DB == "data/axiom_v24_1000_benchmark.sqlite"
