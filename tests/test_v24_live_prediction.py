from __future__ import annotations

import inspect
import sqlite3
import threading
import time

import joblib
import pandas as pd
import pytest

from profit_taker import axiom_budget_benchmark as benchmark
from profit_taker import axiom_peak_structure_impl as peak_impl
from profit_taker import axiom_v24_impl as impl
from profit_taker.db import migrate as migrate_raw


def _insert_observation(conn: sqlite3.Connection, token: str, timestamp: str, market_cap: float) -> None:
    conn.execute(
        """INSERT INTO axiom_observations
        (cycle_id,token_key,snapshot_at,market_cap_usd,volume_usd,
         field_confidence_json,raw_ocr_json,source_json)
        VALUES(NULL,?,?,?,?, '{}','{}','{}')""",
        (token, timestamp, market_cap, market_cap * 2.0),
    )


def _live_db(path) -> None:
    migrate_raw(path)
    with sqlite3.connect(path) as conn:
        impl.migrate(conn)
        _insert_observation(conn, "A", "2026-09-14T21:53:51.724000+00:00", 100.0)
        _insert_observation(conn, "A", "2026-09-19T20:02:06.873000+00:00", 125.0)
        _insert_observation(conn, "B", "2026-09-19T20:02:06.873000+00:00", 250.0)
        conn.commit()


def test_current_inference_is_label_free_and_uses_exact_latest_capture(tmp_path):
    db = tmp_path / "raw.sqlite"
    _live_db(db)
    cfg = impl.V24Config(sequence_windows_minutes=(60,), sequence_segments=1)

    with impl.closing(impl._connect_live_read(str(db))) as conn:
        frame, sequence, latest = impl._current_inference_inputs(conn, cfg)

    assert latest == pd.Timestamp("2026-09-19T20:02:06.873000Z")
    assert set(frame.token_key) == {"A", "B"}
    assert set(sequence.token_key) == {"A", "B"}
    assert set(pd.to_datetime(frame.snapshot_at, utc=True)) == {latest}
    assert set(pd.to_datetime(sequence.snapshot_at, utc=True)) == {latest}


def test_latest_only_feature_builder_matches_full_causal_builder():
    times = pd.date_range("2026-09-19T20:00:00Z", periods=4, freq="min")
    observations = pd.DataFrame({
        "token_key": ["A"] * 4 + ["B"] * 4,
        "snapshot_at": list(times) * 2,
        "market_cap_usd": [100.0, 110.0, 90.0, 135.0, 200.0, 180.0, 220.0, 240.0],
        "volume_usd": [10.0, 12.0, 14.0, 20.0, 30.0, 28.0, 35.0, 40.0],
        "holders": [5, 6, 6, 8, 10, 10, 11, 12],
    })
    full = peak_impl.build_fallback_features(observations)
    expected = full[full.snapshot_at == times[-1]].sort_values("token_key").reset_index(drop=True)
    actual = peak_impl.build_fallback_features(
        observations, emit_at=times[-1]
    ).sort_values("token_key").reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)


def test_in_memory_sequence_encoding_batches_all_current_tokens(monkeypatch):
    timestamp = pd.Timestamp("2026-09-19T20:02:06.873000Z")
    raw = pd.DataFrame({
        "token_key": ["A", "B"],
        "snapshot_at": [timestamp, timestamp],
        "seqraw__market_cap_usd__60m__change": [0.25, -0.10],
    })
    calls = []

    def fake_encoder(frame, _encoder):
        calls.append(len(frame))
        return frame[["token_key", "snapshot_at"]].assign(seqenc__00=[1.0, 2.0])

    monkeypatch.setattr(impl, "apply_sequence_encoder", fake_encoder)
    encoded = impl._encode_sequence_keys(
        raw,
        raw[["token_key", "snapshot_at"]],
        {"columns": ["seqraw__market_cap_usd__60m__change"]},
    )

    assert calls == [2]
    assert list(encoded.token_key) == ["A", "B"]


def test_predict_current_never_enters_training_frame_and_publishes_atomically(
    tmp_path, monkeypatch
):
    db = tmp_path / "raw.sqlite"
    _live_db(db)
    model = tmp_path / "champion.joblib"
    output = tmp_path / "predictions.csv"
    bundle = {
        "schema_version": impl.SCHEMA_VERSION,
        "stable_training_cutoff": "2026-09-09T00:00:00+00:00",
        "sequence_encoder": {"columns": [], "components": 0, "scaler": None, "pca": None, "impute": {}},
    }
    joblib.dump(bundle, model)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("live prediction entered the label-dependent training frame")

    def fake_predict(_conn, _bundle, current, _sequence, _cfg):
        result = current[["token_key", "snapshot_at"]].copy()
        result["p_first_peak_by_240m"] = 0.5
        return result

    monkeypatch.setattr(impl, "load_v24_frame", forbidden)
    monkeypatch.setattr(impl, "predict_frame", fake_predict)

    out = impl.predict_current(
        str(db),
        str(model),
        str(output),
        impl.V24Config(sequence_windows_minutes=(60,), sequence_segments=1),
    )

    latest = pd.Timestamp("2026-09-19T20:02:06.873000Z")
    assert set(pd.to_datetime(out.snapshot_at, utc=True)) == {latest}
    published = pd.read_csv(output)
    assert set(pd.to_datetime(published.snapshot_at, utc=True)) == {latest}
    assert set(published.token_key) == {"A", "B"}
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            f"SELECT COUNT(*) FROM {impl.PREDICTION_LEDGER}"
        ).fetchone()[0] == 2
        assignments = conn.execute(
            f"SELECT token_key,first_seen_at,forecast_role,policy_role "
            f"FROM {impl.TOKEN_ASSIGNMENT_TABLE} ORDER BY token_key"
        ).fetchall()
    assert [row[0] for row in assignments] == ["A", "B"]
    assert assignments[0][1] == "2026-09-14T21:53:51.724000+00:00"
    assert all(row[2] in {"train", "promotion", "audit"} for row in assignments)
    assert all(row[3] in {"train", "promotion", "audit"} for row in assignments)

    benchmark_db = tmp_path / "benchmark.sqlite"
    benchmark.init_benchmark(
        str(benchmark_db), benchmark.BenchmarkConfig(), reset=False
    )
    cycle = benchmark.cycle(
        str(db),
        str(benchmark_db),
        str(output),
        str(model),
        str(tmp_path / "intentionally-absent-bootstrap-policy.joblib"),
        benchmark.BenchmarkConfig(),
    )
    assert cycle["processed"] is True
    assert cycle["snapshot_at"] == latest.isoformat()
    assert cycle["policy_version"] == "bootstrap"
    refreshed = benchmark.refresh_predictions(str(db), str(model), str(output))
    assert refreshed["skipped"] is True
    assert refreshed["reason"] == "prediction_already_current_for_frozen_model"


def test_atomic_prediction_publish_preserves_last_good_file_on_failure(tmp_path, monkeypatch):
    output = tmp_path / "predictions.csv"
    output.write_text("last-known-good\n", encoding="utf-8")
    original = pd.DataFrame.to_csv

    def fail_after_write(self, path, *args, **kwargs):
        original(self, path, *args, **kwargs)
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_after_write)
    with pytest.raises(OSError, match="synthetic publication failure"):
        impl._atomic_write_prediction_csv(pd.DataFrame({"x": [1]}), str(output))

    assert output.read_text(encoding="utf-8") == "last-known-good\n"
    assert not list(tmp_path.glob(".predictions.csv.*.tmp"))


def test_isolated_prediction_stays_read_only_while_collector_holds_writer_lock(
    tmp_path, monkeypatch
):
    db = tmp_path / "raw.sqlite"
    _live_db(db)
    model = tmp_path / "champion.joblib"
    output = tmp_path / "predictions.csv"
    joblib.dump({
        "schema_version": impl.SCHEMA_VERSION,
        "stable_training_cutoff": "2026-09-09T00:00:00+00:00",
        "sequence_encoder": {
            "columns": [], "components": 0, "scaler": None, "pca": None, "impute": {}
        },
    }, model)

    def fake_predict(_conn, _bundle, current, _sequence, _cfg):
        return current[["token_key", "snapshot_at"]].assign(p_first_peak_by_240m=0.5)

    def forbidden_write(*_args, **_kwargs):
        raise AssertionError("isolated benchmark attempted to write the source database")

    monkeypatch.setattr(impl, "predict_frame", fake_predict)
    monkeypatch.setattr(impl, "_run_live_write_with_retry", forbidden_write)

    writer = sqlite3.connect(db)
    try:
        writer.execute("BEGIN IMMEDIATE")
        out = impl.predict_current(
            str(db),
            str(model),
            str(output),
            impl.V24Config(sequence_windows_minutes=(60,), sequence_segments=1),
            persist_source=False,
        )
    finally:
        writer.rollback()
        writer.close()

    assert len(out) == 2
    assert output.exists()
    with sqlite3.connect(db) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM {impl.PREDICTION_LEDGER}").fetchone()[0] == 0
        assert conn.execute(f"SELECT COUNT(*) FROM {impl.TOKEN_ASSIGNMENT_TABLE}").fetchone()[0] == 0


def test_benchmark_refresh_explicitly_disables_source_persistence(tmp_path, monkeypatch):
    model = tmp_path / "champion.joblib"
    output = tmp_path / "predictions.csv"
    joblib.dump({"schema_version": impl.SCHEMA_VERSION}, model)
    captured = {}

    def fake_predict(_db, _model, _output, _cfg, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"token_key": ["A"], "snapshot_at": [pd.Timestamp.now(tz="UTC")]})

    monkeypatch.setattr(benchmark.v24, "predict_current", fake_predict)
    result = benchmark.refresh_predictions(str(tmp_path / "raw.sqlite"), str(model), str(output))

    assert captured == {"persist_source": False}
    assert result["source_db_writes"] is False
    assert result["training_feedback"] == "disabled"


def test_isolated_prediction_can_publish_consistent_snapshot_after_source_advances(
    tmp_path, monkeypatch
):
    db = tmp_path / "raw.sqlite"
    _live_db(db)
    model = tmp_path / "champion.joblib"
    output = tmp_path / "predictions.csv"
    joblib.dump({
        "schema_version": impl.SCHEMA_VERSION,
        "stable_training_cutoff": "2026-09-09T00:00:00+00:00",
    }, model)
    original_inputs = impl._current_inference_inputs
    advanced = False

    def advancing_inputs(conn, cfg):
        nonlocal advanced
        result = original_inputs(conn, cfg)
        if not advanced:
            advanced = True
            with sqlite3.connect(db) as writer:
                _insert_observation(
                    writer, "C", "2026-09-19T20:03:06.873000+00:00", 300.0
                )
                writer.commit()
        return result

    def fake_predict(_conn, _bundle, current, _sequence, _cfg):
        return current[["token_key", "snapshot_at"]].assign(p_first_peak_by_240m=0.5)

    monkeypatch.setattr(impl, "_current_inference_inputs", advancing_inputs)
    monkeypatch.setattr(impl, "predict_frame", fake_predict)
    out = impl.predict_current(
        str(db), str(model), str(output),
        impl.V24Config(sequence_windows_minutes=(60,), sequence_segments=1),
        persist_source=False, require_latest=False,
    )

    assert set(pd.to_datetime(out.snapshot_at, utc=True)) == {
        pd.Timestamp("2026-09-19T20:02:06.873000Z")
    }
    assert impl._latest_observation_timestamp(str(db)) == pd.Timestamp(
        "2026-09-19T20:03:06.873000Z"
    )


def test_live_write_waits_for_collector_lock_then_commits(tmp_path):
    db = tmp_path / "lock.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE writes(value INTEGER NOT NULL)")
        conn.commit()

    ready = threading.Event()

    def collector_write():
        with sqlite3.connect(db) as conn:
            conn.execute("BEGIN IMMEDIATE")
            ready.set()
            time.sleep(1.0)
            conn.commit()

    thread = threading.Thread(target=collector_write)
    thread.start()
    assert ready.wait(timeout=2.0)

    impl._run_live_write_with_retry(
        str(db), lambda conn: conn.execute("INSERT INTO writes(value) VALUES(1)")
    )
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT value FROM writes").fetchall() == [(1,)]


def test_benchmark_keeps_stale_guard_and_does_not_write_source_heartbeat():
    source = inspect.getsource(benchmark._cycle_v24)
    assert "Prediction CSV is stale" not in source  # Guard remains in _read_current.
    assert "record_capture_heartbeat" not in source
    read_source = inspect.getsource(benchmark._read_current)
    assert "Prediction CSV is stale" in read_source
    assert "PRAGMA query_only=ON" in read_source
