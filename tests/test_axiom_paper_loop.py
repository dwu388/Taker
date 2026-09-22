from contextlib import closing
import multiprocessing as mp
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace
import zlib

import joblib
import pandas as pd
import pytest

from profit_taker import axiom_paper_loop as loop
from profit_taker import axiom_budget_benchmark as benchmark
from profit_taker import axiom_v24 as v24
from profit_taker.collection_admin import initialize_collection
from profit_taker.collector_lock import collector_lock


def args_for(tmp_path):
    args = SimpleNamespace(db=str(tmp_path / 'raw.sqlite'),
        queue_db=str(tmp_path / 'queue.sqlite'), config=str(tmp_path / 'config.json'),
        output_dir=str(tmp_path / 'artifacts'), clipboard_file=None, database_only=True,
        benchmark_db=str(tmp_path / 'wallet.sqlite'), predictions=str(tmp_path / 'pred.csv'),
        forecast_model=str(tmp_path / 'model.joblib'), policy_model=str(tmp_path / 'missing.joblib'),
        interval_seconds=0.02, max_snapshot_age_seconds=180)
    Path(args.config).write_text('{}')
    initialize_collection(args.db)
    loop.init_queue(args.queue_db, args.db)
    return args


def parsed_row():
    return {'token_key': 'abc...pump', 'short_address_hint': 'abc...pump',
            'name': 'Synthetic', 'market_cap_usd': 12345.0, 'source': {}, 'field_confidence': {}}


def mock_parser(monkeypatch):
    monkeypatch.setattr(loop.collector, 'clipboard_looks_like_axiom', lambda _: True)
    monkeypatch.setattr(loop.collector, 'rows_from_clipboard', lambda _: [parsed_row()])


def test_queue_ingestion_retains_exact_payload_time_and_no_review_files(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    mock_parser(monkeypatch)
    stamp, text = '2026-09-22T12:00:00.123+00:00', 'exact UTF-8 π\nMC\n$12.3K'
    item = loop.enqueue(args.queue_db, stamp, stamp, text)
    assert loop.ingest_one(args, item)
    with closing(sqlite3.connect(args.db)) as conn:
        assert conn.execute('SELECT snapshot_at FROM axiom_observations').fetchone()[0] == stamp
        assert zlib.decompress(conn.execute('SELECT payload FROM capture_payloads').fetchone()[0]).decode() == text
        assert conn.execute('SELECT source FROM capture_attempts').fetchone()[0] == 'interactive_clipboard'
    assert loop.pending_ids(args.queue_db) == []
    assert not Path(args.output_dir).exists()


def test_crash_after_commit_recovers_without_duplicate_cycle(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    mock_parser(monkeypatch)
    stamp, text = loop.now_iso(), 'MC\n$12.3K'
    loop.collector.process_capture(args, text, stamp)
    item = loop.enqueue(args.queue_db, stamp, stamp, text)
    assert loop.ingest_one(args, item)
    with closing(sqlite3.connect(args.db)) as conn:
        assert conn.execute('SELECT COUNT(*) FROM capture_cycles').fetchone()[0] == 1


def test_storage_failure_leaves_item_queued(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    stamp = loop.now_iso()
    item = loop.enqueue(args.queue_db, stamp, stamp, 'MC')
    def fail(*_args, **_kwargs):
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(loop.collector, 'process_capture', fail)
    with pytest.raises(sqlite3.OperationalError):
        loop.ingest_one(args, item)
    assert loop.pending_ids(args.queue_db) == [item]


def test_partial_capture_and_copy_failure_are_logged_without_observations(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    mock_parser(monkeypatch)
    stamp = loop.now_iso()
    partial = loop.enqueue(args.queue_db, stamp, stamp, 'MC\nMC')
    assert not loop.ingest_one(args, partial)
    failed = loop.enqueue(args.queue_db, stamp, stamp, '', {'type': 'RuntimeError', 'message': 'copy failed'})
    assert not loop.ingest_one(args, failed)
    with closing(sqlite3.connect(args.db)) as conn:
        assert conn.execute('SELECT COUNT(*) FROM axiom_observations').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM capture_attempts WHERE success=0').fetchone()[0] == 2
    assert not loop.pending_ids(args.queue_db)


def _busy_worker(entered, release):
    entered.set()
    release.wait(10)


def test_capture_continues_while_spawned_worker_is_blocked(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    context = mp.get_context('spawn')
    entered, release = context.Event(), context.Event()
    worker = context.Process(target=_busy_worker, args=(entered, release))
    stop = threading.Event()
    counts = []
    def capture(_cfg, count):
        counts.append(count)
        if count == 4:
            stop.set()
        return f'payload {count}', loop.now_iso()
    monkeypatch.setattr(loop.collector, '_capture_clipboard', capture)
    worker.start()
    try:
        assert entered.wait(10)
        loop.capture_forever(args, worker, stop)
        assert counts == [1, 2, 3, 4]
        assert len(loop.pending_ids(args.queue_db)) == 4
        assert not release.is_set()
    finally:
        release.set()
        worker.join(10)
    assert worker.exitcode == 0


def test_collector_ownership_and_queue_source_binding(tmp_path):
    args = args_for(tmp_path)
    with collector_lock(args.db):
        with pytest.raises(RuntimeError, match='Another collector'):
            with collector_lock(args.db):
                pass
    with collector_lock(args.db):
        pass
    with pytest.raises(ValueError, match='another source'):
        loop.init_queue(args.queue_db, str(tmp_path / 'other.sqlite'))


def test_shutdown_suppresses_wallet_and_snapshot_age_is_rechecked(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    mock_parser(monkeypatch)
    loop.collector.process_capture(args, 'MC', loop.now_iso())
    calls = []
    stop = threading.Event()
    def refresh(*_args):
        calls.append('predict')
        stop.set()
        return {}
    fake = SimpleNamespace(refresh_predictions=refresh,
        _cycle_v24=lambda *_a, **_kw: calls.append('wallet'), BenchmarkConfig=lambda: None)
    loop.predict_and_trade(args, fake, stop)
    assert calls == ['predict']
    stop.clear()
    def slow(*_args):
        calls.append('predict')
        args.max_snapshot_age_seconds = 0
        return {}
    fake.refresh_predictions = slow
    loop.predict_and_trade(args, fake, stop)
    assert calls == ['predict', 'predict']


def test_live_orders_cannot_fill_on_boards_copied_during_prediction(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    joblib.dump({'schema_version': v24.SCHEMA_VERSION}, args.forecast_model)
    benchmark.init_benchmark(args.benchmark_db, benchmark.BenchmarkConfig())
    snapshot = pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=2)
    frame = pd.DataFrame({'token_key': ['A'], 'market_cap_usd': [100.0],
        'p_first_peak_by_720m': [0.9], 'pred_next_substantial_peak_multiple_q50': [2.0],
        'pred_time_to_next_substantial_peak_minutes_q50': [10.0],
        'p_death_by_720m': [0.0], 'p_hit_minus50_by_720m': [0.0], 'v24_model_hash': ['frozen']})
    monkeypatch.setattr(benchmark, '_read_current', lambda *_: (snapshot, frame))
    def cycle():
        return benchmark._cycle_v24(args.db, args.benchmark_db, args.predictions,
            args.forecast_model, args.policy_model, benchmark.BenchmarkConfig(), live_decisions=True)
    before = pd.Timestamp.now(tz='UTC')
    first = cycle()
    assert len(first['pending_entries']) == 1
    with closing(sqlite3.connect(args.benchmark_db)) as conn:
        available = pd.Timestamp(conn.execute('SELECT decision_at FROM benchmark_pending_entries_v24').fetchone()[0])
    assert available >= before
    snapshot += pd.Timedelta(minutes=1)  # Captured during prior computation.
    assert cycle()['entries'] == []
    snapshot = available + pd.Timedelta(seconds=1)
    assert len(cycle()['entries']) == 1
    # Force an exit decision and apply the same availability rule to sells.
    frame['pred_next_substantial_peak_multiple_q50'] = 0.5
    snapshot += pd.Timedelta(minutes=3)
    cycle()
    with closing(sqlite3.connect(args.benchmark_db)) as conn:
        pending_exit = conn.execute("SELECT pending_exit_at FROM benchmark_positions_v22 WHERE status='open'").fetchone()[0]
    assert pending_exit is not None
    assert pd.Timestamp(pending_exit) >= snapshot
    # Simulate an exit calculation completing after another board was copied.
    available_exit = snapshot + pd.Timedelta(minutes=2)
    with closing(sqlite3.connect(args.benchmark_db)) as conn, conn:
        conn.execute("UPDATE benchmark_positions_v22 SET pending_exit_at=? WHERE status='open'",
                     (available_exit.isoformat(),))
    snapshot += pd.Timedelta(minutes=1)
    assert cycle()['exits'] == []
    with closing(sqlite3.connect(args.benchmark_db)) as conn:
        assert pd.Timestamp(conn.execute("SELECT pending_exit_at FROM benchmark_positions_v22 WHERE status='open'").fetchone()[0]) == available_exit
    snapshot = available_exit + pd.Timedelta(seconds=1)
    assert len(cycle()['exits']) == 1


def test_spawned_worker_drains_on_stop_and_closes_run(tmp_path):
    args = args_for(tmp_path)
    joblib.dump({'schema_version': v24.SCHEMA_VERSION}, args.forecast_model)
    context = mp.get_context('spawn')
    stop, ready = context.Event(), context.Event()
    worker = context.Process(target=loop.worker_main, args=(args, stop, ready))
    worker.start()
    try:
        assert ready.wait(20)
        stamp = loop.now_iso()
        loop.enqueue(args.queue_db, stamp, stamp, '', {'type': 'RuntimeError', 'message': 'copy failed'})
        stop.set()
        worker.join(20)
        assert worker.exitcode == 0
        assert not loop.pending_ids(args.queue_db)
        with closing(sqlite3.connect(args.db)) as conn:
            assert conn.execute('SELECT status FROM axiom_v24_collection_run_sessions').fetchone()[0] == 'stopped'
            assert conn.execute('SELECT COUNT(*) FROM capture_attempts').fetchone()[0] == 1
    finally:
        stop.set()
        worker.join(5)
        if worker.is_alive():
            worker.terminate()
            worker.join()
