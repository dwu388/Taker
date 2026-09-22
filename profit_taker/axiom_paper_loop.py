"""Independent clipboard producer, raw ingester and latest-state paper trader.

The producer only copies and journals text.  A dedicated ingestion process
persists every capture in FIFO order while an independent trading process
coalesces superseded boards and evaluates the newest durable snapshot.  Paper
prediction is pinned to one SQLite read snapshot, so raw collection may advance
without changing the board on which that decision is evaluated.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import signal
import sqlite3
import time
import zlib

# Use the same hardened public collector facade as run_axiom_loop.bat.  Importing
# the base implementation directly allowed the independent paper loop to drift
# from production clipboard sentinel and extension-hook behavior.
from . import axiom_migrated_runner as collector
from . import axiom_manual_stop as manual_stop
from .collection_admin import initialize_collection
from .common import load_json
from .db import RAW_DB_DEFAULT
from .collector_lock import collector_lock


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def emit(**values) -> None:
    print(json.dumps(values, default=str), flush=True)


def init_queue(path: str, source_db: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS queue_source (source_db TEXT NOT NULL)")
        existing = conn.execute("SELECT source_db FROM queue_source").fetchone()
        source = str(Path(source_db).resolve())
        if existing and existing[0] != source:
            raise ValueError("Capture queue belongs to another source database")
        if not existing:
            conn.execute("INSERT INTO queue_source VALUES(?)", (source,))
        conn.execute("""CREATE TABLE IF NOT EXISTS pending (
            id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL,
            captured_at TEXT NOT NULL, payload BLOB NOT NULL, error TEXT)""")


def enqueue(path: str, started_at: str, captured_at: str, text: str, error=None) -> int:
    # No model/raw-db work and no lock held during clipboard control or prediction.
    payload = zlib.compress(text.encode("utf-8"))
    with closing(sqlite3.connect(path, timeout=2)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO pending(started_at,captured_at,payload,error) VALUES(?,?,?,?)",
            (started_at, captured_at, payload, json.dumps(error) if error else None),
        )
        return int(cursor.lastrowid)


def pending_ids(path: str) -> list[int]:
    with closing(sqlite3.connect(path, timeout=2)) as conn:
        # Finite batch: new captures cannot make the worker drain forever.
        return [int(row[0]) for row in conn.execute("SELECT id FROM pending ORDER BY id")]


def canonical_snapshot(value: str) -> str:
    """Return one UTC identity for equivalent ISO-8601 snapshot timestamps."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def latest_snapshot(db: str) -> str | None:
    with closing(sqlite3.connect(db, timeout=10)) as conn:
        conn.execute("PRAGMA busy_timeout=10000")
        row = conn.execute(
            "SELECT snapshot_at FROM axiom_observations ORDER BY snapshot_at DESC LIMIT 1"
        ).fetchone()
    return str(row[0]) if row else None


def persisted_cycle(db: str, timestamp: str, text: str) -> int | None:
    # Recover a crash after raw COMMIT but before queue acknowledgement.
    with closing(sqlite3.connect(db)) as conn:
        row = conn.execute(
            """SELECT c.cycle_id,p.sha256 FROM capture_cycles c
            JOIN capture_payloads p USING(cycle_id)
            WHERE c.captured_at=? AND c.completed=1 AND c.clipboard_valid=1""",
            (timestamp,),
        ).fetchone()
    if row is None:
        return None
    if row[1] != hashlib.sha256(text.encode("utf-8")).hexdigest():
        raise RuntimeError("Queued timestamp collides with a different durable payload")
    return int(row[0])


def log_failure(args, started_at: str, captured_at: str, text: str, error: dict) -> None:
    collector.record_failed_capture_attempt(
        args.db, started_at=started_at, completed_at=captured_at,
        source="interactive_clipboard", raw_clipboard_text=text,
        error_type=error["type"], error_message=error["message"],
        clipboard_valid=error.get("clipboard_valid", False),
        rows_detected=error.get("rows_detected", 0),
    )


def ingest_one(args, item_id: int) -> bool:
    with closing(sqlite3.connect(args.queue_db, timeout=2)) as conn:
        row = conn.execute(
            "SELECT started_at,captured_at,payload,error FROM pending WHERE id=?", (item_id,)
        ).fetchone()
    if row is None:
        return False
    started_at, captured_at, payload, error_json = row
    text = zlib.decompress(payload).decode("utf-8")
    success = False
    if error_json:
        log_failure(args, started_at, captured_at, text, json.loads(error_json))
    else:
        cycle_id = persisted_cycle(args.db, captured_at, text)
        if cycle_id is None:
            try:
                collector._sync_test_and_extension_hooks()
                result = collector.process_capture(
                    args, text, captured_at, attempt_started_at=started_at,
                )
            except (RuntimeError, ValueError) as exc:
                # Validation/identity rejection is durable diagnostic evidence.
                # SQLite/I/O/unexpected errors escape, leaving the item for retry.
                error = {"type": type(exc).__name__, "message": str(exc),
                         "clipboard_valid": bool(getattr(exc, "clipboard_valid", False)),
                         "rows_detected": int(getattr(exc, "rows_detected", 0))}
                log_failure(args, started_at, captured_at, text, error)
                emit(rejected_capture=item_id, **error)
            else:
                success = True
                emit(stored_capture=item_id, cycle_id=result["cycle_id"],
                     rows=result["rows_inserted"], captured_at=captured_at)
        else:
            success = True
            emit(recovered_capture=item_id, cycle_id=cycle_id)
    with closing(sqlite3.connect(args.queue_db, timeout=2)) as conn, conn:
        conn.execute("DELETE FROM pending WHERE id=?", (item_id,))
    return success


def predict_and_trade(args, benchmark, stop) -> str | None:
    requested_snapshot = latest_snapshot(args.db)
    if requested_snapshot is None:
        return None
    timestamp = datetime.fromisoformat(requested_snapshot.replace("Z", "+00:00"))
    def age():
        return (datetime.now(timezone.utc) - timestamp).total_seconds()
    if age() > args.max_snapshot_age_seconds:
        emit(wallet_skipped="capture_too_old", snapshot_at=requested_snapshot)
        return requested_snapshot
    emit(processing="prediction", snapshot_at=requested_snapshot)
    predictions = benchmark.refresh_predictions(
        args.db, args.forecast_model, args.predictions, require_latest=False,
    )
    predicted_snapshot = str(predictions.get("snapshot_at") or requested_snapshot)
    timestamp = datetime.fromisoformat(predicted_snapshot.replace("Z", "+00:00"))
    if stop.is_set() or age() > args.max_snapshot_age_seconds:
        emit(wallet_skipped="stopping_or_prediction_too_old", snapshot_at=predicted_snapshot)
        return predicted_snapshot
    # live_decisions pins the board read to the prediction timestamp. Captures
    # persisted during inference therefore become the next coalesced decision,
    # never an accidental same-board fill or a reason to retry stale work.
    result = benchmark._cycle_v24(
        args.db, args.benchmark_db, args.predictions, args.forecast_model,
        args.policy_model, benchmark.BenchmarkConfig(), live_decisions=True,
    )
    emit(predictions=predictions, benchmark=result)
    return predicted_snapshot


def ingestion_worker_main(args, stop, ready) -> None:
    # Ctrl+C belongs to the producer, which stops capture and requests a drain.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    initialize_collection(args.db, purpose="v24_production_raw_collection")
    # Recover interrupted ingestion before establishing the new run boundary.
    for item_id in pending_ids(args.queue_db):
        ingest_one(args, item_id)
    run_id = manual_stop.start_collection_session(args.db, source="axiom_paper_loop")
    try:
        ready.set()
        while True:
            for item_id in pending_ids(args.queue_db):
                ingest_one(args, item_id)
            if stop.is_set():
                if not pending_ids(args.queue_db):
                    break
                continue
            stop.wait(0.2)
    finally:
        # Unexpected worker failures retain both pending data and the active run
        # so recovery can ingest first, then establish the unclean-stop boundary.
        if stop.is_set() and not pending_ids(args.queue_db):
            emit(collection_stop=manual_stop.stop_collection_session(args.db, run_id))


def trading_worker_main(args, stop, ingestion_ready, ready) -> None:
    """Evaluate newest durable snapshots without consuming the capture queue."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from . import axiom_budget_benchmark as benchmark
    import joblib

    benchmark._require_v21()
    bundle = joblib.load(args.forecast_model)
    if bundle.get("schema_version") != benchmark.v24.SCHEMA_VERSION:
        raise RuntimeError("Paper loop requires a V24 forecast champion")
    if Path(args.policy_model).exists():
        joblib.load(args.policy_model)  # Fail early for a corrupt policy file.
    while not ingestion_ready.wait(0.2):
        if stop.is_set():
            return
    emit(wallet=benchmark.init_benchmark(args.benchmark_db, benchmark.BenchmarkConfig()))
    recovered = latest_snapshot(args.db)
    # Recovery history is collection-only. Snapshot identity is temporal rather
    # than textual because SQLite and Pandas may serialize the same instant with
    # different fractional precision or UTC suffixes.
    last_attempted = canonical_snapshot(recovered) if recovered is not None else None
    ready.set()
    while not stop.is_set():
        newest = latest_snapshot(args.db)
        newest_identity = canonical_snapshot(newest) if newest is not None else None
        if newest_identity is None or newest_identity == last_attempted:
            stop.wait(0.2)
            continue
        # Claim before expensive work. A failure is retried on the next durable
        # board rather than hot-looping and starving collection resources.
        last_attempted = newest_identity
        try:
            processed = predict_and_trade(args, benchmark, stop)
            if processed is not None:
                last_attempted = canonical_snapshot(processed)
        except Exception as exc:
            emit(processing_error=type(exc).__name__, message=str(exc), snapshot_at=newest)


def capture_forever(args, workers, stop) -> None:
    if hasattr(workers, "is_alive"):
        workers = (workers,)
    cfg = load_json(args.config, {})
    count = 0
    deadline = time.monotonic()
    while not stop.is_set():
        failed = [worker for worker in workers if not worker.is_alive()]
        if failed:
            worker = failed[0]
            raise RuntimeError(
                f"{worker.name} exited ({worker.exitcode}); queued captures retained"
            )
        remaining = deadline - time.monotonic()
        if remaining > 0:
            stop.wait(min(1.0, remaining))
            continue
        count += 1
        started_at = now_iso()
        error = None
        try:
            # The base macro installs/verifies a unique sentinel itself and shares
            # the production click, retry, Ctrl+A/C and 601-cycle refresh logic.
            text, captured_at = collector._capture_clipboard(cfg, count)
        except Exception as exc:
            text, captured_at = getattr(exc, "clipboard_text", ""), now_iso()
            error = {"type": type(exc).__name__, "message": str(exc)}
        item_id = enqueue(args.queue_db, started_at, captured_at, text, error)
        emit(capture_queued=item_id, captured_at=captured_at, error=error)
        deadline += args.interval_seconds
        # Do not burst-copy missed slots if the desktop/disk itself stalls.
        while deadline <= time.monotonic():
            deadline += args.interval_seconds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", "--source-db", default=RAW_DB_DEFAULT)
    parser.add_argument("--config", default="axiom_migrated_config.json")
    parser.add_argument("--queue-db", help="Defaults to SOURCE_DB.paper_queue.sqlite")
    parser.add_argument("--benchmark-db", default="data/axiom_v24_1000_benchmark.sqlite")
    parser.add_argument("--predictions", default="data/axiom_predictions_v24.csv")
    parser.add_argument("--forecast-model", default="models/axiom_v24/champion.joblib")
    parser.add_argument("--policy-model", default="models/axiom_policy_v24/champion.joblib")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--max-snapshot-age-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    import math
    for value in (args.interval_seconds, args.max_snapshot_age_seconds):
        if not math.isfinite(value) or value <= 0:
            parser.error("Timing values must be finite and positive")
    args.queue_db = args.queue_db or args.db + ".paper_queue.sqlite"
    paths = [args.db, args.queue_db, args.benchmark_db, args.predictions,
             args.forecast_model, args.policy_model]
    if len({str(Path(path).resolve()).casefold() for path in paths}) != len(paths):
        parser.error("Raw, queue, wallet, prediction and model paths must be distinct")
    if not Path(args.config).is_file():
        parser.error(f"Config does not exist: {args.config}")
    if not Path(args.forecast_model).is_file():
        parser.error(f"Forecast champion does not exist: {args.forecast_model}")
    args.database_only = True
    args.clipboard_file = None
    args.output_dir = "data/axiom_migrated"  # Required by shared validator; not written.
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    with collector_lock(args.db):
        init_queue(args.queue_db, args.db)
        context = mp.get_context("spawn")
        stop = context.Event()
        ingestion_ready, trading_ready = context.Event(), context.Event()
        ingester = context.Process(
            target=ingestion_worker_main, args=(args, stop, ingestion_ready),
            name="axiom-paper-ingester",
        )
        trader = context.Process(
            target=trading_worker_main,
            args=(args, stop, ingestion_ready, trading_ready),
            name="axiom-paper-trader",
        )
        workers = (ingester, trader)
        for worker in workers:
            worker.start()
        try:
            while not trading_ready.wait(0.2):
                failed = [worker for worker in workers if not worker.is_alive()]
                if failed:
                    worker = failed[0]
                    raise RuntimeError(f"{worker.name} startup failed ({worker.exitcode})")
            emit(mode="independent_paper_loop", capture_interval_seconds=args.interval_seconds)
            capture_forever(args, workers, stop)
        except KeyboardInterrupt:
            emit(stopping="Capture stopped; finishing current work and saving queued captures")
        finally:
            stop.set()
            # Remain responsive while waiting; never terminate an in-flight commit.
            while any(worker.is_alive() for worker in workers):
                try:
                    for worker in workers:
                        worker.join(0.25)
                except KeyboardInterrupt:
                    emit(stopping="Still draining; closing this window leaves durable queue for restart")
            failed = [worker for worker in workers if worker.exitcode]
            if failed:
                worker = failed[0]
                raise RuntimeError(
                    f"{worker.name} failed ({worker.exitcode}); queue retained for restart"
                )


if __name__ == "__main__":
    main()
