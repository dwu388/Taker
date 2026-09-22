"""Lossless collection plus gated V24 forecast and policy maintenance.

Clipboard capture is deliberately isolated from both SQLite ingestion and model
training.  Captures are first journaled to a small queue database at the normal
one-minute cadence.  The ingester drains that queue in order whenever the
official V24 training pipeline is not writing the raw database.  This keeps a
long model fit from turning into missing collection minutes.

The trainer does not implement alternate model logic.  It invokes the official
``v24_contract_runtime_v4`` commands in this order:

* bootstrap a missing forecast champion, otherwise maintain/challenge it;
* generate leakage-safe rolling-origin policy predictions;
* train/challenge the distributional policy.

Every existing readiness, maturity, one-use cohort, and promotion gate therefore
remains authoritative.  A failed or not-yet-ready training attempt is reported
and collection continues; existing champions are never deleted or reset.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import zlib

from . import axiom_collection_context_runner as collection_context
from . import axiom_paper_loop as queued_capture
from .collection_admin import initialize_collection
from .collector_lock import collector_lock
from .common import load_json
from .db import RAW_DB_DEFAULT


RUNTIME_MODULE = "profit_taker.v24_contract_runtime_v4"
DEFAULT_TRAINING_INTERVAL_HOURS = 24.0
DEFAULT_POLICY_CROSSFIT_FOLDS = 5
DEFAULT_STORAGE_RETRY_SECONDS = 2.0


def emit(**values) -> None:
    print(json.dumps(values, default=str), flush=True)


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _queued_item(path: str, item_id: int) -> tuple[str, str, str, str | None] | None:
    with closing(sqlite3.connect(path, timeout=2)) as conn:
        row = conn.execute(
            "SELECT started_at,captured_at,payload,error FROM pending WHERE id=?",
            (item_id,),
        ).fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1]), zlib.decompress(row[2]).decode("utf-8"), row[3]


def ingest_one(args, item_id: int) -> bool:
    """Persist one queued capture and preserve standard-loop diagnostics."""
    item = _queued_item(args.queue_db, item_id)
    if item is None:
        return False
    started_at, captured_at, text, error_json = item
    processing_started = time.perf_counter()
    success = queued_capture.ingest_one(args, item_id)
    post_ms = (time.perf_counter() - processing_started) * 1000.0
    if not success or error_json:
        return success

    cycle_id = queued_capture.persisted_cycle(args.db, captured_at, text)
    followup: dict[str, object] = {"cycle_id": cycle_id}
    if cycle_id is not None:
        try:
            control_ms = max(0.0, (_utc(captured_at) - _utc(started_at)).total_seconds() * 1000.0)
            collection_context._last_capture_control_ms = control_ms
            followup["capture_context"] = collection_context._record_context(
                args.db, int(cycle_id), post_ms
            )
        except Exception as exc:
            # Context is diagnostic and cannot invalidate an already durable raw
            # capture, matching the original collection-context runner contract.
            followup["capture_context"] = {
                "stored": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            followup["performance_report"] = (
                queued_capture.collector._maybe_refresh_performance_report(args)
            )
        except Exception as exc:
            followup["performance_report"] = {
                "generated": False,
                "reason": "report_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    emit(collection_followup=followup)
    return success


def _is_storage_contention(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return isinstance(exc, sqlite3.OperationalError) and (
        "locked" in message or "busy" in message
    )


def _wait_without_shutdown_spin(stop, seconds: float) -> None:
    if stop.is_set():
        time.sleep(seconds)
    else:
        stop.wait(seconds)


def ingestion_worker_main(args, stop, training_active, database_gate, ready) -> None:
    """Drain every queued capture without competing with planned training writes."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    initialize_collection(args.db, purpose="v24_production_raw_collection")

    # Interrupted captures predate this process and must be recovered before a
    # new run boundary is opened or training is allowed to start.
    for item_id in queued_capture.pending_ids(args.queue_db):
        ingest_one(args, item_id)

    run_id = queued_capture.manual_stop.start_collection_session(
        args.db, source="axiom_learning_loop"
    )
    last_contention_notice = 0.0
    try:
        ready.set()
        while True:
            if training_active.is_set():
                _wait_without_shutdown_spin(stop, args.storage_retry_seconds)
                continue

            item_ids = queued_capture.pending_ids(args.queue_db)
            blocked = False
            for item_id in item_ids:
                if training_active.is_set():
                    blocked = True
                    break
                try:
                    # The event prevents new work once training is announced;
                    # the lock closes the small check/start race with an ingest
                    # that was already entering its SQLite transaction.
                    with database_gate:
                        ingest_one(args, item_id)
                except Exception as exc:
                    if not _is_storage_contention(exc):
                        raise
                    blocked = True
                    now = time.monotonic()
                    if now - last_contention_notice >= 30.0:
                        emit(
                            ingestion_waiting="sqlite_writer_busy",
                            queued_captures=len(queued_capture.pending_ids(args.queue_db)),
                            message=str(exc),
                        )
                        last_contention_notice = now
                    break

            if stop.is_set() and not queued_capture.pending_ids(args.queue_db):
                break
            if blocked or not item_ids:
                _wait_without_shutdown_spin(
                    stop, args.storage_retry_seconds if blocked else 0.2
                )
    finally:
        if stop.is_set() and not queued_capture.pending_ids(args.queue_db):
            emit(
                collection_stop=queued_capture.manual_stop.stop_collection_session(
                    args.db, run_id
                )
            )


def _common_runtime_args(args) -> list[str]:
    return [
        "--db", args.db,
        "--cohort-hours", str(args.cohort_hours),
        "--promotion-every", str(args.promotion_every),
        "--audit-every", str(args.audit_every),
        "--warmup-blocks", str(args.warmup_blocks),
    ]


def training_commands(args) -> list[tuple[str, list[str]]]:
    """Return the gated official-runtime pipeline for the current model state."""
    champion = Path(args.model_root) / "champion.joblib"
    common = _common_runtime_args(args)
    if champion.is_file():
        forecast = (
            "forecast_maintenance",
            [sys.executable, "-m", RUNTIME_MODULE, "maintain", *common,
             "--model-root", args.model_root],
        )
    else:
        forecast = (
            "forecast_bootstrap",
            [sys.executable, "-m", RUNTIME_MODULE, "bootstrap", *common,
             "--model-root", args.model_root, "--profile", args.bootstrap_profile],
        )
    return [
        forecast,
        (
            "policy_crossfit",
            [sys.executable, "-m", RUNTIME_MODULE, "crossfit-policy-predictions",
             *common, "--max-folds", str(args.policy_max_folds)],
        ),
        (
            "policy_training",
            [sys.executable, "-m", RUNTIME_MODULE, "train-policy", *common,
             "--policy-root", args.policy_root],
        ),
    ]


def _run_runtime_command(command, *, check=False):
    kwargs = {"check": check}
    if os.name == "nt":
        # Keep Ctrl+C on the producer/supervisor. The trainer finishes its current
        # model/SQLite operation before shutdown rather than interrupting a model
        # serialization or champion promotion.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.run(command, **kwargs)


def run_training_cycle(args, stop=None, run_command=_run_runtime_command) -> dict:
    """Run one ordered maintenance cycle, stopping policy work on stale inputs."""
    results: list[dict[str, object]] = []
    commands = training_commands(args)
    for index, (stage, command) in enumerate(commands):
        if stop is not None and stop.is_set():
            return {"completed": False, "reason": "stopping", "stages": results}
        emit(training_stage=stage, state="starting", command=command)
        completed = run_command(command, check=False)
        code = int(completed.returncode)
        result = {"stage": stage, "exit_code": code}
        results.append(result)
        emit(training_stage=stage, state="finished", exit_code=code)
        if code != 0:
            # Bootstrap/readiness failures are expected while history matures.
            # Do not cross-fit or train a policy after a failed forecast stage,
            # and do not promote from policy inputs whose refresh just failed.
            return {"completed": False, "failed_stage": stage, "stages": results}
        if index == 0 and not (Path(args.model_root) / "champion.joblib").is_file():
            return {
                "completed": False,
                "failed_stage": stage,
                "reason": "forecast command returned success without a champion",
                "stages": results,
            }
    return {"completed": True, "stages": results}


def training_worker_main(
    args, stop, ingestion_ready, training_active, database_gate, ready
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while not ingestion_ready.wait(0.2):
        if stop.is_set():
            return
    ready.set()

    cycle = 0
    while not stop.is_set():
        cycle += 1
        started = time.monotonic()
        training_active.set()
        try:
            with database_gate:
                result = run_training_cycle(args, stop=stop)
        except Exception as exc:
            result = {
                "completed": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        finally:
            training_active.clear()
        emit(training_cycle=cycle, result=result)

        elapsed = time.monotonic() - started
        wait_seconds = max(0.0, args.training_interval_seconds - elapsed)
        if stop.wait(wait_seconds):
            break


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", "--source-db", default=RAW_DB_DEFAULT)
    parser.add_argument("--config", default="axiom_migrated_config.json")
    parser.add_argument("--queue-db", help="Defaults to SOURCE_DB.learning_queue.sqlite")
    parser.add_argument("--output-dir", default="data/axiom_migrated")
    parser.add_argument("--interval-seconds", type=float)
    parser.add_argument(
        "--training-interval-hours", type=float,
        default=DEFAULT_TRAINING_INTERVAL_HOURS,
        help="Hours between complete gated forecast/policy maintenance attempts",
    )
    parser.add_argument("--model-root", default="models/axiom_v24")
    parser.add_argument("--policy-root", default="models/axiom_policy_v24")
    parser.add_argument(
        "--bootstrap-profile", choices=("first_model", "full"), default="first_model"
    )
    parser.add_argument("--policy-max-folds", type=int, default=DEFAULT_POLICY_CROSSFIT_FOLDS)
    parser.add_argument("--cohort-hours", type=int, default=24)
    parser.add_argument("--promotion-every", type=int, default=4)
    parser.add_argument("--audit-every", type=int, default=5)
    parser.add_argument("--warmup-blocks", type=int, default=7)
    parser.add_argument("--storage-retry-seconds", type=float, default=DEFAULT_STORAGE_RETRY_SECONDS)
    # Kept for launcher compatibility. Queued ingestion is always database-only.
    parser.add_argument("--database-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if not Path(args.config).is_file():
        parser.error(f"Config does not exist: {args.config}")
    cfg = load_json(args.config, {}) or {}
    if args.interval_seconds is None:
        capture_cfg = cfg.get("capture") if isinstance(cfg, dict) else None
        capture_cfg = capture_cfg if isinstance(capture_cfg, dict) else {}
        try:
            args.interval_seconds = float(capture_cfg.get("cycle_seconds", 60))
        except (TypeError, ValueError):
            parser.error("capture.cycle_seconds in the config must be numeric")

    numeric_positive = {
        "interval-seconds": args.interval_seconds,
        "training-interval-hours": args.training_interval_hours,
        "storage-retry-seconds": args.storage_retry_seconds,
    }
    for name, value in numeric_positive.items():
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be finite and positive")
    if args.storage_retry_seconds > 60:
        parser.error("--storage-retry-seconds must not exceed 60")
    for name in ("policy_max_folds", "cohort_hours", "promotion_every", "audit_every"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_blocks < 0:
        parser.error("--warmup-blocks must be nonnegative")

    args.training_interval_seconds = args.training_interval_hours * 3600.0
    args.queue_db = args.queue_db or args.db + ".learning_queue.sqlite"
    if str(Path(args.db).resolve()).casefold() == str(Path(args.queue_db).resolve()).casefold():
        parser.error("Raw and queue databases must be distinct")
    if str(Path(args.model_root).resolve()).casefold() == str(Path(args.policy_root).resolve()).casefold():
        parser.error("Forecast and policy model roots must be distinct")
    args.database_only = True
    args.clipboard_file = None
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    with collector_lock(args.db):
        queued_capture.init_queue(args.queue_db, args.db)
        context = mp.get_context("spawn")
        stop = context.Event()
        ingestion_ready = context.Event()
        training_active = context.Event()
        training_ready = context.Event()
        database_gate = context.Lock()
        ingester = context.Process(
            target=ingestion_worker_main,
            args=(args, stop, training_active, database_gate, ingestion_ready),
            name="axiom-learning-ingester",
        )
        trainer = context.Process(
            target=training_worker_main,
            args=(
                args, stop, ingestion_ready, training_active, database_gate,
                training_ready,
            ),
            name="axiom-v24-trainer",
        )
        workers = (ingester, trainer)
        for worker in workers:
            worker.start()
        try:
            while not training_ready.wait(0.2):
                failed = [worker for worker in workers if not worker.is_alive()]
                if failed:
                    worker = failed[0]
                    raise RuntimeError(f"{worker.name} startup failed ({worker.exitcode})")
            emit(
                mode="collection_and_gated_v24_training",
                capture_interval_seconds=args.interval_seconds,
                training_interval_hours=args.training_interval_hours,
                forecast_champion=str(Path(args.model_root) / "champion.joblib"),
                policy_champion=str(Path(args.policy_root) / "champion.joblib"),
            )
            queued_capture.capture_forever(args, workers, stop)
        except KeyboardInterrupt:
            emit(
                stopping=(
                    "Capture stopped; allowing active training to finish, then "
                    "draining every queued capture"
                )
            )
        finally:
            stop.set()
            while any(worker.is_alive() for worker in workers):
                try:
                    for worker in workers:
                        worker.join(0.25)
                except KeyboardInterrupt:
                    emit(
                        stopping=(
                            "Still finishing durable work; closing this window leaves "
                            "the capture queue recoverable on restart"
                        )
                    )
            failed = [worker for worker in workers if worker.exitcode]
            if failed:
                worker = failed[0]
                raise RuntimeError(
                    f"{worker.name} failed ({worker.exitcode}); capture queue retained"
                )


if __name__ == "__main__":
    main()
