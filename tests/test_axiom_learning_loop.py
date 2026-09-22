from contextlib import closing
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from profit_taker import axiom_learning_loop as loop
from profit_taker import axiom_migrated_runner as production_collector
from profit_taker.collection_admin import initialize_collection


def args_for(tmp_path, *, champion=False):
    model_root = tmp_path / "models" / "forecast"
    policy_root = tmp_path / "models" / "policy"
    model_root.mkdir(parents=True)
    policy_root.mkdir(parents=True)
    if champion:
        (model_root / "champion.joblib").write_bytes(b"existing champion")
    return SimpleNamespace(
        db=str(tmp_path / "raw.sqlite"),
        model_root=str(model_root),
        policy_root=str(policy_root),
        bootstrap_profile="first_model",
        policy_max_folds=5,
        cohort_hours=24,
        promotion_every=4,
        audit_every=5,
        warmup_blocks=7,
    )


def test_learning_loop_uses_the_hardened_production_capture_path():
    assert loop.queued_capture.collector is production_collector


def test_queued_ingestion_preserves_standard_collection_context(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{"performance_report":{"enabled":false}}', encoding="utf-8")
    args = SimpleNamespace(
        db=str(tmp_path / "raw.sqlite"),
        queue_db=str(tmp_path / "queue.sqlite"),
        config=str(config),
        output_dir=str(tmp_path / "artifacts"),
        clipboard_file=None,
        database_only=True,
    )
    initialize_collection(args.db)
    loop.queued_capture.init_queue(args.queue_db, args.db)
    monkeypatch.setattr(loop.queued_capture.collector, "clipboard_looks_like_axiom", lambda _: True)
    monkeypatch.setattr(
        loop.queued_capture.collector,
        "rows_from_clipboard",
        lambda _: [{
            "token_key": "abc...pump",
            "short_address_hint": "abc...pump",
            "name": "Synthetic",
            "market_cap_usd": 12345.0,
            "source": {},
            "field_confidence": {},
        }],
    )
    started = "2026-09-22T12:00:00.000+00:00"
    captured = "2026-09-22T12:00:00.750+00:00"
    item_id = loop.queued_capture.enqueue(
        args.queue_db, started, captured, "MC\n$12.3K"
    )

    assert loop.ingest_one(args, item_id) is True

    with closing(sqlite3.connect(args.db)) as conn:
        context = conn.execute(
            "SELECT capture_control_latency_ms,visible_tokens "
            "FROM axiom_v24_capture_context"
        ).fetchone()
    assert context == (750.0, 1)
    assert loop.queued_capture.pending_ids(args.queue_db) == []
    assert not Path(args.output_dir).exists()


def test_missing_forecast_champion_runs_full_ordered_pipeline(tmp_path):
    args = args_for(tmp_path)
    calls = []

    def run(command, check):
        assert check is False
        calls.append(command)
        if "bootstrap" in command:
            (Path(args.model_root) / "champion.joblib").write_bytes(b"new champion")
        return SimpleNamespace(returncode=0)

    result = loop.run_training_cycle(args, run_command=run)

    assert result["completed"] is True
    assert [command[3] for command in calls] == [
        "bootstrap", "crossfit-policy-predictions", "train-policy"
    ]
    assert "--profile" in calls[0]
    assert "--model-root" in calls[0]
    assert "--max-folds" in calls[1]
    assert "--policy-root" in calls[2]


def test_existing_champion_uses_maintenance_before_policy_training(tmp_path):
    args = args_for(tmp_path, champion=True)
    calls = []

    def run(command, check):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    result = loop.run_training_cycle(args, run_command=run)

    assert result["completed"] is True
    assert [command[3] for command in calls] == [
        "maintain", "crossfit-policy-predictions", "train-policy"
    ]


def test_failed_forecast_stage_cannot_train_policy_from_stale_inputs(tmp_path):
    args = args_for(tmp_path, champion=True)
    calls = []

    def run(command, check):
        calls.append(command)
        return SimpleNamespace(returncode=2)

    result = loop.run_training_cycle(args, run_command=run)

    assert result["completed"] is False
    assert result["failed_stage"] == "forecast_maintenance"
    assert len(calls) == 1


def test_failed_crossfit_cannot_promote_policy_from_stale_inputs(tmp_path):
    args = args_for(tmp_path, champion=True)
    calls = []

    def run(command, check):
        calls.append(command)
        code = 3 if "crossfit-policy-predictions" in command else 0
        return SimpleNamespace(returncode=code)

    result = loop.run_training_cycle(args, run_command=run)

    assert result["completed"] is False
    assert result["failed_stage"] == "policy_crossfit"
    assert [command[3] for command in calls] == [
        "maintain", "crossfit-policy-predictions"
    ]


def test_launcher_selects_learning_loop_and_database_only_mode():
    launcher = Path(__file__).resolve().parents[1] / "run_axiom_loop.bat"
    text = launcher.read_text(encoding="utf-8")
    assert "profit_taker.axiom_learning_loop" in text
    assert "--database-only" in text


def test_storage_contention_is_retryable_but_other_io_failures_are_fatal():
    import sqlite3

    assert loop._is_storage_contention(sqlite3.OperationalError("database is locked"))
    assert loop._is_storage_contention(sqlite3.OperationalError("database table is busy"))
    assert not loop._is_storage_contention(sqlite3.OperationalError("disk I/O error"))
