import sqlite3
from contextlib import closing
from unittest.mock import patch

import pandas as pd
import pytest

from profit_taker import axiom_v24_impl as impl
from profit_taker import v24_preflight as preflight
from profit_taker.db import migrate


@pytest.mark.parametrize("reverse", [False, True])
def test_lifetime_mixed_precision_preserves_episode_mapping(reverse):
    with sqlite3.connect(":memory:") as conn:
        impl.migrate(conn)
        stamps = ["2026-09-09T10:36:27.123456+00:00", "2026-09-09T10:36:27+00:00"]
        if reverse:
            stamps.reverse()
        for i, stamp in enumerate(stamps):
            conn.execute(f"INSERT INTO {impl.LIFETIME_TABLE} "
                         "(lifetime_id,token_key,episode_index,first_seen_at,last_seen_at,observation_count) "
                         "VALUES(?,?,0,?,?,1)", (f"life{i}", str(i), stamp, stamp))
        frame = pd.DataFrame({"token_key": ["0", "1"],
                              "snapshot_at": pd.to_datetime(stamps, format="ISO8601", utc=True)})
        out = impl._attach_calendar_and_lifetime(conn, frame)
        assert out.lifetime_id.tolist() == ["life0", "life1"]
        conn.execute(f"UPDATE {impl.LIFETIME_TABLE} SET first_seen_at='invalid'")
        with pytest.raises(ValueError):
            impl._attach_calendar_and_lifetime(conn, frame)


def make_db(path, reverse=False):
    migrate(str(path))
    with closing(sqlite3.connect(path)) as conn, conn:
        for token in ([1, 0] if reverse else [0, 1]):
            for minute, price in enumerate((100., 150., 120.)):
                stamp = pd.Timestamp("2026-09-09T10:36:27Z") + pd.Timedelta(
                    minutes=minute, microseconds=123456 if token == 0 else 0)
                conn.execute("INSERT INTO axiom_observations "
                             "(token_key,snapshot_at,market_cap_usd,field_confidence_json,raw_ocr_json,source_json) "
                             "VALUES(?,?,?,'{}','{}','{}')", (str(token), stamp.isoformat(), price))


def make_mature_trusted_db(path):
    migrate(str(path))
    with closing(sqlite3.connect(path)) as conn, conn:
        start = pd.Timestamp("2026-08-01T00:00:00Z")
        conn.execute(
            """INSERT INTO collection_sessions
               (session_id,started_at,purpose,collector_schema)
               VALUES('session',?,'v24_production_raw_collection','v24_clipboard_raw_v2')""",
            (start.isoformat(),),
        )
        for day in range(16):
            for minute in range(30):
                stamp = start + pd.Timedelta(days=day, minutes=minute)
                created = stamp + pd.Timedelta(seconds=5)
                sha = f"sha-{day}-{minute}"
                cycle = conn.execute(
                    """INSERT INTO capture_cycles
                       (session_id,captured_at,clipboard_valid,rows_detected,completed,created_at)
                       VALUES('session',?,1,1,1,?)""",
                    (stamp.isoformat(), created.isoformat()),
                ).lastrowid
                conn.execute(
                    "INSERT INTO capture_payloads(cycle_id,sha256,byte_count,compression,payload) "
                    "VALUES(?,?,1,'zlib',?)",
                    (cycle, sha, b"x"),
                )
                conn.execute(
                    """INSERT INTO capture_attempts
                       (session_id,started_at,completed_at,success,clipboard_valid,rows_detected,
                        cycle_id,source,raw_payload_sha256,raw_payload_bytes)
                       VALUES('session',?,?,1,1,1,?,'interactive_clipboard',?,1)""",
                    (stamp.isoformat(), created.isoformat(), cycle, sha),
                )
                conn.execute(
                    """INSERT INTO axiom_observations
                       (cycle_id,token_key,snapshot_at,market_cap_usd,
                        field_confidence_json,raw_ocr_json,source_json,created_at)
                       VALUES(?,?,?,?, '{}','{}','{}',?)""",
                    (cycle, f"token-{day}", stamp.isoformat(), 100.0 + minute, created.isoformat()),
                )
def test_timestamp_check_read_only_and_missing_path(tmp_path):
    db = tmp_path / "raw.sqlite"
    make_db(db)
    before = db.read_bytes()
    assert preflight.main(["--db", str(db)]) == 0
    assert db.read_bytes() == before
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE axiom_observations SET snapshot_at='bad' WHERE rowid=1")
    assert preflight.main(["--db", str(db)]) == 1
    missing = tmp_path / "missing.sqlite"
    assert preflight.main(["--db", str(missing)]) == 1
    assert not missing.exists()


@pytest.mark.parametrize("profile", ["full", "first_model"])
def test_frame_check_runs_real_loader_without_training_or_source_writes(tmp_path, profile, capsys):
    from profit_taker import v24_contract_runtime_v4 as official
    db = tmp_path / "raw.sqlite"
    make_db(db)
    before = db.read_bytes()
    cohort = {"cohort_id": "test-promotion", "start_at": "2026-09-10T00:00:00+00:00"}
    eligibility = {"eligible_training_rows": 200}
    with patch.object(official.v24, "bootstrap_v24", side_effect=AssertionError("must not train")), \
         patch.object(impl, "fit_batch_bundle", side_effect=AssertionError("must not fit")), \
         patch.object(official.v24, "next_one_use_promotion_cohort", return_value=cohort), \
         patch.object(official.v24, "training_history_diagnostics", return_value=eligibility):
        assert preflight.main(["--db", str(db), "--stage", "frame", "--profile", profile]) == 0
    output = capsys.readouterr().out
    assert '"frame_rows": 6' in output
    assert '"sequence_rows": 6' in output
    assert '"tokens": 2' in output
    assert '"eligible_training_rows": 200' in output
    assert db.read_bytes() == before


def test_frame_check_refuses_zero_leakage_safe_training_rows(tmp_path, capsys):
    from profit_taker import v24_contract_runtime_v4 as official
    db = tmp_path / "raw.sqlite"
    make_db(db)
    cohort = {"cohort_id": "test-promotion", "start_at": "2026-09-10T00:00:00+00:00"}
    diagnostics = {
        "total_frame_rows": 6,
        "allowed_cohort_rows": 4,
        "mature_label_rows": 4,
        "vintage_known_rows": 0,
        "eligible_training_rows": 0,
    }
    with patch.object(official.v24, "next_one_use_promotion_cohort", return_value=cohort), \
         patch.object(official.v24, "training_history_diagnostics", return_value=diagnostics):
        assert preflight.main(["--db", str(db), "--stage", "frame", "--profile", "full"]) == 1
    output = capsys.readouterr().out
    assert '"passed": false' in output
    assert "Insufficient leakage-safe V24 training history before fitting" in output
    assert '\\"vintage_known_rows\\": 0' in output


def test_frame_check_accepts_reconstructed_canonical_vintage_end_to_end(tmp_path, capsys):
    db = tmp_path / "raw.sqlite"
    make_mature_trusted_db(db)

    assert preflight.main(["--db", str(db), "--stage", "frame", "--profile", "full"]) == 0
    output = capsys.readouterr().out
    assert '"passed": true' in output
    assert '"mature_promotion_available": true' in output
    assert '"eligible_training_rows": 270' in output
    assert '"canonical_prospective_insert": 270' in output


@pytest.mark.parametrize("reverse", [False, True])
def test_observation_loader_retains_both_timestamp_precisions(tmp_path, reverse):
    db = tmp_path / "raw.sqlite"
    make_db(db, reverse=reverse)
    with sqlite3.connect(db) as conn:
        obs, _ = impl.peak.load_observations(conn)
        assert len(obs) == 6
        assert set(obs.token_key) == {"0", "1"}
        assert obs[obs.token_key == "0"].snapshot_at.dt.microsecond.eq(123456).all()
        assert obs[obs.token_key == "1"].snapshot_at.dt.microsecond.eq(0).all()
