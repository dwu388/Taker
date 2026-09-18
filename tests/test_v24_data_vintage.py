from __future__ import annotations

import sqlite3
from contextlib import closing

import pandas as pd

from profit_taker import axiom_v24_impl as impl
from profit_taker.db import migrate


EVENT = "2026-09-01T12:00:00+00:00"
CREATED = "2026-09-01T12:00:30+00:00"


def _database(path, *, purpose: str = "v24_production_raw_collection") -> None:
    migrate(path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT INTO collection_sessions(session_id,started_at,purpose,collector_schema) "
            "VALUES('session',?,?, 'v24_clipboard_raw_v2')",
            (EVENT, purpose),
        )
        cycle = conn.execute(
            """INSERT INTO capture_cycles
               (session_id,captured_at,clipboard_valid,rows_detected,completed,created_at)
               VALUES('session',?,1,1,1,?)""",
            (EVENT, CREATED),
        ).lastrowid
        conn.execute(
            "INSERT INTO capture_payloads(cycle_id,sha256,byte_count,compression,payload) "
            "VALUES(?, 'payload-sha', 1, 'zlib', ?)",
            (cycle, b"x"),
        )
        conn.execute(
            """INSERT INTO capture_attempts
               (session_id,started_at,completed_at,success,clipboard_valid,rows_detected,
                cycle_id,source,raw_payload_sha256,raw_payload_bytes)
               VALUES('session',?,?,1,1,1,?,'interactive_clipboard','payload-sha',1)""",
            (EVENT, CREATED, cycle),
        )
        conn.execute(
            """INSERT INTO axiom_observations
               (cycle_id,token_key,snapshot_at,market_cap_usd,
                field_confidence_json,raw_ocr_json,source_json,created_at)
               VALUES(?,'token',?,100,'{}','{}','{}',?)""",
            (cycle, EVENT, CREATED),
        )


def _observations(conn: sqlite3.Connection) -> pd.DataFrame:
    observations, _ = impl.peak.load_observations(conn)
    return observations


def test_canonical_capture_reconstructs_original_vintage(tmp_path):
    db = str(tmp_path / "raw.sqlite")
    _database(db)
    with closing(sqlite3.connect(db)) as conn, conn:
        impl.migrate(conn)
        observations = _observations(conn)
        result = impl.refresh_data_vintage(conn, observations)
        row = conn.execute(
            f"SELECT first_ingested_at,last_corrected_at,ingestion_provenance "
            f"FROM {impl.DATA_VINTAGE_TABLE}"
        ).fetchone()

    assert result == {"inserted": 1, "corrected": 0, "reconstructed": 0}
    assert pd.Timestamp(row[0]) == pd.Timestamp(CREATED)
    assert pd.Timestamp(row[1]) == pd.Timestamp(CREATED)
    assert row[2] == "canonical_prospective_insert"


def test_existing_unknown_vintage_is_repaired_only_when_fingerprint_is_unchanged(tmp_path):
    db = str(tmp_path / "raw.sqlite")
    _database(db)
    with closing(sqlite3.connect(db)) as conn, conn:
        impl.migrate(conn)
        observations = _observations(conn)
        safe = [c for c in observations.columns if c != "_rowid_"]
        rec = dict(zip(safe, observations[safe].itertuples(index=False, name=None).__next__()))
        fingerprint = impl._stable_hash(rec)
        event = impl._iso(observations.iloc[0].snapshot_at)
        conn.execute(
            f"""INSERT INTO {impl.DATA_VINTAGE_TABLE}
                (token_key,event_time,first_ingested_at,last_corrected_at,
                 value_version,row_fingerprint,ingestion_provenance)
                VALUES('token',?,'2026-09-18T00:00:00+00:00',
                       '2026-09-18T00:00:00+00:00',1,?,'legacy_unknown_vintage')""",
            (event, fingerprint),
        )
        result = impl.refresh_data_vintage(conn, observations)
        row = conn.execute(
            f"SELECT first_ingested_at,last_corrected_at,value_version,ingestion_provenance "
            f"FROM {impl.DATA_VINTAGE_TABLE}"
        ).fetchone()

    assert result == {"inserted": 0, "corrected": 0, "reconstructed": 1}
    assert pd.Timestamp(row[0]) == pd.Timestamp(CREATED)
    assert pd.Timestamp(row[1]) == pd.Timestamp(CREATED)
    assert row[2] == 1
    assert row[3] == "canonical_prospective_reconstructed"


def test_replay_capture_cannot_manufacture_historical_vintage(tmp_path):
    db = str(tmp_path / "raw.sqlite")
    _database(db, purpose="v24_replay_experiment")
    with closing(sqlite3.connect(db)) as conn, conn:
        impl.migrate(conn)
        result = impl.refresh_data_vintage(conn, _observations(conn))
        row = conn.execute(
            f"SELECT first_ingested_at,ingestion_provenance FROM {impl.DATA_VINTAGE_TABLE}"
        ).fetchone()

    assert result == {"inserted": 1, "corrected": 0, "reconstructed": 0}
    assert pd.Timestamp(row[0]) > pd.Timestamp(CREATED)
    assert row[1] == "legacy_unknown_vintage"


def test_training_history_diagnostics_identify_vintage_and_exclusion_losses():
    with closing(sqlite3.connect(":memory:")) as conn, conn:
        impl.migrate(conn)
        conn.execute(
            f"""INSERT INTO {impl.COHORT_TABLE}
                (cohort_id,ordinal,start_at,end_at,role,status,created_at)
                VALUES('train',0,'2026-09-01T00:00:00+00:00',
                       '2026-09-02T00:00:00+00:00','train','available',?)""",
            (CREATED,),
        )
        frame = pd.DataFrame({
            "token_key": ["A", "B", "C"],
            "calendar_cohort_ordinal": [0, 0, 0],
            "snapshot_at": pd.to_datetime([
                "2026-09-01T01:00:00Z", "2026-09-01T02:00:00Z", "2026-09-01T03:00:00Z"
            ]),
            "label_interval_end": pd.to_datetime([
                "2026-09-02T01:00:00Z", "2026-09-02T02:00:00Z", "2026-09-02T03:00:00Z"
            ]),
            "first_ingested_at": [CREATED, CREATED, "2026-09-04T00:00:00+00:00"],
            "last_corrected_at": [CREATED, CREATED, "2026-09-04T00:00:00+00:00"],
            "ingestion_provenance": [
                "canonical_prospective_insert", "canonical_prospective_insert", "legacy_unknown_vintage"
            ],
        })
        diagnostics = impl.training_history_diagnostics(
            conn,
            frame,
            pd.Timestamp("2026-09-03T00:00:00Z"),
            impl.V24Config(promotion_embargo_hours=0),
            exclude_tokens={"B"},
        )

    assert diagnostics["allowed_cohort_rows"] == 3
    assert diagnostics["mature_label_rows"] == 3
    assert diagnostics["vintage_known_rows"] == 2
    assert diagnostics["excluded_evaluation_rows"] == 1
    assert diagnostics["eligible_training_rows"] == 1
