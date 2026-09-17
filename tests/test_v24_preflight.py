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
    with patch.object(official.v24, "bootstrap_v24", side_effect=AssertionError("must not train")), \
         patch.object(impl, "fit_batch_bundle", side_effect=AssertionError("must not fit")):
        assert preflight.main(["--db", str(db), "--stage", "frame", "--profile", profile]) == 0
    output = capsys.readouterr().out
    assert '"frame_rows": 6' in output
    assert '"sequence_rows": 6' in output
    assert '"tokens": 2' in output
    assert db.read_bytes() == before


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
