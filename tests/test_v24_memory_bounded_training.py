from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from profit_taker import axiom_peak_structure_impl as peak_impl
from profit_taker import axiom_v24_impl as impl


def _timestamps(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-09-01T00:00:00Z", periods=n, freq="min")


def test_sequence_cache_load_is_lazy_and_decodes_only_requested_rows(monkeypatch):
    with sqlite3.connect(":memory:") as conn:
        impl.migrate(conn)
        cfg = impl.V24Config(sequence_windows_minutes=(60,), sequence_segments=1)
        observations = pd.DataFrame({
            "token_key": ["A", "A", "B"],
            "snapshot_at": _timestamps(3),
            "market_cap_usd": [100.0, 110.0, 120.0],
        })
        feature = "seqraw__market_cap_usd__60m__points"
        for row in observations.itertuples(index=False):
            conn.execute(
                f"INSERT INTO {impl.SEQUENCE_CACHE_TABLE}"
                "(token_key,snapshot_at,fingerprint_json,config_hash,created_at) "
                "VALUES(?,?,?,?,?)",
                (
                    row.token_key,
                    impl._iso(row.snapshot_at),
                    json.dumps({feature: float(row.market_cap_usd)}),
                    "test",
                    impl._now_iso(),
                ),
            )
        conn.commit()

        monkeypatch.setattr(impl, "refresh_sequence_fingerprint_cache", lambda *_args, **_kwargs: {})
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        source = impl.load_sequence_fingerprint_cache(conn, observations, cfg)

        assert isinstance(source, impl.SequenceFingerprintCache)
        assert len(source) == 3
        assert not any("fingerprint_json" in sql.lower() for sql in statements)

        calls = 0
        original_loads = impl._loads

        def counting_loads(value):
            nonlocal calls
            calls += 1
            return original_loads(value)

        monkeypatch.setattr(impl, "_loads", counting_loads)
        requested = observations.iloc[[1]][["token_key", "snapshot_at"]]
        raw = impl._sequence_raw_for_keys(source, requested)

        assert calls == 1
        assert raw.token_key.tolist() == ["A"]
        assert raw.snapshot_at.tolist() == [pd.Timestamp("2026-09-01T00:01:00Z")]
        assert raw[feature].dtype == np.float32
        assert float(raw.iloc[0][feature]) == 110.0


def test_lazy_sequence_encoding_matches_eager_encoding(monkeypatch):
    with sqlite3.connect(":memory:") as conn:
        impl.migrate(conn)
        cfg = impl.V24Config(
            sequence_windows_minutes=(60,),
            sequence_segments=1,
            sequence_components=2,
        )
        rows = []
        feature_a = "seqraw__market_cap_usd__60m__points"
        feature_b = "seqraw__market_cap_usd__60m__change"
        for token_no, token in enumerate(("A", "B")):
            for minute, stamp in enumerate(_timestamps(5)):
                rows.append({
                    "token_key": token,
                    "snapshot_at": stamp,
                    feature_a: float(minute + 1 + token_no),
                    feature_b: float((minute + 1) * (token_no + 1)) / 10.0,
                })
        eager = pd.DataFrame(rows)
        observations = eager[["token_key", "snapshot_at"]].copy()
        observations["market_cap_usd"] = 100.0
        for row in eager.itertuples(index=False):
            payload = {feature_a: getattr(row, feature_a), feature_b: getattr(row, feature_b)}
            conn.execute(
                f"INSERT INTO {impl.SEQUENCE_CACHE_TABLE}"
                "(token_key,snapshot_at,fingerprint_json,config_hash,created_at) VALUES(?,?,?,?,?)",
                (row.token_key, impl._iso(row.snapshot_at), json.dumps(payload), "test", impl._now_iso()),
            )
        conn.commit()
        monkeypatch.setattr(impl, "refresh_sequence_fingerprint_cache", lambda *_args, **_kwargs: {})
        source = impl.load_sequence_fingerprint_cache(conn, observations, cfg)

        encoder = impl.fit_sequence_encoder(eager, cfg)
        expected = impl.apply_sequence_encoder(eager, encoder).sort_values(
            ["token_key", "snapshot_at"]
        ).reset_index(drop=True)
        actual = impl._encode_sequence_keys(
            source, eager[["token_key", "snapshot_at"]], encoder
        ).sort_values(["token_key", "snapshot_at"]).reset_index(drop=True)

        assert actual[["token_key", "snapshot_at"]].equals(expected[["token_key", "snapshot_at"]])
        np.testing.assert_allclose(
            actual.filter(like="seqenc__").to_numpy(),
            expected.filter(like="seqenc__").to_numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


def test_bounded_training_rows_are_deterministic_balanced_and_cover_endpoints():
    rows = []
    for token, count in (("A", 10), ("B", 5), ("C", 2)):
        for minute, stamp in enumerate(_timestamps(count)):
            rows.append({"token_key": token, "snapshot_at": stamp, "value": minute})
    frame = pd.DataFrame(rows)
    cfg = impl.V24Config(model_max_training_rows=7, model_max_rows_per_token=4)

    first = impl._bounded_model_training_rows(frame.sample(frac=1.0, random_state=1), cfg)
    second = impl._bounded_model_training_rows(frame.sample(frac=1.0, random_state=2), cfg)

    keys = ["token_key", "snapshot_at"]
    assert len(first) == 7
    assert set(first.token_key) == {"A", "B", "C"}
    assert first[keys].equals(second[keys])
    for token, group in frame.groupby("token_key"):
        sampled = first[first.token_key == token]
        assert sampled.snapshot_at.min() == group.snapshot_at.min()
        assert sampled.snapshot_at.max() == group.snapshot_at.max()


def test_shared_grid_expansion_keeps_only_fit_columns_and_float32_features():
    data = pd.DataFrame({
        "token_key": ["A", "B", "C"],
        "feature": [1.0, 2.0, 3.0],
        "target_a": [0, 1, 1],
        "target_b": [1, 0, np.nan],
        "unused_large_column": ["x" * 1000] * 3,
    })
    long, features = impl._shared_grid_long(
        data,
        ["feature"],
        [("target_a", 0.5, 60.0), ("target_b", 1.0, 240.0)],
        "joint",
    )

    assert len(long) == 5
    assert "unused_large_column" not in long
    assert "target_a" not in long and "target_b" not in long
    assert long["feature"].dtype == np.float32
    assert features == ["feature", "joint__a", "joint__log_b", "joint__a_log_b"]


def test_model_memory_errors_are_never_silently_dropped(monkeypatch):
    class ExhaustedModel:
        def fit(self, *_args, **_kwargs):
            raise MemoryError("synthetic exhaustion")

    data = pd.DataFrame({
        "token_key": [str(i % 3) for i in range(24)],
        "feature": np.arange(24, dtype=float),
        "target": [i % 2 for i in range(24)],
    })
    monkeypatch.setattr(impl, "_binary_components", lambda _n: [("exhausted", ExhaustedModel())])
    with pytest.raises(MemoryError, match="synthetic exhaustion"):
        impl._fit_binary_head(data, ["feature"], "target", 10)


def test_legacy_json_feature_enrichment_is_selected_and_decoded_in_chunks(monkeypatch):
    with sqlite3.connect(":memory:") as conn:
        conn.execute(
            "CREATE TABLE legacy_v18_features ("
            "token_key TEXT, snapshot_at TEXT, features_json TEXT, unused_payload TEXT)"
        )
        conn.executemany(
            "INSERT INTO legacy_v18_features VALUES(?,?,?,?)",
            [
                ("A", impl._iso(stamp), json.dumps({"feature": index}), "x" * 1000)
                for index, stamp in enumerate(_timestamps(3))
            ],
        )
        conn.commit()
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        chunksizes: list[int | None] = []
        original_read_sql = pd.read_sql_query

        def recording_read_sql(*args, **kwargs):
            chunksizes.append(kwargs.get("chunksize"))
            return original_read_sql(*args, **kwargs)

        monkeypatch.setattr(pd, "read_sql_query", recording_read_sql)
        features = peak_impl._load_existing_features(conn)

        assert chunksizes == [2048]
        select = next(sql for sql in statements if "FROM \"legacy_v18_features\"" in sql)
        assert "unused_payload" not in select
        assert features is not None
        assert features["feature"].dtype == np.float32
        assert features["feature"].tolist() == [0.0, 1.0, 2.0]


def test_load_v24_frame_reuses_the_already_loaded_observations(monkeypatch):
    with sqlite3.connect(":memory:") as conn:
        impl.migrate(conn)
        stamp = pd.Timestamp("2026-09-01T00:00:00Z")
        observations = pd.DataFrame({
            "token_key": ["A"],
            "snapshot_at": [stamp],
            "market_cap_usd": [100.0],
        })
        source = {"table": "axiom_observations", "token": "token_key", "time": "snapshot_at", "mc": "market_cap_usd"}
        calls = 0

        def load_observations(_conn):
            nonlocal calls
            calls += 1
            return observations, source

        frame = pd.DataFrame({
            "token_key": ["A"],
            "snapshot_at": [stamp],
            "decision_at": [stamp],
            "path_end_at": [stamp],
        })

        monkeypatch.setattr(impl.peak, "load_observations", load_observations)
        monkeypatch.setattr(impl, "refresh_capture_heartbeats", lambda _c, obs: {"rows": len(obs)})
        monkeypatch.setattr(impl, "refresh_data_vintage", lambda _c, obs: {"rows": len(obs)})
        monkeypatch.setattr(impl, "refresh_lifetimes", lambda _c, _cfg, observations=None: {"rows": len(observations)})
        monkeypatch.setattr(impl, "refresh_calendar_cohorts", lambda *_args: {})
        monkeypatch.setattr(impl, "refresh_policy_cohorts", lambda *_args: {})
        monkeypatch.setattr(impl, "refresh_token_assignments", lambda *_args: {})
        monkeypatch.setattr(impl, "_attach_calendar_and_lifetime", lambda _c, value: value)

        def load_training_frame(_conn, observations=None, observation_source=None):
            assert observations is not None and observations is globals_observations
            assert observation_source == source
            return frame.copy(), "cache", source

        globals_observations = observations
        monkeypatch.setattr(impl.peak, "load_training_frame", load_training_frame)
        lazy = impl.SequenceFingerprintCache(conn, 1, ())
        monkeypatch.setattr(impl, "load_sequence_fingerprint_cache", lambda *_args: lazy)

        loaded, sequence, _ = impl.load_v24_frame(conn, impl.V24Config())

        assert calls == 1
        assert len(loaded) == 1
        assert sequence is lazy
