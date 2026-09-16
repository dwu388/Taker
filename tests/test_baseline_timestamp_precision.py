import sqlite3

import pandas as pd
import pytest

from profit_taker import pretraining_contract as base
from profit_taker import pretraining_contract_v5 as contract


@pytest.mark.parametrize("fraction_first", [True, False])
@pytest.mark.parametrize("parsed_observations", [True, False])
def test_baselines_accept_mixed_iso_precision_without_changing_results(tmp_path, monkeypatch, fraction_first, parsed_observations):
    db = str(tmp_path / "baseline.sqlite")
    observations = []
    with sqlite3.connect(db) as conn:
        base.migrate(conn)
        for i in range(10):
            start = pd.Timestamp("2026-08-30T01:38:27Z") + pd.Timedelta(hours=i)
            fractional = (i % 2 == 0) == fraction_first
            decision = start + pd.Timedelta(minutes=1, microseconds=123456 if fractional else 0)
            for stamp, price in ((start, 100.), (decision, 101. + i)):
                observations.append({"token_key": str(i), "snapshot_at": stamp.isoformat(), "market_cap_usd": price})
            conn.execute(
                f"""INSERT INTO {base.TARGET_TABLE}
                (token_key,decision_at,target_kind,horizon_minutes,up_barrier,down_barrier,
                 outcome,target_definition_hash,details_json)
                VALUES(?,?,'triple_barrier',60,0.3,-0.2,?,'test','{{}}')""",
                (str(i), decision.isoformat(), "up_first" if i % 2 else "down_first"),
            )
    obs = pd.DataFrame(observations)
    if parsed_observations:
        obs["snapshot_at"] = pd.to_datetime(obs.snapshot_at, format="ISO8601", utc=True)
    monkeypatch.setattr(base.peak, "load_observations", lambda conn: (obs.copy(), "test"))
    mixed = contract.evaluate_baselines(db, refresh_targets=False)
    assert mixed["available"]
    assert mixed["split"] == {"train_tokens": 7, "eval_tokens": 3}
    assert mixed["age_bucket"]["rows"] == 3

    # Uniform formatting represents exactly the same instants and must yield
    # identical chronological splits, causal histories, and evaluation metrics.
    obs["snapshot_at"] = [pd.Timestamp(x).isoformat(timespec="microseconds") for x in obs.snapshot_at]
    with sqlite3.connect(db) as conn:
        for rowid, stamp in conn.execute(f"SELECT rowid,decision_at FROM {base.TARGET_TABLE}").fetchall():
            conn.execute(f"UPDATE {base.TARGET_TABLE} SET decision_at=? WHERE rowid=?",
                         (pd.Timestamp(stamp).isoformat(timespec="microseconds"), rowid))
    assert contract.evaluate_baselines(db, refresh_targets=False) == mixed

    # Invalid timestamps must still stop evaluation rather than disappear as NaT.
    with sqlite3.connect(db) as conn:
        conn.execute(f"UPDATE {base.TARGET_TABLE} SET decision_at='invalid' WHERE token_key='9'")
    with pytest.raises(ValueError):
        contract.evaluate_baselines(db, refresh_targets=False)
