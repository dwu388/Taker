import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from profit_taker import v24_contract_runtime_v4 as official
from profit_taker import pretraining_contract_v5 as contract
from profit_taker import pretraining_contract_v2 as counts
from profit_taker.db import migrate


@pytest.mark.usefixtures("closed_sqlite_connections")
class FreshBootstrapReadinessTests(unittest.TestCase):
    def test_raw_only_bootstrap_materializes_gate_evidence_before_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "raw.sqlite")
            migrate(db)
            with closing(sqlite3.connect(db)) as conn, conn:
                for day in range(16):
                    for minute, mc in enumerate((100., 150., 120.)):
                        stamp = pd.Timestamp("2026-08-01", tz="UTC") + pd.Timedelta(days=day, minutes=minute)
                        conn.execute(
                            """INSERT INTO axiom_observations
                            (token_key,short_address_hint,snapshot_at,market_cap_usd,
                             field_confidence_json,raw_ocr_json,source_json)
                            VALUES(?,?,?,?,'{}','{}','{}')""",
                            (f"token{day}", f"token{day}", stamp.isoformat(), mc),
                        )

            class ReachedReadiness(Exception):
                pass

            def inspect(db_path, cfg):
                with closing(sqlite3.connect(db_path)) as conn, conn:
                    peaks = conn.execute("SELECT COUNT(DISTINCT token_key) FROM axiom_peak_events_v21").fetchone()[0]
                    self.assertEqual(peaks, 16)
                    self.assertGreaterEqual(counts._mature_development_blocks(
                        conn, pd.Timestamp("2026-08-16T00:02:00Z")), 6)
                    roles = dict(conn.execute(
                        "SELECT forecast_role, COUNT(*) FROM axiom_v24_token_assignment GROUP BY forecast_role"))
                    self.assertGreater(roles["train"], 0)
                    self.assertGreater(roles["promotion"], 0)
                    self.assertEqual(sum(roles.values()), 16)
                    self.assertEqual(conn.execute(
                        "SELECT COUNT(*) FROM axiom_v24_calendar_cohorts WHERE status != 'available' AND role != 'audit'"
                    ).fetchone()[0], 0)
                raise ReachedReadiness

            # Exercise the real label/cohort preparation; skip unrelated costly
            # barrier materialization and stop before any model training.
            with patch.object(official.runtime, "_pretraining_for_command", return_value={}), \
                 patch.object(official.contract, "assert_training_ready", side_effect=inspect), \
                 patch.object(official.runtime.v24, "bootstrap_v24") as fit:
                for profile in ("full", "first_model"):
                    with self.assertRaises(ReachedReadiness):
                        official.main(["bootstrap", "--db", db, "--profile", profile])
                fit.assert_not_called()

    def test_genuine_shortage_still_refused_with_counts(self):
        report = {"ready": False, "gates": {
            "confirmed_peak_tokens": {"value": 16, "minimum": 50, "pass": False},
            "mature_development_blocks": {"value": 2, "minimum": 6, "pass": False},
            "operational_death_tokens": {"value": 0, "minimum": 50, "pass": False},
        }}
        with patch.object(contract, "training_readiness", return_value=report):
            with self.assertRaises(RuntimeError) as error:
                contract.assert_training_ready("unused")
        self.assertIn("confirmed_peak_tokens (value=16, minimum=50)", str(error.exception))
        self.assertIn("mature_development_blocks (value=2, minimum=6)", str(error.exception))
        self.assertNotIn("operational_death_tokens", str(error.exception))


if __name__ == "__main__":
    unittest.main()
