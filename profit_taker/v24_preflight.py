"""Staged checks for fresh V24 bootstrap. Never fits or publishes a model."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd


def _open_source(db):
    # mode=ro also prevents a misspelled path from creating an empty database.
    return sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)


def check_timestamps(db):
    counts = {}
    with closing(_open_source(db)) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'axiom_%'"
        )]
        if "axiom_observations" not in tables:
            raise ValueError("Missing axiom_observations table")
        rows = conn.execute("SELECT COUNT(*) FROM axiom_observations").fetchone()[0]
        if not rows:
            raise ValueError("axiom_observations is empty")
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            columns = [r[1] for r in conn.execute(f"PRAGMA table_info({quoted})")
                       if r[1].endswith("_at") or r[1] == "event_time"]
            for column in columns:
                col = '"' + column.replace('"', '""') + '"'
                cursor = conn.execute(f"SELECT {col} FROM {quoted} WHERE {col} IS NOT NULL")
                count = 0
                while chunk := cursor.fetchmany(10000):
                    values = pd.Series([r[0] for r in chunk])
                    try:
                        parsed = pd.to_datetime(values, format="ISO8601", utc=True)
                        if parsed.isna().any():
                            raise ValueError("non-null timestamp parsed as NaT")
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"Invalid timestamp in {table}.{column}: {exc}") from exc
                    count += len(chunk)
                counts[f"{table}.{column}"] = count
        if counts.get("axiom_observations.snapshot_at", 0) != rows:
            raise ValueError("Missing observation snapshot_at timestamps")
    return {"observations": rows, "timestamp_counts": counts}


def check_frame(db, args):
    from . import v24_contract_runtime_v4 as official
    v24 = official.v24
    runtime = official.runtime
    pcfg = official.contract.PretrainingConfig()
    runtime._install_recurrent_target_hash()
    runtime.shared.install_target_hash_contract(pcfg)
    cfg, _ = runtime._runtime_cfg(args, "bootstrap")
    if args.profile == "first_model":
        runtime._apply_first_model_profile(cfg, pcfg)
    # SQLite backup includes committed WAL data. Every derived write is isolated.
    with tempfile.TemporaryDirectory(prefix="taker-preflight-", dir=args.temp_dir) as tmp:
        copy = str(Path(tmp) / "preflight.sqlite")
        print("Copying database for isolated frame check...", flush=True)
        with closing(_open_source(db)) as source, closing(sqlite3.connect(copy)) as dest:
            source.backup(dest)
        print("Building labels and loading the training frame; no model fitting...", flush=True)
        peak_cfg = v24.peak.PeakStructureConfig(
            horizon_minutes=cfg.horizon_minutes,
            death_gap_minutes=cfg.operational_gap_minutes,
            death_missed_cycles=int(cfg.operational_gap_minutes),
            age_out_minutes=cfg.age_out_minutes,
        )
        v24.peak.refresh_labels(copy, peak_cfg)
        with closing(sqlite3.connect(copy)) as conn, conn:
            conn.row_factory = sqlite3.Row
            frame, seq, sources = v24.load_v24_frame(conn, cfg)
            if frame.empty:
                raise ValueError("Training frame is empty")
            cohort = v24.next_one_use_promotion_cohort(conn, cfg)
            return {"frame_rows": len(frame), "sequence_rows": len(seq),
                    "tokens": int(frame.token_key.nunique()), "sources": sources,
                    "mature_promotion_available": cohort is not None,
                    "note": "Frame check only; readiness, baselines and fitting remain unchecked."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/axiom_v24_raw.sqlite")
    parser.add_argument("--stage", choices=("timestamps", "frame"), default="timestamps")
    parser.add_argument("--profile", choices=("full", "first_model"), default="full")
    parser.add_argument("--temp-dir", help="Directory with space for a database copy (frame stage)")
    parser.add_argument("--cohort-hours", type=int, default=24)
    parser.add_argument("--promotion-every", type=int, default=4)
    parser.add_argument("--audit-every", type=int, default=5)
    parser.add_argument("--warmup-blocks", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        print("Checking stored ISO timestamps (read-only; no training)...", flush=True)
        result = check_timestamps(args.db)
        if args.stage == "frame":
            result["frame"] = check_frame(args.db, args)
        print(json.dumps({"passed": True, "stage": args.stage, **result}, indent=2))
        return 0
    except (ValueError, RuntimeError, sqlite3.Error, OSError, ImportError) as exc:
        print(json.dumps({"passed": False, "stage": args.stage, "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
