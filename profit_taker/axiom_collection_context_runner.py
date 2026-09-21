from __future__ import annotations

"""Production collector entrypoint with additive capture-context instrumentation.

The hardened collector remains authoritative for raw persistence.  This wrapper
measures only wall-clock stages around that existing path, then writes diagnostics
in a separate table after the raw cycle is durable.  A context-write failure is
reported but can never roll back or reclassify the raw capture.
"""

import json
import sqlite3
import sys
import time
from typing import Any

from . import axiom_migrated_runner as runner

CONTEXT_TABLE = "axiom_v24_capture_context"
BOARD_TABLE = "axiom_v24_board_membership"

_last_capture_control_ms: float | None = None


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {CONTEXT_TABLE} (
            cycle_id INTEGER PRIMARY KEY,
            captured_at TEXT NOT NULL,
            visible_tokens INTEGER NOT NULL,
            new_tokens_this_cycle INTEGER NOT NULL,
            full_mint_tokens INTEGER NOT NULL,
            short_identity_tokens INTEGER NOT NULL,
            capture_control_latency_ms REAL,
            post_capture_processing_latency_ms REAL,
            board_layout_version TEXT,
            details_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS {BOARD_TABLE} (
            cycle_id INTEGER NOT NULL,
            token_key TEXT NOT NULL,
            board_position_1based INTEGER NOT NULL,
            board_visible_count INTEGER NOT NULL,
            board_position_fraction REAL NOT NULL,
            PRIMARY KEY(cycle_id,token_key)
        );
        CREATE INDEX IF NOT EXISTS idx_{BOARD_TABLE}_token_cycle ON {BOARD_TABLE}(token_key,cycle_id);
        """
    )
    conn.commit()


def _timed_capture(cfg: dict, cycle_count: int) -> tuple[str, str]:
    global _last_capture_control_ms
    started = time.perf_counter()
    try:
        return _original_capture(cfg, cycle_count)
    finally:
        _last_capture_control_ms = (time.perf_counter() - started) * 1000.0


def _record_context(db: str, cycle_id: int, post_ms: float) -> dict[str, Any]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        migrate(conn)
        cycle = conn.execute("SELECT captured_at,rows_detected FROM capture_cycles WHERE cycle_id=?", (cycle_id,)).fetchone()
        if cycle is None:
            raise RuntimeError(f"capture context cannot find durable cycle {cycle_id}")
        rows = conn.execute(
            "SELECT token_key,token_address,source_json FROM axiom_observations WHERE cycle_id=? ORDER BY observation_id",
            (cycle_id,),
        ).fetchall()
        visible = len(rows)
        positions: list[tuple[str,int]] = []
        for fallback, r in enumerate(rows, start=1):
            pos = fallback
            try:
                src = json.loads(r["source_json"] or "{}")
                raw = ((src.get("clipboard_only") or {}).get("clipboard_card"))
                if raw is not None:
                    pos = int(raw) + 1
            except Exception:
                pass
            positions.append((str(r["token_key"]), max(1, pos)))
        current = {str(r["token_key"]) for r in rows}
        prior = {
            str(r[0]) for r in conn.execute(
                "SELECT DISTINCT token_key FROM axiom_observations WHERE cycle_id < ?", (cycle_id,)
            ).fetchall()
        }
        new_count = len(current - prior)
        full = sum(1 for r in rows if r["token_address"])
        short = visible - full
        layout = "v24-clipboard-only-25h-view"
        details = {
            "latency_semantics": {
                "capture_control_latency_ms": "wall clock inside browser/clipboard capture macro including configured waits",
                "post_capture_processing_latency_ms": "wall clock after clipboard return through parsing, atomic persistence and optional manual-review artifacts",
            },
            "rows_detected": int(cycle["rows_detected"]),
        }
        conn.execute(
            f"""INSERT OR REPLACE INTO {CONTEXT_TABLE}
            (cycle_id,captured_at,visible_tokens,new_tokens_this_cycle,full_mint_tokens,short_identity_tokens,
             capture_control_latency_ms,post_capture_processing_latency_ms,board_layout_version,details_json)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (cycle_id, str(cycle["captured_at"]), visible, new_count, full, short,
             _last_capture_control_ms, float(post_ms), layout, json.dumps(details, sort_keys=True)),
        )
        conn.execute(f"DELETE FROM {BOARD_TABLE} WHERE cycle_id=?", (cycle_id,))
        denom = max(1, visible - 1)
        for token, pos in positions:
            fraction = 0.0 if visible <= 1 else float(pos - 1) / float(denom)
            conn.execute(
                f"INSERT INTO {BOARD_TABLE}(cycle_id,token_key,board_position_1based,board_visible_count,board_position_fraction) VALUES(?,?,?,?,?)",
                (cycle_id, token, pos, visible, fraction),
            )
        conn.commit()
        return {
            "cycle_id": cycle_id,
            "visible_tokens": visible,
            "new_tokens_this_cycle": new_count,
            "full_mint_tokens": full,
            "short_identity_tokens": short,
            "capture_control_latency_ms": _last_capture_control_ms,
            "post_capture_processing_latency_ms": float(post_ms),
        }


def _timed_run_once(*args: Any, **kwargs: Any) -> dict:
    started = time.perf_counter()
    result = _original_run_once(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    post_ms = max(0.0, elapsed_ms - float(_last_capture_control_ms or 0.0))
    collector_args = args[0] if args else kwargs.get("args")
    if collector_args is not None and result.get("cycle_id") is not None:
        try:
            result["capture_context"] = _record_context(str(collector_args.db), int(result["cycle_id"]), post_ms)
        except Exception as exc:
            result["capture_context"] = {"stored": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


_original_capture = runner._capture_clipboard
_original_run_once = runner.run_once
runner._capture_clipboard = _timed_capture
runner._impl._capture_clipboard = _timed_capture
runner.run_once = _timed_run_once
runner._impl.run_once = _timed_run_once


def main() -> None:
    # runner.main handles session provenance, reports, one-minute cadence and the
    # durable Ctrl+C neutral-censor boundary.
    runner.main()


if __name__ == "__main__":
    main()
