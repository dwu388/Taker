from __future__ import annotations

from typing import Any
import sqlite3

import pandas as pd


def verified_absence_minutes(run_start: Any, run_end: Any) -> float:
    """Minutes proven absent by a contiguous run of successful captures.

    A gap before ``run_start`` is collector downtime/censoring and must never be
    counted as token absence.
    """
    start = pd.Timestamp(run_start)
    end = pd.Timestamp(run_end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    return max(0.0, (end - start).total_seconds() / 60.0)


def collector_absence(
    conn: sqlite3.Connection, token_key: str, last_seen: Any, upto: Any,
    *, max_gap_minutes: float, minimum_captures: int, close_minutes: float,
) -> dict[str, Any]:
    """Verify absence against committed raw captures and their actual card identities.

    The first capture after downtime starts a new run. A later appearance, an
    invalid board, or a large capture gap breaks the run. Never infer absence
    from a forecast or a derived model heartbeat.
    """
    def utc(value: Any) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")

    seen, end = utc(last_seen), utc(upto)
    rows = conn.execute(
        """SELECT c.cycle_id,c.captured_at,c.rows_detected,c.completed,c.clipboard_valid,
                  EXISTS(SELECT 1 FROM axiom_observations o
                         WHERE o.cycle_id=c.cycle_id AND o.token_key=?) AS present,
                  (SELECT COUNT(*) FROM axiom_observations o
                   WHERE o.cycle_id=c.cycle_id) AS observed_rows
           FROM capture_cycles c
           WHERE datetime(c.captured_at)>=datetime(?) AND datetime(c.captured_at)<=datetime(?)
           ORDER BY c.cycle_id""", (token_key, seen.isoformat(), end.isoformat()),
        # Captures use ISO timestamps with varying fractions/UTC spellings.
        # SQLite datetime narrows the read; Python compares exact instants.
    ).fetchall()
    times = []
    for cycle_id, raw_time, detected, completed, valid, present, observed in rows:
        at = utc(raw_time)
        if seen < at <= end:
            times.append((at, int(cycle_id), int(detected), bool(completed and valid), bool(present), int(observed)))
    times.sort()
    start = previous = None
    count = 0
    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    for at, cycle_id, detected, valid, present, observed in times:
        # A successful capture without its matching identities cannot prove
        # which token was on the board. Fail closed on partial/ambiguous data.
        if not valid or detected <= 0 or present or observed != detected or (previous is not None and at - previous > max_gap):
            start = previous = None
            count = 0
        if not valid or detected <= 0 or present or observed != detected:
            if present:
                seen = at
            continue
        if start is None:
            start = seen if at - seen <= max_gap else at
        previous = at
        count += 1
    minutes = verified_absence_minutes(start, previous) if count else 0.0
    return {
        "verified_missing_minutes": minutes,
        "verified_absent_captures": count,
        "absence_run_start": start.isoformat() if start is not None else None,
        "absence_run_end": previous.isoformat() if previous is not None else None,
        "terminal": minutes >= close_minutes and count >= minimum_captures,
    }
