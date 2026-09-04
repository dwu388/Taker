from __future__ import annotations

"""Fast collector-heartbeat continuity lookup for the V24 pretraining contract.

The continuity contract itself is unchanged.  Production capture timestamps are
already returned by pretraining_contract._capture_times() as sorted, UTC-aware
pandas Timestamps.  The older V4 gap check nevertheless restarted at capture zero
and reconverted every timestamp for every token-path interval.  On a one-minute
collection this makes target refreshes scale catastrophically as capture history
grows.

This module installs an equivalent lookup that binary-searches directly to the
requested interval and only scans captures inside that interval.  Non-production
callers that pass an unnormalized sequence still get the historical defensive
normalization behavior.
"""

from bisect import bisect_right
from typing import Any, Sequence

import pandas as pd


def _production_ready_timeline(captures: Sequence[Any]) -> bool:
    """Recognize the tuple produced by V4's production capture context.

    _capture_times() already sorts and UTC-normalizes its results before V4 stores
    them as a tuple.  Checking only the endpoints keeps this test O(1); production
    construction is the source of the ordering invariant.
    """
    if not isinstance(captures, tuple) or not captures:
        return False
    first = captures[0]
    last = captures[-1]
    return (
        isinstance(first, pd.Timestamp)
        and isinstance(last, pd.Timestamp)
        and first.tzinfo is not None
        and last.tzinfo is not None
    )


def capture_gap_break(
    start: pd.Timestamp,
    end: pd.Timestamp,
    captures: Sequence[pd.Timestamp],
    max_gap: float,
    *,
    utc,
    gap_minutes,
) -> pd.Timestamp | None:
    """Return the last known-valid time before the first heartbeat gap.

    Semantics match the V4 linear scan, but the production path is
    O(log N + K) rather than O(N), where K is the number of collector captures
    actually inside ``(start, end]``.  For one-minute token paths K is normally
    zero or one regardless of total collection age.
    """
    start = utc(start)
    end = utc(end)
    if not captures or end <= start:
        return None

    if _production_ready_timeline(captures):
        timeline = captures
    else:
        timeline = tuple(sorted(utc(raw) for raw in captures))

    lo = bisect_right(timeline, start)
    hi = bisect_right(timeline, end, lo=lo)
    last = start
    for idx in range(lo, hi):
        current = timeline[idx]
        if gap_minutes(last, current) > max_gap:
            return last
        last = current
    if gap_minutes(last, end) > max_gap:
        return last
    return None


def install(contract):
    """Patch a loaded pretraining_contract_v4 module in place, idempotently."""
    current = getattr(contract, "_capture_gap_break")
    if getattr(current, "_v24_indexed_heartbeat_lookup", False):
        return contract

    def _indexed_capture_gap_break(start, end, captures, max_gap):
        return capture_gap_break(
            start,
            end,
            captures,
            max_gap,
            utc=contract._utc,
            gap_minutes=contract._gap_minutes,
        )

    _indexed_capture_gap_break.__name__ = "_capture_gap_break"
    _indexed_capture_gap_break.__doc__ = current.__doc__
    _indexed_capture_gap_break._v24_indexed_heartbeat_lookup = True
    contract._capture_gap_break = _indexed_capture_gap_break
    return contract
