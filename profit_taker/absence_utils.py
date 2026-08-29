from __future__ import annotations

from typing import Any

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
