from __future__ import annotations

"""Indexed/vectorized triple-barrier evaluation for V24 pretraining.

Production token frames are sorted once by the retained materializer.  The legacy
V4 evaluator rebuilt a pandas slice and iterated it in Python for every
(decision, horizon, up, down) cell.  This equivalent implementation caches the
prepared token arrays and decision/horizon continuity window on the frame, then
uses NumPy to locate barrier touches.  Target semantics and hashes are unchanged.
"""

import math
from typing import Any

import numpy as np
import pandas as pd

_CACHE_KEY = "_v24_fast_barrier_cache_v1"


def _timestamp(ns: int) -> pd.Timestamp:
    return pd.Timestamp(int(ns), tz="UTC")


def _prepared(g: pd.DataFrame) -> dict[str, Any]:
    cached = g.attrs.get(_CACHE_KEY)
    if isinstance(cached, dict):
        return cached
    times = pd.DatetimeIndex(pd.to_datetime(g["snapshot_at"], utc=True, errors="coerce"))
    times_ns = times.asi8
    prices = pd.to_numeric(g["market_cap_usd"], errors="coerce").to_numpy(dtype=float, copy=False)
    cached = {
        "times_ns": times_ns,
        "prices": prices,
        "windows": {},
        "hits": {},
    }
    g.attrs[_CACHE_KEY] = cached
    return cached


def _first_manual_censor(censors, decision: pd.Timestamp, deadline: pd.Timestamp):
    future = [c for c in censors if decision < c <= deadline]
    return min(future) if future else None


def barrier_outcome(contract, g, decision, horizon, up, down, censors,
                    capture_times=None, max_gap_minutes=None):
    decision = contract._utc(decision)
    max_gap = float(contract._active_max_gap.get() if max_gap_minutes is None else max_gap_minutes)
    captures = tuple(contract._active_capture_times.get() if capture_times is None else capture_times)
    data = _prepared(g)
    times_ns = data["times_ns"]
    prices = data["prices"]
    if not len(times_ns):
        return contract._censored("no_next_observation")

    wkey = (int(decision.value), int(horizon), id(censors), max_gap, id(captures))
    window = data["windows"].get(wkey)
    if window is None:
        ref_idx = int(np.searchsorted(times_ns, int(decision.value), side="right"))
        if ref_idx >= len(times_ns):
            window = {"early": contract._censored("no_next_observation")}
            data["windows"][wkey] = window
            return dict(window["early"])

        ref_at = _timestamp(times_ns[ref_idx])
        ref_mc = float(prices[ref_idx])
        if not math.isfinite(ref_mc) or ref_mc <= 0:
            window = {"early": contract._censored("invalid_reference_price", reference_at=ref_at)}
            data["windows"][wkey] = window
            return dict(window["early"])
        if contract._gap_minutes(decision, ref_at) > max_gap:
            window = {"early": contract._censored(
                "next_observation_too_late", event_at=ref_at,
                reference_at=ref_at, reference_mc=ref_mc,
            )}
            data["windows"][wkey] = window
            return dict(window["early"])
        if contract._capture_gap_break(decision, ref_at, captures, max_gap) is not None:
            window = {"early": contract._censored(
                "capture_gap_before_executable_reference", event_at=ref_at,
                reference_at=ref_at, reference_mc=ref_mc,
            )}
            data["windows"][wkey] = window
            return dict(window["early"])

        deadline = decision + pd.Timedelta(minutes=int(horizon))
        censor_at = _first_manual_censor(censors, decision, deadline)
        end = min(deadline, censor_at) if censor_at is not None else deadline
        if ref_at > end:
            window = {"early": contract._censored(
                "reference_after_window_end", event_at=end,
                reference_at=ref_at, reference_mc=ref_mc,
            )}
            data["windows"][wkey] = window
            return dict(window["early"])

        end_idx = int(np.searchsorted(times_ns, int(end.value), side="right")) - 1
        if end_idx < ref_idx:
            window = {"early": contract._censored(
                "empty_observed_path", reference_at=ref_at, reference_mc=ref_mc
            )}
            data["windows"][wkey] = window
            return dict(window["early"])

        break_idx = None
        break_reason = None

        segment_prices = prices[ref_idx:end_idx + 1]
        invalid = np.flatnonzero(~np.isfinite(segment_prices) | (segment_prices <= 0))
        if len(invalid):
            break_idx = ref_idx + int(invalid[0])
            break_reason = "invalid_path_price"

        if end_idx > ref_idx:
            deltas = np.diff(times_ns[ref_idx:end_idx + 1])
            token_gaps = np.flatnonzero(deltas > int(max_gap * 60.0 * 1_000_000_000.0))
            if len(token_gaps):
                idx = ref_idx + int(token_gaps[0]) + 1
                if break_idx is None or idx < break_idx or (idx == break_idx and break_reason != "invalid_path_price"):
                    break_idx = idx
                    break_reason = "token_observation_gap"

            # Preserve V4's exact heartbeat semantics: the legacy evaluator checks
            # collector continuity separately for each adjacent observed-price
            # interval, after the token-gap check and before evaluating that row's
            # barrier touches.  Checking the entire ref->path_end interval at once
            # is stronger and can invent a heartbeat censor in synthetic/inconsistent
            # data where observed prices remain continuous.  The indexed heartbeat
            # helper keeps each adjacency check O(log N + K), and this window is
            # cached once per (decision, horizon), so we retain the speedup without
            # changing target labels.
            for idx in range(ref_idx + 1, end_idx + 1):
                if break_idx is not None and idx >= break_idx:
                    break
                prev_at = _timestamp(times_ns[idx - 1])
                current_at = _timestamp(times_ns[idx])
                if contract._capture_gap_break(prev_at, current_at, captures, max_gap) is not None:
                    break_idx = idx
                    break_reason = "capture_heartbeat_gap"
                    break

        valid_end_idx = end_idx if break_idx is None else break_idx - 1
        window = {
            "early": None,
            "ref_idx": ref_idx,
            "ref_at": ref_at,
            "ref_mc": ref_mc,
            "deadline": deadline,
            "censor_at": censor_at,
            "end_idx": end_idx,
            "valid_end_idx": valid_end_idx,
            "break_idx": break_idx,
            "break_reason": break_reason,
        }
        data["windows"][wkey] = window

    if window.get("early") is not None:
        return dict(window["early"])

    ref_idx = int(window["ref_idx"])
    valid_end_idx = int(window["valid_end_idx"])
    ref_at = window["ref_at"]
    ref_mc = float(window["ref_mc"])

    def first_hit(kind: str, threshold: float):
        hkey = (wkey, kind, float(threshold))
        if hkey in data["hits"]:
            return data["hits"][hkey]
        if valid_end_idx < ref_idx:
            hit = None
        else:
            vals = prices[ref_idx:valid_end_idx + 1]
            mask = vals >= threshold if kind == "up" else vals <= threshold
            found = np.flatnonzero(mask)
            hit = None if not len(found) else ref_idx + int(found[0])
        data["hits"][hkey] = hit
        return hit

    up_idx = first_hit("up", ref_mc * (1.0 + float(up)))
    down_idx = first_hit("down", ref_mc * (1.0 + float(down)))
    if up_idx is not None or down_idx is not None:
        if up_idx is not None and down_idx is not None and up_idx == down_idx:
            idx = up_idx
            return contract._censored(
                "same_snapshot_opposing_touches", event_at=_timestamp(times_ns[idx]),
                reference_at=ref_at, reference_mc=ref_mc, terminal_mc=float(prices[idx]),
            )
        if down_idx is None or (up_idx is not None and up_idx < down_idx):
            idx = int(up_idx)
            mc = float(prices[idx])
            return {
                "outcome": "up_first", "event_at": _timestamp(times_ns[idx]),
                "reference_at": ref_at, "reference_mc": ref_mc, "terminal_mc": mc,
                "gross_return": mc / ref_mc - 1.0, "target_ready_at": _timestamp(times_ns[idx]),
            }
        idx = int(down_idx)
        mc = float(prices[idx])
        return {
            "outcome": "down_first", "event_at": _timestamp(times_ns[idx]),
            "reference_at": ref_at, "reference_mc": ref_mc, "terminal_mc": mc,
            "gross_return": mc / ref_mc - 1.0, "target_ready_at": _timestamp(times_ns[idx]),
        }

    break_idx = window["break_idx"]
    terminal_idx = max(ref_idx, valid_end_idx)
    terminal_mc = float(prices[terminal_idx])
    if break_idx is not None:
        return contract._censored(
            str(window["break_reason"]), event_at=_timestamp(times_ns[int(break_idx)]),
            reference_at=ref_at, reference_mc=ref_mc, terminal_mc=terminal_mc,
        )

    censor_at = window["censor_at"]
    if censor_at is not None:
        return contract._censored(
            "manual_stop", event_at=censor_at, reference_at=ref_at,
            reference_mc=ref_mc, terminal_mc=terminal_mc,
        )

    deadline = window["deadline"]
    prev = _timestamp(times_ns[int(window["end_idx"])])
    if contract._gap_minutes(prev, deadline) > max_gap:
        return contract._censored(
            "token_observation_gap_before_horizon_completion", event_at=deadline,
            reference_at=ref_at, reference_mc=ref_mc, terminal_mc=terminal_mc,
        )
    if contract._capture_gap_break(prev, deadline, captures, max_gap) is not None:
        return contract._censored(
            "capture_heartbeat_gap_before_horizon_completion", event_at=deadline,
            reference_at=ref_at, reference_mc=ref_mc, terminal_mc=terminal_mc,
        )
    return {
        "outcome": "neither", "event_at": deadline, "reference_at": ref_at,
        "reference_mc": ref_mc, "terminal_mc": terminal_mc,
        "gross_return": terminal_mc / ref_mc - 1.0, "target_ready_at": deadline,
    }


def install(contract):
    """Patch V4 and the retained V1 materializer to use the fast evaluator."""
    current = getattr(contract, "_barrier_outcome")
    if getattr(current, "_v24_indexed_barrier_lookup", False):
        return contract

    def _fast(g, decision, horizon, up, down, censors,
              capture_times=None, max_gap_minutes=None):
        return barrier_outcome(
            contract, g, decision, horizon, up, down, censors,
            capture_times=capture_times, max_gap_minutes=max_gap_minutes,
        )

    _fast.__name__ = "_barrier_outcome"
    _fast.__doc__ = current.__doc__
    _fast._v24_indexed_barrier_lookup = True
    contract._barrier_outcome = _fast
    contract._v1._barrier_outcome = _fast
    return contract
