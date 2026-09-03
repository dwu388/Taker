from __future__ import annotations

"""Latest pretraining contract: continuity-safe price targets.

V3 remains the compatibility/idempotence layer.  This layer tightens the price
truth contract so a missing token path or collector heartbeat cannot silently turn
into a resolved price outcome.  It also advances the target-definition hash so
old labels cannot be mistaken for labels produced under the stricter contract.
"""

import hashlib
import json
import math
import sqlite3
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from . import pretraining_contract as _v1
from . import pretraining_contract_v2 as _v2
from . import pretraining_contract_v3 as _compat

# Re-export the public V3 surface without copying private forwarding sentinels.
for _name in dir(_compat):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_compat, _name)

PRETRAINING_SCHEMA_VERSION = "v24_pretraining_contract_v4_continuity"

_original_payload = _compat.target_contract_payload
_original_refresh = _compat.refresh_pretraining_targets
_original_readiness = _compat.training_readiness
_original_baselines = _compat.evaluate_baselines
_original_audit = _compat.collection_audit

_active_capture_times: ContextVar[tuple[pd.Timestamp, ...]] = ContextVar(
    "v24_pretraining_capture_times", default=()
)
_active_max_gap: ContextVar[float] = ContextVar(
    "v24_pretraining_max_price_gap", default=5.0
)


@dataclass(frozen=True)
class PretrainingConfig(_compat.PretrainingConfig):
    # Price-dependent labels may bridge only short observation gaps.  A longer
    # gap represents operational disappearance or collection outage and must
    # censor the price target rather than assume an unseen path.
    max_price_observation_gap_minutes: float = 5.0


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def target_contract_payload(cfg: PretrainingConfig) -> dict[str, Any]:
    base = _original_payload(cfg)
    payload = dict(base)
    payload["schema"] = PRETRAINING_SCHEMA_VERSION
    tri = dict(base.get("triple_barrier") or {})
    tri["max_price_observation_gap_minutes"] = float(cfg.max_price_observation_gap_minutes)
    tri["missing_price_semantics"] = "censor_on_token_disappearance_or_capture_gap"
    payload["triple_barrier"] = tri
    collapse = dict(base.get("economic_collapse") or {})
    collapse["max_price_observation_gap_minutes"] = float(cfg.max_price_observation_gap_minutes)
    collapse["persistence_requires_contiguous_valid_observations"] = True
    payload["economic_collapse"] = collapse
    return payload


def target_contract_hash(cfg: PretrainingConfig) -> str:
    return _hash(target_contract_payload(cfg))


def _utc(value: Any) -> pd.Timestamp:
    return _v1._utc(value)


def _gap_minutes(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return (b - a).total_seconds() / 60.0


def _capture_gap_break(
    start: pd.Timestamp,
    end: pd.Timestamp,
    captures: Sequence[pd.Timestamp],
    max_gap: float,
) -> pd.Timestamp | None:
    """Return the last known-valid time before a collector-heartbeat gap.

    Empty capture history is tolerated for small synthetic/unit-test frames; the
    token-observation continuity check remains authoritative in that case.
    """
    if not captures or end <= start:
        return None
    last = start
    for raw in captures:
        t = _utc(raw)
        if t <= start:
            continue
        if t > end:
            break
        if _gap_minutes(last, t) > max_gap:
            return last
        last = t
    if _gap_minutes(last, end) > max_gap:
        return last
    return None


def _censored(
    reason: str,
    *,
    event_at: pd.Timestamp | None = None,
    reference_at: pd.Timestamp | None = None,
    reference_mc: float | None = None,
    terminal_mc: float | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"outcome": "censored", "censor_reason": reason}
    if event_at is not None:
        out["event_at"] = event_at
    if reference_at is not None:
        out["reference_at"] = reference_at
    if reference_mc is not None:
        out["reference_mc"] = reference_mc
    if terminal_mc is not None:
        out["terminal_mc"] = terminal_mc
        if reference_mc is not None and math.isfinite(reference_mc) and reference_mc > 0:
            out["gross_return"] = terminal_mc / reference_mc - 1.0
    return out


def _barrier_outcome(
    g: pd.DataFrame,
    decision: pd.Timestamp,
    horizon: int,
    up: float,
    down: float,
    censors: list[pd.Timestamp],
    capture_times: Sequence[pd.Timestamp] | None = None,
    max_gap_minutes: float | None = None,
) -> dict[str, Any]:
    decision = _utc(decision)
    max_gap = float(_active_max_gap.get() if max_gap_minutes is None else max_gap_minutes)
    captures = tuple(_active_capture_times.get() if capture_times is None else capture_times)
    ref = _v1._next_observation(g, decision)
    if ref is None:
        return _censored("no_next_observation")
    ref_at = _utc(ref.snapshot_at)
    ref_mc = float(ref.market_cap_usd)
    if not math.isfinite(ref_mc) or ref_mc <= 0:
        return _censored("invalid_reference_price", reference_at=ref_at)
    if _gap_minutes(decision, ref_at) > max_gap:
        return _censored(
            "next_observation_too_late",
            event_at=ref_at,
            reference_at=ref_at,
            reference_mc=ref_mc,
        )
    if _capture_gap_break(decision, ref_at, captures, max_gap) is not None:
        return _censored(
            "capture_gap_before_executable_reference",
            event_at=ref_at,
            reference_at=ref_at,
            reference_mc=ref_mc,
        )

    deadline = decision + pd.Timedelta(minutes=int(horizon))
    future_censors = [c for c in censors if decision < c <= deadline]
    censor_at = min(future_censors) if future_censors else None
    end = min(deadline, censor_at) if censor_at is not None else deadline
    if ref_at > end:
        return _censored(
            "reference_after_window_end",
            event_at=end,
            reference_at=ref_at,
            reference_mc=ref_mc,
        )

    path = g[(g.snapshot_at >= ref_at) & (g.snapshot_at <= end)].copy().sort_values("snapshot_at")
    if path.empty:
        return _censored("empty_observed_path", reference_at=ref_at, reference_mc=ref_mc)

    upper = ref_mc * (1.0 + float(up))
    lower = ref_mc * (1.0 + float(down))
    prev = ref_at
    terminal_mc = ref_mc
    for r in path.itertuples(index=False):
        t = _utc(r.snapshot_at)
        mc = float(r.market_cap_usd)
        if not math.isfinite(mc) or mc <= 0:
            return _censored(
                "invalid_path_price", event_at=t, reference_at=ref_at,
                reference_mc=ref_mc, terminal_mc=terminal_mc,
            )
        if t > prev:
            if _gap_minutes(prev, t) > max_gap:
                return _censored(
                    "token_observation_gap", event_at=t, reference_at=ref_at,
                    reference_mc=ref_mc, terminal_mc=terminal_mc,
                )
            if _capture_gap_break(prev, t, captures, max_gap) is not None:
                return _censored(
                    "capture_heartbeat_gap", event_at=t, reference_at=ref_at,
                    reference_mc=ref_mc, terminal_mc=terminal_mc,
                )
        terminal_mc = mc
        hit_up = mc >= upper
        hit_down = mc <= lower
        if hit_up and hit_down:
            return _censored(
                "same_snapshot_opposing_touches", event_at=t, reference_at=ref_at,
                reference_mc=ref_mc, terminal_mc=mc,
            )
        if hit_up:
            return {
                "outcome": "up_first", "event_at": t, "reference_at": ref_at,
                "reference_mc": ref_mc, "terminal_mc": mc,
                "gross_return": mc / ref_mc - 1.0, "target_ready_at": t,
            }
        if hit_down:
            return {
                "outcome": "down_first", "event_at": t, "reference_at": ref_at,
                "reference_mc": ref_mc, "terminal_mc": mc,
                "gross_return": mc / ref_mc - 1.0, "target_ready_at": t,
            }
        prev = t

    if censor_at is not None:
        return _censored(
            "manual_stop", event_at=censor_at, reference_at=ref_at,
            reference_mc=ref_mc, terminal_mc=terminal_mc,
        )
    if _gap_minutes(prev, deadline) > max_gap:
        return _censored(
            "token_observation_gap_before_horizon_completion",
            event_at=deadline, reference_at=ref_at,
            reference_mc=ref_mc, terminal_mc=terminal_mc,
        )
    if _capture_gap_break(prev, deadline, captures, max_gap) is not None:
        return _censored(
            "capture_heartbeat_gap_before_horizon_completion",
            event_at=deadline, reference_at=ref_at,
            reference_mc=ref_mc, terminal_mc=terminal_mc,
        )
    return {
        "outcome": "neither", "event_at": deadline, "reference_at": ref_at,
        "reference_mc": ref_mc, "terminal_mc": terminal_mc,
        "gross_return": terminal_mc / ref_mc - 1.0, "target_ready_at": deadline,
    }


def _economic_collapse(
    g: pd.DataFrame,
    decision: pd.Timestamp,
    cfg: PretrainingConfig,
    censors: list[pd.Timestamp],
    capture_times: Sequence[pd.Timestamp] | None = None,
) -> dict[str, Any]:
    decision = _utc(decision)
    max_gap = float(getattr(cfg, "max_price_observation_gap_minutes", _active_max_gap.get()))
    captures = tuple(_active_capture_times.get() if capture_times is None else capture_times)
    ref = _v1._next_observation(g, decision)
    if ref is None:
        return _censored("no_next_observation")
    ref_at = _utc(ref.snapshot_at)
    ref_mc = float(ref.market_cap_usd)
    if _gap_minutes(decision, ref_at) > max_gap:
        return _censored("next_observation_too_late", event_at=ref_at, reference_at=ref_at, reference_mc=ref_mc)
    if _capture_gap_break(decision, ref_at, captures, max_gap) is not None:
        return _censored("capture_gap_before_executable_reference", event_at=ref_at, reference_at=ref_at, reference_mc=ref_mc)

    path = g[g.snapshot_at >= ref_at].copy().sort_values("snapshot_at")
    if path.empty:
        return _censored("empty_observed_path", reference_at=ref_at, reference_mc=ref_mc)

    running_high = -math.inf
    run_start: pd.Timestamp | None = None
    prev = ref_at
    terminal_mc = ref_mc
    for r in path.itertuples(index=False):
        t = _utc(r.snapshot_at)
        mc = float(r.market_cap_usd)
        if t > prev:
            if _gap_minutes(prev, t) > max_gap:
                return _censored(
                    "token_observation_gap", event_at=t, reference_at=ref_at,
                    reference_mc=ref_mc, terminal_mc=terminal_mc,
                )
            if _capture_gap_break(prev, t, captures, max_gap) is not None:
                return _censored(
                    "capture_heartbeat_gap", event_at=t, reference_at=ref_at,
                    reference_mc=ref_mc, terminal_mc=terminal_mc,
                )
        crossed = [c for c in censors if prev < c <= t]
        if crossed:
            return _censored(
                "manual_stop", event_at=min(crossed), reference_at=ref_at,
                reference_mc=ref_mc, terminal_mc=terminal_mc,
            )
        if not math.isfinite(mc) or mc <= 0:
            return _censored(
                "invalid_path_price", event_at=t, reference_at=ref_at,
                reference_mc=ref_mc, terminal_mc=terminal_mc,
            )
        terminal_mc = mc
        running_high = max(running_high, mc)
        collapsed = running_high > 0 and mc <= running_high * (1.0 - cfg.economic_collapse_drawdown_pct)
        if collapsed:
            if run_start is None:
                run_start = t
            if _gap_minutes(run_start, t) >= cfg.economic_collapse_sustain_minutes:
                return {
                    "outcome": "economic_collapse", "event_at": t,
                    "reference_at": ref_at, "reference_mc": ref_mc,
                    "terminal_mc": mc, "gross_return": mc / ref_mc - 1.0,
                    "target_ready_at": t, "trailing_peak_mc": running_high,
                }
        else:
            run_start = None
        prev = t
    return {"outcome": "open", "reference_at": ref_at, "reference_mc": ref_mc}


def _capture_context(db: str, cfg: PretrainingConfig):
    try:
        with sqlite3.connect(db) as conn:
            captures = tuple(_v1._capture_times(conn))
    except sqlite3.DatabaseError:
        captures = ()
    tok_captures = _active_capture_times.set(captures)
    tok_gap = _active_max_gap.set(float(cfg.max_price_observation_gap_minutes))
    return tok_captures, tok_gap


def _reset_context(tokens) -> None:
    tok_captures, tok_gap = tokens
    _active_capture_times.reset(tok_captures)
    _active_max_gap.reset(tok_gap)


def refresh_pretraining_targets(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    tokens = _capture_context(db, cfg)
    try:
        return _original_refresh(db, cfg)
    finally:
        _reset_context(tokens)


def training_readiness(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    tokens = _capture_context(db, cfg)
    try:
        return _original_readiness(db, cfg)
    finally:
        _reset_context(tokens)


def assert_training_ready(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    report = training_readiness(db, cfg)
    if not report["ready"]:
        failed = [
            k for k, v in report["gates"].items()
            if not bool(v.get("pass")) and k != "operational_death_tokens"
        ]
        raise RuntimeError(
            "V24 production bootstrap refused by pretraining readiness gates: " + ", ".join(failed)
        )
    return report


def evaluate_baselines(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    tokens = _capture_context(db, cfg)
    try:
        return _original_baselines(db, cfg)
    finally:
        _reset_context(tokens)


def collection_audit(db: str, cfg: PretrainingConfig | None = None) -> dict[str, Any]:
    return _original_audit(db, cfg or PretrainingConfig())


# The retained V1 materializer owns the iteration/storage machinery.  Patch only
# its truth functions and hash so every V2/V3 compatibility path receives the
# continuity-safe semantics while preserving the already-tested schema behavior.
_v1._barrier_outcome = _barrier_outcome
_v1._economic_collapse = _economic_collapse
for _module in (_v1, _v2, _compat):
    _module.target_contract_payload = target_contract_payload
    _module.target_contract_hash = target_contract_hash
