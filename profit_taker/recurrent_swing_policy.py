from __future__ import annotations

"""Causal decision helpers for recurrent swing trading within one token lifetime.

This module consumes only forecast values available at the current decision time.
It does not create labels, modify a trained model, or inspect future observations.
"""

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


WATCH_TABLE = "benchmark_swing_watch_v24"
SHORT_HORIZON_WEIGHTS = ((5, 0.30), (10, 0.25), (15, 0.20), (30, 0.15), (60, 0.10))
CALIBRATION_HORIZON_MINUTES = 60
CALIBRATION_FEATURES = (
    "probability", "conservative_upside", "timing_discount",
    "death_probability", "downside_probability",
)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _first(state: dict[str, float], *names: str) -> float | None:
    for name in names:
        value = _finite(state.get(name))
        if value is not None:
            return value
    return None


def _prob(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    return min(1.0, max(0.0, float(value)))


def _fraction(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    value = float(value)
    if value > 1.5:
        value /= 100.0
    return max(0.0, value)


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS {WATCH_TABLE} (
            watch_id TEXT PRIMARY KEY,
            benchmark_id TEXT NOT NULL,
            token_key TEXT NOT NULL,
            source_position_id TEXT NOT NULL UNIQUE,
            exit_at TEXT NOT NULL,
            exit_mc REAL NOT NULL,
            expires_at TEXT NOT NULL,
            status TEXT NOT NULL,
            later_higher_probability REAL,
            second_peak_gap_minutes REAL,
            second_peak_relative_magnitude REAL,
            exit_context_json TEXT NOT NULL,
            last_evaluated_at TEXT,
            last_retrace_pct REAL,
            reentry_decision_at TEXT,
            reentry_position_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_{WATCH_TABLE}_active
            ON {WATCH_TABLE}(benchmark_id,token_key,status,exit_at);
        """
    )


@dataclass(frozen=True)
class ShortTermSetup:
    available: bool
    qualifies: bool
    score: float
    probability: float
    occurrence_q50_minutes: float | None
    occurrence_spread_minutes: float | None
    confirmation_lag_q50_minutes: float | None
    peak_multiple_q50: float | None
    conservative_upside: float
    death_probability: float
    downside_probability: float
    net_edge: float
    raw_net_edge: float
    calibration_status: str
    calibrated_probability_floor: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "qualifies": self.qualifies,
            "score": self.score,
            "probability": self.probability,
            "occurrence_q50_minutes": self.occurrence_q50_minutes,
            "occurrence_spread_minutes": self.occurrence_spread_minutes,
            "confirmation_lag_q50_minutes": self.confirmation_lag_q50_minutes,
            "peak_multiple_q50": self.peak_multiple_q50,
            "conservative_upside": self.conservative_upside,
            "death_probability": self.death_probability,
            "downside_probability": self.downside_probability,
            "net_edge": self.net_edge,
            "raw_net_edge": self.raw_net_edge,
            "calibration_status": self.calibration_status,
            "calibrated_probability_floor": self.calibrated_probability_floor,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EntryCalibration:
    """Causal calibration learned only from already-mature candidate outcomes."""

    status: str
    sample_size: int
    token_count: int
    model_hash: str | None
    feature_mean: tuple[float, ...] = ()
    feature_scale: tuple[float, ...] = ()
    coefficients: tuple[float, ...] = ()
    covariance: tuple[tuple[float, ...], ...] = ()
    residual_scale: float = 0.0
    probability_bins: tuple[tuple[float, float, int, int, float, float], ...] = ()

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def evaluate(self, metrics: dict[str, float]) -> tuple[float, float | None, str]:
        if not self.ready:
            return float("nan"), None, self.status
        values = np.asarray([metrics[name] for name in CALIBRATION_FEATURES], dtype=float)
        mean = np.asarray(self.feature_mean, dtype=float)
        scale = np.asarray(self.feature_scale, dtype=float)
        x = np.r_[1.0, (values - mean) / scale]
        beta = np.asarray(self.coefficients, dtype=float)
        cov = np.asarray(self.covariance, dtype=float)
        estimate = float(x @ beta)
        # A one-sided 90% lower confidence bound protects against promoting a
        # noisy positive mean into a live paper entry rule.
        standard_error = math.sqrt(max(0.0, float(x @ cov @ x)))
        conservative_edge = estimate - 1.2815515655446004 * standard_error
        probability = float(metrics["probability"])
        selected = None
        for row in self.probability_bins:
            if probability <= row[1] + 1e-12:
                selected = row
                break
        if selected is None and self.probability_bins:
            selected = self.probability_bins[-1]
        if selected is None or selected[5] <= 0.0:
            return conservative_edge, (selected[0] if selected else None), "no_calibrated_profitable_probability_region"
        return conservative_edge, selected[0], "calibrated"

    def as_dict(self) -> dict[str, Any]:
        profitable = [row for row in self.probability_bins if row[5] > 0.0]
        return {
            "status": self.status,
            "mature_samples": self.sample_size,
            "unique_tokens": self.token_count,
            "forecast_model_hash": self.model_hash,
            "profitable_probability_bins": len(profitable),
            "minimum_profitable_probability": min((row[0] for row in profitable), default=None),
            "probability_bins": [
                {
                    "lower": row[0], "upper": row[1], "samples": row[2],
                    "unique_tokens": row[3], "mean_realized_reward": row[4],
                    "lower_confidence_reward": row[5],
                }
                for row in self.probability_bins
            ],
        }


def _setup_metrics(state: dict[str, float], config: Any) -> dict[str, float | None] | None:
    probabilities: list[tuple[float, float]] = []
    for horizon, weight in SHORT_HORIZON_WEIGHTS:
        value = _finite(state.get(f"p_first_peak_by_{horizon}m"))
        if value is not None:
            probabilities.append((weight, _prob(value)))
    if not probabilities:
        return None
    total_weight = sum(weight for weight, _ in probabilities)
    probability = sum(weight * value for weight, value in probabilities) / total_weight
    occurrence = _first(
        state, "next_occurrence_q50", "next_gap_q50",
        "pred_time_to_next_substantial_peak_minutes_q50",
    )
    occurrence_quantiles = [
        _finite(state.get(f"next_occurrence_q{quantile}"))
        for quantile in (10, 25, 50, 75, 90)
    ]
    occurrence_quantiles = [value for value in occurrence_quantiles if value is not None]
    occurrence_spread = (
        max(occurrence_quantiles) - min(occurrence_quantiles)
        if len(occurrence_quantiles) >= 2 else None
    )
    confirmation_lag = _first(state, "next_confirmation_lag_q50")
    multiple_q50 = _first(state, "next_peak_multiple_q50", "pred_next_substantial_peak_multiple_q50")
    multiple_q25 = _first(state, "next_peak_multiple_q25", "pred_next_substantial_peak_multiple_q25")
    median_upside = max(0.0, float(multiple_q50 or 1.0) - 1.0)
    lower_upside = max(0.0, float(multiple_q25 or multiple_q50 or 1.0) - 1.0)
    conservative_upside = 0.5 * median_upside + 0.5 * lower_upside
    timing = max(0.0, float(occurrence if occurrence is not None else 60.0))
    timing_risk = timing + 0.25 * max(0.0, float(occurrence_spread or 0.0)) + 0.25 * max(0.0, float(confirmation_lag or 0.0))
    timing_discount = 1.0 / (1.0 + timing_risk / 60.0)
    death = _prob(_first(state, "p_death_by_60m", "p_death_by_30m", "p_death_by_720m", "p_death_by_1440m"))
    downside = _prob(_first(state, "p_hit_minus50_by_60m", "p_hit_minus50_by_720m", "p_hit_minus50_by_1440m"))
    friction = max(0.0, float(config.friction_bps_round_trip)) / 10000.0
    raw_net_edge = probability * conservative_upside * timing_discount - friction - 0.30 * death - 0.20 * downside
    return {
        "probability": probability,
        "occurrence": occurrence,
        "occurrence_spread": occurrence_spread,
        "confirmation_lag": confirmation_lag,
        "multiple_q50": multiple_q50,
        "conservative_upside": conservative_upside,
        "timing_discount": timing_discount,
        "death_probability": death,
        "downside_probability": downside,
        "raw_net_edge": raw_net_edge,
    }


def short_term_setup(
    state: dict[str, float], config: Any, base_score: float, base_kind: str,
    calibration: EntryCalibration | None = None,
) -> tuple[ShortTermSetup, str]:
    metrics = _setup_metrics(state, config)
    if metrics is None:
        fallback = ShortTermSetup(
            available=False,
            qualifies=math.isfinite(base_score) and base_score > float(config.min_entry_score),
            score=float(base_score),
            probability=0.0,
            occurrence_q50_minutes=None,
            occurrence_spread_minutes=None,
            confirmation_lag_q50_minutes=None,
            peak_multiple_q50=None,
            conservative_upside=0.0,
            death_probability=0.0,
            downside_probability=0.0,
            net_edge=float(base_score),
            raw_net_edge=float(base_score),
            calibration_status="not_applicable",
            calibrated_probability_floor=None,
            reason="legacy_forecast_without_minute_heads",
        )
        return fallback, base_kind
    probability = float(metrics["probability"])
    occurrence = metrics["occurrence"]
    occurrence_spread = metrics["occurrence_spread"]
    confirmation_lag = metrics["confirmation_lag"]
    multiple_q50 = metrics["multiple_q50"]
    conservative_upside = float(metrics["conservative_upside"])
    death = float(metrics["death_probability"])
    downside = float(metrics["downside_probability"])
    friction = max(0.0, float(config.friction_bps_round_trip)) / 10000.0
    raw_net_edge = float(metrics["raw_net_edge"])
    calibration_status = "legacy_absolute_threshold"
    calibrated_probability_floor = None
    calibrated_reason = "legacy_absolute_threshold"
    net_edge = raw_net_edge
    if calibration is not None and calibration.ready:
        net_edge, calibrated_probability_floor, calibrated_reason = calibration.evaluate(
            {name: float(metrics[name]) for name in CALIBRATION_FEATURES}
        )
        calibration_status = calibration.status
    score = net_edge + 0.05 * math.tanh(float(base_score))

    reason = "short_term_setup"
    qualifies = True
    if calibration is not None and calibration.ready and calibrated_reason != "calibrated":
        qualifies, reason = False, calibrated_reason
    elif calibration is None or not calibration.ready:
        if probability < float(config.swing_entry_min_probability):
            qualifies, reason = False, "short_peak_probability_too_low"
    if qualifies and occurrence is not None and occurrence > float(config.swing_entry_max_occurrence_minutes):
        qualifies, reason = False, "next_peak_too_distant"
    elif (
        qualifies
        and (calibration is None or not calibration.ready)
        and conservative_upside < friction + float(config.swing_entry_min_net_upside)
    ):
        qualifies, reason = False, "short_peak_upside_below_friction_buffer"
    elif qualifies and net_edge <= 0.0:
        qualifies, reason = False, (
            "calibrated_net_edge_nonpositive"
            if calibration is not None and calibration.ready
            else "short_peak_net_edge_nonpositive"
        )

    setup = ShortTermSetup(
        available=True,
        qualifies=qualifies,
        score=float(score),
        probability=float(probability),
        occurrence_q50_minutes=occurrence,
        occurrence_spread_minutes=occurrence_spread,
        confirmation_lag_q50_minutes=confirmation_lag,
        peak_multiple_q50=multiple_q50,
        conservative_upside=float(conservative_upside),
        death_probability=float(death),
        downside_probability=float(downside),
        net_edge=float(net_edge),
        raw_net_edge=float(raw_net_edge),
        calibration_status=calibration_status,
        calibrated_probability_floor=calibrated_probability_floor,
        reason=reason,
    )
    return setup, f"recurrent_swing_{base_kind}"


def build_entry_calibration(
    conn: sqlite3.Connection,
    source_db: str,
    benchmark_id: str,
    snapshot: pd.Timestamp,
    forecast_model_hash: str | None,
    config: Any,
) -> EntryCalibration:
    """Fit a causal entry calibration from champion-matched, mature outcomes.

    Repeated minute boards are thinned to one observation per token per 15-minute
    block. This prevents one long-lived token from masquerading as hundreds of
    independent examples. Incomplete paths are excluded unless the V24 lifetime
    ledger has already recorded an operational terminal event.
    """
    cutoff = snapshot - pd.Timedelta(minutes=CALIBRATION_HORIZON_MINUTES)
    limit = int(getattr(config, "swing_calibration_max_samples", 5000))
    minimum = int(getattr(config, "swing_calibration_min_samples", 80))
    minimum_tokens = int(getattr(config, "swing_calibration_min_tokens", 12))
    rows = conn.execute(
        """SELECT snapshot_at,token_key,market_cap_usd,state_json,forecast_model_hash
           FROM benchmark_candidates_v22
           WHERE benchmark_id=? ORDER BY snapshot_at DESC LIMIT ?""",
        (benchmark_id, limit * 4),
    ).fetchall()
    candidates: list[dict[str, Any]] = []
    seen_blocks: set[tuple[str, int]] = set()
    for row in rows:
        at = pd.Timestamp(row[0])
        at = at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")
        if at > cutoff:
            continue
        row_hash = str(row[4]) if row[4] is not None else None
        if forecast_model_hash is not None and row_hash != forecast_model_hash:
            continue
        token = str(row[1])
        block = int(at.timestamp() // (15 * 60))
        if (token, block) in seen_blocks:
            continue
        seen_blocks.add((token, block))
        try:
            state = json.loads(str(row[3]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        metrics = _setup_metrics(state, config)
        market_cap = _finite(row[2])
        if metrics is None or market_cap is None or market_cap <= 0:
            continue
        candidates.append({"snapshot_at": at, "token_key": token, "market_cap": market_cap, "metrics": metrics})
        if len(candidates) >= limit:
            break
    if not candidates:
        return EntryCalibration("no_mature_champion_candidates", 0, 0, forecast_model_hash)

    tokens = sorted({row["token_key"] for row in candidates})
    start = min(row["snapshot_at"] for row in candidates)
    end = max(row["snapshot_at"] for row in candidates) + pd.Timedelta(minutes=CALIBRATION_HORIZON_MINUTES)
    observations: list[tuple[Any, ...]] = []
    terminal_rows: list[tuple[Any, ...]] = []
    try:
        with sqlite3.connect(source_db, timeout=10.0) as source:
            source.execute("PRAGMA busy_timeout=10000")
            source.execute("PRAGMA query_only=ON")
            for offset in range(0, len(tokens), 800):
                batch = tokens[offset:offset + 800]
                marks = ",".join("?" for _ in batch)
                observations.extend(source.execute(
                    f"""SELECT token_key,snapshot_at,market_cap_usd FROM axiom_observations
                        WHERE token_key IN ({marks}) AND snapshot_at>=? AND snapshot_at<=?
                          AND market_cap_usd>0 ORDER BY token_key,snapshot_at""",
                    (*batch, start.isoformat(), end.isoformat()),
                ).fetchall())
            lifetime_table = "axiom_v24_token_lifetimes"
            exists = source.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (lifetime_table,)
            ).fetchone()
            if exists:
                for offset in range(0, len(tokens), 800):
                    batch = tokens[offset:offset + 800]
                    marks = ",".join("?" for _ in batch)
                    terminal_rows.extend(source.execute(
                        f"""SELECT token_key,terminal_at FROM {lifetime_table}
                            WHERE token_key IN ({marks}) AND terminal_at IS NOT NULL
                              AND terminal_at>=? AND terminal_at<=?""",
                        (*batch, start.isoformat(), end.isoformat()),
                    ).fetchall())
    except sqlite3.DatabaseError:
        return EntryCalibration("source_outcomes_unavailable", 0, 0, forecast_model_hash)

    by_token: dict[str, pd.DataFrame] = {}
    if observations:
        observed = pd.DataFrame(observations, columns=["token_key", "snapshot_at", "market_cap_usd"])
        observed["snapshot_at"] = pd.to_datetime(observed["snapshot_at"], utc=True, errors="coerce")
        observed["market_cap_usd"] = pd.to_numeric(observed["market_cap_usd"], errors="coerce")
        observed = observed.dropna(subset=["snapshot_at", "market_cap_usd"])
        by_token = {str(token): group.sort_values("snapshot_at") for token, group in observed.groupby("token_key")}
    terminals: dict[str, list[pd.Timestamp]] = {}
    for token, terminal_at in terminal_rows:
        parsed = pd.Timestamp(terminal_at)
        parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
        terminals.setdefault(str(token), []).append(parsed)

    training: list[dict[str, Any]] = []
    friction = max(0.0, float(config.friction_bps_round_trip)) / 10000.0
    for candidate in candidates:
        at = candidate["snapshot_at"]
        deadline = at + pd.Timedelta(minutes=CALIBRATION_HORIZON_MINUTES)
        frame = by_token.get(candidate["token_key"])
        path = frame[(frame.snapshot_at > at) & (frame.snapshot_at <= deadline)] if frame is not None else pd.DataFrame()
        terminal = any(at < value <= deadline for value in terminals.get(candidate["token_key"], ()))
        complete = bool(not path.empty and path.snapshot_at.max() >= deadline - pd.Timedelta(minutes=2))
        if path.empty or (not complete and not terminal):
            continue
        entry = float(candidate["market_cap"])
        future = path.market_cap_usd.to_numpy(dtype=float)
        best = float(np.nanmax(future) / entry - 1.0)
        worst = float(np.nanmin(future) / entry - 1.0)
        ending = float(future[-1] / entry - 1.0)
        if terminal:
            recognition = float(getattr(config, "disappearance_profit_recognition_fraction", 0.0))
            best = min(0.0, best) + recognition * max(0.0, best)
            ending = min(0.0, ending) + recognition * max(0.0, ending)
        # Same mature 60-minute economic objective used by policy training:
        # reward peak opportunity while charging terminal loss, drawdown and fees.
        reward = 0.70 * best + 0.30 * ending - 0.50 * abs(min(0.0, worst)) - friction
        record = {name: float(candidate["metrics"][name]) for name in CALIBRATION_FEATURES}
        record.update(token_key=candidate["token_key"], reward=float(np.clip(reward, -1.0, 5.0)))
        training.append(record)

    token_count = len({row["token_key"] for row in training})
    if len(training) < minimum or token_count < minimum_tokens:
        return EntryCalibration("insufficient_mature_outcomes", len(training), token_count, forecast_model_hash)
    data = pd.DataFrame(training)
    features = data[list(CALIBRATION_FEATURES)].to_numpy(dtype=float)
    target = data.reward.to_numpy(dtype=float)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale])
    token_frequency = data.token_key.value_counts().to_dict()
    weights = np.asarray([1.0 / token_frequency[token] for token in data.token_key], dtype=float)
    weights *= len(weights) / weights.sum()
    weighted = design * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)
    ridge = np.eye(design.shape[1]) * 1.0
    ridge[0, 0] = 0.0
    precision = weighted.T @ weighted + ridge
    inverse = np.linalg.pinv(precision)
    coefficients = inverse @ weighted.T @ weighted_target
    residual = target - design @ coefficients
    degrees = max(1.0, float(weights.sum() - design.shape[1]))
    sigma2 = float(np.sum(weights * residual ** 2) / degrees)
    covariance = inverse * sigma2

    probability = data.probability.to_numpy(dtype=float)
    bins: list[tuple[float, float, int, int, float, float]] = []
    unique_probability = np.unique(probability)
    probability_groups = np.array_split(unique_probability, min(5, len(unique_probability)))
    for group in probability_groups:
        lower, upper = float(group[0]), float(group[-1])
        mask = (probability >= lower) & (probability <= upper)
        values = target[mask]
        if not len(values):
            continue
        group_tokens = int(data.loc[mask, "token_key"].nunique())
        avg = float(np.mean(values))
        lower_confidence = avg - 1.2815515655446004 * float(np.std(values, ddof=1) if len(values) > 1 else 0.0) / math.sqrt(len(values))
        if len(values) < max(10, minimum // 10) or group_tokens < max(4, minimum_tokens // 3):
            lower_confidence = min(lower_confidence, 0.0)
        bins.append((lower, upper, int(len(values)), group_tokens, avg, lower_confidence))
    return EntryCalibration(
        "ready", len(training), token_count, forecast_model_hash,
        tuple(float(value) for value in mean), tuple(float(value) for value in scale),
        tuple(float(value) for value in coefficients),
        tuple(tuple(float(value) for value in row) for row in covariance),
        math.sqrt(max(0.0, sigma2)), tuple(bins),
    )


def later_peak_context(state: dict[str, float]) -> dict[str, float | None]:
    marked = [
        _finite(value)
        for key, value in state.items()
        if str(key).startswith("p_later_higher_") and "_by_" in str(key)
    ]
    marked = [value for value in marked if value is not None]
    later_probability = max((_prob(value) for value in marked), default=0.0)
    second_gap = _first(state, "second_occurrence_gap_q50", "second_gap_q50")
    second_relative = _first(
        state,
        "second_peak_relative_q50",
        "pred_later_higher_peak_multiple_vs_next_q50",
    )
    if second_relative is not None and second_relative > 1.0:
        second_relative -= 1.0
    return {
        "later_higher_probability": later_probability,
        "second_peak_gap_minutes": second_gap,
        "second_peak_relative_magnitude": second_relative,
    }


def peak_boundary_decision(
    state: dict[str, float], position_return: float, config: Any
) -> dict[str, Any]:
    setup, _ = short_term_setup(state, config, 0.0, "bootstrap")
    occurrence = setup.occurrence_q50_minutes
    near_probability = _prob(_first(state, "p_first_peak_by_10m", "p_first_peak_by_5m", "p_first_peak_by_15m"))
    friction = max(0.0, float(config.friction_bps_round_trip)) / 10000.0
    min_realized = max(float(config.swing_exit_min_return), 2.0 * friction)
    at_boundary = bool(
        setup.available
        and occurrence is not None
        and occurrence <= float(config.swing_peak_boundary_minutes)
        and near_probability >= float(config.swing_peak_boundary_probability)
        and float(position_return) >= min_realized
    )

    context = later_peak_context(state)
    later_probability = float(context["later_higher_probability"] or 0.0)
    second_gap = context["second_peak_gap_minutes"]
    second_relative = max(0.0, float(context["second_peak_relative_magnitude"] or 0.0))
    retrace = _fraction(_first(state, "pred_post_next_peak_retracement_pct_q50"))
    immediate_hold_value = max(0.0, setup.net_edge)
    gap_discount = math.exp(-max(0.0, float(second_gap or 240.0)) / 60.0)
    chained_hold_value = later_probability * second_relative * gap_discount
    hold_through_value = immediate_hold_value + chained_hold_value
    capital_release_option = min(0.04, max(0.0, float(second_gap or 0.0) - 30.0) / 6000.0)
    sell_reentry_value = later_probability * retrace + capital_release_option - friction

    strong_near_continuation = bool(
        later_probability >= float(config.swing_hold_later_probability)
        and second_relative >= friction + float(config.swing_hold_min_second_upside)
        and second_gap is not None
        and second_gap <= float(config.swing_hold_max_second_gap_minutes)
    )
    sell = bool(at_boundary and not strong_near_continuation and sell_reentry_value > hold_through_value)
    reason = "recurrent_swing_peak_boundary" if sell else (
        "hold_strong_near_second_peak" if at_boundary and strong_near_continuation
        else "not_at_peak_boundary"
    )
    return {
        "at_peak_boundary": at_boundary,
        "sell": sell,
        "reason": reason,
        "near_peak_probability": near_probability,
        "hold_through_value": hold_through_value,
        "sell_reentry_value": sell_reentry_value,
        "predicted_retracement": retrace,
        **context,
    }


def create_watch(
    conn: sqlite3.Connection,
    benchmark_id: str,
    position: sqlite3.Row,
    exit_at: pd.Timestamp,
    exit_mc: float,
    state: dict[str, Any],
    config: Any,
) -> str:
    context = later_peak_context(state)
    watch_id = str(uuid.uuid4())
    expires = exit_at + pd.Timedelta(minutes=float(config.swing_watch_max_minutes))
    conn.execute(
        f"""INSERT OR REPLACE INTO {WATCH_TABLE}
        (watch_id,benchmark_id,token_key,source_position_id,exit_at,exit_mc,expires_at,status,
         later_higher_probability,second_peak_gap_minutes,second_peak_relative_magnitude,exit_context_json)
        VALUES(?,?,?,?,?,?,?,'active',?,?,?,?)""",
        (
            watch_id, benchmark_id, str(position["token_key"]), str(position["position_id"]),
            exit_at.isoformat(), float(exit_mc), expires.isoformat(),
            context["later_higher_probability"], context["second_peak_gap_minutes"],
            context["second_peak_relative_magnitude"], json.dumps(state, sort_keys=True),
        ),
    )
    return watch_id


def active_watch(
    conn: sqlite3.Connection, benchmark_id: str, token: str, snapshot: pd.Timestamp
) -> sqlite3.Row | None:
    conn.execute(
        f"""UPDATE {WATCH_TABLE} SET status='expired'
        WHERE benchmark_id=? AND token_key=? AND status='active' AND expires_at<?""",
        (benchmark_id, token, snapshot.isoformat()),
    )
    return conn.execute(
        f"""SELECT * FROM {WATCH_TABLE}
        WHERE benchmark_id=? AND token_key=? AND status='active' AND expires_at>=?
        ORDER BY exit_at DESC LIMIT 1""",
        (benchmark_id, token, snapshot.isoformat()),
    ).fetchone()


def reentry_eligibility(
    conn: sqlite3.Connection,
    benchmark_id: str,
    token: str,
    snapshot: pd.Timestamp,
    market_cap: float,
    setup: ShortTermSetup,
    config: Any,
) -> tuple[bool, str, str | None]:
    last = conn.execute(
        """SELECT closed_at,exit_mc FROM benchmark_positions_v22
        WHERE benchmark_id=? AND token_key=? AND status='closed'
        ORDER BY closed_at DESC LIMIT 1""",
        (benchmark_id, token),
    ).fetchone()
    if last is None or not last[0]:
        return True, "new_token_setup", None
    closed_at = pd.Timestamp(last[0])
    if closed_at.tzinfo is None:
        closed_at = closed_at.tz_localize("UTC")
    age_minutes = max(0.0, (snapshot - closed_at).total_seconds() / 60.0)
    if age_minutes >= float(config.reentry_cooldown_minutes):
        return True, "standard_cooldown_elapsed", None

    watch = active_watch(conn, benchmark_id, token, snapshot)
    if watch is None:
        return False, "reentry_cooldown", None
    exit_mc = max(1e-12, float(watch["exit_mc"]))
    retrace = max(0.0, 1.0 - float(market_cap) / exit_mc)
    conn.execute(
        f"""UPDATE {WATCH_TABLE} SET last_evaluated_at=?,last_retrace_pct=? WHERE watch_id=?""",
        (snapshot.isoformat(), retrace, str(watch["watch_id"])),
    )
    friction = max(0.0, float(config.friction_bps_round_trip)) / 10000.0
    required_retrace = max(float(config.swing_reentry_min_retrace), 2.0 * friction)
    if age_minutes < float(config.swing_reentry_min_minutes):
        return False, "swing_reentry_minimum_wait", str(watch["watch_id"])
    watch_later = _prob(_finite(watch["later_higher_probability"]))
    watch_second = max(0.0, float(_finite(watch["second_peak_relative_magnitude"]) or 0.0))
    if (
        watch_later < float(config.swing_watch_min_later_probability)
        or watch_second < float(config.swing_watch_min_second_upside)
    ):
        return False, "swing_reentry_lifecycle_context_weak", str(watch["watch_id"])
    if retrace < required_retrace:
        return False, "swing_reentry_reset_not_reached", str(watch["watch_id"])
    if not setup.available or not setup.qualifies:
        return False, "swing_reentry_short_setup_not_confirmed", str(watch["watch_id"])
    return True, "swing_reentry_after_reset", str(watch["watch_id"])


def mark_watch_pending(
    conn: sqlite3.Connection, watch_id: str | None, decision_at: pd.Timestamp
) -> None:
    if watch_id:
        conn.execute(
            f"""UPDATE {WATCH_TABLE} SET status='reentry_pending',reentry_decision_at=?
            WHERE watch_id=? AND status='active'""",
            (decision_at.isoformat(), watch_id),
        )


def mark_watch_filled(conn: sqlite3.Connection, watch_id: str | None, position_id: str) -> None:
    if watch_id:
        conn.execute(
            f"""UPDATE {WATCH_TABLE} SET status='reentered',reentry_position_id=?
            WHERE watch_id=? AND status='reentry_pending'""",
            (position_id, watch_id),
        )


def release_watch(conn: sqlite3.Connection, watch_id: str | None) -> None:
    if watch_id:
        conn.execute(
            f"""UPDATE {WATCH_TABLE} SET status='active',reentry_decision_at=NULL
            WHERE watch_id=? AND status='reentry_pending'""",
            (watch_id,),
        )
