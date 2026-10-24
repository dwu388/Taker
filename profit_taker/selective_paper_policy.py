"""Selective paper entries supported by subsequent, executable shadow trades.

The forecast's peak probability is a ranking signal, not a probability of a
profitable sale. Reference tokens define score regions. Later, disjoint tokens
estimate each region's win frequency and net payoff using next-observation
entries/exits, the same exit rule as the wallet, and the proposed dollar size.
This screen is empirical evidence, not a promised live win rate.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
import json
import math
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from . import paper_execution_costs as costs
from . import recurrent_swing_policy as swing

VERSION = "selective_net_exit_v1"
# One-sided 99% bounds for each of at most three fixed reference regions.
CONFIDENCE_Z = 2.3263478740408408


def validate(config: Any) -> None:
    for name in ("selective_min_win_lower_bound", "selective_max_death_probability",
                 "selective_max_downside_probability"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be strictly between zero and one.")
    for name in ("selective_min_net_edge", "selective_take_profit_net",
                 "selective_stop_loss_net", "selective_max_hold_minutes",
                 "selective_max_entry_delay_minutes", "selective_max_path_gap_minutes"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if config.selective_stop_loss_net >= 1:
        raise ValueError("Selective stop loss must be below 100%.")
    if config.selective_reference_tokens < 20 or config.selective_min_validation_tokens < 20:
        raise ValueError("Selective reference and validation samples need at least 20 distinct tokens.")


def exit_plan(config: Any) -> dict[str, float]:
    return {"take_profit_net": float(config.selective_take_profit_net),
            "stop_loss_net": float(config.selective_stop_loss_net),
            "max_hold_minutes": float(config.selective_max_hold_minutes)}


def exit_reason(net_return: float, held_minutes: float, plan: dict[str, float]) -> str | None:
    """Signal now, fill on a later observable board; thresholds are not fills."""
    if net_return <= -plan["stop_loss_net"]:
        return "selective_net_stop_loss"
    if net_return >= plan["take_profit_net"]:
        return "selective_net_take_profit"
    if held_minutes >= plan["max_hold_minutes"]:
        return "selective_time_exit"
    return None


def signal(state: dict[str, Any], config: Any, cash: float,
           decision_mc: float = 1.0, fill_mc: float = 1.0) -> dict[str, Any]:
    """Apply mandatory forecast, downside and actual-size fee constraints."""
    values = [swing._finite(state.get(f"p_first_peak_by_{h}m"))
              for h, _ in swing.SHORT_HORIZON_WEIGHTS]
    q25 = swing._first(state, "next_peak_multiple_q25", "pred_next_substantial_peak_multiple_q25")
    q50 = swing._first(state, "next_peak_multiple_q50", "pred_next_substantial_peak_multiple_q50")
    occurrence = swing._first(state, "next_occurrence_q50", "next_gap_q50",
                              "pred_time_to_next_substantial_peak_minutes_q50")
    death = swing._finite(state.get("p_death_by_60m"))
    downside = swing._finite(state.get("p_hit_minus50_by_60m"))
    result = {"eligible": False, "reason": "incomplete_selective_forecast", "probability": 0.0,
              "q25": q25, "q50": q50, "occurrence": occurrence,
              "death": death, "downside": downside, "net_peak_q25": None}
    # The active V24 hazard model supplies death, but some champions do not
    # train a separate minus-50% barrier head. Never pretend that missing head
    # is a measured zero risk; executable outcome validation covers losses too.
    result["downside_head_available"] = downside is not None
    if any(value is None for value in (*values, q25, q50, occurrence, death)):
        return result
    if (any(not 0 <= v <= 1 for v in (*values, death))
            or (downside is not None and not 0 <= downside <= 1)
            or any(a > b + 1e-8 for a, b in zip(values, values[1:]))
            or q25 <= 0 or q50 < q25 or occurrence < 0):
        result["reason"] = "invalid_selective_forecast"
        return result
    result["probability"] = sum(v * w for v, (_, w) in zip(values, swing.SHORT_HORIZON_WEIGHTS))
    if cash <= costs.fixed_fee_usd(config) or not all(math.isfinite(v) and v > 0 for v in (cash, decision_mc, fill_mc)):
        result["reason"] = "selective_trade_cannot_cover_fees"
        return result
    _, _, units = costs.entry_fill(cash, fill_mc, config)
    proceeds, _ = costs.exit_fill(units, decision_mc * q25, config)
    result["net_peak_q25"] = proceeds / cash - 1.0
    if occurrence > min(config.swing_entry_max_occurrence_minutes, config.selective_max_hold_minutes):
        result["reason"] = "selective_peak_too_distant"
    elif death > config.selective_max_death_probability or (downside is not None and downside > config.selective_max_downside_probability):
        result["reason"] = "selective_downside_risk_too_high"
    elif result["net_peak_q25"] < config.selective_take_profit_net + config.swing_entry_min_net_upside:
        result["reason"] = "selective_lower_upside_below_cost_buffer"
    elif costs.round_trip_cost_fraction(config, cash) >= config.selective_stop_loss_net:
        result["reason"] = "selective_cost_exceeds_loss_budget"
    else:
        result.update(eligible=True, reason="selective_forecast_eligible")
    return result


@dataclass(frozen=True)
class Trial:
    token: str
    at: pd.Timestamp
    available_at: pd.Timestamp
    decision_mc: float
    state: dict[str, Any]
    times: tuple[pd.Timestamp, ...]
    prices: tuple[float, ...]
    decision_times: tuple[pd.Timestamp, ...] = ()


def replay(trial: Trial, cash: float, config: Any) -> tuple[float | None, str]:
    """An unavailable exit counts as a full loss for screening, never a win.

    No fill means no trade. After a fill, a collection gap or missing sell price
    is unresolved evidence; its conservative screening payoff is -100%. This
    does not fabricate a sale or change the wallet's settlement accounting.
    """
    decision_times = trial.decision_times or trial.times
    points = [(t, p, ready) for t, p, ready in zip(trial.times, trial.prices, decision_times)
              if t > trial.available_at]
    if not points or (points[0][0] - trial.available_at).total_seconds() > 60 * config.selective_max_entry_delay_minutes:
        return None, "unfilled"
    entry_at, entry_mc, _ = points[0]
    if not signal(trial.state, config, cash, trial.decision_mc, entry_mc)["eligible"]:
        return None, "cancelled_at_fill"
    _, _, units = costs.entry_fill(cash, entry_mc, config)
    previous = entry_at
    pending = None
    pending_available = None
    for at, mc, available in points[1:]:
        if (at - previous).total_seconds() > 60 * config.selective_max_path_gap_minutes:
            return -1.0, "unresolved_gap"
        proceeds, _ = costs.exit_fill(units, mc, config)
        net = proceeds / cash - 1.0
        if pending is not None and at > pending_available:
            return net, pending
        if pending is None:
            pending = exit_reason(net, (at - entry_at).total_seconds() / 60, exit_plan(config))
            pending_available = available if pending else None
        previous = at
    return -1.0, "unresolved_exit"


def wilson_lower(wins: int, count: int) -> float:
    if count == 0:
        return 0.0
    p, z2 = wins / count, CONFIDENCE_Z ** 2
    return (p + z2 / (2 * count) - CONFIDENCE_Z * math.sqrt(
        p * (1 - p) / count + z2 / (4 * count ** 2))) / (1 + z2 / count)


@dataclass
class Calibration:
    trials: list[Trial]
    config: Any
    model_hash: str | None
    asof: pd.Timestamp
    status: str = "collecting_selective_evidence"
    cache: dict[float, dict[str, Any]] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return len(self.trials) >= self.config.selective_reference_tokens + self.config.selective_min_validation_tokens

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "decision_policy_version": VERSION,
                "mature_unique_tokens": len(self.trials), "forecast_model_hash": self.model_hash,
                "asof": self.asof.isoformat(), "required_win_lower_bound": self.config.selective_min_win_lower_bound,
                "minimum_validation_tokens_per_region": self.config.selective_min_validation_tokens,
                "confidence_per_region": 0.99, "maximum_regions": 3,
                "outcome": "net next-observation exit; unresolved filled paths count as losses"}

    def regions(self, cash: float) -> dict[str, Any]:
        if cash in self.cache:
            return self.cache[cash]
        # Partition by time before reading outcomes. One deterministic candidate
        # per token prevents repeated minutes/reentries inflating sample counts.
        nref = int(self.config.selective_reference_tokens)
        reference, later = self.trials[:nref], self.trials[nref:]
        result: dict[str, Any] = {"status": "insufficient_selective_reference", "regions": []}
        if len(reference) < nref:
            self.cache[cash] = result
            return result
        scores = [signal(t.state, self.config, cash)["probability"] for t in reference]
        edges = np.unique(np.quantile(scores, [0, 1/3, 2/3, 1]))
        # Embargo validation until all reference paths could have matured.
        embargo = reference[-1].available_at + pd.Timedelta(minutes=(
            self.config.selective_max_entry_delay_minutes + self.config.selective_max_hold_minutes
            + self.config.selective_max_path_gap_minutes))
        intervals = list(zip(edges[:-1], edges[1:])) if len(edges) > 1 else [(edges[0], edges[0])]
        for index, (lower, upper) in enumerate(intervals):
            outcomes, unresolved = [], 0
            for trial in later:
                if trial.available_at <= embargo:
                    continue
                info = signal(trial.state, self.config, cash)
                probability = info["probability"]
                if not info["eligible"] or not (lower <= probability <= upper):
                    continue
                if index > 0 and probability == lower:
                    continue
                net, kind = replay(trial, cash, self.config)
                if net is None:
                    continue
                outcomes.append(net)
                unresolved += int(kind.startswith("unresolved"))
            n = len(outcomes)
            wins = sum(value > 0 for value in outcomes)
            mean = float(np.mean(outcomes)) if n else 0.0
            error = float(np.std(outcomes, ddof=1)) / math.sqrt(n) if n > 1 else float("inf")
            result["regions"].append({"lower": float(lower), "upper": float(upper),
                "validation_tokens": n, "wins": wins, "unresolved": unresolved,
                "win_rate": wins / n if n else None, "win_lower_bound": wilson_lower(wins, n),
                "mean_net_return": mean, "net_return_lower_bound": mean - CONFIDENCE_Z * error if n > 1 else None})
        result["status"] = "selective_validation"
        self.cache[cash] = result
        return result

    def assess(self, state: dict[str, Any], cash: float) -> tuple[swing.ShortTermSetup, str]:
        info = signal(state, self.config, cash)
        reason = info["reason"]
        evidence = {**self.as_dict(), "cash_usd": cash, "net_peak_q25": info["net_peak_q25"],
                    "downside_head_available": info["downside_head_available"]}
        net_edge, qualifies, floor = 0.0, False, None
        if info["eligible"]:
            result = self.regions(cash)
            selected = next((r for r in result["regions"]
                             if r["lower"] <= info["probability"] <= r["upper"]), None)
            reason = result["status"] if not result["regions"] else "outside_selective_reference_support"
            if selected is not None:
                evidence.update(selected)
                floor = selected["lower"]
                net_edge = selected["net_return_lower_bound"] or 0.0
                if selected["validation_tokens"] < self.config.selective_min_validation_tokens:
                    reason = "insufficient_selective_validation_tokens"
                elif selected["win_lower_bound"] < self.config.selective_min_win_lower_bound:
                    reason = "selective_win_evidence_below_target"
                elif net_edge < self.config.selective_min_net_edge:
                    reason = "selective_net_edge_below_buffer"
                else:
                    qualifies, reason = True, "selective_evidence_qualified"
        evidence["reason"] = reason
        setup = swing.ShortTermSetup(
            available=info["q25"] is not None, qualifies=qualifies, score=net_edge,
            probability=info["probability"], occurrence_q50_minutes=info["occurrence"],
            occurrence_spread_minutes=None, confirmation_lag_q50_minutes=None,
            peak_multiple_q50=info["q50"], conservative_upside=max(0.0, (info["q25"] or 1.0) - 1),
            death_probability=info["death"] or 0.0, downside_probability=info["downside"] or 0.0,
            net_edge=net_edge, raw_net_edge=info["net_peak_q25"] or 0.0,
            calibration_status=evidence["status"], calibrated_probability_floor=floor,
            reason=reason, precision_evidence=evidence)
        return setup, VERSION


def build_calibration(conn: sqlite3.Connection, source_db: str, benchmark_id: str,
                      snapshot: pd.Timestamp, model_hash: str | None, config: Any) -> Calibration:
    result = Calibration([], config, model_hash, snapshot)
    maturity_minutes = (config.selective_max_entry_delay_minutes + config.selective_max_hold_minutes
                        + config.selective_max_path_gap_minutes)
    cutoff = snapshot - pd.Timedelta(minutes=maturity_minutes)
    rows = conn.execute(
        """SELECT token_key,snapshot_at,market_cap_usd,state_json FROM benchmark_candidates_v22
           WHERE benchmark_id=? AND forecast_model_hash=? AND snapshot_at<?
           ORDER BY snapshot_at DESC LIMIT ?""",
        (benchmark_id, model_hash, (cutoff + pd.Timedelta(seconds=1)).isoformat(),
         config.swing_calibration_max_samples * 4)).fetchall()
    candidates: dict[str, tuple[Any, ...]] = {}
    for token, at, mc, encoded in reversed(rows):
        if token in candidates:
            continue
        try:
            state = json.loads(encoded)
            if state.get("selective_policy_version") != VERSION:
                continue
            # Old opportunity-score records lacked real decision availability.
            # Do not reinterpret those rows as executable precision evidence.
            available = pd.Timestamp(state.get("selective_available_at"))
            at = pd.Timestamp(at)
            if available.tzinfo is None or at.tzinfo is None:
                continue
            available, at = available.tz_convert("UTC"), at.tz_convert("UTC")
            if available is None or pd.isna(available) or available < at or available > cutoff:
                continue
            mc = float(mc)
            if not math.isfinite(mc) or mc <= 0:
                continue
            if not signal(state, config, config.initial_cash_usd * config.ordinary_position_fraction)["eligible"]:
                continue
            candidates[token] = (at, available, mc, state)
        except (TypeError, ValueError, KeyError):
            continue
    if not candidates:
        return result
    clock_rows = conn.execute(
        "SELECT snapshot_at,available_at FROM benchmark_decision_clock_v24 WHERE benchmark_id=? AND snapshot_at<=?",
        (benchmark_id, snapshot.isoformat())).fetchall()
    clocks = {pd.Timestamp(at): pd.Timestamp(available) for at, available in clock_rows
              if pd.Timestamp(available) <= snapshot}
    tokens = list(candidates)
    paths: dict[str, dict[pd.Timestamp, float]] = {token: {} for token in tokens}
    try:
        with closing(sqlite3.connect(source_db, timeout=10.0)) as source:
            source.execute("PRAGMA query_only=ON")
            for offset in range(0, len(tokens), 250):
                batch = tokens[offset:offset + 250]
                bounds = []
                for token in batch:
                    available = candidates[token][1]
                    end = min(snapshot, available + pd.Timedelta(minutes=maturity_minutes))
                    bounds.extend((token, (available - pd.Timedelta(seconds=1)).isoformat(),
                                   (end + pd.Timedelta(seconds=1)).isoformat()))
                # Each token needs only its decision's execution window, not
                # every observation over the entire history of all candidates.
                marks = ",".join("(?,?,?)" for _ in batch)
                records = source.execute(
                    f"""WITH windows(token,begin_at,end_at) AS (VALUES {marks})
                        SELECT o.token_key,o.snapshot_at,o.market_cap_usd
                        FROM windows w JOIN axiom_observations o
                        ON o.token_key=w.token AND o.snapshot_at>=w.begin_at AND o.snapshot_at<=w.end_at
                        ORDER BY o.observation_id""", bounds).fetchall()
                if not records:
                    continue
                data = pd.DataFrame(records, columns=["token", "at", "price"])
                data["at"] = pd.to_datetime(data["at"], utc=True, format="ISO8601", errors="coerce")
                for token, at, price in data.itertuples(index=False, name=None):
                    available = candidates[token][1]
                    price = swing._finite(price)
                    if at in clocks and price is not None and price > 0 and available < at <= min(
                            snapshot, available + pd.Timedelta(minutes=maturity_minutes)):
                        paths[token][at] = price
    except sqlite3.DatabaseError:
        result.status = "selective_source_outcomes_unavailable"
        return result
    for token, (at, available, mc, state) in candidates.items():
        observations = sorted(paths[token].items())
        result.trials.append(Trial(token, at, available, mc, state,
                                   tuple(t for t, _ in observations), tuple(p for _, p in observations),
                                   tuple(clocks[t] for t, _ in observations)))
    result.trials.sort(key=lambda trial: (trial.available_at, trial.token))
    result.status = "selective_validation" if result.ready else "collecting_selective_evidence"
    return result
