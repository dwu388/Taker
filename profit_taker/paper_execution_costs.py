"""Explicit, per-transaction assumptions for the isolated Axiom paper wallet.

Rates are per side. Fixed SOL costs are converted with a saved USD/SOL scenario;
the collector does not provide an authenticated, current SOL quote or pool route.
"""
from __future__ import annotations

import math
from typing import Any


def variable_rates(config: Any) -> tuple[float, float]:
    if config.execution_cost_model == "legacy_round_trip":
        half = float(config.friction_bps_round_trip) / 20000.0
        return half, half
    per_side = (
        float(config.axiom_net_fee_bps_per_side)
        + float(config.pool_fee_bps_per_side)
        + float(config.execution_slippage_bps_per_side)
    ) / 10000.0
    return per_side, per_side


def fixed_fee_usd(config: Any) -> float:
    if config.execution_cost_model == "legacy_round_trip":
        return 0.0
    return (
        float(config.priority_fee_sol_per_side)
        + float(config.bribe_sol_per_side)
        + float(config.network_fee_sol_per_side)
    ) * float(config.sol_usd_reference)


def entry_fill(cash_spend: float, market_cap: float, config: Any) -> tuple[float, float, float]:
    """Return exposure notional, total entry cost, and units for a cash budget."""
    entry_rate, _ = variable_rates(config)
    usable = float(cash_spend) - fixed_fee_usd(config)
    if usable <= 0 or market_cap <= 0:
        raise ValueError("Trade cannot cover the fixed entry cost.")
    notional = usable / (1.0 + entry_rate)
    return notional, float(cash_spend) - notional, notional / float(market_cap)


def exit_fill(units: float, market_cap: float, config: Any) -> tuple[float, float]:
    """Return net sale proceeds and total exit cost (including price haircut)."""
    _, exit_rate = variable_rates(config)
    gross = max(0.0, float(units) * float(market_cap))
    proceeds = max(0.0, gross * (1.0 - exit_rate) - fixed_fee_usd(config))
    return proceeds, gross - proceeds


def gross_gain_multiplier(config: Any, cash_spend: float) -> float:
    """Change in net proceeds for one unit of gross market-cap return."""
    if cash_spend <= fixed_fee_usd(config):
        return 0.0
    _, _, units = entry_fill(cash_spend, 1.0, config)
    _, exit_rate = variable_rates(config)
    return units * (1.0 - exit_rate) / cash_spend


def gross_break_even_return(config: Any, cash_spend: float) -> float:
    gain = gross_gain_multiplier(config, cash_spend)
    if gain <= 0:
        return float("inf")
    return round_trip_cost_fraction(config, cash_spend) / gain


def round_trip_cost_fraction(config: Any, cash_spend: float) -> float:
    """Fraction lost on a same-price buy and sell of this cash size."""
    if cash_spend <= fixed_fee_usd(config):
        return 1.0
    _, _, units = entry_fill(cash_spend, 1.0, config)
    proceeds, _ = exit_fill(units, 1.0, config)
    return 1.0 - proceeds / cash_spend


def validate(config: Any) -> None:
    if config.execution_cost_model not in {"axiom_cost_scenario", "legacy_round_trip"}:
        raise ValueError("Unknown paper execution cost model.")
    fields = (
        "friction_bps_round_trip", "axiom_net_fee_bps_per_side",
        "pool_fee_bps_per_side", "execution_slippage_bps_per_side",
        "priority_fee_sol_per_side", "bribe_sol_per_side",
        "network_fee_sol_per_side", "sol_usd_reference",
    )
    for field in fields:
        value = float(getattr(config, field))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative.")
    entry_rate, exit_rate = variable_rates(config)
    if entry_rate >= 1 or exit_rate >= 1:
        raise ValueError("Per-side cost rates must be below 100%.")
    if config.execution_cost_model != "legacy_round_trip" and config.sol_usd_reference <= 0:
        raise ValueError("Provide a positive USD/SOL reference for fixed Solana costs.")
