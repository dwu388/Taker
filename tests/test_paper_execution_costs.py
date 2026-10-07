from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from profit_taker import paper_execution_costs as costs


def scenario(**overrides):
    values = dict(
        execution_cost_model="axiom_cost_scenario",
        friction_bps_round_trip=100.0,
        axiom_net_fee_bps_per_side=85.0,
        pool_fee_bps_per_side=125.0,
        execution_slippage_bps_per_side=50.0,
        priority_fee_sol_per_side=0.001,
        bribe_sol_per_side=0.001,
        network_fee_sol_per_side=0.000005,
        sol_usd_reference=150.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class PaperExecutionCostsTests(unittest.TestCase):
    def test_default_gold_scenario_charges_each_component_on_both_sides(self):
        cfg = scenario()
        costs.validate(cfg)
        self.assertEqual(costs.variable_rates(cfg), (0.026, 0.026))
        self.assertAlmostEqual(costs.fixed_fee_usd(cfg), 0.30075)
        notional, entry_fee, units = costs.entry_fill(50.0, 100.0, cfg)
        self.assertAlmostEqual(notional, (50.0 - 0.30075) / 1.026)
        self.assertAlmostEqual(entry_fee, 50.0 - notional)
        proceeds, exit_fee = costs.exit_fill(units, 100.0, cfg)
        self.assertAlmostEqual(proceeds, notional * 0.974 - 0.30075)
        self.assertAlmostEqual(exit_fee, notional - proceeds)
        self.assertAlmostEqual(costs.round_trip_cost_fraction(cfg, 50.0), 1.0 - proceeds / 50.0)
        breakeven = costs.gross_break_even_return(cfg, 50.0)
        sale, _ = costs.exit_fill(units, 100.0 * (1.0 + breakeven), cfg)
        self.assertAlmostEqual(sale, 50.0)

    def test_fixed_sol_cost_makes_small_trades_less_attractive(self):
        cfg = scenario()
        self.assertGreater(costs.round_trip_cost_fraction(cfg, 25.0), costs.round_trip_cost_fraction(cfg, 100.0))
        self.assertGreater(costs.round_trip_cost_fraction(scenario(sol_usd_reference=250), 50.0),
                           costs.round_trip_cost_fraction(cfg, 50.0))
        with self.assertRaises(ValueError):
            costs.entry_fill(0.25, 100.0, cfg)

    def test_legacy_wallet_retains_original_fee_math(self):
        cfg = scenario(execution_cost_model="legacy_round_trip")
        self.assertEqual(costs.variable_rates(cfg), (0.005, 0.005))
        self.assertEqual(costs.fixed_fee_usd(cfg), 0.0)
        notional, fee, units = costs.entry_fill(50.0, 100.0, cfg)
        self.assertAlmostEqual(notional, 50.0 / 1.005)
        proceeds, _ = costs.exit_fill(units, 100.0, cfg)
        self.assertAlmostEqual(proceeds, notional * 0.995)

    def test_invalid_cost_configuration_rejected(self):
        for cfg in (scenario(sol_usd_reference=0), scenario(bribe_sol_per_side=-0.001),
                    scenario(pool_fee_bps_per_side=math.nan)):
            with self.assertRaises(ValueError):
                costs.validate(cfg)


if __name__ == "__main__":
    unittest.main()
