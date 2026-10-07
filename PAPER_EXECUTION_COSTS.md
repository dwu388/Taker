# Paper execution cost scenario

The Axiom board supplies market-cap observations and model forecasts, not a verified
pool address, trade quote, signed transaction, current SOL/USD quote, or actual fill.
The wallet therefore uses a saved and configurable **scenario**, not a claim that
these costs were charged on-chain. The same scenario prices candidates, calibration,
pending fills, liquidation marks, and closed-trade P&L.

## Default assumptions for a new wallet

| Component | Buy | Sell | Basis |
| --- | ---: | ---: | --- |
| Axiom Gold net fee after cashback | 0.85% | 0.85% | Trade notional, each side |
| Unknown-route pool fee scenario | 1.25% | 1.25% | Trade notional, each side |
| Assumed realized price slippage | 0.50% | 0.50% | Execution haircut, each side |
| Axiom priority setting | 0.001 SOL | 0.001 SOL | Fixed transaction spend |
| Axiom bribe setting | 0.001 SOL | 0.001 SOL | Fixed transaction spend |
| Solana base transaction fee | 0.000005 SOL | 0.000005 SOL | One signature |
| USD/SOL reference | $150 | $150 | **Scenario input, not a live quote** |

Thus a $50 cash buy has $0.30075 fixed cost before the 2.60% variable
entry cost. Selling at the same observed market cap pays the 2.60% variable
exit cost and another $0.30075 fixed cost. The exact break-even movement is
computed from those fills; a flat-price round trip loses roughly 6.25% of
the $50 budget. The configured `swing_entry_min_net_upside` is an additional
margin above the fees.

Axiom publishes its **net fee per trade** by cashback tier: Wood 0.95%,
Bronze 0.90%, Silver 0.875%, Gold 0.85%, Platinum 0.825%, Diamond
0.80%, Champion 0.75%. Gold is the scenario because the prior request
specified 0.85%; it is not Axiom's default tier. The gross platform fee
and cashback must not both be charged when modeling the published net fee.

Axiom lists 0.001 SOL as the default priority setting and 0.001 SOL as the
default bribe setting. They are separate from the platform fee. The bribe
is optional and Axiom may recommend a different amount in its UI. Solana
charges 5,000 lamports per signature, plus the configured priority fee.
For failed on-chain transactions, network fees may still be charged; failed
orders are not simulated by this wallet.

The pool route is **not known** from a verified quote. Pump's bonding curve
and its smallest SOL canonical PumpSwap band charge 1.25% per trade,
with lower bands at higher market caps. Raydium's CLMM and CPMM fees
depend on the actual on-chain pool configuration; 0.25% is a common tier,
and much higher tiers exist. The 1.25% scenario can overstate or understate
a particular route. Token-2022 transfer fees, multi-hop routes, account
creation/rent, dynamic pool fees, failed attempts, price impact beyond the
slippage scenario, and SOL/USD movements are not measured here.

Axiom's slippage **limit** is the maximum allowed price change, not the
realized cost. The 0.50% figure is an explicit assumption and can be
changed. The next-observable market-cap fill remains in place, so some
minute-to-minute price movement is already reflected separately.

## Changing the assumptions

For a **new** paper wallet, the benchmark init CLI accepts
`--axiom-net-fee-bps-per-side`, `--pool-fee-bps-per-side`,
`--execution-slippage-bps-per-side`, `--priority-fee-sol-per-side`,
`--bribe-sol-per-side`, `--network-fee-sol-per-side`, and
`--sol-usd-reference`. `run_paper_loop.bat` accepts the same scenario
overrides for a newly initialized wallet, for example:

```bat
run_paper_loop.bat --sol-usd-reference 115.47 --pool-fee-bps-per-side 125
```

Choose a current USD/SOL reference from your actual trading environment.
The value is saved with the wallet; rerunning with another value does
not rewrite historic trades. To adopt a new scenario, stop the paper loop,
then retire the active account in the **benchmark database only**:

```bat
v24_benchmark.bat init --benchmark-db data\axiom_v24_1000_benchmark.sqlite --reset --sol-usd-reference 115.47
run_paper_loop.bat --sol-usd-reference 115.47
```

This leaves previous paper wallet records and the raw collection and
champion model files intact. Prior wallets whose saved configs have only
`friction_bps_round_trip` continue on their original legacy fee model.

Sources (accessed October 7, 2026):

- [Axiom net fees by tier](https://docs.axiom.trade/getting-started/fees/axiom-fees)
- [Axiom Solana priority and bribe settings](https://docs.axiom.trade/getting-started/fees/solana-fees)
- [Solana fee structure](https://solana.com/docs/core/fees/fee-structure)
- [Pump fee schedule](https://pump.fun/docs/fees)
- [Raydium fee comparison](https://docs.raydium.io/reference/fee-comparison)
- [Jito tip behavior](https://docs.jito.wtf/lowlatencytxnsend/)
