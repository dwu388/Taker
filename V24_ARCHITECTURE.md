# V24 Architecture

V24 is the active modeling line. This document records the invariants that this cleanup preserves rather than redefining them.

## Data and assignment

- Axiom capture is clipboard-only at a one-minute cadence.
- Raw observations are the durable source record; the collector does not precompute the retired V18 feature table.
- Promotion/audit role is assigned at token first-seen for the token lifetime, preventing decision-time holdout leakage.
- Collector heartbeat state is separated from token presence so outages are not interpreted as token death.
- Immutable definition hashes and data-vintage state are part of the V24 audit contract.

## Targets

- Swing/recurrent peak targets preserve both `peak_at` and `confirmed_at`; information is not usable before confirmation.
- Recurrent peaks are explicitly marked.
- V24 uses shared multi-horizon probability outputs with missing-head-safe monotonic projection.
- Hazard/event training equalizes total token contribution; obsolete extra event weighting is not part of the V24 contract.

## Execution and policy learning

- Paper entries and exits execute on the next observable price after the decision, not the price that caused the decision.
- The policy has an independent one-use challenger/promotion system.
- Candidate action propensities and counterfactual HOLD targets support doubly robust/off-policy evaluation rather than training entry/HOLD only from behavior-policy outcomes.
- Forecast and policy promotion use paired token-level bootstrap evidence.

## Adaptation and calibration

- Probability calibration is based on out-of-fold/online-safe predictions.
- Stable and adapter families use separate clocks, weights, and drift handling.
- The sequence challenger is causal/TS2Vec-style and requires PyTorch.

## Friction and audit

- Liquidity/slippage estimates use historical cutoffs only; friction stress is part of evaluation.
- V24 CLI supports bootstrap, maintenance, prediction, policy training/cross-fitting, status, sequence-cache rebuild, and sealed audit workflows.

## Compatibility modules retained intentionally

`axiom_peak_structure.py`, `axiom_self_teach.py`, and `axiom_budget_benchmark.py` retain some historical schema/table names because V24 imports and relies on their hardened behavior. They are compatibility substrate, not obsolete V21/V22 model entry points.
