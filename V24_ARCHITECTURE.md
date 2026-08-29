# V24 Architecture

V24 is the active modeling line. This document records the invariants preserved by
the 24-hour lifecycle conversion.

## Lifecycle boundary

- The predictive lifecycle is exactly 24 hours (1,440 minutes).
- The operator configures Axiom to expose tokens through 25 hours (1,500 minutes).
- The final visible hour is a collection/confirmation buffer only; no V24 target may extend beyond 24 hours.
- Natural age-out begins at the 24-hour model boundary and is not a death event.
- Operational disappearance still requires a contiguous successful-capture gap, so collector outages remain censored.
- The former 72-hour target definition is a different model generation. Existing raw observations may be relabeled, but old derived labels/champions cannot be silently reused.

## Data and assignment

- Axiom capture is clipboard-only at a one-minute cadence.
- Raw observations are the durable source record; the collector does not precompute the retired V18 feature table.
- Promotion/audit role is assigned at token first-seen for the token lifetime, preventing decision-time holdout leakage.
- Collector heartbeat state is separated from token presence so outages are not interpreted as token death.
- Immutable definition hashes and data-vintage state are part of the V24 audit contract.
- A changed lifecycle target changes schema/target hashes and forces a clean model generation.

## Targets

- Swing/recurrent peak targets preserve both `peak_at` and `confirmed_at`; information is not usable before confirmation.
- Recurrent peaks are explicitly marked.
- Competing-risk survival ends at 24h with bins through 1,440 minutes.
- Shared upside probability outputs cover 1h, 4h, 8h, 12h and 24h.
- Marked recurrent/higher-peak modeling remains bounded to the 24h lifecycle.
- V24 uses shared multi-horizon probability outputs with missing-head-safe monotonic projection.
- Hazard/event training equalizes total token contribution; obsolete extra event weighting is not part of the V24 contract.

## Sequence representation

The causal sequence representation is re-centered on the shorter lifecycle, using
1h, 4h, 6h, 12h and 24h windows. The TS2Vec-style challenger, sequence-vintage
controls, late-insert invalidation and PCA/stable representation machinery remain.

## Validation clocks

- Calendar cohorts remain 24-hour UTC blocks.
- Token-first promotion/audit assignment remains immutable for the whole token lifetime.
- Promotion maturity remains conservative: cohort end + token lifecycle + outcome horizon. Under the active contract this is approximately 24h + 24h instead of 72h + 72h.
- Sealed audit retains the additional configured audit delay after the complete prospective lifecycle/outcome window.
- CPCV/purge/embargo, paired token bootstrap and fixed required-head coverage remain deployment gates.

## Execution and policy learning

- Paper entries and exits execute on the next observable price after the decision, not the price that caused the decision.
- The policy has an independent one-use challenger/promotion system.
- Candidate action propensities and counterfactual HOLD targets support doubly robust/off-policy evaluation rather than training entry/HOLD only from behavior-policy outcomes.
- Forecast and policy promotion use paired token-level bootstrap evidence.
- A policy champion whose target/execution/schema hashes do not match the active 24h generation is excluded rather than compared as if equivalent.

## Adaptation and calibration

- Probability calibration is based on out-of-fold/online-safe predictions.
- Stable and adapter families use separate clocks, weights, and drift handling.
- Shorter-horizon heads may adapt faster while longer 24h heads remain more strongly anchored to the stable model.
- The sequence challenger is causal/TS2Vec-style and requires PyTorch.

## Friction and audit

- Liquidity/slippage estimates use historical cutoffs only; friction stress is part of evaluation.
- V24 CLI supports bootstrap, maintenance, prediction, policy training/cross-fitting, status, sequence-cache rebuild, and sealed audit workflows.
- Sealed audit remains token-first, prospective, confirmation-safe and development-ineligible.

## Compatibility modules retained intentionally

`axiom_peak_structure.py`, `axiom_self_teach.py`, and `axiom_budget_benchmark.py`
retain historical table/column names because V24 imports and relies on their hardened
behavior. In particular, some durable label column identifiers still contain `72h`.
Those strings are storage compatibility identifiers only; the active truth boundary
is controlled by the 24h configuration, and a target-contract mismatch triggers a
full derived-label rebuild from raw observations.
