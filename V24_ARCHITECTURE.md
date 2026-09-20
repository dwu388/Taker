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
- First-bootstrap vintage can be reconstructed from canonical immutable inserts only
  when the production collection session, completed clipboard-valid cycle, successful
  attempt and archived payload independently corroborate the observation timestamp and
  original insertion time. Replayed, altered or uncorroborated history remains blocked.
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

Wide sequence fingerprints remain durable JSON in SQLite and are decoded one token
at a time. Training retains only the compact PCA/TS2Vec embeddings in memory, so a
mature one-minute cache is never duplicated into a full list of Python dictionaries
and a second wide pandas frame. Encoder fitting uses deterministic, evenly spaced
token-balanced rows.

Estimator fitting is bounded to 50,000 leakage-safe development rows and at most 384
evenly spaced rows per token by default. Every development token remains represented,
first/last lifecycle coverage is retained whenever a token receives at least two
rows, and token weights still equalize total contribution. This bound applies only
to fitting and internal CPCV/calibration. One-use promotion and sealed-audit scoring
remain unsampled and use every eligible evaluation row.

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

## Live inference and collector concurrency

- Live prediction is label-free. It calculates causal current-state and sequence features directly from the newest durable raw capture; it never selects the newest row from the label-dependent training frame.
- Every published prediction frame must contain exactly the latest usable raw `snapshot_at`. If collection advances during calculation, prediction retries and refuses to replace the last known-good CSV unless it catches up.
- Feature and sequence calculations are read-only. Token-first cohort assignment, capture heartbeat and prediction-ledger provenance are committed together in one short, bounded-retry transaction so the collector retains write priority.
- Prediction CSV publication is atomic. A failed or interrupted write cannot expose a partial CSV to paper trading.
- The benchmark keeps its independent stale-prediction guard and does not write duplicate capture heartbeats into the source database. Repeated loop iterations skip model inference when the current frozen model already has predictions for the newest capture.
- Adaptive calibration learning remains label-dependent maintenance work. Live prediction applies the latest durable calibration state; `maintain` resolves and records new calibration evidence after refreshing mature labels.
- The isolated budget benchmark has training feedback disabled and calls live prediction with `persist_source=False`. Its refresh is completely read-only against the collector database; decisions and provenance stay in the separate benchmark database. Ordinary V24 prediction commands keep source-ledger persistence enabled by default.

## Adaptation, calibration and full-model graduation

- Probability calibration is based on out-of-fold/online-safe predictions.
- Stable and adapter families use separate clocks, weights, and drift handling.
- Shorter-horizon heads may adapt faster while longer 24h heads remain more strongly anchored to the stable model.
- The sequence challenger is causal/TS2Vec-style and requires PyTorch.
- The default first production bootstrap remains deliberately reduced to the 1h/4h forecast and sequence contract. Its fixed promotion requirements are restricted to heads it actually produces.
- `v24_maintain.bat` recognizes that reduced champion and does not keep warm-adapting it indefinitely. Once a fresh one-use forecast-promotion cohort is fully mature, maintenance trains a clean full-contract challenger using the current 1h/4h/8h/12h/24h defaults while preserving the champion's holdout cadence, execution assumptions and other non-profile settings.
- A reduced champion cannot be scored on heads it never produced. Graduation therefore applies the ordinary paired token-level promotion rule only to fixed components shared by both models. The full challenger must also have complete fixed coverage for every newly introduced required component, at least the configured independent-token minimum, and each new binary Brier component must beat the 0.25 error of an uninformative `p=0.5` predictor.
- The graduation cohort is consumed whether the full challenger wins or loses. A rejected challenger leaves the reduced champion untouched and a later attempt must use a genuinely new promotion cohort. A promoted full challenger persists the full configuration in `champion.joblib`, after which ordinary adapter/compaction maintenance proceeds under that full contract.

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
