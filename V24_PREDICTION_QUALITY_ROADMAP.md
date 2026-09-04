# V24 prediction-quality review update

This note records the September 2026 review of the current `main` prediction stack and separates changes that are safe correctness fixes from changes that should remain explicit challengers.

## Implemented in 0.24.2

### Recurrent count heads use the active contract

The retained implementation still carried historical recurrent-count horizon literals. The active 24-hour V24 contract defines 240m, 480m, 720m and 1440m recurrent targets, but the retained fitter trained 240m, 720m, 1440m and retired 4320m heads. The compatibility layer now derives count horizons from the active probability grid and refits every available recurrent count target. This restores the missing 480m head and prevents retired horizons from surviving as model heads.

Recurrent peak counts are non-negative integer-like targets, so the refit uses the existing Poisson regression path instead of ordinary squared-error regression. Timing and magnitude quantile heads remain unchanged.

### CPCV fold caps are balanced instead of lexicographic

When the full combinatorial purged-CV set exceeds `cpcv_max_splits`, the retained implementation previously took the first lexicographically ordered combinations. That can over-represent early calendar blocks. V24 now builds the valid purged folds and deterministically chooses a subset that balances block coverage and test-block separation before model diagnostics and calibration are fit.

The one-use promotion and sealed-audit cohorts remain unchanged and continue to be the deployment gates.

## High-value options deliberately not mixed into this patch

The following remain worthwhile, but each changes the feature, target or policy contract enough to deserve its own review and regression suite rather than being bundled into a training-correctness patch:

1. Join `axiom_v24_board_membership` and `axiom_v24_capture_context` into the strictly-as-of feature frame. Candidate features include board-rank fraction, rank velocity, board turnover, visible-token count and new-token fraction.
2. Engineer the already-collected deployer fields (`dev_migrations`, `dev_creations`, funding recency, DS status, DEX-paid state and image reuse) into numerical causal features and interactions.
3. Persist ticker text and add cheap character/name-reuse features before considering larger text models.
4. Add cross-sectional capture ranks and endogenous Axiom-board regime features.
5. Add 15m/30m upside probability heads. This changes the target-definition hash and should be introduced together with explicit maturity/censoring tests for the recent adapter.
6. Make recent adapter barrier supervision explicitly head-mature or IPCW-corrected so recent positives cannot enter a head before equivalent unresolved negatives are eligible.
7. Learn LightGBM/XGBoost and stable/adapter blend weights from purged OOS development predictions rather than equal/heuristic blending.
8. Expand the policy from the current 60m counterfactual training slice to the already-materialized 5m/15m/30m/60m/240m curve.
9. Expose per-head uncertainty/calibration radii rather than only the current aggregate radius.
10. Treat direct raw-series TS2Vec/multi-task neural survival models as challengers after the tabular signal and validation fixes above are exhausted.

## Promotion principle

None of the deferred features or model families should replace the current champion merely because they improve in-sample fit. Promotion should continue to require token-purged, time-purged OOS improvement and should be checked for calibration, candidate ranking and execution-adjusted utility under realistic friction assumptions.
