# V24 Minute-Sensitive Peak Timing

This extension keeps the existing primary upside-probability grid at **1h, 4h, 8h, 12h and 24h**. It does not replace those targets with noisy minute-by-minute barrier classifiers.

Instead, V24 now separates three concepts that were previously partially conflated:

1. **Peak occurrence** — when the actual swing high (`peak_at`) happened.
2. **Peak confirmation** — when the subsequent retracement made that swing high safe to call a substantial peak (`confirmed_at`).
3. **Confirmation lag** — `confirmed_at - peak_at`.

A high that occurs at T+10m and is confirmed at T+15m is therefore represented as a 10-minute occurrence with a 5-minute confirmation lag, rather than a 15-minute peak.

## New first-peak timing outputs

The competing-risk hazard grid now includes confirmation checkpoints at:

- 5 minutes
- 10 minutes
- 15 minutes
- 30 minutes
- 60 minutes

These produce cumulative `p_first_peak_by_*m` / `p_death_by_*m` / event-free outputs. They refer to **confirmation-safe event time**, not the actual earlier swing-high occurrence.

The actual high occurrence is modeled separately with conditional quantiles:

- `next_occurrence_q10`
- `next_occurrence_q25`
- `next_occurrence_q50`
- `next_occurrence_q75`
- `next_occurrence_q90`

The existing `next_gap_q25/q50/q75` confirmation-time family remains for compatibility, with new q10/q90 tails. Confirmation uncertainty is also explicit:

- `next_confirmation_lag_q25`
- `next_confirmation_lag_q50`
- `next_confirmation_lag_q75`

## New secondary-peak timing outputs

For the next confirmed substantial peak after the first one, V24 now trains occurrence-gap quantiles from:

`second_peak_at - first_peak_at`

Outputs:

- `second_occurrence_gap_q25`
- `second_occurrence_gap_q50`
- `second_occurrence_gap_q75`

It separately predicts the second peak's own confirmation lag:

- `second_confirmation_lag_q25`
- `second_confirmation_lag_q50`
- `second_confirmation_lag_q75`

The retained confirmation-to-confirmation gap and second-peak relative-height families now also receive q25/q75 bounds around their median outputs.

## Leakage / target-boundary rule

Minute-sensitive occurrence targets are only populated from **substantial peaks that are themselves confirmation-safe inside the active 24-hour follow-up**. The training feature row still contains only information available at the decision timestamp. `peak_at` is used as the retrospective occurrence label only after the later confirmation exists.

The extension therefore does not turn an unconfirmed local high into a positive label and does not allow a confirmation beyond the 24-hour target boundary to back-date a peak into the training target.

## Calibration and adaptation

Sub-hour first-peak confirmation hazards use the same purged out-of-fold calibration path as longer-horizon V24 probabilities. The online adapter receives explicit per-head weights for the new sub-hour hazard and timing heads. Quantile outputs are projected monotonically while preserving missing heads as missing rather than inventing zero-valued constraints.

## Model-generation compatibility

The minute-sensitive timing contract changes the V24 target-definition hash and schema version. Existing raw observations remain usable, but an older champion does not silently acquire these outputs. A new compatible champion must be trained/promoted under the updated target contract.
