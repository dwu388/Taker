# Axiom V18 ML pipeline

```text
Axiom Migrated page
  -> Ctrl+A / Ctrl+C selected DOM text
  -> Esc
  -> Ctrl+Shift+E scrolling screenshot
  -> dynamic card anchors
  -> fixed semantic cells
  -> PP-OCRv6 + Tesseract validation
  -> clipboard/OCR reconciliation
  -> raw observations in SQLite
  -> causal five-minute feature reconstruction
  -> per-decision lifecycle outcomes
  -> token-grouped chronological training
  -> LightGBM + XGBoost model bank
  -> multi-dimensional inference
  -> local future-API handoff queue
```

## Model head families

Maximum available heads: 44.

- 20 upside classifiers: +30/+50/+100/+200/+400% by 1h/4h/12h/24h.
- 8 downside classifiers: -30/-50% by 1h/4h/12h/24h.
- 3 competing-event classifiers.
- 3 death classifiers: death within 1h/4h/12h.
- 5 peak-multiple quantile regressors: q10/q25/q50/q75/q90.
- 5 continuous regressions: expected log peak multiple, log time-to-peak, drawdown below entry before peak, peak-to-trough drawdown before peak, and log time-to-death conditional on death.

Each head trains only when its own target is known and has enough class/row/token support. A missing or single-class head is skipped rather than fabricated.

## Death semantics

A token becomes operationally dead only after 10 consecutive completed capture cycles without a sighting. Reappearance before miss 10 resets the counter. If the last observed age is approximately 23h or older, the disappearance is marked `age_out_ambiguity` rather than death.

For a decision timestamp, +100%, -50%, and death are competing future events. If +100% occurs first, that decision remains successful even if death happens later.

No fake -100% return is invented at death.
