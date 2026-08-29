# Validation report

Validation performed in the rebuild environment:

- Python compile check across all package modules and tests: passed.
- Core regression suite: **8 passed**.
- Clipboard parser/reconciliation: partial-card identity matching and value correction passed.
- OCR parsers: separated numeric groups are not concatenated; dev-pair and duration grammar tests passed.
- Causal feature test: appending a future observation does not alter an earlier decision's features.
- Lifecycle test: an early 2x decision remains successful when the same token later reaches 10 missed cycles; a later decision on that token fails if death occurs first.
- Reappearance test: a sighting before missed cycle 10 resets the death counter.
- Natural age-out test: disappearance after an approximately 23h last-seen age is classified as `age_out_ambiguity`, not death.
- Legacy SQLite compatibility test: older observation history was discovered, imported, grouped into capture cycles, and visibility reconstructed.
- Synthetic end-to-end ML exercise: **12 tokens / 24 rows**, LightGBM + XGBoost training, expected peak regression, five peak quantile heads, model persistence, and inference all completed successfully. The synthetic bundle trained 7 viable heads and produced monotonic q10/q25/q50/q75/q90 peak outputs.

Environment limitation: RapidOCR/ONNX Runtime are not installed in the rebuild container, so PP-OCRv6 itself could not be runtime-executed here. The project therefore retains the V15 fail-closed production behavior and includes `install_v15_ocr.bat` plus `validate_v15_ocr.bat` for the Windows machine. Tesseract/parser logic and OCR integration syntax were validated locally.

No live Helius/Shyft/Birdeye calls were made during rebuild validation.
