# Profit Taker — Clean V18 Rebuild

This directory is a clean reconstruction of the latest project lineage documented in the supplied project sources. It folds the later patches into one coherent tree instead of requiring V11/V13/V15/V17/V18 patch-over-patch installation.

## Final behavior carried forward

- **5-minute Axiom collection** from the user's Downloads folder.
- Capture sequence: **Ctrl+A → wait → Ctrl+C → wait → Esc → Ctrl+Shift+E**.
- Clipboard-selected Axiom text is parsed and reconciled against screenshot OCR when identities match.
- Cleaned Axiom compact cards use **named semantic fields**, not generic `risk_slot_*` columns.
- Screenshot card count is dynamic; thumbnail anchors determine candidate cards.
- **PP-OCRv6 via RapidOCR** is the primary recognizer; Tesseract is an independent validator/fallback inside the hybrid decision logic.
- Dangerous numeric fields fail closed on disagreement. Trader semantic contradictions are rejected to `NULL` rather than repaired.
- All usable raw observations are stored. There is no subjective bullish/bearish Axiom PreScore in model admission.
- Neutral, causal five-minute trajectory features are reconstructed at 5/10/15/30/60/90/180 minute scales.
- Axiom persistence is a **positive-only operational visibility bonus** and is excluded from model features.
- Missing tokens create no synthetic market observation and no negative feature.
- A token is operationally dead after **10 consecutive completed capture cycles** without a sighting, except near the natural ~24h Axiom age-out boundary.
- Death is a **competing terminal event per decision timestamp**. Earlier successful decision points remain successful even if the token dies later.
- V18 trains a bank of LightGBM + XGBoost heads for multi-horizon upside/downside probabilities, death risk, expected peak, peak quantiles, time-to-peak, and pre-peak drawdown.
- Training rows are token-grouped chronologically, production uses a 144-hour purge, and total training weight is normalized per token.
- Provider code remains available, but **no provider command is called by the normal Axiom collector**.

## First setup on Windows

```bat
setup_venv.bat
```

Install Tesseract separately if it is not already present. The extractor automatically recognizes the conventional path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

RapidOCR/ONNX validation:

```bat
validate_v15_ocr.bat
```

## Collect one cycle

Open Axiom Pulse/Migrated in the browser and make that page active, then:

```bat
run_axiom_once_debug.bat
```

The capture automation executes:

```text
Ctrl+A
  wait 2.0 s
Ctrl+C
  wait 2.5 s
read clipboard
Esc
  wait 0.75 s
Ctrl+Shift+E
wait for the new scrolling PNG to become stable
```

The default screenshot directory is:

```text
C:\Users\<you>\Downloads
```

Outputs are written beneath:

```text
data\axiom_migrated\
```

including rows CSV/JSON plus `.selection.txt` and `.selection.json` clipboard/OCR diagnostics.

## Continuous five-minute collection

```bat
run_axiom_loop.bat
```

This path is local/offline apart from the browser session used to capture Axiom. It does not load provider credentials or invoke Helius, Shyft, or Birdeye.

## Prepare historical V18 training data

Back up the live database first:

```bat
copy data\live.sqlite data\live_before_v18.sqlite
```

Then:

```bat
prepare_v18_max_inference.bat
```

This rebuilds causal features from all stored observations and then recalculates per-decision 24-hour lifecycle targets.

## Train

Experimental/plumbing model:

```bat
train_v18_max_inference.bat
```

Production-gated model:

```bat
train_v18_production.bat
```

The production trainer refuses to train until the documented minimums are met: roughly 30 calendar days of complete labels, at least 500 positive primary decision examples, and at least 30 distinct tokens.

## Predict

```bat
predict_v18_max_inference.bat
```

Output:

```text
data\axiom_predictions_24h.csv
```

Examples of inference columns, when the corresponding head has enough data to train:

```text
p_hit_plus30_by_1h ... p_hit_plus400_by_24h
p_hit_minus30_by_1h ... p_hit_minus50_by_24h
p_success_2x_before_50dd_or_dead_24h
p_dead_within_1h_24h / 4h / 12h
p_survive_1h / 4h / 12h
pred_peak_multiple_expected
pred_peak_multiple_q10/q25/q50/q75/q90
pred_peak_market_cap_usd_q10...q90
pred_time_to_peak_minutes_24h
pred_drawdown_before_peak_pct_24h
pred_peak_to_trough_before_peak_pct_24h
pred_time_to_death_minutes_if_death_24h
```

Quantiles are monotonically corrected at inference time so later quantiles cannot fall below earlier ones.

## Model isolation

The learned market model intentionally does **not** receive operational visibility/capture-density fields such as:

```text
capture_count
visibility_bonus
consecutive_capture_count
reappearance_count
observation_index
interval_minutes
obs_count_*
```

Repeated captures help only by creating more genuine causal trajectory information. Per-token weighting prevents long-lived tokens from dominating simply because they have more rows.

## Local future-API handoff

Prediction can populate:

```text
axiom_api_handoff_queue
```

and exports:

```text
data\api_handoff_candidates.csv
data\api_handoff_queue.csv
```

The local handoff priority is:

```text
min(1.0, primary_model_probability + visibility_bonus)
```

with visibility capped at +0.12. The queue states are `ready_future_api` when a full mint is known and `awaiting_mint` otherwise. Creating this queue does not make any API request.

## Optional provider path

Provider code from the earlier architecture is retained under `profit_taker/providers/` and is explicit-only. To configure keys:

```bat
copy .env.example .env
configure_and_check.bat
```

To intentionally execute the provider cycle:

```bat
run_provider_cycle.bat
```

That command can consume provider quotas. It is never called by `run_axiom_once.bat`, `run_axiom_loop.bat`, or `run_axiom_loop_with_model.bat`.

## Tests

```bat
.venv\Scripts\python.exe -m pytest
```

The tests cover the critical reconstructed behavior: clipboard reconciliation, non-concatenating numeric parsers, causal features, early-success/later-death lifecycle labeling, reappearance reset, and natural Axiom age-out protection.

## Reusing an older `live.sqlite`

`prepare_v18_max_inference.bat` first runs the defensive legacy importer. If the new `axiom_observations` table is empty, it inspects older SQLite tables for recognizable timestamp/token/MC/volume/TX columns, imports real observed rows, reconstructs completed capture cycles, and rebuilds positive-only visibility. It ignores rows that contain no real observed card/market value, so old synthetic disappearance placeholders are not imported as market observations.

You can run that compatibility step by itself with:

```bat
import_legacy_history.bat
```
