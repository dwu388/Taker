# Axiom clipboard-only collector

This patch removes screenshots and OCR from the **active Axiom collection path**.

The collector now does only:

```text
focus Axiom
-> optional browser zoom-out
-> Ctrl+A
-> Ctrl+C
-> validate copied Axiom structure
-> parse the copied DOM text directly
-> write observations
-> build existing V18 features / predictions
-> wait for the next five-minute cycle
```

There is no `Ctrl+Shift+E`, no screenshot wait, no `extract_screenshot()`, no RapidOCR/Tesseract call, and no clipboard/OCR reconciliation.

Every newly collected row has:

```text
data_origin = clipboard_only
field_source = clipboard_primary (or missing)
ocr_diagnostics = {}
```

The shortened address is mandatory and becomes the token key. The collector fails closed when the clipboard does not look like a valid Axiom Migrated selection, so stale clipboard text is not reused.

## Install

From the extracted patch folder:

```bat
.venv\Scripts\python.exe apply_clipboard_only_patch.py "C:\Users\yiumi\Downloads\Profit Taker"
```

Or copy these two files over the matching project files:

```text
profit_taker\axiom_clipboard.py
profit_taker\axiom_migrated_runner.py
```

The installer backs up the existing copies first.

## Normal operation

One cycle:

```bat
.venv\Scripts\python.exe -m profit_taker.axiom_migrated_runner --db data\live.sqlite --once
```

Continuous five-minute collection:

```bat
.venv\Scripts\python.exe -m profit_taker.axiom_migrated_runner --db data\live.sqlite
```

With the trained model:

```bat
.venv\Scripts\python.exe -m profit_taker.axiom_migrated_runner --db data\live.sqlite --with-model
```

## Validate a saved clipboard capture

```bat
.venv\Scripts\python.exe validate_clipboard_only.py "Axiom-Pulse.selection.txt" --rows
```

## OCR files

Existing OCR modules and installed OCR packages are intentionally not deleted. Other historical/debug tools may still refer to them. The live collector no longer imports or invokes them, so they cannot contaminate newly collected observations.

This patch does not rewrite historical database rows. Previously stored OCR/hybrid observations remain in `data/live.sqlite` unless they are separately purged/rebuilt.
