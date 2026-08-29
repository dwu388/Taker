# Taker V24

This branch is the clean V24-only line of the Profit Taker research system.

## Active architecture

- `profit_taker/axiom_migrated_runner.py`: one-minute clipboard-only Axiom capture. No screenshots or OCR.
- `profit_taker/axiom_clipboard.py`: Axiom clipboard parsing and structural validation.
- `profit_taker/axiom_migrated_process.py`: raw observation persistence only.
- `profit_taker/axiom_peak_structure.py`: confirmation-safe recurrent peak structure used by V24. The historical V21 schema name is retained for database compatibility.
- `profit_taker/axiom_self_teach.py`: paper-policy compatibility substrate, including next-observable fills and behavior-policy accounting. Historical table/schema names are retained where required for compatibility.
- `profit_taker/axiom_v24.py`: V24 forecaster/policy/audit implementation.
- `profit_taker/axiom_budget_benchmark.py`: V24-aware isolated budget benchmark.

V24 model outputs default to `models/axiom_v24`, policy artifacts to `models/axiom_policy_v24`, and predictions to `data/axiom_predictions_v24.csv`.

## Windows setup

```bat
setup_v24.bat
```

## Collection

Continuous one-minute clipboard-only capture:

```bat
run_axiom_loop.bat
```

One capture:

```bat
run_axiom_once.bat
```

Capture once and then run the V24 predictor:

```bat
run_v24_capture_predict_once.bat
```

## V24 operations

```bat
v24_bootstrap.bat
v24_maintain.bat
v24_status.bat
v24_rebuild_sequence_cache.bat
v24_audit_manifest.bat
v24_benchmark.bat
```

See `V24_ARCHITECTURE.md`, `INSTALL_V24.txt`, and `VALIDATION_V24.txt` for the retained hardening contract and validation procedure.

## Repository hygiene

Generated data, SQLite databases, model artifacts, virtual environments, Python bytecode, secrets, OCR artifacts, V15-V18 model code, and optional provider experiments are intentionally excluded from this V24-only branch.
