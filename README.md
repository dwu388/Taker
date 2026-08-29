# Taker V24

This is the clean V24-only line of the Profit Taker research system.

## Current production-research status

The supported authoritative raw-data path is `data/axiom_v24_raw.sqlite`. The
collector is clipboard-only at a one-minute cadence and is designed to fail closed:
each browser copy is preceded by a unique verified clipboard sentinel, partial or
identity-invalid captures are rejected, successful captures store exactly one row
per parsed card, and the exact accepted UTF-8 clipboard payload is retained in
SQLite with SHA-256 + zlib.

`collection_status.bat` is the operational readiness gate. Do not treat a dataset as
an authoritative fresh collection unless it reports `ready_to_collect=true`.

## Active 24-hour lifecycle contract

V24 now predicts the token lifecycle through **24 hours** while Axiom is configured
to expose tokens through **25 hours**. The last visible hour is deliberately a
collection buffer; it is not a target horizon and disappearance at the model boundary
is not learned as operational death.

- prediction horizon: 1,440 minutes (24h)
- Axiom view: 1,500 minutes (25h)
- natural model age-out guard: 1,440 minutes (24h)
- collection cadence: 1 minute
- operational disappearance gap: 50 minutes of contiguous successful captures
- primary probability horizons: 1h, 4h, 8h, 12h and 24h
- causal sequence windows: 1h, 4h, 6h, 12h and 24h

Changing from the former 72h target is a model-generation change. Raw prospective
observations remain usable, but derived labels are rebuilt under the 24h contract and
old 72h forecaster/policy champions are not warm-promoted into the new generation.

## Active architecture

- `profit_taker/axiom_migrated_runner.py`: production-hardened one-minute clipboard-only Axiom capture. No screenshots or OCR.
- `profit_taker/axiom_clipboard.py`: Axiom clipboard parsing, complete-card validation and full-mint matching.
- `profit_taker/axiom_migrated_process.py`: atomic raw observation + exact-payload persistence; silent duplicate loss is forbidden.
- `profit_taker/collection_admin.py`: fresh-session initialization and end-to-end raw collection integrity audit.
- `profit_taker/axiom_peak_structure.py`: 24h confirmation-safe recurrent peak structure, natural-age-out protection and causal feature cache; operational DB IDs are excluded from model inputs.
- `profit_taker/axiom_self_teach.py`: paper-policy compatibility substrate, including next-observable fills and behavior-policy accounting.
- `profit_taker/axiom_v24.py`: 24h V24 forecaster/policy/audit implementation, with token-first holdouts, data vintage, sequence-cache invalidation and sealed prospective audit.
- `profit_taker/axiom_budget_benchmark.py`: V24-aware isolated $1,000 paper benchmark.

V24 model outputs default to `models/axiom_v24`, policy artifacts to
`models/axiom_policy_v24`, and predictions to `data/axiom_predictions_v24.csv`.

## Windows setup

```bat
setup_v24.bat
```

## Start a fresh authoritative collection

Configure the Axiom page to show the 25-hour window, smoke-test one capture, then
inspect readiness:

```bat
run_axiom_once.bat
collection_status.bat
```

If `ready_to_collect=true`, start/resume continuous collection:

```bat
run_axiom_loop.bat
```

The collector does **not** contain a hardcoded training-day trigger. Collection
duration is an evidence/maturity decision, not a stop condition.

## V24 operations

```bat
v24_bootstrap.bat
v24_maintain.bat
v24_status.bat
v24_rebuild_sequence_cache.bat
v24_audit_manifest.bat
v24_benchmark.bat
```

Direct `python -m profit_taker.axiom_v24 ...` commands also default to the canonical
raw DB unless an explicit `--db` is supplied.

See `V24_ARCHITECTURE.md`, `INSTALL_V24.txt`, and `VALIDATION_V24.txt` for the
hardening contract and release gate.

## Scope

Repository readiness for fresh research-data collection is not evidence that the
model already has profitable alpha. Real-money autonomous execution should remain
off until genuinely later one-use promotion cohorts, sealed prospective audit,
paper-policy results, friction stress and execution/liquidity evidence support it.

## Repository hygiene

Generated data, SQLite databases, model artifacts, virtual environments, Python
bytecode, secrets, OCR artifacts, V15-V18 model code, and optional provider
experiments are intentionally excluded from this V24-only branch.
