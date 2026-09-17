# Check a fresh bootstrap before fitting

Run each command separately from the Taker directory in Windows Command Prompt.
Stop if a command fails. These checks do not fit or publish any model.

1. Check installed dependency consistency (usually quick):

```bat
.venv\Scripts\python.exe -m pip check
```

2. Scan stored timestamps, including lifetime boundaries (read-only):

```bat
.venv\Scripts\python.exe -m profit_taker.v24_preflight --db data\axiom_v24_raw.sqlite
```

Expect `"passed": true`. This scans all non-null timestamp columns in axiom
tables in batches of 10,000, without rebuilding labels, targets or caches.
Runtime depends on database size. Missing/empty databases and invalid timestamps
fail with exit code 1. Optional NULL timestamps are allowed; observation timestamps
are required. ISO timestamps with and without fractional seconds are accepted.

3. Exercise real label preparation and the training-frame loader, including the
previously failing lifetime join:

```bat
.venv\Scripts\python.exe -m profit_taker.v24_preflight --db data\axiom_v24_raw.sqlite --stage frame --profile full
```

Expect `"passed": true` with nonzero frame rows. This stage can take appreciable
time because it builds labels and sequence features. It uses a temporary SQLite
backup, including committed WAL data; it does not alter the source database,
consume promotion cohorts, evaluate sealed audits, or write model files. Ensure
enough temporary disk space for the database and derived tables. Use
`--temp-dir D:\TakerTemp` to choose an existing directory on another drive.
`mature_promotion_available: false` means a bootstrap requirement is still missing.
This check does not certify model dependencies, training sufficiency, or model
quality. It does not run the pretraining barriers or baseline evaluation.

4. Inspect the cached readiness snapshot without rebuilding pretraining targets:

```bat
.venv\Scripts\python.exe -m profit_taker.v24_contract_runtime_v4 status --db data\axiom_v24_raw.sqlite
```

A missing/stale snapshot is not evidence of readiness. The full bootstrap still
runs authoritative readiness gates and baseline evaluation before fitting.
Do not use `--refresh-pretraining`, `pretraining_cli readiness`, or
`pretraining_cli baselines` expecting a cheap check: those can rebuild historical
targets. `first_model` is a different training profile, not a dry run.

Only after reviewing these checks, run the original expensive command separately:

```bat
.venv\Scripts\python.exe -m profit_taker.v24_contract_runtime_v4 bootstrap --db data\axiom_v24_raw.sqlite --profile full
```

The parser fix uses `format="ISO8601"` for V24 timestamp conversions, preserving
UTC instants and fractional precision. It does not rewrite collected data or
change readiness thresholds, training profiles, or cohort assignments.
