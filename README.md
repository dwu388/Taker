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
- `profit_taker/axiom_learning_loop.py`: lossless one-minute collection plus scheduled, gated V24 forecast and policy maintenance. Clipboard snapshots are durably queued while training owns the raw database, then ingested in order.
- `run_axiom_loop.bat`: the single research-learning launcher. It collects database-only raw history, bootstraps or maintains the forecast champion when eligible, refreshes rolling-origin policy predictions, and trains/challenges the policy champion.
- `profit_taker/collection_admin.py`: fresh-session initialization and end-to-end raw collection integrity audit.
- `profit_taker/axiom_peak_structure.py`: 24h confirmation-safe recurrent peak structure, natural-age-out protection and causal feature cache; operational DB IDs are excluded from model inputs.
- `profit_taker/axiom_self_teach.py`: paper-policy compatibility substrate, including next-observable fills and behavior-policy accounting.
- `profit_taker/axiom_v24.py`: 24h V24 forecaster/policy/audit implementation, with token-first holdouts, data vintage, sequence-cache invalidation and sealed prospective audit.
- `profit_taker/axiom_budget_benchmark.py`: V24-aware isolated $1,000 paper benchmark.

The V24 benchmark leaves the trained prediction and policy models unchanged and
applies conviction only at execution: ordinary/strong/exceptional entries target
5%/7.5%/10% of execution-conservative equity. Open positions plus pending orders
are capped at 30%, at least 70% remains unreserved cash, and empirically highly
correlated tokens share a 15% risk-bucket cap. Existing benchmark databases gain
these defaults through the additive migration; no model retraining is required.

The paper wallet also treats each token as a recurrent swing lifecycle. When the
minute-sensitive heads are present, a new entry must have a positive, friction-
adjusted setup across the 5/10/15/30/60-minute peak probabilities, next-occurrence
quantiles and next-peak magnitude. Broader later-higher, second-peak, downside and
death forecasts provide lifecycle context rather than substituting for a timely
entry signal.

An approaching first peak is a decision boundary, not an automatic sale. The
wallet compares remaining hold-through value with the value of selling, releasing
capital and retaining an option to re-enter. Strong, nearby second-peak evidence
keeps the position open. A swing sale creates a durable watch record. Before the
normal 20-minute cooldown expires, the same token can re-enter only after all of
these are observed causally: a minimum three-minute wait, enough retracement to
cover at least twice modeled round-trip friction, lifecycle evidence for another
peak, and a newly recomputed qualifying short-term setup. This prevents immediate
SELL/BUY churn while allowing an early second swing when the evidence supports it.

The strategy remains next-observation executable and paper-only. It does not alter
V24 training labels, retrain a champion, feed benchmark results into training, or
use future data. `v24_benchmark.bat status` reports active watches, completed
re-entries, tokens with multiple swings and the highest swing sequence; benchmark
exports include the swing-watch ledger.

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

This command attempts the complete gated forecast/policy maintenance pipeline at
startup and every 24 hours thereafter. It never promotes on elapsed time alone:
the existing readiness checks, mature one-use cohorts, leakage-safe evaluation and
paired token-level promotion rules remain authoritative. If history is not ready,
the attempt is reported and collection continues. Override only the check cadence,
if needed, with `--training-interval-hours N`.

If a compatible `models/axiom_v24/champion.joblib` was intentionally preserved
across a clean raw-database restart, maintenance links its hash and persisted
training provenance into the new database without deleting, overwriting, or
pretending to re-promote it. The new observations are immediately available to
current-state/sequence inference and derived-label/calibration refreshes. Model
weight changes remain protected by the normal mature one-use promotion gate.
`CURRENT_MODEL_PERFORMANCE.txt` reports the preserved-artifact link separately
from current-database promotion evidence and shows the latest semantic result of
each learning stage.

Clipboard capture remains on its one-minute clock during a long fit. Captures are
compressed into `data\axiom_v24_raw.sqlite.learning_queue.sqlite`; ingestion pauses
while training writes, then drains every queued board in timestamp order. On
restart, pending captures are recovered before a new collection session or training
attempt begins. Existing raw data, candidates, and champions are never reset.

Press Ctrl+C once to stop new captures. Allow active training to finish and the
queue to drain so the neutral collection-stop boundary can be recorded cleanly.

## Execute the trained strategy in the paper wallet

Stop the existing collector and benchmark loops, then run:

```bat
run_paper_loop.bat
```

For the first-model baseline files, use:

```bat
run_paper_loop.bat --forecast-model models\axiom_v24\baseline_first_champion.joblib --policy-model models\axiom_policy_v24\baseline_bootstrap_policy.joblib
```

The launcher resumes the existing $1,000 benchmark wallet (or initializes one if
absent). This is the execution launcher: it does not reset balances, change sizing,
train, or promote models. Stop `run_axiom_loop.bat` before starting it because both
commands own the desktop clipboard collector and canonical raw database.
Default model paths are `models/axiom_v24/champion.joblib` and
`models/axiom_policy_v24/champion.joblib`; an absent policy uses the existing
bootstrap rules. A valid V24 forecast model is required before capture starts.

Clipboard copying runs in the foreground every 60 seconds, independently of two
separate processing processes. It reuses the production click coordinates, verified
clipboard sentinel, Ctrl+A / Ctrl+C retries and 601-cycle refresh. It journals only
compressed clipboard text and capture timestamps to
`data/axiom_v24_raw.sqlite.paper_queue.sqlite`. The ingester validates and saves all
queued observations and exact payloads into the usual raw database. Independently,
the trader predicts on the newest durable board and updates the paper wallet. The queue deletes
acknowledged items and reuses its space; it survives interruption. SQLite can also
create temporary WAL/SHM sidecars, and a small ownership marker prevents two
updated collectors from controlling the same source database simultaneously.

This mode skips per-capture TXT/JSON/CSV review exports, the hourly performance
report and capture-context diagnostics. It keeps the required prediction CSV and
wallet database. Use `performance_report.bat` when a manual report is needed.

If prediction takes longer than a minute, clipboard capture and durable raw
ingestion continue without waiting for it. When inference completes, the trader
immediately advances to the newest durable board instead of replaying a FIFO
decision backlog; superseded intermediate boards remain in raw history but may not
receive individual wallet decisions. Each forecast is pinned to the exact board
read at inference start, even if ingestion advances concurrently. Pending
entries/exits can fill only on a board captured after the decision became available.
Wallet activity is skipped if its snapshot is older than 180 seconds, checked both
before and after prediction (`--max-snapshot-age-seconds` overrides this limit).
Desktop stalls and resource exhaustion can still delay capture; this is not a
hard real-time scheduler.

Press Ctrl+C once to stop new captures, finish in-flight work, drain the queue and
record a neutral collection-stop boundary. Allow that shutdown to complete. On an
unclean restart, pending captures are recovered before a new collection run starts;
old queued boards are not traded at startup. Storage failures stop the ingester and
leave unacknowledged captures queued for recovery. Do not run another collector,
benchmark loop or training/maintenance writer against these files concurrently.

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
