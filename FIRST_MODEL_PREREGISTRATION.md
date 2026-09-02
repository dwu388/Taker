# V24 First-Model Preregistration

This note is part of the target/training contract and must be committed before the first production model is trained.

## Frozen scope

The predictive facade remains the production V24 24-hour lifecycle over the 25-hour Axiom view. The first fitted profile is intentionally reduced; this does not change the raw collection horizon or the durable V24 lifecycle contract.

The first production model must use `profit_taker.v24_pretraining_bootstrap` / `v24_bootstrap.bat`. `--allow-small` is not an approved production-first-model path.

## Primary first-model targets

The reduced profile prioritizes short-horizon momentum/economic decisions:

- triple barriers at 60m and 240m;
- upside barriers +30% and +50%;
- downside barriers -20% and -35%;
- ordered outcome is `up_first`, `down_first`, `neither`, or `censored`;
- the executable reference is the first observation strictly after the decision;
- same-snapshot opposing touches are censored because within-minute ordering is unknowable;
- operational disappearance is not silently converted into a downside price barrier;
- economic collapse is a separate observed event: >=85% drawdown from trailing observed peak sustained >=10 valid minutes;
- economic collapse keeps the observed market return. A -100% settlement is stress analysis only.

## Counterfactual friction

Default round-trip friction is 100 bps. Stress reports are required at 1%, 3%, 5%, and 10% round trip.

ENTRY targets include future entry plus exit cost. HOLD-versus-EXIT comparisons do not recharge already-sunk entry friction; they compare future exit costs. Gross and net returns are both retained, and policy learning is intended to use net targets.

## Readiness gates

The production bootstrap refuses until the core gates pass:

- >=14 calendar days of observed collection span;
- >=300 distinct usable tokens;
- >=50 distinct tokens with confirmed substantial peaks;
- >=1 retained triple-barrier cell with >=30 distinct tokens in each resolved class (`up_first`, `down_first`, `neither`).

Operational-death modeling is modular and remains disabled until >=50 distinct operational-death tokens exist. TS2Vec remains disabled for the first-model profile and is not eligible below 120 distinct training tokens.

Counts are always independent token counts, never one-minute row counts.

## Baselines

Before learned-model bootstrap, two preregistered baselines must be evaluable on a chronological token split:

1. age-bucket base rates;
2. causal 15-minute market-cap momentum (`slope_15m`) with training-only monotone probability calibration.

Baseline metrics are token-balanced Brier loss and token-balanced log loss. The same target and token split must be used for both baselines.

## First learned-model claim threshold

The first learned model should not be described as demonstrating incremental predictive value unless all are true:

- it improves the preregistered primary loss by at least 3% relative to the better simple baseline;
- paired token-bootstrap 95% confidence interval for improvement is above zero;
- no required target family is materially degraded by more than 6%;
- all one-use promotion/audit rules in V24 remain satisfied.

The ordinary later challenger margin can remain smaller; this stronger threshold is specifically for the first claim that the ML system adds value beyond simple rules.

## Deferred model families

The first-model production profile uses <=150 trees/component and short-horizon causal features. TS2Vec, recurrent peak-count heads, later-higher-peak heads, and longer-horizon heads are deferred rather than deleted. They may be enabled only after their own data-readiness gates and one-use promotion evidence are available.
