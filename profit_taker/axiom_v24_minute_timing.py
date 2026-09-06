from __future__ import annotations

"""Minute-sensitive peak-timing extension for the production V24 facade.

The long-horizon upside grid remains 1h/4h/8h/12h/24h. This module adds a
separate minute-resolution timing contract: early confirmation hazards plus
quantile regressions for the actual swing-high occurrence timestamp, the lag
until that high becomes confirmation-safe, and the occurrence timing of the
second substantial peak.
"""

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

MINUTE_PEAK_CONFIRMATION_HORIZONS_MINUTES = (5, 10, 15, 30, 60)
NEXT_PEAK_OCCURRENCE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
PEAK_CONFIRMATION_LAG_QUANTILES = (0.25, 0.50, 0.75)
SECOND_PEAK_OCCURRENCE_QUANTILES = (0.25, 0.50, 0.75)
# Preserve the durable 24h-v2 schema family. Minute timing is a model-target
# generation change carried by target_definition_hash, not a storage-layout
# rename that would unnecessarily invalidate compatible durable label tables.
MINUTE_TIMING_SCHEMA_VERSION = "v24_lifetime_purged_adaptive_event_policy_minute_timing_24h_v2"


def _qtag(q: float) -> str:
    return str(int(round(float(q) * 100.0)))


def _minute_contract_payload() -> dict[str, Any]:
    return {
        "first_peak_confirmation_horizons_minutes": list(MINUTE_PEAK_CONFIRMATION_HORIZONS_MINUTES),
        "next_peak_occurrence_target": "confirmed_substantial_peak.peak_at_minus_decision_at",
        "next_peak_occurrence_quantiles": list(NEXT_PEAK_OCCURRENCE_QUANTILES),
        "peak_confirmation_lag_target": "confirmed_at_minus_peak_at",
        "peak_confirmation_lag_quantiles": list(PEAK_CONFIRMATION_LAG_QUANTILES),
        "second_peak_occurrence_target": "second_peak_at_minus_first_peak_at",
        "second_peak_occurrence_quantiles": list(SECOND_PEAK_OCCURRENCE_QUANTILES),
        "confirmation_boundary": "peak_at_and_confirmed_at_must_be_inside_active_24h_followup",
    }


def _project_quantile_family(
    pred: dict[str, np.ndarray],
    keys: tuple[str, ...],
    *,
    floor: float | None = None,
) -> None:
    present = [k for k in keys if k in pred]
    if not present:
        return
    mat = np.column_stack([np.asarray(pred[k], dtype=float) for k in present])
    for i in range(mat.shape[0]):
        mask = np.isfinite(mat[i])
        if mask.sum() > 1:
            mat[i, mask] = np.sort(mat[i, mask])
    if floor is not None:
        finite = np.isfinite(mat)
        mat[finite] = np.maximum(mat[finite], float(floor))
    for j, key in enumerate(present):
        pred[key] = mat[:, j]


def _fit_minute_timing_heads(
    impl,
    model_frame: pd.DataFrame,
    features: list[str],
    recurrent: dict[str, Any] | None,
    n_estimators: int,
) -> dict[str, Any]:
    out = dict(recurrent or {})
    specs: tuple[tuple[str, str, tuple[float, ...]], ...] = (
        ("next_occurrence", "recurrent_next_occurrence_gap_minutes", NEXT_PEAK_OCCURRENCE_QUANTILES),
        ("next_confirmation_lag", "recurrent_next_confirmation_lag_minutes", PEAK_CONFIRMATION_LAG_QUANTILES),
        ("second_occurrence_gap", "recurrent_second_occurrence_gap_minutes", SECOND_PEAK_OCCURRENCE_QUANTILES),
        ("second_confirmation_lag", "recurrent_second_confirmation_lag_minutes", SECOND_PEAK_OCCURRENCE_QUANTILES),
        ("next_gap", "recurrent_next_gap_minutes", (0.10, 0.90)),
        ("second_gap", "recurrent_second_gap_minutes", (0.25, 0.75)),
        ("second_peak_relative", "recurrent_second_peak_relative_to_first", (0.25, 0.75)),
    )
    for prefix, target, quantiles in specs:
        if target not in model_frame.columns:
            continue
        d = model_frame[pd.to_numeric(model_frame[target], errors="coerce").notna()].copy()
        if d.empty:
            continue
        for q in quantiles:
            fit = impl._fit_blended_regression(d, features, target, n_estimators, quantile=float(q))
            if fit:
                out[f"{prefix}_q{_qtag(q)}"] = fit
    return out


def install(public, core, base, impl) -> None:
    if getattr(public, "_minute_peak_timing_installed", False):
        return

    BaseConfig = core.V24Config

    @dataclass
    class MinuteSensitiveV24Config(BaseConfig):
        survival_bins_minutes: tuple[int, ...] = (
            5, 10, 15, 30, 60, 120, 240, 480, 720, 1440
        )

        def __post_init__(self) -> None:
            parent = getattr(super(), "__post_init__", None)
            if parent is not None:
                parent()
            horizon = int(self.horizon_minutes)
            bins = tuple(int(x) for x in self.survival_bins_minutes)
            if tuple(sorted(set(bins))) != bins:
                raise ValueError("survival_bins_minutes must be strictly increasing and unique")
            if any(x <= 0 or x > horizon for x in bins):
                raise ValueError("survival_bins_minutes must be inside the active horizon")
            missing = set(MINUTE_PEAK_CONFIRMATION_HORIZONS_MINUTES) - set(bins)
            if missing:
                raise ValueError(f"minute peak confirmation bins missing: {sorted(missing)}")

    MinuteSensitiveV24Config.__name__ = "V24Config"
    MinuteSensitiveV24Config.__qualname__ = "V24Config"

    modules = (public, core, base, impl)
    for module in modules:
        setattr(module, "V24Config", MinuteSensitiveV24Config)
        setattr(module, "SCHEMA_VERSION", MINUTE_TIMING_SCHEMA_VERSION)
        setattr(module, "MINUTE_PEAK_CONFIRMATION_HORIZONS_MINUTES", MINUTE_PEAK_CONFIRMATION_HORIZONS_MINUTES)
        setattr(module, "NEXT_PEAK_OCCURRENCE_QUANTILES", NEXT_PEAK_OCCURRENCE_QUANTILES)
        setattr(module, "PEAK_CONFIRMATION_LAG_QUANTILES", PEAK_CONFIRMATION_LAG_QUANTILES)
        setattr(module, "SECOND_PEAK_OCCURRENCE_QUANTILES", SECOND_PEAK_OCCURRENCE_QUANTILES)

    original_target_definition_hash = core.target_definition_hash

    def target_definition_hash(cfg) -> str:
        payload = {
            "base_target_definition_hash": str(original_target_definition_hash(cfg)),
            "minute_peak_timing": _minute_contract_payload(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    for module in modules:
        setattr(module, "target_definition_hash", target_definition_hash)

    original_add_recurrent_targets = core.add_recurrent_targets

    def add_recurrent_targets(conn, frame, cfg, *, as_of=None):
        out = original_add_recurrent_targets(conn, frame, cfg, as_of=as_of)
        for col in (
            "recurrent_next_confirmation_lag_minutes",
            "recurrent_second_occurrence_gap_minutes",
            "recurrent_second_confirmation_lag_minutes",
        ):
            out[col] = np.nan

        peaks = impl._future_peak_lists(conn)
        for i, row in out.iterrows():
            decision = impl._utc(row.decision_at)
            known_end = impl._known_followup_end(row, cfg, as_of)
            events = peaks.get(str(row.token_key), [])
            eligible = [
                event for event in events
                if decision < event["peak_at"] <= decision + pd.Timedelta(minutes=cfg.horizon_minutes)
                and decision < event["confirmed_at"] <= known_end
            ]
            eligible.sort(key=lambda event: event["peak_at"])
            if eligible:
                first = eligible[0]
                out.at[i, "recurrent_next_confirmation_lag_minutes"] = max(
                    0.0, (first["confirmed_at"] - first["peak_at"]).total_seconds() / 60.0
                )
            if len(eligible) >= 2:
                first, second = eligible[0], eligible[1]
                out.at[i, "recurrent_second_occurrence_gap_minutes"] = max(
                    0.0, (second["peak_at"] - first["peak_at"]).total_seconds() / 60.0
                )
                out.at[i, "recurrent_second_confirmation_lag_minutes"] = max(
                    0.0, (second["confirmed_at"] - second["peak_at"]).total_seconds() / 60.0
                )
        return out

    for module in modules:
        setattr(module, "add_recurrent_targets", add_recurrent_targets)

    original_derive_adapter_head_weights = core._derive_adapter_head_weights

    def _derive_adapter_head_weights(adapter: dict, cfg) -> dict[str, float]:
        out = dict(original_derive_adapter_head_weights(adapter, cfg))
        base_weights = adapter.get("weights") or {}
        hazard_weight = float(base_weights.get("hazard", 0.0))
        recurrent_weight = float(base_weights.get("recurrent", 0.0))
        for h in MINUTE_PEAK_CONFIRMATION_HORIZONS_MINUTES:
            scale = max(0.75, min(1.35, (60.0 / float(h)) ** 0.20))
            weight = min(float(cfg.adapter_max_weight), hazard_weight * scale)
            out[f"p_first_peak_by_{h}m"] = weight
            out[f"p_death_by_{h}m"] = weight
            out[f"p_event_free_through_{h}m"] = weight
            out[f"p_event_free_by_{h}m"] = weight
        timing_keys = [
            *(f"next_occurrence_q{_qtag(q)}" for q in NEXT_PEAK_OCCURRENCE_QUANTILES),
            *(f"next_confirmation_lag_q{_qtag(q)}" for q in PEAK_CONFIRMATION_LAG_QUANTILES),
            *(f"second_occurrence_gap_q{_qtag(q)}" for q in SECOND_PEAK_OCCURRENCE_QUANTILES),
            *(f"second_confirmation_lag_q{_qtag(q)}" for q in SECOND_PEAK_OCCURRENCE_QUANTILES),
            "next_gap_q10", "next_gap_q90", "second_gap_q25", "second_gap_q75",
            "second_peak_relative_q25", "second_peak_relative_q75",
        ]
        for key in timing_keys:
            out[key] = recurrent_weight
        return out

    for module in modules:
        setattr(module, "_derive_adapter_head_weights", _derive_adapter_head_weights)

    original_project_recurrent_outputs = core.project_recurrent_outputs

    def project_recurrent_outputs(pred: dict[str, np.ndarray]) -> None:
        original_project_recurrent_outputs(pred)
        _project_quantile_family(pred, tuple(f"next_occurrence_q{_qtag(q)}" for q in NEXT_PEAK_OCCURRENCE_QUANTILES), floor=0.0)
        _project_quantile_family(pred, ("next_gap_q10", "next_gap_q25", "next_gap_q50", "next_gap_q75", "next_gap_q90"), floor=0.0)
        _project_quantile_family(pred, tuple(f"next_confirmation_lag_q{_qtag(q)}" for q in PEAK_CONFIRMATION_LAG_QUANTILES), floor=0.0)
        _project_quantile_family(pred, ("next_peak_multiple_q25", "next_peak_multiple_q50", "next_peak_multiple_q75"), floor=0.0)
        _project_quantile_family(pred, tuple(f"second_occurrence_gap_q{_qtag(q)}" for q in SECOND_PEAK_OCCURRENCE_QUANTILES), floor=0.0)
        _project_quantile_family(pred, tuple(f"second_confirmation_lag_q{_qtag(q)}" for q in SECOND_PEAK_OCCURRENCE_QUANTILES), floor=0.0)
        _project_quantile_family(pred, ("second_gap_q25", "second_gap_q50", "second_gap_q75"), floor=0.0)
        _project_quantile_family(pred, ("second_peak_relative_q25", "second_peak_relative_q50", "second_peak_relative_q75"))

    for module in modules:
        setattr(module, "project_recurrent_outputs", project_recurrent_outputs)

    original_fit_cpcv_calibrators = impl.fit_cpcv_calibrators

    def fit_cpcv_calibrators(conn, train, seqraw, cfg, allow_small):
        expanded = tuple(sorted(set(cfg.promotion_required_horizons_minutes) | set(MINUTE_PEAK_CONFIRMATION_HORIZONS_MINUTES)))
        calibration_cfg = replace(cfg, promotion_required_horizons_minutes=expanded)
        return original_fit_cpcv_calibrators(conn, train, seqraw, calibration_cfg, allow_small)

    for module in modules:
        setattr(module, "fit_cpcv_calibrators", fit_cpcv_calibrators)

    original_fit_batch_bundle = core.fit_batch_bundle

    def fit_batch_bundle(conn, frame, seqraw, cutoff, cfg, *, allow_small=False, generation=1, exclude_tokens=None):
        bundle = original_fit_batch_bundle(conn, frame, seqraw, cutoff, cfg, allow_small=allow_small, generation=generation, exclude_tokens=exclude_tokens)
        train = impl.training_history_before(conn, frame, cutoff, cfg, exclude_tokens=exclude_tokens)
        model_frame, _ = impl._prepare_model_frame(conn, train, seqraw, cfg, encoder=bundle["sequence_encoder"], sequence_challenger=bundle.get("sequence_challenger"), as_of=cutoff)
        n_estimators = cfg.small_estimators if allow_small else cfg.stable_estimators
        bundle["recurrent"] = _fit_minute_timing_heads(impl, model_frame, bundle["features"], bundle.get("recurrent"), n_estimators)
        bundle["minute_peak_timing_contract"] = _minute_contract_payload()
        return bundle

    for module in modules:
        setattr(module, "fit_batch_bundle", fit_batch_bundle)

    original_fit_online_adapter = core.fit_online_adapter

    def fit_online_adapter(conn, champion, frame, seqraw, cutoff, cfg, *, allow_small=False, exclude_tokens=None):
        adapter = original_fit_online_adapter(conn, champion, frame, seqraw, cutoff, cfg, allow_small=allow_small, exclude_tokens=exclude_tokens)
        stable_cutoff = impl._utc(champion["stable_training_cutoff"])
        recent = impl.adapter_history_before(conn, frame, cutoff, stable_cutoff, cfg, exclude_tokens=exclude_tokens)
        model_frame, _ = impl._prepare_model_frame(conn, recent, seqraw, cfg, encoder=champion["sequence_encoder"], sequence_challenger=champion.get("sequence_challenger"), as_of=cutoff)
        n_estimators = max(30, cfg.adapter_estimators // (2 if allow_small else 1))
        adapter["recurrent"] = _fit_minute_timing_heads(impl, model_frame, champion["features"], adapter.get("recurrent"), n_estimators)
        adapter["head_weights"] = _derive_adapter_head_weights(adapter, cfg)
        adapter["minute_peak_timing_contract"] = _minute_contract_payload()
        return adapter

    for module in modules:
        setattr(module, "fit_online_adapter", fit_online_adapter)

    original_lifecycle_contract = core.lifecycle_contract

    def lifecycle_contract() -> dict[str, Any]:
        out = dict(original_lifecycle_contract())
        out.update(
            schema_version=MINUTE_TIMING_SCHEMA_VERSION,
            first_peak_confirmation_horizons_minutes=list(MINUTE_PEAK_CONFIRMATION_HORIZONS_MINUTES),
            next_peak_occurrence_quantiles=list(NEXT_PEAK_OCCURRENCE_QUANTILES),
            peak_confirmation_lag_quantiles=list(PEAK_CONFIRMATION_LAG_QUANTILES),
            second_peak_occurrence_quantiles=list(SECOND_PEAK_OCCURRENCE_QUANTILES),
            peak_timing_semantics={
                "occurrence": "peak_at - decision_at",
                "confirmation": "confirmed_at - decision_at",
                "confirmation_lag": "confirmed_at - peak_at",
                "secondary_occurrence_gap": "second_peak_at - first_peak_at",
            },
        )
        return out

    for module in modules:
        setattr(module, "lifecycle_contract", lifecycle_contract)
        setattr(module, "_fit_minute_timing_heads", _fit_minute_timing_heads)

    public._minute_peak_timing_installed = True
