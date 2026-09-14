from __future__ import annotations

"""Single-refresh execution layer for the V24 pretraining contract.

The continuity-safe V4 target definitions remain authoritative. This layer only
changes materialization orchestration: a top-level command may rebuild the large
historical target table once and readiness/baseline consumers in that same command
reuse it. Separate commands still refresh by default, so no stale cross-command
cache is introduced.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from . import pretraining_contract as _v1
from . import pretraining_contract_v4 as _compat
from . import pretraining_capture_index

# Direct pretraining CLI use should receive the same indexed collector-heartbeat
# lookup as the official runtime, not only the bootstrap command path.
pretraining_capture_index.install(_compat)

# Freeze the V4 call targets before the official runtime installs the wrappers
# back onto the public V4 module for compatibility with existing imports/tests.
_v4_refresh = _compat.refresh_pretraining_targets
_v4_training_readiness = _compat.training_readiness
_v4_evaluate_baselines = _compat.evaluate_baselines

# Re-export the complete public V4 contract surface. Target semantics and hashes
# are intentionally unchanged by this execution-only optimization.
for _name in dir(_compat):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_compat, _name)


# V2/V3 readiness helpers ultimately resolve the V1 materializer from the V1
# module globals. Save the real materializer, then replace that one binding with a
# context-aware guard so nested compatibility calls cannot trigger another full
# historical rebuild while a consumer is reading an already-materialized table.
_v1_materializer = _v1.refresh_pretraining_targets
_suppress_nested_refresh: ContextVar[bool] = ContextVar(
    "v24_pretraining_suppress_nested_refresh", default=False
)
_command_scope_id: ContextVar[object | None] = ContextVar(
    "v24_pretraining_command_scope_id", default=None
)
_scope_materialized_keys: ContextVar[frozenset[tuple[str, str]]] = ContextVar(
    "v24_pretraining_scope_materialized_keys", default=frozenset()
)


def _refresh_key(db: str, cfg: PretrainingConfig) -> tuple[str, str]:
    return (str(db), str(target_contract_hash(cfg)))


def _guarded_v1_refresh(
    db: str,
    cfg: PretrainingConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or PretrainingConfig()
    if _suppress_nested_refresh.get():
        return {
            "written": 0,
            "skipped": True,
            "reason": "nested_refresh_suppressed_already_materialized",
            "target_definition_hash": target_contract_hash(cfg),
        }
    return _v1_materializer(db, cfg)


# Patch only the V1 name used by the retained V2/V3 compatibility call chain.
# The actual V4 public refresh below still reaches the saved materializer when the
# guard is not active, preserving all continuity-safe barrier/collapse semantics.
_v1.refresh_pretraining_targets = _guarded_v1_refresh


@contextmanager
def command_refresh_scope() -> Iterator[None]:
    """Create a short-lived scope in which one successful refresh may be reused.

    The marker is process-local and reset when the command returns. It is never a
    persistent cache and therefore cannot make a later CLI invocation believe old
    targets are current.
    """

    scope_token = _command_scope_id.set(object())
    materialized_token = _scope_materialized_keys.set(frozenset())
    try:
        yield
    finally:
        _scope_materialized_keys.reset(materialized_token)
        _command_scope_id.reset(scope_token)


def _mark_materialized(db: str, cfg: PretrainingConfig) -> None:
    if _command_scope_id.get() is None:
        return
    keys = set(_scope_materialized_keys.get())
    keys.add(_refresh_key(db, cfg))
    _scope_materialized_keys.set(frozenset(keys))


def _materialized_in_current_scope(db: str, cfg: PretrainingConfig) -> bool:
    return (
        _command_scope_id.get() is not None
        and _refresh_key(db, cfg) in _scope_materialized_keys.get()
    )


def refresh_pretraining_targets(
    db: str,
    cfg: PretrainingConfig | None = None,
) -> dict[str, Any]:
    """Run one real V4 materialization and advertise it only inside this command."""

    cfg = cfg or PretrainingConfig()
    out = _v4_refresh(db, cfg)
    _mark_materialized(db, cfg)
    if isinstance(out, dict):
        out = dict(out)
        out.setdefault("materialization_mode", "full_refresh")
    return out


def _ensure_materialized(
    db: str,
    cfg: PretrainingConfig,
    refresh_targets: bool | None,
) -> bool:
    """Return True when this call performed the full refresh itself."""

    if refresh_targets is False:
        return False
    if refresh_targets is None and _materialized_in_current_scope(db, cfg):
        return False
    refresh_pretraining_targets(db, cfg)
    return True


def _without_nested_refresh(fn, *args, **kwargs):
    token = _suppress_nested_refresh.set(True)
    try:
        return fn(*args, **kwargs)
    finally:
        _suppress_nested_refresh.reset(token)


def training_readiness(
    db: str,
    cfg: PretrainingConfig | None = None,
    *,
    refresh_targets: bool | None = None,
) -> dict[str, Any]:
    """Evaluate readiness from one current materialization, never two nested ones."""

    cfg = cfg or PretrainingConfig()
    refreshed_here = _ensure_materialized(db, cfg, refresh_targets)
    report = _without_nested_refresh(_v4_training_readiness, db, cfg)
    if isinstance(report, dict):
        report = dict(report)
        report["pretraining_targets_refreshed_here"] = bool(refreshed_here)
        report["pretraining_targets_reused"] = bool(not refreshed_here)
    return report


def assert_training_ready(
    db: str,
    cfg: PretrainingConfig | None = None,
    *,
    refresh_targets: bool | None = None,
) -> dict[str, Any]:
    report = training_readiness(db, cfg, refresh_targets=refresh_targets)
    if not report["ready"]:
        failed = [
            k
            for k, v in report["gates"].items()
            if not bool(v.get("pass")) and k != "operational_death_tokens"
        ]
        raise RuntimeError(
            "V24 production bootstrap refused by pretraining readiness gates: "
            + ", ".join(failed)
        )
    return report


def evaluate_baselines(
    db: str,
    cfg: PretrainingConfig | None = None,
    *,
    refresh_targets: bool | None = None,
) -> dict[str, Any]:
    """Evaluate preregistered baselines without rematerializing current targets."""

    cfg = cfg or PretrainingConfig()
    refreshed_here = _ensure_materialized(db, cfg, refresh_targets)
    out = _without_nested_refresh(_v4_evaluate_baselines, db, cfg)
    if isinstance(out, dict):
        out = dict(out)
        out["pretraining_targets_refreshed_here"] = bool(refreshed_here)
        out["pretraining_targets_reused"] = bool(not refreshed_here)
    return out
