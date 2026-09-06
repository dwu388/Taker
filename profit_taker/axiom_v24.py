from __future__ import annotations

"""Production V24 facade with minute-sensitive peak timing installed."""

import sys

from . import axiom_v24_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from . import axiom_v24_minute_timing as _minute_timing

_actual_base = _base
_minute_timing.install(sys.modules[__name__], _core, _actual_base, _impl)


# Persisted pre-0.24.4 champions can legitimately lack the newly introduced
# 10-minute survival checkpoint. They must remain deserializable so runtime
# provenance/contract checks can reject or compare them explicitly rather than
# failing while merely reading their config. New training, however, is not
# allowed to claim the minute-sensitive target contract without all required bins.
def _legacy_readable_post_init(self) -> None:
    bins = tuple(int(x) for x in self.survival_bins_minutes)
    if tuple(sorted(set(bins))) != bins:
        raise ValueError("survival_bins_minutes must be strictly increasing and unique")
    horizon = int(self.horizon_minutes)
    if any(x <= 0 or x > horizon for x in bins):
        raise ValueError("survival_bins_minutes must be inside the active horizon")


V24Config.__post_init__ = _legacy_readable_post_init
_core.V24Config.__post_init__ = _legacy_readable_post_init
_impl.V24Config.__post_init__ = _legacy_readable_post_init
_actual_base.V24Config.__post_init__ = _legacy_readable_post_init


def _require_minute_timing_training_contract(cfg) -> None:
    missing = set(_minute_timing.MINUTE_PEAK_CONFIRMATION_HORIZONS_MINUTES) - set(
        int(x) for x in cfg.survival_bins_minutes
    )
    if missing:
        raise RuntimeError(
            "Minute-sensitive V24 training requires first-peak confirmation bins "
            f"{list(_minute_timing.MINUTE_PEAK_CONFIRMATION_HORIZONS_MINUTES)}; "
            f"missing {sorted(missing)}. Legacy champions remain readable but must "
            "not be extended as minute-timing models under an incomplete contract."
        )


_installed_fit_batch_bundle = fit_batch_bundle
_installed_fit_online_adapter = fit_online_adapter


def fit_batch_bundle(*args, **kwargs):
    cfg = args[4] if len(args) > 4 else kwargs.get("cfg")
    _require_minute_timing_training_contract(cfg)
    return _installed_fit_batch_bundle(*args, **kwargs)


def fit_online_adapter(*args, **kwargs):
    cfg = args[5] if len(args) > 5 else kwargs.get("cfg")
    _require_minute_timing_training_contract(cfg)
    return _installed_fit_online_adapter(*args, **kwargs)


# Batch/adapter fitting is resolved from several historical facade layers. Make
# the training guard uniform without changing prediction-only compatibility.
for _module in (_core, _actual_base, _impl):
    setattr(_module, "fit_batch_bundle", fit_batch_bundle)
    setattr(_module, "fit_online_adapter", fit_online_adapter)


class _LayerProxy:
    """Forward compatibility patches to both the retained base and moved core.

    The official runtime deliberately monkey-patches ``v24._base`` so target hashes,
    recurrent grids and projection semantics stay synchronized across facade layers.
    Since the former outer facade now lives in ``axiom_v24_core``, every mutation
    aimed at the historical base layer must also reach that core module.
    """

    def __init__(self, base_module, core_module):
        object.__setattr__(self, "_base_module", base_module)
        object.__setattr__(self, "_core_module", core_module)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base_module"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_base_module"), name, value)
        setattr(object.__getattribute__(self, "_core_module"), name, value)


# Preserve the public ``v24._base`` patch surface expected by all retained runtime
# layers while ensuring those patches also update the moved facade globals.
_base = _LayerProxy(_actual_base, _core)


# Preserve the original outer-facade policy-helper forwarding contract. Tests and
# runtime patches intentionally monkey-patch helpers on ``profit_taker.axiom_v24``;
# the moved core function has its own global namespace, so copy those public helper
# bindings into the core immediately before delegating.
def train_distributional_policy(db, policy_root, cfg, *, allow_small=False):
    for name in (
        "migrate", "refresh_counterfactual_policy_targets", "refresh_policy_cohorts",
        "refresh_token_assignments", "next_one_use_policy_cohort",
    ):
        helper = globals()[name]
        setattr(_core, name, helper)
        setattr(_actual_base, name, helper)
    return _core.train_distributional_policy(
        db, policy_root, cfg, allow_small=allow_small
    )


_impl.train_distributional_policy = train_distributional_policy


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
