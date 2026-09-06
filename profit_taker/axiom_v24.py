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

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
