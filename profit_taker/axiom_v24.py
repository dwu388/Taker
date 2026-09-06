from __future__ import annotations

"""Production V24 facade with minute-sensitive peak timing installed."""

import sys

from . import axiom_v24_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from . import axiom_v24_minute_timing as _minute_timing

_minute_timing.install(sys.modules[__name__], _core, _base, _impl)

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
