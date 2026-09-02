from __future__ import annotations

"""Compatibility entrypoint for the preregistered first bootstrap.

There is one production bootstrap implementation: v24_contract_runtime_v4.  Keep
this historical module only as a thin alias so documentation and old commands
cannot bypass the latest target, readiness, friction, or first-model contracts.
"""

import sys
from typing import Sequence

from . import v24_contract_runtime_v4 as runtime


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return runtime.main(["bootstrap", *args])


if __name__ == "__main__":
    raise SystemExit(main())
