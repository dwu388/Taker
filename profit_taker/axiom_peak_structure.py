"""V24 peak-structure facade with manual collection-stop right censoring."""
from __future__ import annotations

import sqlite3

from . import axiom_peak_structure_pre_manual_stop as _current
from . import axiom_manual_stop as manual_stop

for _name in dir(_current):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_current, _name)

_impl = _current._impl
_base = _current._base
_original_refresh_labels = _current.refresh_labels


def refresh_labels(db: str, config: PeakStructureConfig, full_rebuild: bool = False) -> dict[str, object]:
    result = _original_refresh_labels(db, config, full_rebuild=full_rebuild)
    with sqlite3.connect(db) as conn:
        censor_result = manual_stop.apply_peak_label_censors(conn)
    if isinstance(result, dict):
        result["manual_stop_censoring"] = censor_result
    return result


# All callers, including retained V24 internals, must see the censor-aware refresh.
_impl.refresh_labels = refresh_labels
_base.refresh_labels = refresh_labels


def __getattr__(name: str):
    return getattr(_current, name)


if __name__ == "__main__":  # pragma: no cover
    main()
