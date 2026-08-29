from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> dict[str, str]:
    p = Path(path)
    values: dict[str, str] = {}
    if p.exists():
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("HELIUS_API_KEY", "SHYFT_API_KEY", "BIRDEYE_API_KEY"):
        if os.getenv(k):
            values[k] = os.environ[k]
    return values


def key(values: dict[str, str], name: str) -> str:
    v = values.get(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing {name}. Add it to .env or the environment.")
    return v
