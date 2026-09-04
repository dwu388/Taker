from __future__ import annotations

"""Official V24 runtime bound to the latest pretraining contract.

V3 retains the training/maintenance CLI.  V4 installs continuity-safe target
semantics, indexed heartbeat lookup, friction-safe policy refreshes, and a fast
read-only status path.  Ordinary status must never rebuild every historical
price-path target merely to display operational state.
"""

import argparse
import json
import sys
from typing import Sequence

from . import axiom_v24 as v24
from . import pretraining_contract_v4 as contract
from . import pretraining_capture_index
from . import pretraining_status
from . import v24_contract_runtime_v3 as runtime

# Keep V4 target semantics unchanged while replacing the pathological collector-
# heartbeat prefix scan with an indexed interval lookup.  This patch is in-place
# so every retained V4 barrier/collapse function resolves the optimized helper.
pretraining_capture_index.install(contract)

runtime.contract = contract
runtime.shared.target_contract_hash = contract.target_contract_hash

_original_cf_refresh = v24._impl.refresh_counterfactual_policy_targets


def _refresh_counterfactual_policy_targets_net(conn, cfg):
    out = _original_cf_refresh(conn, cfg)
    friction = contract.enrich_counterfactual_friction(conn, contract.PretrainingConfig())
    if isinstance(out, dict):
        out = dict(out)
        out["friction"] = friction
    return out


# train_distributional_policy resolves this name from axiom_v24_impl globals at
# call time.  Patch every facade reference too so explicit refreshes share exactly
# the same economics.
v24._impl.refresh_counterfactual_policy_targets = _refresh_counterfactual_policy_targets_net
v24._base.refresh_counterfactual_policy_targets = _refresh_counterfactual_policy_targets_net
v24.refresh_counterfactual_policy_targets = _refresh_counterfactual_policy_targets_net


def _status_main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser(
        prog="v24_contract_runtime_v4 status",
        description="Fast V24 status; historical pretraining refresh is opt-in.",
    )
    p.add_argument("--db", default=v24.MODEL_DB_DEFAULT)
    p.add_argument("--cohort-hours", type=int, default=24)
    p.add_argument("--promotion-every", type=int, default=4)
    p.add_argument("--audit-every", type=int, default=5)
    p.add_argument("--warmup-blocks", type=int, default=7)
    p.add_argument(
        "--refresh-pretraining",
        action="store_true",
        help="Explicitly rebuild historical pretraining targets before reporting status.",
    )
    args = p.parse_args(list(argv)[1:])

    pcfg = contract.PretrainingConfig()
    runtime.shared.target_contract_hash = contract.target_contract_hash
    runtime.shared.install_target_hash_contract(pcfg)
    cfg = runtime._cfg(args)

    if args.refresh_pretraining:
        pretraining = runtime._refresh(args.db, pcfg)
        pretraining["mode"] = "full_refresh"
        pretraining["readiness_snapshot"] = pretraining_status.status_snapshot(args.db, pcfg)
    else:
        pretraining = {
            "mode": "read_only_snapshot",
            "targets_refreshed": False,
            "readiness_snapshot": pretraining_status.status_snapshot(args.db, pcfg),
        }

    out = v24.status(args.db, cfg)
    if not isinstance(out, dict):
        out = {"status": out}
    out.setdefault("pretraining_contract", pretraining)
    out.setdefault("pretraining_target_definition_hash", contract.target_contract_hash(pcfg))
    out.setdefault("combined_v24_target_definition_hash", v24.target_definition_hash(cfg))
    print(json.dumps(out, indent=2, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "status":
        return _status_main(args)
    return runtime.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
