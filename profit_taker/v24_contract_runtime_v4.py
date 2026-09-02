from __future__ import annotations

"""Official V24 runtime bound to the latest pretraining contract.

This wrapper is intentionally small: V3 owns the reduced first-model CLI, while
V4 installs the continuity-safe target contract and makes policy counterfactual
refreshes friction-safe at their source so train-policy cannot consume freshly
recreated gross compatibility aliases.
"""

from . import axiom_v24 as v24
from . import pretraining_contract_v4 as contract
from . import v24_contract_runtime_v3 as runtime

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

main = runtime.main

if __name__ == "__main__":
    raise SystemExit(main())
