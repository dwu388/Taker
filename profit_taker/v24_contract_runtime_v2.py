from __future__ import annotations

from . import pretraining_contract_v2 as contract
from . import v24_contract_runtime as runtime

# Rebind the runtime's imported contract functions to the tightened definitions.
runtime.PretrainingConfig = contract.PretrainingConfig
runtime.assert_training_ready = contract.assert_training_ready
runtime.enrich_counterfactual_friction = contract.enrich_counterfactual_friction
runtime.evaluate_baselines = contract.evaluate_baselines
runtime.first_model_profile = contract.first_model_profile
runtime.refresh_pretraining_targets = contract.refresh_pretraining_targets
runtime.target_contract_hash = contract.target_contract_hash

main = runtime.main

if __name__ == "__main__":
    raise SystemExit(main())
