from __future__ import annotations

from . import pretraining_contract_v3 as contract
from . import v24_contract_runtime_v3 as runtime

runtime.contract = contract
runtime.shared.target_contract_hash = contract.target_contract_hash

main = runtime.main

if __name__ == "__main__":
    raise SystemExit(main())
