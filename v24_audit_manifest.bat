@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.v24_contract_runtime_v2 audit-manifest --db data\axiom_v24_raw.sqlite %*
