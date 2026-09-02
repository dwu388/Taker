@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.pretraining_contract_v2 baselines --db data\axiom_v24_raw.sqlite %*
