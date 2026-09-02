@echo off
setlocal
cd /d "%~dp0"
set "RAW_DB=data\axiom_v24_raw.sqlite"
.venv\Scripts\python.exe -m profit_taker.collection_admin init --db "%RAW_DB%" --purpose v24_production_raw_collection || exit /b 1
.venv\Scripts\python.exe -m profit_taker.axiom_collection_context_runner --db "%RAW_DB%" %*
