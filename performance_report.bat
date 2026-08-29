@echo off
setlocal
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" -m profit_taker.performance_report --db data\axiom_v24_raw.sqlite --benchmark-db data\axiom_v24_1000_benchmark.sqlite --output data\CURRENT_MODEL_PERFORMANCE.txt --force
endlocal
