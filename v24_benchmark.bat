@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  .venv\Scripts\python.exe -m profit_taker.axiom_budget_benchmark status --benchmark-db data\axiom_v24_1000_benchmark.sqlite
) else (
  .venv\Scripts\python.exe -m profit_taker.axiom_budget_benchmark %*
)
