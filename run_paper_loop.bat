@echo off
setlocal
cd /d "%~dp0"

".venv\Scripts\python.exe" -m profit_taker.axiom_paper_loop ^
  --db "data\axiom_v24_raw.sqlite" ^
  --queue-db "data\axiom_v24_raw.sqlite.paper_queue.sqlite" ^
  --benchmark-db "data\axiom_v24_1000_benchmark.sqlite" ^
  --predictions "data\axiom_predictions_v24.csv" ^
  --forecast-model "models\axiom_v24\champion.joblib" ^
  --policy-model "models\axiom_policy_v24\champion.joblib" ^
  --config "axiom_migrated_config.json" ^
  --interval-seconds 60 ^
  --max-snapshot-age-seconds 180 ^
  %*

set "exit_code=%errorlevel%"
endlocal & exit /b %exit_code%