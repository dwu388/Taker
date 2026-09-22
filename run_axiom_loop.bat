@echo off
setlocal
cd /d "%~dp0"
set "RAW_DB=data\axiom_v24_raw.sqlite"
.venv\Scripts\python.exe -m profit_taker.axiom_learning_loop --db "%RAW_DB%" --database-only %*

set "exit_code=%errorlevel%"
endlocal & exit /b %exit_code%
