@echo off
cd /d "%~dp0"
echo Checking for legacy observation history...
.venv\Scripts\python.exe -m profit_taker.legacy_import --db data\live.sqlite --auto
if errorlevel 1 exit /b 1
echo Rebuilding causal V18 features...
.venv\Scripts\python.exe -m profit_taker.axiom_features --db data\live.sqlite
if errorlevel 1 exit /b 1
echo Rebuilding lifecycle-aware 24h targets...
.venv\Scripts\python.exe -m profit_taker.axiom_24h refresh --db data\live.sqlite
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m profit_taker.axiom_24h status --db data\live.sqlite
