@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.axiom_24h refresh --db data\live.sqlite
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m profit_taker.axiom_24h status --db data\live.sqlite
