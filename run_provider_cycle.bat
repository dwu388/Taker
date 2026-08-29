@echo off
cd /d "%~dp0"
echo WARNING: This is an EXPLICIT provider command and can consume provider quotas.
.venv\Scripts\python.exe -m profit_taker.provider_sync --tokens data\tokens.csv --db data\live.sqlite --env-file .env
