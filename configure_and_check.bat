@echo off
cd /d "%~dp0"
if not exist .env copy .env.example .env >nul
.venv\Scripts\python.exe -m profit_taker.provider_health --env-file .env
