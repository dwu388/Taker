@echo off
setlocal
if not exist .venv\Scripts\python.exe py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt || exit /b 1
.venv\Scripts\python.exe -m pytest || exit /b 1
.venv\Scripts\python.exe -m profit_taker.axiom_v24 status --db data\live.sqlite
