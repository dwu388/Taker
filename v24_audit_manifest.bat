@echo off
.venv\Scripts\python.exe -m profit_taker.axiom_v24 audit-manifest --db data\live.sqlite %*
