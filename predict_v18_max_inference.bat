@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m profit_taker.axiom_predict_24h --db data\live.sqlite --model models\axiom24\latest.joblib --out data\axiom_predictions_24h.csv --rank-by p_hit_plus100_by_12h
