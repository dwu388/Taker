@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m profit_taker.migrate_v4 --db data\live.sqlite
.venv\Scripts\python.exe -m pytest
if errorlevel 1 exit /b 1
echo.
echo Setup complete.
echo Tesseract must also be installed at C:\Program Files\Tesseract-OCR\tesseract.exe or available on PATH.
endlocal
