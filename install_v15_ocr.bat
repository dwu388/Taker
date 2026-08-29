@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Missing .venv. Run setup_venv.bat first.
  exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade "rapidocr>=3.9.2,<4" "onnxruntime>=1.23,<2" "pytesseract>=0.3.13,<0.4" "opencv-python>=4.10,<5"
.venv\Scripts\python.exe -m pip check
.venv\Scripts\rapidocr.exe check 2>nul
if errorlevel 1 .venv\Scripts\python.exe -m rapidocr check
endlocal
