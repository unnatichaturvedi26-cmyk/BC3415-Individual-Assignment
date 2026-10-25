@echo off
cd /d "%~dp0"
if not exist .venv (py -3.11 -m venv .venv || py -3 -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt || pause && exit /b 1
python -m streamlit run streamlit_app.py --server.port 8501
pause
