@echo off
REM Activate the virtual environment
call .venv\Scripts\activate

REM Run the Streamlit app
start streamlit run app.py --server.headless true

REM Wait for 5 seconds
timeout /t 5 >nul

REM Deactivate the virtual environment
deactivate