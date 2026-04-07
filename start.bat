@echo off
setlocal enabledelayedexpansion

REM Track start time
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a:%%b)
set START_TIME=%mydate% %mytime%
echo [%START_TIME%] ===== STARTUP BEGIN =====
echo.

REM Step 1: Activate virtual environment
echo [%START_TIME%] [1/3] Activating Virtual Environment...
call venv\Scripts\activate
if !errorlevel! equ 0 (
    echo [*] Virtual environment activated successfully
) else (
    echo [ERROR] Failed to activate virtual environment
    exit /b 1
)

REM Step 2: Check dependencies (optimized - skip already installed)
echo.
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a:%%b)
echo [%mydate% %mytime%] [2/3] Checking dependencies...
echo [*] Note: Only missing/outdated packages will be installed
pip install -q -r requirements.txt --upgrade-strategy only-if-needed
if !errorlevel! equ 0 (
    echo [*] Dependencies check complete
) else (
    echo [ERROR] Failed to check dependencies
    exit /b 1
)

REM Step 3: Start backend server
echo.
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a:%%b)
echo [%mydate% %mytime%] [3/3] Starting Enterprise Backend...
echo [*] Server: http://127.0.0.1:8000
echo [*] API Docs: http://127.0.0.1:8000/docs
echo [*] Database: ./documents.db
echo [*] Press Ctrl+C to stop
echo.
uvicorn app.main:app --reload --port 8000

REM Cleanup on exit
echo.
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a:%%b)
echo [%mydate% %mytime%] ===== SERVER SHUTDOWN ====