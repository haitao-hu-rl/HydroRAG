@echo off
REM ===========================================================================
REM  HydroRAG launcher (double-click to start)
REM
REM  Usage:
REM    start.bat            start service and open the browser
REM    start.bat -Port 8080 use another port
REM    start.bat -Stop      stop the running service
REM
REM  NOTE: keep this file ASCII-only. cmd.exe on a zh-CN system reads .bat as
REM  code page 936 (GBK), so UTF-8 Chinese comments corrupt the script.
REM ===========================================================================
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo [FAILED] exit code %EXITCODE%
    echo See the messages above, or check data\logs\server.log
    echo.
    pause
)

endlocal & exit /b %EXITCODE%
