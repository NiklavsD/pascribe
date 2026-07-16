@echo off
setlocal
pushd "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "exit_code=%ERRORLEVEL%"
popd
if not "%exit_code%"=="0" (
  echo.
  echo Pascribe installation failed. Review the message above.
  pause
)
exit /b %exit_code%
