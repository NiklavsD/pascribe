@echo off
setlocal
pushd "%~dp0"
if not exist "%~dp0venv\Scripts\python.exe" (
  echo Pascribe is not installed yet. Run install.bat first.
  popd
  exit /b 1
)
"%~dp0venv\Scripts\python.exe" -m unittest discover -s tests -v
set "exit_code=%ERRORLEVEL%"
popd
exit /b %exit_code%
