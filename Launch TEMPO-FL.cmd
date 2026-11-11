@echo off
setlocal
title TEMPO-FL
set "NO_COLOR=1"
set "TERM=dumb"
set "FORCE_COLOR="
set "CLICOLOR_FORCE="
set "PYTHONUTF8=1"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project Python environment is missing.
  echo Follow Installation in README.md, then run this launcher again.
  if "%~1"=="" pause
  exit /b 1
)
".venv\Scripts\python.exe" -m implementation %*
set "TEMPO_FL_EXIT_CODE=%ERRORLEVEL%"
if not "%TEMPO_FL_EXIT_CODE%"=="0" if not "%TEMPO_FL_EXIT_CODE%"=="130" (
  echo.
  echo TEMPO-FL could not start. See the message above and Installation in README.md.
  if "%~1"=="" pause
)
exit /b %TEMPO_FL_EXIT_CODE%
