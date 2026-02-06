@echo off
setlocal

cd /d "%~dp0"

echo.
echo ===== git pull =====
git pull
if errorlevel 1 (
  echo ERRO no git pull
  pause
  exit /b 1
)

echo.
echo ===== OK: pull concluido =====
pause
