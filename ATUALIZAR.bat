@echo off
setlocal

REM Vai para a pasta onde está este .bat (assumindo que é a raiz do repo)
cd /d "%~dp0"

REM Data de hoje no formato YYYY-MM-DD (funciona bem em pt-BR)
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "TODAY=%%i"

echo.
echo ===== git add . =====
git add .
if errorlevel 1 (
  echo ERRO no git add .
  pause
  exit /b 1
)

echo.
echo ===== git commit =====
git commit -m "Dash Hedge IPCA - %TODAY%"
if errorlevel 1 (
  echo ERRO no git commit (talvez nao tenha mudancas para commitar)
  pause
  exit /b 1
)

echo.
echo ===== git push =====
git push
if errorlevel 1 (
  echo ERRO no git push
  pause
  exit /b 1
)

echo.
echo ===== OK: push concluido =====
pause
