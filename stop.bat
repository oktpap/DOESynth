@echo off
chcp 65001 >nul
cd /d "%~dp0"
docker compose down
echo DOESynth остановлен.
timeout /t 3 >nul 2>nul
exit /b 0
