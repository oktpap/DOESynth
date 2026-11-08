@echo off
rem Запуск DOESynth в Docker одним двойным щелчком (Windows).
chcp 65001 >nul
cd /d "%~dp0"
title DOESynth: запуск

where docker >nul 2>&1
if errorlevel 1 (
  echo Docker не найден. Установите Docker Desktop: https://www.docker.com/products/docker-desktop/
  pause
  exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
  echo Запускаю Docker Desktop...
  if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
  powershell -NoProfile -Command "for($i=0;$i -lt 90;$i++){docker info *> $null; if($LASTEXITCODE -eq 0){exit 0}; Start-Sleep 2}; exit 1"
  if errorlevel 1 (
    echo Docker не запустился. Запустите Docker Desktop вручную и повторите.
    pause
    exit /b 1
  )
)

if "%DOE_PORT%"=="" set DOE_PORT=8000
echo Сборка и запуск контейнера (первый раз — несколько минут)...
docker compose up -d --build
if errorlevel 1 (
  echo Не удалось собрать или запустить контейнер.
  pause
  exit /b 1
)

echo Ожидание готовности приложения...
powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){try{Invoke-WebRequest -UseBasicParsing http://localhost:%DOE_PORT%/api/config -TimeoutSec 2 | Out-Null; exit 0}catch{Start-Sleep 1}}; exit 1"
set "URL=http://localhost:%DOE_PORT%"
rem Отдельное окно без адресной строки и вкладок: режим приложения Edge (есть в любой Windows 10/11)
rem или Chrome. Если их нет — обычный браузер по умолчанию.
set "APPBROWSER="
if exist "%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe" set "APPBROWSER=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
if not defined APPBROWSER if exist "%ProgramFiles%\Microsoft\Edge\Application\msedge.exe" set "APPBROWSER=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
if not defined APPBROWSER if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set "APPBROWSER=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not defined APPBROWSER if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set "APPBROWSER=%LocalAppData%\Google\Chrome\Application\chrome.exe"
if not defined APPBROWSER goto open_default
start "" "%APPBROWSER%" --app=%URL%/?window=app --window-size=1440,900
call :wait_window && goto opened
rem Браузер, который только что запустился в фоне, иногда не показывает окно с первого раза
start "" "%APPBROWSER%" --app=%URL%/?window=app --window-size=1440,900
call :wait_window && goto opened
:open_default
start "" %URL%
:opened
echo.
echo DOESynth открыт: %URL%
echo Окно можно закрыть и снова открыть через start.bat.
echo Остановить: stop.bat
timeout /t 5 >nul 2>nul
exit /b 0

:wait_window
rem Ждём до 8 с видимое окно Edge/Chrome (класс Chrome_WidgetWin_1) с заголовком страницы DOESynth
powershell -NoProfile -Command "$w = Add-Type -Name W -Namespace DOESynth -PassThru -MemberDefinition '[DllImport(\"user32.dll\")] public static extern System.IntPtr FindWindowEx(System.IntPtr p, System.IntPtr a, string c, string t); [DllImport(\"user32.dll\")] public static extern bool IsWindowVisible(System.IntPtr h);'; for ($i = 0; $i -lt 16; $i++) { $h = [System.IntPtr]::Zero; while (($h = $w::FindWindowEx([System.IntPtr]::Zero, $h, 'Chrome_WidgetWin_1', 'DOESynth')) -ne [System.IntPtr]::Zero) { if ($w::IsWindowVisible($h)) { exit 0 } }; Start-Sleep -Milliseconds 500 }; exit 1"
exit /b %errorlevel%
