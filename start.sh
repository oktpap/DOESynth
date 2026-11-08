#!/usr/bin/env sh
# Запуск DOESynth в Docker (Linux, macOS): ./start.sh
set -e
cd "$(dirname "$0")"
PORT="${DOE_PORT:-8000}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker не найден: https://docs.docker.com/get-docker/"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  if [ "$(uname)" = "Darwin" ]; then open -a Docker || true; fi
  echo "Ожидание запуска Docker..."
  i=0
  until docker info >/dev/null 2>&1; do
    i=$((i + 1)); [ "$i" -gt 90 ] && { echo "Docker не запустился"; exit 1; }
    sleep 2
  done
fi

echo "Сборка и запуск контейнера (первый раз — несколько минут)..."
DOE_PORT="$PORT" docker compose up -d --build

echo "Ожидание готовности приложения..."
i=0
until curl -fs "http://localhost:$PORT/api/config" >/dev/null 2>&1 \
  || python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/api/config', timeout=2)" >/dev/null 2>&1; do
  i=$((i + 1)); [ "$i" -gt 120 ] && break
  sleep 1
done

URL="http://localhost:$PORT"
APP_URL="$URL/?window=app"
# Отдельное окно без адресной строки и вкладок: режим приложения Chrome, Chromium или Edge.
# Если их нет — обычный браузер по умолчанию.
open_window() {
  if [ "$(uname)" = "Darwin" ]; then
    for app in "Google Chrome" "Microsoft Edge" "Chromium"; do
      if [ -d "/Applications/$app.app" ]; then
        open -na "$app" --args --app="$APP_URL" --window-size=1440,900 && return 0
      fi
    done
    open "$URL"
    return 0
  fi
  for b in google-chrome google-chrome-stable chromium chromium-browser microsoft-edge microsoft-edge-stable; do
    if command -v "$b" >/dev/null 2>&1; then
      nohup "$b" --app="$APP_URL" --window-size=1440,900 >/dev/null 2>&1 &
      return 0
    fi
  done
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true; fi
}
open_window
echo "Приложение: $URL  (остановить: ./stop.sh)"
