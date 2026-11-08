FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/app

WORKDIR /app

# Зависимости — отдельным слоем: при правке кода не переустанавливаются
COPY requirements.txt .
RUN pip install -r requirements.txt

# Веса и примеры меняются редко — раньше кода
COPY weights ./weights
COPY examples ./examples
COPY benchmark ./benchmark
COPY doesynth ./doesynth

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=4s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/config', timeout=3)"

ENTRYPOINT ["python", "-m", "doesynth"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
