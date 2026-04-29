FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CAPTION_DOCKER_VERSION=20260427-test-09

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libsm6 \
    libstdc++6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/caption/requirements.txt /tmp/requirements-caption.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements-caption.txt

COPY operators /app/operators
COPY docker/hand_analysis/bcecmd /app/bcecmd
COPY docker/caption/entrypoint.sh /app/entrypoint.sh
COPY docker/caption/platform_runner.py /app/platform_runner.py
COPY docker/caption/runner.py /app/runner.py

RUN chmod +x /app/entrypoint.sh /app/bcecmd

ENTRYPOINT ["/app/entrypoint.sh"]
