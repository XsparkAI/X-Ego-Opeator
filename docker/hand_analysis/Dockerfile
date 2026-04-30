FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

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

COPY docker/hand_analysis/requirements.txt /tmp/requirements-hand-cpu.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install -r /tmp/requirements-hand-cpu.txt

COPY operators /app/operators
COPY weights/detector.pt /app/weights/detector.pt
COPY operators/video_path.py /app/video_path.py
COPY docker/hand_analysis/bcecmd /app/bcecmd
COPY docker/platform_input.py /app/platform_input.py
COPY docker/hand_analysis/entrypoint.sh /app/entrypoint.sh
COPY docker/hand_analysis/platform_runner.py /app/platform_runner.py
COPY docker/hand_analysis/runner.py /app/runner.py

RUN chmod +x /app/entrypoint.sh /app/bcecmd

ENTRYPOINT ["/app/entrypoint.sh"]
