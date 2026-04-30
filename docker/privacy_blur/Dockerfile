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

COPY docker/privacy_blur/requirements.txt /tmp/requirements-privacy-blur.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install -r /tmp/requirements-privacy-blur.txt

COPY operators /app/operators
RUN rm -rf /app/operators/privacy_blur/weights
COPY weights/yolo26s_face.pt /app/weights/yolo26s_face.pt
COPY weights/yolo_lp.pt /app/weights/yolo_lp.pt
COPY operators/video_path.py /app/video_path.py
COPY docker/hand_analysis/bcecmd /app/bcecmd
COPY docker/platform_input.py /app/platform_input.py
COPY docker/hand_analysis/platform_runner.py /app/hand_platform_helpers.py
COPY docker/privacy_blur/entrypoint.sh /app/entrypoint.sh
COPY docker/privacy_blur/platform_runner.py /app/platform_runner.py
COPY docker/privacy_blur/runner.py /app/runner.py

RUN chmod +x /app/entrypoint.sh /app/bcecmd

ENTRYPOINT ["/app/entrypoint.sh"]
