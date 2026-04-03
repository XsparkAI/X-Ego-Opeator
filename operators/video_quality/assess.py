#!/usr/bin/env python3
"""
视频质量评估 — 整合脚本
调用三个独立算子：画面质量 / 视频稳定性 / 曝光与光照
全部基于 OpenCV + NumPy + SciPy，无深度学习依赖，CPU 运行。

用法:
    python -m video_quality.assess <video> [--sample-fps 2] [-o result.json]
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2

from . import op_quality, op_stability, op_exposure

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)


def read_frames(video_path: str, sample_fps: Optional[float] = None):
    """读取视频帧，返回灰度帧列表、帧索引、视频元信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step = int(round(fps / sample_fps)) if sample_fps and sample_fps < fps else 1

    log.info(
        f"Video: {width}x{height}, {fps:.1f}fps, {total_frames} frames, "
        f"sample_step={step}"
    )

    frames_gray = []
    frame_indices = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if width > 1920:
                scale = 1920 / width
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames_gray.append(gray)
            frame_indices.append(idx)
        idx += 1
    cap.release()

    meta = {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "step": step,
    }
    return frames_gray, frame_indices, meta


def process_video(video_path: str, sample_fps: Optional[float] = None) -> dict:
    """处理视频，返回完整评估结果字典"""
    t0 = time.time()
    frames_gray, frame_indices, meta = read_frames(video_path, sample_fps)
    log.info(f"Read {len(frames_gray)} frames in {time.time() - t0:.2f}s")

    fps = meta["fps"]
    step = meta["step"]

    # ---- 算子1: 画面质量 ----
    t1 = time.time()
    quality_results = [
        op_quality.assess_frame(g, fi) for g, fi in zip(frames_gray, frame_indices)
    ]
    t1_elapsed = time.time() - t1
    log.info(f"Operator 1 (quality):   {t1_elapsed:.3f}s  ({t1_elapsed/len(frames_gray)*1000:.2f} ms/frame)")

    # ---- 算子2: 稳定性 ----
    t2 = time.time()
    stability_result = op_stability.assess(frames_gray, fps / step)
    t2_elapsed = time.time() - t2
    log.info(f"Operator 2 (stability): {t2_elapsed:.3f}s  ({t2_elapsed/len(frames_gray)*1000:.2f} ms/frame)")

    # ---- 算子3: 曝光 ----
    t3 = time.time()
    exposure_results = [
        op_exposure.assess_frame(g, fi) for g, fi in zip(frames_gray, frame_indices)
    ]
    t3_elapsed = time.time() - t3
    log.info(f"Operator 3 (exposure):  {t3_elapsed:.3f}s  ({t3_elapsed/len(frames_gray)*1000:.2f} ms/frame)")

    total_time = time.time() - t0

    return {
        "video_path": video_path,
        "num_frames": len(frames_gray),
        "fps": fps,
        "resolution": f"{meta['width']}x{meta['height']}",
        "processing_time_sec": round(total_time, 3),
        "quality": op_quality.summarize(quality_results),
        "stability": op_stability.summarize(stability_result),
        "exposure": op_exposure.summarize(exposure_results),
    }


def main():
    parser = argparse.ArgumentParser(description="视频质量评估（经典方案）")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument(
        "--sample-fps", type=float, default=None,
        help="采样帧率，默认全帧处理。设为 2 表示每秒采样 2 帧",
    )
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    result = process_video(args.video, args.sample_fps)
    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        log.info(f"Results saved to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
