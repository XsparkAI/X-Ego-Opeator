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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import subprocess

import cv2
import numpy as np

from ..frame_cache.cache_utils import (
    build_sample_fps_frame_ids,
    load_or_build_cached_quality_frames,
    probe_video,
)
from . import op_quality, op_stability, op_exposure

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)


def _probe_video(video_path: str) -> dict:
    """用 ffprobe 获取视频元信息"""
    cmd = [
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name",
        "-of", "json", video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)["streams"][0]
    num, den = map(int, info["r_frame_rate"].split("/"))
    fps = num / den
    # nb_frames 可能为 "N/A"，此时回退到 OpenCV 获取
    nb = info.get("nb_frames", "N/A")
    total_frames = int(nb) if nb != "N/A" else None
    return {
        "width": int(info["width"]),
        "height": int(info["height"]),
        "fps": fps,
        "total_frames": total_frames,
        "codec_name": info.get("codec_name", ""),
    }


def _read_frames_nvdec(video_path: str, sample_fps: Optional[float], probe: dict):
    """通过 ffmpeg hevc_cuvid (NVDEC) 硬件解码读取帧"""
    width, height, fps = probe["width"], probe["height"], probe["fps"]
    step = int(round(fps / sample_fps)) if sample_fps and sample_fps < fps else 1

    # 用 scale 降采样到 <=1920 宽度，在 GPU 端完成
    if width > 1920:
        scale_w = 1920
        scale_h = int(height * 1920 / width)
        # 确保偶数（ffmpeg 要求）
        scale_h = scale_h + (scale_h % 2)
        vf = f"scale={scale_w}:{scale_h}"
    else:
        scale_w, scale_h = width, height
        vf = None

    # 构建 filter chain：先帧选择，再缩放
    filters = []
    if step > 1:
        filters.append(f"select=not(mod(n\\,{step}))")
    if vf:
        filters.append(vf)
    vf_str = ",".join(filters) if filters else None

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-c:v", "hevc_cuvid",
        "-i", video_path,
    ]
    if vf_str:
        cmd += ["-vf", vf_str]
    cmd += [
        "-vsync", "vfr",
        "-pix_fmt", "gray",
        "-f", "rawvideo", "-"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_size = scale_w * scale_h
    frames_gray = []
    frame_indices = []
    out_idx = 0

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            gray = np.frombuffer(raw, dtype=np.uint8).reshape(scale_h, scale_w)
            frames_gray.append(gray)
            frame_indices.append(out_idx * step)
            out_idx += 1
    finally:
        proc.stdout.close()
        proc.wait()

    total_frames = probe["total_frames"] or idx
    meta = {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "step": step,
        "decoder": "nvdec",
    }
    return frames_gray, frame_indices, meta


def _read_frames_cpu(video_path: str, sample_fps: Optional[float]):
    """OpenCV CPU 软解码读取帧（回退方案）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    step = int(round(fps / sample_fps)) if sample_fps and sample_fps < fps else 1

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
        "decoder": "cpu",
    }
    return frames_gray, frame_indices, meta


def read_frames(video_path: str, sample_fps: Optional[float] = None):
    """读取视频帧，优先 NVDEC 硬件解码，失败回退 CPU 软解码"""
    episode_dir = Path(video_path).parent
    try:
        fps, total_frames = probe_video(Path(video_path))
        frame_ids = build_sample_fps_frame_ids(fps, total_frames, sample_fps)
        cached = load_or_build_cached_quality_frames(episode_dir, frame_ids)
        if cached is not None:
            frames_gray, frame_indices, meta = cached
            step = (
                max(1, int(round(fps / sample_fps)))
                if sample_fps and sample_fps < fps else 1
            )
            meta["step"] = step
            log.info(
                f"Video: {meta['width']}x{meta['height']}, {meta['fps']:.1f}fps, "
                f"{meta['total_frames']} frames, sample_step={step}, "
                f"decoder=frame_cache"
            )
            return frames_gray, frame_indices, meta
    except Exception as e:
        log.warning(f"frame_cache unavailable ({e}), falling back to decoder")

    try:
        probe = _probe_video(video_path)
        if probe["codec_name"] == "hevc":
            frames_gray, frame_indices, meta = _read_frames_nvdec(
                video_path, sample_fps, probe
            )
            if frames_gray:
                log.info(
                    f"Video: {meta['width']}x{meta['height']}, {meta['fps']:.1f}fps, "
                    f"{meta['total_frames']} frames, sample_step={meta['step']}, "
                    f"decoder=NVDEC"
                )
                return frames_gray, frame_indices, meta
        log.info("NVDEC not applicable, falling back to CPU decode")
    except Exception as e:
        log.warning(f"NVDEC decode failed ({e}), falling back to CPU decode")

    frames_gray, frame_indices, meta = _read_frames_cpu(video_path, sample_fps)
    log.info(
        f"Video: {meta['width']}x{meta['height']}, {meta['fps']:.1f}fps, "
        f"{meta['total_frames']} frames, sample_step={meta['step']}, "
        f"decoder=CPU"
    )
    return frames_gray, frame_indices, meta


def process_video(
    video_path: str,
    sample_fps: Optional[float] = None,
    check_quality: bool = True,
    check_stability: bool = True,
    check_exposure: bool = True,
) -> dict:
    """处理视频，返回完整评估结果字典。可通过开关控制各子算子。"""
    t0 = time.time()
    frames_gray, frame_indices, meta = read_frames(video_path, sample_fps)
    log.info(f"Read {len(frames_gray)} frames in {time.time() - t0:.2f}s")

    fps = meta["fps"]
    step = meta["step"]
    n = len(frames_gray)

    quality_summary = None
    stability_summary = None
    exposure_summary = None

    def _run_quality():
        t1 = time.time()
        quality_results = [
            op_quality.assess_frame(g, fi) for g, fi in zip(frames_gray, frame_indices)
        ]
        elapsed = time.time() - t1
        return "quality", elapsed, op_quality.summarize(quality_results, fps=fps)

    def _run_stability():
        t2 = time.time()
        stability_result = op_stability.assess(frames_gray, fps / step, frame_indices)
        elapsed = time.time() - t2
        return "stability", elapsed, op_stability.summarize(stability_result, fps=fps)

    def _run_exposure():
        t3 = time.time()
        exposure_results = [
            op_exposure.assess_frame(g, fi) for g, fi in zip(frames_gray, frame_indices)
        ]
        elapsed = time.time() - t3
        return "exposure", elapsed, op_exposure.summarize(exposure_results, fps=fps)

    tasks = []
    if check_quality:
        tasks.append(_run_quality)
    else:
        log.info("Operator 1 (quality):   跳过")
    if check_stability:
        tasks.append(_run_stability)
    else:
        log.info("Operator 2 (stability): 跳过")
    if check_exposure:
        tasks.append(_run_exposure)
    else:
        log.info("Operator 3 (exposure):  跳过")

    if tasks:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            for name, elapsed, summary in [f.result() for f in [pool.submit(task) for task in tasks]]:
                if name == "quality":
                    quality_summary = summary
                    log.info(f"Operator 1 (quality):   {elapsed:.3f}s  ({elapsed/n*1000:.2f} ms/frame)")
                elif name == "stability":
                    stability_summary = summary
                    log.info(f"Operator 2 (stability): {elapsed:.3f}s  ({elapsed/n*1000:.2f} ms/frame)")
                elif name == "exposure":
                    exposure_summary = summary
                    log.info(f"Operator 3 (exposure):  {elapsed:.3f}s  ({elapsed/n*1000:.2f} ms/frame)")

    total_time = time.time() - t0

    # 整体合格判定：仅考虑启用的算子
    enabled_summaries = [s for s in [quality_summary, stability_summary, exposure_summary] if s is not None]
    overall_pass = all(s.get("pass", True) for s in enabled_summaries)

    result = {
        "video_path": video_path,
        "num_frames": n,
        "fps": fps,
        "resolution": f"{meta['width']}x{meta['height']}",
        "processing_time_sec": round(total_time, 3),
        "pass": overall_pass,
    }
    if quality_summary is not None:
        result["quality"] = quality_summary
    if stability_summary is not None:
        result["stability"] = stability_summary
    if exposure_summary is not None:
        result["exposure"] = exposure_summary

    return result


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
