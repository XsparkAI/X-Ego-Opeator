#!/usr/bin/env python3
"""算子3: 曝光与光照检测 — 过曝/欠曝、动态范围（纯 OpenCV + NumPy 接口）"""

from dataclasses import dataclass

import cv2
import numpy as np

OVEREXPOSURE_THRESH = 220
UNDEREXPOSURE_THRESH = 30

# 单帧异常判定阈值：超过此值则该帧被标记为过曝/欠曝
FRAME_OVEREXPOSURE_LIMIT = 0.05   # 超过 5% 像素过曝 → 该帧异常
FRAME_UNDEREXPOSURE_LIMIT = 0.20  # 超过 20% 像素欠曝 → 该帧异常
UNDEREXPOSURE_BRIGHTNESS_GATE = 0.35  # 平均亮度低于此值才允许判定欠曝（排除深色物体误报）
OVEREXPOSURE_BRIGHTNESS_GATE = 0.55   # 平均亮度高于此值才允许判定过曝（排除白色物体误报）


@dataclass
class FrameExposure:
    """单帧曝光指标"""
    frame_idx: int
    mean_brightness: float     # 平均亮度 (0-1)
    overexposure_ratio: float
    underexposure_ratio: float
    dynamic_range: float       # (0-1)
    histogram_entropy: float


def assess_frame(gray: np.ndarray, frame_idx: int) -> FrameExposure:
    """评估单帧曝光质量"""
    total = gray.size
    mean_b = float(gray.mean()) / 255.0
    over_ratio = float((gray > OVEREXPOSURE_THRESH).sum()) / total
    under_ratio = float((gray < UNDEREXPOSURE_THRESH).sum()) / total

    p1, p99 = np.percentile(gray, [1, 99])
    dynamic_range = (p99 - p1) / 255.0

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    nonzero = hist[hist > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    return FrameExposure(
        frame_idx=frame_idx,
        mean_brightness=mean_b,
        overexposure_ratio=over_ratio,
        underexposure_ratio=under_ratio,
        dynamic_range=dynamic_range,
        histogram_entropy=entropy,
    )


def summarize(results: list[FrameExposure], fps: float = 30.0) -> dict:
    """汇总多帧曝光结果，包含逐帧异常检测和合格判定"""
    over_vals = [e.overexposure_ratio for e in results]
    under_vals = [e.underexposure_ratio for e in results]
    dr_vals = [e.dynamic_range for e in results]
    ent_vals = [e.histogram_entropy for e in results]

    # 逐帧异常检测
    overexposed_frames = []
    underexposed_frames = []
    for e in results:
        if e.overexposure_ratio > FRAME_OVEREXPOSURE_LIMIT and e.mean_brightness > OVEREXPOSURE_BRIGHTNESS_GATE:
            overexposed_frames.append({
                "frame_idx": e.frame_idx,
                "time_sec": round(e.frame_idx / fps, 2),
                "overexposure_ratio": round(e.overexposure_ratio, 4),
                "mean_brightness": round(e.mean_brightness, 4),
            })
        if e.underexposure_ratio > FRAME_UNDEREXPOSURE_LIMIT and e.mean_brightness < UNDEREXPOSURE_BRIGHTNESS_GATE:
            underexposed_frames.append({
                "frame_idx": e.frame_idx,
                "time_sec": round(e.frame_idx / fps, 2),
                "underexposure_ratio": round(e.underexposure_ratio, 4),
                "mean_brightness": round(e.mean_brightness, 4),
            })

    has_anomaly = len(overexposed_frames) > 0 or len(underexposed_frames) > 0

    return {
        "mean_overexposure": round(float(np.mean(over_vals)), 4),
        "mean_underexposure": round(float(np.mean(under_vals)), 4),
        "mean_dynamic_range": round(float(np.mean(dr_vals)), 4),
        "mean_entropy": round(float(np.mean(ent_vals)), 4),
        "pass": not has_anomaly,
        "overexposed_frames": overexposed_frames,
        "underexposed_frames": underexposed_frames,
    }
