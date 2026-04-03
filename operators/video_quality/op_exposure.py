#!/usr/bin/env python3
"""算子3: 曝光与光照检测 — 过曝/欠曝、动态范围（纯 OpenCV + NumPy 接口）"""

from dataclasses import dataclass

import cv2
import numpy as np

OVEREXPOSURE_THRESH = 220
UNDEREXPOSURE_THRESH = 30


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


def summarize(results: list[FrameExposure]) -> dict:
    """汇总多帧曝光结果"""
    over_vals = [e.overexposure_ratio for e in results]
    under_vals = [e.underexposure_ratio for e in results]
    dr_vals = [e.dynamic_range for e in results]
    ent_vals = [e.histogram_entropy for e in results]
    return {
        "mean_overexposure": round(float(np.mean(over_vals)), 4),
        "mean_underexposure": round(float(np.mean(under_vals)), 4),
        "mean_dynamic_range": round(float(np.mean(dr_vals)), 4),
        "mean_entropy": round(float(np.mean(ent_vals)), 4),
    }
