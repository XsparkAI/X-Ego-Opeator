#!/usr/bin/env python3
"""算子1: 画面质量检测 — 清晰度、亮度、模糊度"""

from dataclasses import dataclass

import cv2
import numpy as np

BLUR_THRESHOLD = 100.0  # Laplacian 方差低于此值视为模糊


@dataclass
class FrameQuality:
    """单帧画面质量指标"""
    frame_idx: int
    laplacian_var: float       # 清晰度（Laplacian 方差），越高越清晰
    tenengrad: float           # Tenengrad 梯度能量
    mean_brightness: float     # 平均亮度 (0-255)
    is_blurry: bool


def assess_frame(gray: np.ndarray, frame_idx: int) -> FrameQuality:
    """评估单帧画面质量"""
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))
    mean_brightness = float(gray.mean())

    return FrameQuality(
        frame_idx=frame_idx,
        laplacian_var=lap_var,
        tenengrad=tenengrad,
        mean_brightness=mean_brightness,
        is_blurry=lap_var < BLUR_THRESHOLD,
    )


def summarize(results: list[FrameQuality]) -> dict:
    """汇总多帧质量结果"""
    lap_vals = [q.laplacian_var for q in results]
    ten_vals = [q.tenengrad for q in results]
    bri_vals = [q.mean_brightness for q in results]
    blur_count = sum(1 for q in results if q.is_blurry)
    return {
        "mean_laplacian": round(float(np.mean(lap_vals)), 2),
        "mean_tenengrad": round(float(np.mean(ten_vals)), 2),
        "mean_brightness": round(float(np.mean(bri_vals)), 2),
        "blur_ratio": round(blur_count / len(results), 4),
    }
