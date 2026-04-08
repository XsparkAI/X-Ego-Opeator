#!/usr/bin/env python3
"""算子2: 视频稳定性检测 — 基于光流追踪 + 仿射估计（纯 OpenCV 接口）"""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

FEATURE_MAX_CORNERS = 500
FEATURE_QUALITY = 0.01
FEATURE_MIN_DIST = 30

# 帧对级异常阈值（速度形式，自动适配不同采样率）
JITTER_TRANSLATION_SPEED = 1500.0  # 帧间平移速度超过 1500 像素/秒 → 标记为剧烈抖动
JITTER_ROTATION_SPEED = 0.50       # 帧间旋转速度超过 0.50 弧度/秒 (~28.6°/s) → 标记为剧烈抖动


@dataclass
class FramePairMotion:
    """单帧对的运动信息"""
    frame_idx_from: int
    frame_idx_to: int
    translation: float   # 帧间平移量 (像素)
    rotation: float      # 帧间旋转量 (弧度)
    matched_points: int


@dataclass
class StabilityResult:
    """视频稳定性结果"""
    translation_std: float   # 帧间平移标准差 (像素)
    rotation_std: float      # 帧间旋转标准差 (弧度)
    mean_matched_points: float  # 平均匹配特征点数
    pair_motions: list  # list[FramePairMotion]


def _assess_pair(args: tuple[np.ndarray, np.ndarray, int, int]) -> tuple[float, float, float, int, FramePairMotion] | None:
    prev, curr, frame_idx_from, frame_idx_to = args

    pts = cv2.goodFeaturesToTrack(prev, FEATURE_MAX_CORNERS, FEATURE_QUALITY, FEATURE_MIN_DIST)
    if pts is None or len(pts) < 4:
        return None

    pts_new, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None)
    good_old = pts[status.flatten() == 1]
    good_new = pts_new[status.flatten() == 1]

    if len(good_old) < 4:
        return None

    mat, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
    if mat is None:
        return None

    dx = float(mat[0, 2])
    dy = float(mat[1, 2])
    da = float(np.arctan2(mat[1, 0], mat[0, 0]))
    mc = int(inliers.sum()) if inliers is not None else len(good_old)
    trans = float(np.sqrt(dx ** 2 + dy ** 2))

    return (
        dx,
        dy,
        da,
        mc,
        FramePairMotion(
            frame_idx_from=frame_idx_from,
            frame_idx_to=frame_idx_to,
            translation=trans,
            rotation=float(abs(da)),
            matched_points=mc,
        ),
    )


def assess(frames_gray: list[np.ndarray], fps: float,
           frame_indices: list[int] | None = None) -> StabilityResult:
    """评估视频稳定性：Shi-Tomasi 特征 + Lucas-Kanade 光流 + 仿射估计"""
    dx_list, dy_list, da_list = [], [], []
    matched_counts = []
    pair_motions = []

    if frame_indices is None:
        frame_indices = list(range(len(frames_gray)))

    pair_args = [
        (frames_gray[i - 1], frames_gray[i], frame_indices[i - 1], frame_indices[i])
        for i in range(1, len(frames_gray))
    ]
    max_workers = min(8, max(1, len(pair_args)))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(_assess_pair, pair_args):
            if result is None:
                continue
            dx, dy, da, mc, pair_motion = result
            dx_list.append(dx)
            dy_list.append(dy)
            da_list.append(da)
            matched_counts.append(mc)
            pair_motions.append(pair_motion)

    if not dx_list:
        return StabilityResult(0.0, 0.0, 0.0, [])

    trans_std = float(np.sqrt(np.std(dx_list) ** 2 + np.std(dy_list) ** 2))
    rot_std = float(np.std(da_list))
    mean_pts = float(np.mean(matched_counts))

    return StabilityResult(trans_std, rot_std, mean_pts, pair_motions)


def summarize(result: StabilityResult, fps: float = 30.0) -> dict:
    """汇总稳定性结果，包含剧烈抖动帧检测和合格判定"""
    jitter_frames = []
    for pm in result.pair_motions:
        # 帧间时间差 → 计算速度
        dt = (pm.frame_idx_to - pm.frame_idx_from) / fps if fps > 0 else 1.0
        trans_speed = pm.translation / dt if dt > 0 else 0.0
        rot_speed = pm.rotation / dt if dt > 0 else 0.0

        if trans_speed > JITTER_TRANSLATION_SPEED or rot_speed > JITTER_ROTATION_SPEED:
            jitter_frames.append({
                "frame_idx_from": pm.frame_idx_from,
                "frame_idx_to": pm.frame_idx_to,
                "time_sec": round(pm.frame_idx_from / fps, 2),
                "translation": round(pm.translation, 2),
                "translation_speed": round(trans_speed, 1),
                "rotation": round(pm.rotation, 4),
                "rotation_speed": round(rot_speed, 4),
                "matched_points": pm.matched_points,
            })

    passed = len(jitter_frames) == 0

    return {
        "translation_std": result.translation_std,
        "rotation_std": result.rotation_std,
        "mean_matched_points": result.mean_matched_points,
        "pass": passed,
        "jitter_frames": jitter_frames,
    }
