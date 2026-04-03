#!/usr/bin/env python3
"""算子2: 视频稳定性检测 — 基于光流追踪 + 仿射估计（纯 OpenCV 接口）"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

FEATURE_MAX_CORNERS = 500
FEATURE_QUALITY = 0.01
FEATURE_MIN_DIST = 30


@dataclass
class StabilityResult:
    """视频稳定性结果"""
    translation_std: float   # 帧间平移标准差 (像素)
    rotation_std: float      # 帧间旋转标准差 (弧度)
    mean_matched_points: float  # 平均匹配特征点数


def assess(frames_gray: list[np.ndarray], fps: float) -> StabilityResult:
    """评估视频稳定性：Shi-Tomasi 特征 + Lucas-Kanade 光流 + 仿射估计"""
    dx_list, dy_list, da_list = [], [], []
    matched_counts = []

    for i in range(1, len(frames_gray)):
        prev, curr = frames_gray[i - 1], frames_gray[i]

        pts = cv2.goodFeaturesToTrack(prev, FEATURE_MAX_CORNERS, FEATURE_QUALITY, FEATURE_MIN_DIST)
        if pts is None or len(pts) < 4:
            continue

        pts_new, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None)
        good_old = pts[status.flatten() == 1]
        good_new = pts_new[status.flatten() == 1]

        if len(good_old) < 4:
            continue

        mat, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
        if mat is None:
            continue

        dx_list.append(mat[0, 2])
        dy_list.append(mat[1, 2])
        da_list.append(np.arctan2(mat[1, 0], mat[0, 0]))
        matched_counts.append(int(inliers.sum()) if inliers is not None else len(good_old))

    if not dx_list:
        return StabilityResult(0.0, 0.0, 0.0)

    trans_std = float(np.sqrt(np.std(dx_list) ** 2 + np.std(dy_list) ** 2))
    rot_std = float(np.std(da_list))
    mean_pts = float(np.mean(matched_counts))

    return StabilityResult(trans_std, rot_std, mean_pts)


def summarize(result: StabilityResult) -> dict:
    return asdict(result)
