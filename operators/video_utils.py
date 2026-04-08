"""Shared video utilities for Ego-X operators.

Provides rotation-safe frame extraction that correctly handles OpenCV's
auto-orientation feature (CAP_PROP_ORIENTATION_AUTO, available since OpenCV 4.5.1).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def _get_video_rotation_metadata(video_path: str) -> int:
    """Read rotation degrees from video container metadata via ffprobe.

    Returns 0, 90, 180, or 270.
    """
    # 1) Try legacy rotate tag
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream_tags=rotate", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=5,
        )
        if out.stdout.strip():
            return int(out.stdout.strip()) % 360
    except Exception:
        pass

    # 2) Try side_data displaymatrix (newer containers)
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream_side_data=rotation", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=5,
        )
        if out.stdout.strip():
            rot = float(out.stdout.strip().split('\n')[0])
            # displaymatrix rotation is often negative (e.g. -90 means 90° CW)
            return int(-rot) % 360
    except Exception:
        pass

    return 0


def get_manual_rotation(video_path: str) -> int:
    """Return rotation degrees that need to be applied manually.

    If OpenCV's CAP_PROP_ORIENTATION_AUTO is enabled (OpenCV ≥ 4.5.1),
    frames are already correctly oriented — returns 0.
    Otherwise, returns the rotation from container metadata.
    """
    cap = cv2.VideoCapture(video_path)
    # CAP_PROP_ORIENTATION_AUTO = 49: 1.0 means OpenCV auto-rotates
    auto_orient = cap.get(49)
    cap.release()
    if auto_orient == 1.0:
        return 0
    return _get_video_rotation_metadata(video_path)


def apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate frame by the given degrees (0, 90, 180, 270)."""
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame
