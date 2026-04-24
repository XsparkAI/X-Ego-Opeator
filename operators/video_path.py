"""Helpers for resolving the configured input video inside an episode directory."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_INPUT_VIDEO_PATH = "rgb.mp4"
ENV_INPUT_VIDEO_PATH = "EGOX_INPUT_VIDEO_PATH"
DIRECT_VIDEO_NAMES = ("rgb.mp4", "cam_head.mp4", "video.mp4")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def get_configured_input_video_path() -> str:
    value = os.getenv(ENV_INPUT_VIDEO_PATH, "").strip()
    return value or DEFAULT_INPUT_VIDEO_PATH


def resolve_episode_video_path(episode_dir: str | Path) -> Path:
    episode_dir = Path(episode_dir)
    if episode_dir.is_file() and episode_dir.suffix.lower() in VIDEO_SUFFIXES:
        return episode_dir

    configured_rel = Path(get_configured_input_video_path())
    configured_path = episode_dir / configured_rel
    if configured_path.exists():
        return configured_path

    default_path = episode_dir / DEFAULT_INPUT_VIDEO_PATH
    if configured_rel != Path(DEFAULT_INPUT_VIDEO_PATH) and default_path.exists():
        return default_path

    for name in DIRECT_VIDEO_NAMES:
        candidate = episode_dir / name
        if candidate.exists():
            return candidate

    if episode_dir.is_dir():
        direct_videos = sorted(
            path for path in episode_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        if len(direct_videos) == 1:
            return direct_videos[0]

    return configured_path


def episode_has_input_video(episode_dir: str | Path) -> bool:
    return resolve_episode_video_path(episode_dir).exists()
