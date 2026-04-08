"""Helpers for resolving the configured input video inside an episode directory."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_INPUT_VIDEO_PATH = "rgb.mp4"
ENV_INPUT_VIDEO_PATH = "EGOX_INPUT_VIDEO_PATH"


def get_configured_input_video_path() -> str:
    value = os.getenv(ENV_INPUT_VIDEO_PATH, "").strip()
    return value or DEFAULT_INPUT_VIDEO_PATH


def resolve_episode_video_path(episode_dir: str | Path) -> Path:
    episode_dir = Path(episode_dir)
    configured_rel = Path(get_configured_input_video_path())
    configured_path = episode_dir / configured_rel
    if configured_path.exists():
        return configured_path

    default_path = episode_dir / DEFAULT_INPUT_VIDEO_PATH
    if configured_rel != Path(DEFAULT_INPUT_VIDEO_PATH) and default_path.exists():
        return default_path
    return configured_path


def episode_has_input_video(episode_dir: str | Path) -> bool:
    return resolve_episode_video_path(episode_dir).exists()
