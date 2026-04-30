from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DIRECT_VIDEO_NAMES = ("rgb.mp4", "cam_head.mp4", "video.mp4")
COLLECTION_KEYS = (
    "clips",
    "clipPaths",
    "clip_paths",
    "videos",
    "videoPaths",
    "video_paths",
    "files",
    "items",
    "entries",
    "artifacts",
    "segments",
    "segment_dirs",
)
PATH_KEYS = (
    "clipPath",
    "clip_path",
    "payloadPath",
    "manifestPath",
    "partitionDir",
    "rootPath",
    "root_path",
)


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def drop_parent_videos_when_child_inputs_exist(
    paths: list[Path],
    *,
    direct_video_names: tuple[str, ...] = DIRECT_VIDEO_NAMES,
) -> list[Path]:
    resolved = [(path, path.resolve()) for path in paths]
    filtered: list[Path] = []
    for path, real_path in resolved:
        if path.name in direct_video_names and any(
            other != real_path and real_path.parent in other.parents
            for _, other in resolved
        ):
            continue
        filtered.append(path)
    return filtered


def resolve_local_video_inputs(
    artifact: dict,
    *,
    read_json_file: Callable[[Path], object],
    video_keys: tuple[str, ...],
    video_suffixes: set[str] = VIDEO_SUFFIXES,
    direct_video_names: tuple[str, ...] = DIRECT_VIDEO_NAMES,
) -> list[Path]:
    visited_json_files: set[Path] = set()
    visited_dirs: set[Path] = set()

    def resolve_path(candidate: str | Path, base_dir: Path | None = None) -> Path:
        path = Path(candidate)
        if path.is_absolute() or base_dir is None:
            return path
        return (base_dir / path).resolve()

    def direct_video_files(directory: Path) -> list[Path]:
        configured_name = os.getenv("EGOX_INPUT_VIDEO_PATH", "rgb.mp4")
        for name in (configured_name, *direct_video_names):
            candidate = directory / name
            if candidate.is_file():
                return [candidate]
        return sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in video_suffixes
        )

    def collect_from_dir(directory: Path) -> list[Path]:
        real_dir = directory.resolve()
        if real_dir in visited_dirs:
            return []
        visited_dirs.add(real_dir)

        direct_videos = direct_video_files(real_dir)
        if direct_videos:
            return direct_videos

        child_videos: list[Path] = []
        child_dirs = sorted(
            path for path in real_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        for child in child_dirs:
            child_videos.extend(direct_video_files(child))
        if child_videos:
            return unique_paths(child_videos)

        json_candidates = sorted(
            path
            for path in real_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        )
        for child in child_dirs:
            json_candidates.extend(
                sorted(
                    path
                    for path in child.iterdir()
                    if path.is_file() and path.suffix.lower() == ".json"
                )
            )

        collected: list[Path] = []
        for json_path in json_candidates[:500]:
            collected.extend(collect_from_path(json_path, real_dir))
        return unique_paths(collected)

    def collect_from_path(candidate: str | Path, base_dir: Path | None = None) -> list[Path]:
        path = resolve_path(candidate, base_dir)
        if path.is_file() and path.suffix.lower() in video_suffixes:
            return [path]
        if path.is_file() and path.suffix.lower() == ".json":
            real_path = path.resolve()
            if real_path in visited_json_files:
                return []
            visited_json_files.add(real_path)
            return collect_from_payload(read_json_file(real_path), real_path.parent)
        if path.is_dir():
            return collect_from_dir(path)
        return []

    def collect_from_payload(payload: object, base_dir: Path | None = None) -> list[Path]:
        if isinstance(payload, str):
            if payload.startswith("bos://"):
                return []
            return collect_from_path(payload, base_dir)

        if isinstance(payload, list):
            collected: list[Path] = []
            for item in payload:
                collected.extend(collect_from_payload(item, base_dir))
            return unique_paths(collected)

        if not isinstance(payload, dict):
            return []

        for key in COLLECTION_KEYS:
            if key in payload:
                collected = collect_from_payload(payload[key], base_dir)
                if collected:
                    return unique_paths(collected)

        for key in ("clipPath", "clip_path", *video_keys, *PATH_KEYS):
            value = payload.get(key)
            if isinstance(value, str) and value:
                collected = collect_from_payload(value, base_dir)
                if collected:
                    return unique_paths(collected)

        collected: list[Path] = []
        for value in payload.values():
            collected.extend(collect_from_payload(value, base_dir))
        return unique_paths(collected)

    collected: list[Path] = []
    artifact_path = artifact.get("path")
    if isinstance(artifact_path, str) and artifact_path and not artifact_path.startswith("bos://"):
        collected.extend(collect_from_path(artifact_path))

    artifact_data = artifact.get("data")
    collected.extend(collect_from_payload(artifact_data))

    for data_ref in (
        artifact.get("dataRef"),
        artifact_data.get("dataRef") if isinstance(artifact_data, dict) else None,
    ):
        if isinstance(data_ref, dict):
            for key in ("payloadPath", "manifestPath", "partitionDir"):
                value = data_ref.get(key)
                if isinstance(value, str) and value and not value.startswith("bos://"):
                    collected.extend(collect_from_path(value))

    return drop_parent_videos_when_child_inputs_exist(
        unique_paths(collected),
        direct_video_names=direct_video_names,
    )
