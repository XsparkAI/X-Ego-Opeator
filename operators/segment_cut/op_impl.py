"""Split ``rgb.mp4`` into task-level or atomic-action clips.

Reads ``caption_v2t.json`` (produced by video_segmentation) and extracts
either task-level or atomic-action-level clips from the unified caption schema.

Cuts are re-encoded on purpose instead of stream-copied. With H.264/H.265
inputs, non-keyframe stream-copy cuts can leak earlier GOP content into the
output clip and create repeated frames at segment boundaries.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..operator_base import OperatorResult
from ..video_path import resolve_episode_video_path
from ..vlm_limit import cpu_task_slot

log = logging.getLogger(__name__)

_MIN_FILE_BYTES = 1024  # skip re-cut if file already > 1 KB


@dataclass
class SegmentCutConfig:
    min_duration_sec: float = 0.5
    granularity: str = "atomic_action"
    ffmpeg_timeout_sec: float = 180.0


class SegmentCutOperator:
    name = "segment_cut"

    def __init__(self, config: SegmentCutConfig | None = None):
        self.config = config or SegmentCutConfig()

    def run(self, episode_dir: Path, **kwargs: Any) -> OperatorResult:
        with cpu_task_slot():
            caption_path = episode_dir / "caption_v2t.json"
            video_path = resolve_episode_video_path(episode_dir)

            if not caption_path.exists():
                return OperatorResult(
                    status="error", operator=self.name,
                    errors=[f"caption_v2t.json not found in {episode_dir}"],
                )
            if not video_path.exists():
                return OperatorResult(
                    status="error", operator=self.name,
                    errors=[f"Input video not found in {episode_dir}"],
                )

            caption = json.loads(caption_path.read_text(encoding="utf-8"))
            granularity = self.config.granularity.lower()
            segments = self._select_segments(caption, video_path, granularity)
            fps = caption.get("fps")
            if caption.get("tasks"):
                fps = fps or self._probe_fps(video_path)

            if fps is None:
                fps = self._probe_fps(video_path)
            if fps is None or fps <= 0:
                return OperatorResult(
                    status="error", operator=self.name,
                    errors=["Cannot determine video fps"],
                )

            segments_root = episode_dir / "segments"
            segments_root.mkdir(exist_ok=True)

            segment_dirs: list[str] = []
            cut_count = 0
            skip_count = 0

            for i, seg in enumerate(segments):
                fi = seg.get("frame_interval", [])
                if len(fi) < 2:
                    continue

                start_frame, end_frame = fi[0], fi[1]
                start_sec = start_frame / fps
                duration_sec = (end_frame - start_frame) / fps

                if duration_sec < self.config.min_duration_sec:
                    skip_count += 1
                    continue

                seg_dir = segments_root / f"seg_{i:03d}"
                seg_dir.mkdir(exist_ok=True)
                seg_video = seg_dir / "rgb.mp4"

                if seg_video.exists() and seg_video.stat().st_size > _MIN_FILE_BYTES:
                    segment_dirs.append(str(seg_dir))
                    self._write_info(seg_dir, seg, fps)
                    continue

                # Use input-side seek so long videos do not need to decode from frame 0
                # before each cut. We still re-encode the selected range to keep segment
                # boundaries stable on GOP-compressed inputs.
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start_sec:.6f}",
                    "-i", str(video_path),
                    "-t", f"{duration_sec:.6f}",
                    "-map", "0:v:0",
                    "-map", "0:a?",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-movflags", "+faststart",
                    str(seg_video),
                ]
                try:
                    subprocess.run(
                        cmd,
                        capture_output=True,
                        timeout=max(1.0, float(self.config.ffmpeg_timeout_sec)),
                        check=True,
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    log.warning(f"ffmpeg cut failed for seg_{i:03d}: {e}")
                    continue

                if seg_video.exists() and seg_video.stat().st_size > _MIN_FILE_BYTES:
                    segment_dirs.append(str(seg_dir))
                    self._write_info(seg_dir, seg, fps)
                    cut_count += 1
                else:
                    log.warning(f"seg_{i:03d} output too small, skipping")

            log.info(
                f"segment_cut: {len(segment_dirs)} segments "
                f"(cut={cut_count}, cached={len(segment_dirs) - cut_count}, "
                f"skipped={skip_count})"
            )

            return OperatorResult(
                status="ok",
                operator=self.name,
                output_files=[str(segments_root)],
                metrics={
                    "segment_dirs": segment_dirs,
                    "total_segments": len(segment_dirs),
                    "cut_count": cut_count,
                    "skipped_short": skip_count,
                    "granularity": granularity,
                },
            )

    @staticmethod
    def _write_info(seg_dir: Path, seg: dict, fps: float) -> None:
        fi = seg["frame_interval"]
        info = {
            "start_frame": fi[0],
            "end_frame": fi[1],
            "fps": fps,
            "duration_sec": round((fi[1] - fi[0]) / fps, 4),
            "instruction": seg.get("instruction", ""),
        }
        if "task_instruction" in seg:
            info["task_instruction"] = seg["task_instruction"]
        if "action_name" in seg:
            info["action_name"] = seg["action_name"]
        (seg_dir / "segment_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    @staticmethod
    def _tasks_to_segments(tasks: list[dict], video_path: Path) -> list[dict]:
        fps = SegmentCutOperator._probe_fps(video_path)
        if fps is None or fps <= 0:
            raise ValueError("Cannot determine video fps for hierarchical caption")

        segments = []
        for task in tasks:
            fi = task.get("frame_interval", [])
            if len(fi) < 2:
                continue
            segments.append(
                {
                    "frame_interval": [int(fi[0]), int(fi[1])],
                    "instruction": task.get("instruction") or "",
                    "task_instruction": task.get("instruction", ""),
                }
            )
        return segments

    @staticmethod
    def _actions_to_segments(tasks: list[dict], video_path: Path) -> list[dict]:
        fps = SegmentCutOperator._probe_fps(video_path)
        if fps is None or fps <= 0:
            raise ValueError("Cannot determine video fps for hierarchical caption")

        segments = []
        for task in tasks:
            task_instruction = task.get("instruction", "")
            for action in task.get("atomic_actions", []):
                fi = action.get("frame_interval", [])
                if len(fi) < 2:
                    continue
                action_name = action.get("caption", "")
                segments.append(
                    {
                        "frame_interval": [int(fi[0]), int(fi[1])],
                        "instruction": action_name or task_instruction,
                        "task_instruction": task_instruction,
                        "action_name": action_name,
                    }
                )
        return segments

    @staticmethod
    def _select_segments(caption: dict, video_path: Path, granularity: str) -> list[dict]:
        tasks = caption.get("tasks", [])

        if not tasks:
            return []

        if granularity == "task":
            return SegmentCutOperator._tasks_to_segments(tasks, video_path)
        if granularity == "atomic_action":
            return SegmentCutOperator._actions_to_segments(tasks, video_path)

        raise ValueError(
            f"segment_cut.granularity must be 'atomic_action' or 'task', got {granularity!r}"
        )

    @staticmethod
    def _probe_fps(video_path: Path) -> float | None:
        try:
            out = subprocess.check_output(
                [
                    "ffprobe", "-v", "quiet",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate",
                    "-of", "csv=p=0",
                    str(video_path),
                ],
                text=True, timeout=10,
            )
            num, den = out.strip().split("/")
            return float(num) / float(den)
        except Exception:
            return None
