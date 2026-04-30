"""Pipeline adapter for video segmentation backends."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from ..operator_base import OperatorResult
from ..segment_cut.op_impl import cut_caption_segments
from ..video_path import resolve_episode_video_path
from .vlm_api import get_api_key

log = logging.getLogger(__name__)


def _count_caption_actions(tasks: list[dict]) -> int:
    return sum(len(task.get("atomic_actions", [])) for task in tasks)


def _count_caption_segments(tasks: list[dict]) -> int:
    actions = _count_caption_actions(tasks)
    return actions if actions else len(tasks)


@dataclass
class SegmentationConfig:
    method: str = "task"
    batch_enabled: bool = True
    batch_poll_interval_sec: int = 20
    window_sec: float = 10.0
    step_sec: float = 5.0
    frames_per_window: int = 12
    snap_radius: int = 45
    preview: bool = False
    dry_run: bool = False
    max_workers: int = 8
    task_window_sec: float = 12.0
    task_step_sec: float = 6.0
    task_frames_per_window: int = 12
    action_window_sec: float = 6.0
    action_step_sec: float = 3.0
    action_frames_per_window: int = 8
    segment_cut_enabled: bool = False
    segment_cut_granularity: str = "task"
    segment_cut_output_dir_name: str = "segments"


class SegmentationOperator:
    name = "video_segmentation"
    OUTPUT_FILENAME = "caption.json"
    # Persist async batch submission info so dependent stages can collect later.
    STATE_FILENAME = ".video_segmentation_batch_state.json"

    def __init__(self, config: SegmentationConfig | None = None):
        self.config = config or SegmentationConfig()

    @staticmethod
    def _normalize_method(method: str | None) -> str:
        normalized = str(method or "task").strip().lower()
        aliases = {
            "segment_v2t": "task",
            "task_v2t": "task",
            "task_action_v2t": "atomic_action",
            "atomic": "atomic_action",
            "atomic-action": "atomic_action",
        }
        return aliases.get(normalized, normalized)

    def _state_path(self, episode_dir: Path) -> Path:
        return episode_dir / self.STATE_FILENAME

    def _write_state(self, episode_dir: Path, state: dict) -> None:
        self._state_path(episode_dir).write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_state(self, episode_dir: Path) -> dict | None:
        state_path = self._state_path(episode_dir)
        if not state_path.exists():
            return None
        return json.loads(state_path.read_text(encoding="utf-8"))

    def _clear_state(self, episode_dir: Path) -> None:
        state_path = self._state_path(episode_dir)
        if state_path.exists():
            state_path.unlink()

    def collect(self, episode_dir: Path) -> OperatorResult:
        state = self._load_state(episode_dir)
        if not state:
            return OperatorResult(
                status="error", operator=self.name,
                errors=["No pending video_segmentation batch state found"],
            )

        video_path = resolve_episode_video_path(episode_dir)
        output_path = episode_dir / self.OUTPUT_FILENAME

        try:
            method = self._normalize_method(state["method"])
            preview_fn = None
            if method == "task":
                from .segment_v2t import collect_segment_job, generate_preview

                caption = collect_segment_job(
                    state["submission"],
                    poll_interval_sec=self.config.batch_poll_interval_sec,
                )
                preview_fn = generate_preview
            elif method == "atomic_action":
                from .task_action_v2t import collect_segment_job

                caption = collect_segment_job(
                    state["submission"],
                    poll_interval_sec=self.config.batch_poll_interval_sec,
                )
            else:
                raise ValueError(f"Unsupported pending segmentation method: {method!r}")

            output_path.write_text(
                json.dumps(caption, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._clear_state(episode_dir)

            tasks = caption.get("tasks", [])
            metrics = {
                "num_segments": _count_caption_segments(tasks),
                "num_tasks": len(tasks),
                "num_actions": _count_caption_actions(tasks),
                "method": method,
            }
            output_files = [str(output_path)]
            segment_metrics = None

            if preview_fn is not None:
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    cap.release()
                    preview_path = preview_fn(video_path, caption, fps)
                    output_files.append(str(preview_path))
                    log.info(f"Caption preview saved: {preview_path}")
                except Exception as e:
                    log.warning(f"caption preview failed (non-fatal): {e}")

            if self.config.segment_cut_enabled:
                segment_result = cut_caption_segments(
                    caption_path=output_path,
                    video_path=video_path,
                    output_dir=episode_dir / self.config.segment_cut_output_dir_name,
                    granularity=self.config.segment_cut_granularity,
                )
                if segment_result.status != "ok":
                    raise RuntimeError("; ".join(segment_result.errors) or "segment_cut failed")
                output_files.extend(segment_result.output_files)
                segment_metrics = segment_result.metrics

            if segment_metrics is not None:
                metrics["segment_cut"] = segment_metrics

            return OperatorResult(
                status="ok", operator=self.name,
                output_files=output_files,
                metrics=metrics,
            )
        except Exception as e:
            log.exception("video_segmentation collect failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )

    def run(self, episode_dir: Path, **kwargs) -> OperatorResult:
        video_path = resolve_episode_video_path(episode_dir)
        output_path = episode_dir / self.OUTPUT_FILENAME

        if not video_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"Video not found: {video_path}"],
            )
        if not get_api_key():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[
                    "VLM API key is not set. Configure VLM_API_KEY, DASHSCOPE_API_KEY, "
                    "ARK_API_KEY, or vlm_api_key in pipeline_config.yaml."
                ],
            )

        try:
            method = self._normalize_method(self.config.method)
            preview_fn = None
            if method == "task":
                from .segment_v2t import generate_preview, segment, submit_segment_job

                preview_fn = generate_preview
                if self.config.batch_enabled:
                    submission = submit_segment_job(
                        video_path,
                        window_sec=self.config.window_sec,
                        step_sec=self.config.step_sec,
                        frames_per_window=self.config.frames_per_window,
                    )
                    self._write_state(episode_dir, {"method": method, "submission": submission})
                    return OperatorResult(
                        status="pending", operator=self.name,
                        metrics={"method": method, "pending": True},
                    )
                caption = segment(
                    video_path,
                    window_sec=self.config.window_sec,
                    step_sec=self.config.step_sec,
                    frames_per_window=self.config.frames_per_window,
                    max_workers=self.config.max_workers,
                    batch_enabled=False,
                )
            elif method == "segment_v2t_desc_only":
                raise ValueError("video_segmentation.method=segment_v2t_desc_only is not supported by this operator adapter")
            elif method == "atomic_action":
                from .task_action_v2t import segment, submit_segment_job

                if self.config.batch_enabled:
                    submission = submit_segment_job(
                        video_path,
                        task_window_sec=self.config.task_window_sec,
                        task_step_sec=self.config.task_step_sec,
                        task_frames_per_window=self.config.task_frames_per_window,
                        action_window_sec=self.config.action_window_sec,
                        action_step_sec=self.config.action_step_sec,
                        action_frames_per_window=self.config.action_frames_per_window,
                        max_workers=self.config.max_workers,
                    )
                    self._write_state(episode_dir, {"method": method, "submission": submission})
                    return OperatorResult(
                        status="pending", operator=self.name,
                        metrics={"method": method, "pending": True},
                    )
                caption = segment(
                    video_path,
                    task_window_sec=self.config.task_window_sec,
                    task_step_sec=self.config.task_step_sec,
                    task_frames_per_window=self.config.task_frames_per_window,
                    action_window_sec=self.config.action_window_sec,
                    action_step_sec=self.config.action_step_sec,
                    action_frames_per_window=self.config.action_frames_per_window,
                    max_workers=self.config.max_workers,
                    batch_enabled=False,
                )
            else:
                raise ValueError(
                    f"video_segmentation.method must be 'task' or 'atomic_action', got {self.config.method!r}"
                )

            output_path.write_text(
                json.dumps(caption, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            tasks = caption.get("tasks", [])
            metrics = {
                "num_segments": _count_caption_segments(tasks),
                "num_tasks": len(tasks),
                "num_actions": _count_caption_actions(tasks),
                "method": method,
            }
            output_files = [str(output_path)]
            segment_metrics = None

            # Preview generation is optional metadata output and should not fail the operator.
            if preview_fn is not None:
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    cap.release()
                    preview_path = preview_fn(video_path, caption, fps)
                    output_files.append(str(preview_path))
                    log.info(f"Caption preview saved: {preview_path}")
                except Exception as e:
                    log.warning(f"caption preview failed (non-fatal): {e}")

            if self.config.segment_cut_enabled:
                segment_result = cut_caption_segments(
                    caption_path=output_path,
                    video_path=video_path,
                    output_dir=episode_dir / self.config.segment_cut_output_dir_name,
                    granularity=self.config.segment_cut_granularity,
                )
                if segment_result.status != "ok":
                    raise RuntimeError("; ".join(segment_result.errors) or "segment_cut failed")
                output_files.extend(segment_result.output_files)
                segment_metrics = segment_result.metrics

            if segment_metrics is not None:
                metrics["segment_cut"] = segment_metrics

            return OperatorResult(
                status="ok", operator=self.name,
                output_files=output_files,
                metrics=metrics,
            )
        except Exception as e:
            log.exception("video_segmentation failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )
