#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent):
    if (candidate / "operators").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

from operators.caption.segment_v2t import generate_preview, segment as run_segment_v2t
from operators.caption.task_action_v2t import segment as run_task_action_v2t
from operators.caption.vlm_api import get_api_key
from operators.segment_cut.op_impl import cut_caption_segments
from operators.video_path import resolve_episode_video_path


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def _default_method() -> str:
    return _normalize_method(os.getenv("METHOD", "task"))


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


def _configure_vlm_env(args: argparse.Namespace) -> None:
    if args.vlm_provider:
        os.environ["VLM_API_PROVIDER"] = args.vlm_provider
    if args.vlm_api_key:
        os.environ["VLM_API_KEY"] = args.vlm_api_key
    if args.dashscope_api_key:
        os.environ["DASHSCOPE_API_KEY"] = args.dashscope_api_key
    if args.ark_api_key:
        os.environ["ARK_API_KEY"] = args.ark_api_key
    if args.vlm_model:
        os.environ["VLM_MODEL"] = args.vlm_model


def _resolve_video_path(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.video:
        return args.video, args.video.parent
    episode_dir = args.episode
    return resolve_episode_video_path(episode_dir), episode_dir


def _default_output_path(args: argparse.Namespace, video_path: Path, base_dir: Path) -> Path:
    if args.output:
        return args.output
    return base_dir / "caption.json" if args.episode else video_path.with_name("caption.json")


def _caption_summary(method: str, caption: dict, preview_path: Path | None = None) -> dict:
    tasks = caption.get("tasks", []) if isinstance(caption, dict) else []
    actions = sum(len(task.get("atomic_actions", [])) for task in tasks if isinstance(task, dict))
    return {
        "method": method,
        "instruction": caption.get("instruction", "") if isinstance(caption, dict) else "",
        "scene": caption.get("scene", "unknown") if isinstance(caption, dict) else "unknown",
        "numTasks": len(tasks),
        "numActions": actions,
        "previewGenerated": preview_path is not None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified caption runner for Docker")
    parser.add_argument(
        "--method",
        choices=["task", "atomic_action", "segment_v2t", "task_action_v2t"],
        default=_default_method(),
        help="Caption method (default: %(default)s)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to input video")
    group.add_argument("--episode", type=Path, help="Episode directory")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--preview", action="store_true", default=_bool_from_env("PREVIEW"))
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("MAX_WORKERS", "8")))
    parser.add_argument("--no-batch", action="store_true", default=_bool_from_env("NO_BATCH"))
    parser.add_argument("--segment-cut", action="store_true", default=_bool_from_env("SEGMENT_CUT"))
    parser.add_argument(
        "--segment-granularity",
        choices=["task", "atomic_action"],
        default=os.getenv("SEGMENT_GRANULARITY", "task"),
    )
    parser.add_argument(
        "--segment-output-dir",
        type=Path,
        default=Path(os.getenv("SEGMENT_OUTPUT_DIR")) if os.getenv("SEGMENT_OUTPUT_DIR") else None,
    )
    parser.add_argument("--window-sec", type=float, default=float(os.getenv("WINDOW_SEC", "10.0")))
    parser.add_argument("--step-sec", type=float, default=float(os.getenv("STEP_SEC", "5.0")))
    parser.add_argument("--frames-per-window", type=int, default=int(os.getenv("FRAMES_PER_WINDOW", "12")))
    parser.add_argument("--task-name", default=os.getenv("TASK_NAME", ""))

    parser.add_argument("--task-window-sec", type=float, default=float(os.getenv("TASK_WINDOW_SEC", "12.0")))
    parser.add_argument("--task-step-sec", type=float, default=float(os.getenv("TASK_STEP_SEC", "6.0")))
    parser.add_argument("--task-frames-per-window", type=int, default=int(os.getenv("TASK_FRAMES_PER_WINDOW", "12")))
    parser.add_argument("--action-window-sec", type=float, default=float(os.getenv("ACTION_WINDOW_SEC", "6.0")))
    parser.add_argument("--action-step-sec", type=float, default=float(os.getenv("ACTION_STEP_SEC", "3.0")))
    parser.add_argument("--action-frames-per-window", type=int, default=int(os.getenv("ACTION_FRAMES_PER_WINDOW", "8")))

    parser.add_argument("--vlm-provider", default=os.getenv("VLM_API_PROVIDER", ""))
    parser.add_argument("--vlm-api-key", default=os.getenv("VLM_API_KEY", ""))
    parser.add_argument("--dashscope-api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--ark-api-key", default=os.getenv("ARK_API_KEY", ""))
    parser.add_argument(
        "--vlm-model",
        default=os.getenv("VLM_MODEL", ""),
    )
    return parser


def run_caption(args: argparse.Namespace) -> dict:
    _configure_vlm_env(args)
    if not get_api_key():
        raise SystemExit(
            "VLM API key is not set. Provide one of: "
            "VLM_API_KEY, DASHSCOPE_API_KEY, ARK_API_KEY, or the matching CLI flag."
        )

    video_path, base_dir = _resolve_video_path(args)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_path = _default_output_path(args, video_path, base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preview_path: Path | None = None
    method = _normalize_method(args.method)

    if method == "atomic_action":
        caption = run_task_action_v2t(
            video_path,
            task_window_sec=args.task_window_sec,
            task_step_sec=args.task_step_sec,
            task_frames_per_window=args.task_frames_per_window,
            action_window_sec=args.action_window_sec,
            action_step_sec=args.action_step_sec,
            action_frames_per_window=args.action_frames_per_window,
            max_workers=args.max_workers,
            batch_enabled=not args.no_batch,
        )
    else:
        caption = run_segment_v2t(
            video_path,
            task_name=args.task_name or None,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
            frames_per_window=args.frames_per_window,
            max_workers=args.max_workers,
            batch_enabled=not args.no_batch,
        )
        if args.preview:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            preview_path = generate_preview(video_path, caption, fps)

    output_path.write_text(json.dumps(caption, indent=2, ensure_ascii=False), encoding="utf-8")
    segment_result = None
    if args.segment_cut:
        segment_output_dir = args.segment_output_dir or output_path.parent / "segments"
        segment_result = cut_caption_segments(
            caption_path=output_path,
            video_path=video_path,
            output_dir=segment_output_dir,
            granularity=args.segment_granularity,
        )
        if segment_result.status != "ok":
            raise RuntimeError("; ".join(segment_result.errors) or "segment_cut failed")

    return {
        "method": method,
        "video_path": video_path,
        "output_path": output_path,
        "preview_path": preview_path,
        "segment_result": segment_result,
        "result": caption,
        "summary": _caption_summary(method, caption, preview_path=preview_path),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_caption(args)


if __name__ == "__main__":
    main()
