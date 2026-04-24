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

from operators.caption.vlm_api import get_api_key
from operators.hand.detect_hand_in_frame import (
    detect_hands_in_video,
    generate_preview,
)
from operators.hand.vlm_hand_audit import audit_video
from operators.video_path import resolve_episode_video_path


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def _default_backend() -> str:
    return os.getenv("BACKEND", os.getenv("HAND_BACKEND", "yolo")).strip().lower() or "yolo"


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
        video_path = args.video
        parent_dir = video_path.parent
        return video_path, parent_dir

    episode_dir = args.episode
    video_path = resolve_episode_video_path(episode_dir)
    return video_path, episode_dir


def _default_output_path(args: argparse.Namespace, video_path: Path, base_dir: Path) -> Path:
    if args.output:
        return args.output
    return base_dir / "hand_analysis.json" if args.episode else video_path.with_name("hand_analysis.json")


def _normalize_yolo_result(result: dict) -> dict:
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    frame_results = result.get("frame_results", []) if isinstance(result.get("frame_results"), list) else []
    sampled_frames = int(summary.get("processed_frames", summary.get("total_frames_processed", len(frame_results))) or 0)
    valid_frames = sampled_frames
    left_frames = int(summary.get("frames_with_left_hand", 0) or 0)
    right_frames = int(summary.get("frames_with_right_hand", 0) or 0)
    at_least_one_hand = int(
        summary.get("frames_with_at_least_one_hand", summary.get("frames_with_any_hand", summary.get("frames_with_hands", 0)))
        or 0
    )
    both_hands = int(summary.get("frames_with_both_hands", 0) or 0)
    no_hands = int(summary.get("frames_with_no_hands", max(sampled_frames - at_least_one_hand, 0)) or 0)
    avg_ego_hand_count = round((left_frames + right_frames) / sampled_frames, 3) if sampled_frames > 0 else 0.0
    return {
        "backend": "yolo",
        "summary": {
            "backend": "yolo",
            "totalFrames": summary.get("total_frames"),
            "sampledFrames": sampled_frames,
            "validFrames": valid_frames,
            "failedFrames": 0,
            "avgEgoHandCount": avg_ego_hand_count,
            "framesWithAtLeastOneHand": at_least_one_hand,
            "atLeastOneHandRatio": summary.get("at_least_one_hand_ratio", summary.get("any_hand_ratio")),
            "framesWithBothHands": both_hands,
            "bothHandsRatio": summary.get("both_hands_ratio"),
            "framesWithNoHands": no_hands,
            "noHandsRatio": summary.get("no_hands_ratio"),
            "activeManipulationRatio": None,
            "singlePersonOperationRatio": None,
        },
        "frameResults": frame_results,
        "rawSummary": summary,
    }


def _normalize_vlm_result(result: dict) -> dict:
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    frame_results = result.get("frame_results", []) if isinstance(result.get("frame_results"), list) else []
    valid_frames = int(summary.get("valid_responses", len(frame_results)) or 0)
    sampled_frames = int(summary.get("total_frames_sampled", valid_frames) or 0)
    failed_frames = int(summary.get("failed_responses", max(sampled_frames - valid_frames, 0)) or 0)
    hand_counts = [
        int(item.get("ego_hand_count", -1))
        for item in frame_results
        if isinstance(item, dict) and int(item.get("ego_hand_count", -1)) >= 0
    ]
    frames_with_no_hands = sum(1 for count in hand_counts if count == 0)
    frames_with_at_least_one_hand = sum(1 for count in hand_counts if count >= 1)
    frames_with_both_hands = sum(1 for count in hand_counts if count >= 2)
    return {
        "backend": "vlm",
        "summary": {
            "backend": "vlm",
            "totalFrames": summary.get("total_frames"),
            "sampledFrames": sampled_frames,
            "validFrames": valid_frames,
            "failedFrames": failed_frames,
            "avgEgoHandCount": summary.get("avg_ego_hand_count"),
            "framesWithAtLeastOneHand": frames_with_at_least_one_hand,
            "atLeastOneHandRatio": summary.get("ego_1_hand_ratio", 0) + summary.get("ego_2_hands_ratio", 0),
            "framesWithBothHands": frames_with_both_hands,
            "bothHandsRatio": summary.get("ego_2_hands_ratio"),
            "framesWithNoHands": frames_with_no_hands,
            "noHandsRatio": summary.get("ego_0_hands_ratio"),
            "activeManipulationRatio": summary.get("active_manipulation_ratio"),
            "singlePersonOperationRatio": summary.get("single_person_operation_ratio"),
        },
        "frameResults": frame_results,
        "rawSummary": summary,
    }


def _normalize_hand_analysis_result(backend: str, result: dict) -> dict:
    if backend == "vlm":
        return _normalize_vlm_result(result)
    return _normalize_yolo_result(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified hand analysis runner for Docker")
    parser.add_argument(
        "--backend",
        choices=["yolo", "vlm"],
        default=_default_backend(),
        help="Hand analysis backend (default: %(default)s)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to input video")
    group.add_argument("--episode", type=Path, help="Episode directory")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")

    parser.add_argument("--frame-step", type=int, default=int(os.getenv("FRAME_STEP", os.getenv("STEP", "1"))))
    parser.add_argument("--conf", type=float, default=float(os.getenv("CONF", "0.3")))
    parser.add_argument("--resize", type=int, default=int(os.getenv("RESIZE", "720")))
    parser.add_argument("--preview", action="store_true", default=_bool_from_env("PREVIEW"))

    parser.add_argument("--max-workers", type=int, default=int(os.getenv("MAX_WORKERS", "4")))
    parser.add_argument("--no-batch", action="store_true", default=_bool_from_env("NO_BATCH"))
    parser.add_argument("--vlm-provider", default=os.getenv("VLM_API_PROVIDER", ""))
    parser.add_argument("--vlm-api-key", default=os.getenv("VLM_API_KEY", ""))
    parser.add_argument("--dashscope-api-key", default=os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--ark-api-key", default=os.getenv("ARK_API_KEY", ""))
    parser.add_argument("--vlm-model", default=os.getenv("VLM_MODEL", os.getenv("VLM_HAND_MODEL", os.getenv("VLM_DEFAULT_MODEL", ""))))
    return parser


def run_yolo(args: argparse.Namespace, video_path: Path, output_path: Path) -> dict:
    raw_result = detect_hands_in_video(
        video_path,
        conf_thresh=args.conf,
        frame_step=args.frame_step,
        input_height=args.resize,
    )
    result = _normalize_hand_analysis_result("yolo", raw_result)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.preview:
        preview_path = output_path.with_name(output_path.stem + "_preview.mp4")
        generate_preview(video_path, raw_result, preview_path)
    return result


def run_vlm(args: argparse.Namespace, video_path: Path, output_path: Path) -> dict:
    _configure_vlm_env(args)
    if not get_api_key():
        raise SystemExit(
            "VLM API key is not set. Provide one of: "
            "VLM_API_KEY, DASHSCOPE_API_KEY, ARK_API_KEY, or the matching CLI flag."
        )
    raw_result = audit_video(
        video_path,
        frame_step=args.frame_step,
        max_workers=args.max_workers,
        batch_enabled=not args.no_batch,
    )
    result = _normalize_hand_analysis_result("vlm", raw_result)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_hand_analysis(args: argparse.Namespace) -> dict:
    video_path, base_dir = _resolve_video_path(args)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_path = _default_output_path(args, video_path, base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "vlm":
        result = run_vlm(args, video_path, output_path)
    else:
        result = run_yolo(args, video_path, output_path)

    return {
        "backend": args.backend,
        "video_path": video_path,
        "output_path": output_path,
        "result": result,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_hand_analysis(args)


if __name__ == "__main__":
    main()
