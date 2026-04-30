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

from operators.privacy_blur.blur_privacy import blur_video, generate_preview
from operators.video_path import resolve_episode_video_path


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def _optional_float_from_env(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return None if not value else float(value)


def _default_yolo_path(env_name: str, image_path: str, repo_path: str) -> str | None:
    configured = os.getenv(env_name, "").strip()
    if configured:
        return configured
    for candidate in (Path(image_path), Path(repo_path)):
        if candidate.exists():
            return str(candidate)
    return image_path


def _none_if_zero(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def _resolve_video_path(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.video:
        return args.video, args.video.parent
    video_path = resolve_episode_video_path(args.episode)
    return video_path, args.episode


def _default_output_path(args: argparse.Namespace, video_path: Path, base_dir: Path) -> Path:
    if args.output:
        return args.output
    if args.episode:
        return base_dir / "rgb_blurred.mp4"
    return video_path.with_name(f"{video_path.stem}_blurred{video_path.suffix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privacy blur runner for Docker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to input video")
    group.add_argument("--episode", type=Path, help="Episode directory")
    parser.add_argument("--output", type=Path, default=None, help="Output video path")
    parser.add_argument(
        "--detection-mode",
        choices=["sampling_expand", "legacy_per_frame"],
        default=os.getenv("DETECTION_MODE", "sampling_expand"),
    )
    parser.add_argument(
        "--frame-sampling-step",
        type=int,
        default=int(os.getenv("FRAME_SAMPLING_STEP", os.getenv("STEP", "30"))),
    )
    parser.add_argument("--scale", type=float, default=float(os.getenv("SCALE", "1.0")))
    parser.add_argument(
        "--blur-targets",
        choices=["face", "lp", "both"],
        default=os.getenv("BLUR_TARGETS", "face"),
        help="Blur targets: face, lp, or both. Defaults to face only.",
    )
    parser.add_argument("--face-thresh", type=float, default=_optional_float_from_env("FACE_THRESH"))
    parser.add_argument("--lp-thresh", type=float, default=_optional_float_from_env("LP_THRESH"))
    parser.add_argument(
        "--yolo-face-model-path",
        default=_default_yolo_path(
            "YOLO_FACE_MODEL_PATH",
            "/app/weights/yolo26s_face.pt",
            "weights/yolo26s_face.pt",
        ),
    )
    parser.add_argument(
        "--yolo-lp-model-path",
        default=_default_yolo_path(
            "YOLO_LP_MODEL_PATH",
            "/app/weights/yolo_lp.pt",
            "/home/pc/Desktop/zijian/ego/Ego-X_Operator/weights/yolo_lp.pt",
        ),
    )
    parser.add_argument("--yolo-conf-thresh", type=float, default=float(os.getenv("YOLO_CONF_THRESH", "0.25")))
    parser.add_argument("--yolo-input-size", type=int, default=int(os.getenv("YOLO_INPUT_SIZE", "960")))
    parser.add_argument(
        "--use-frame-cache",
        action=argparse.BooleanOptionalAction,
        default=_bool_from_env("USE_FRAME_CACHE", True),
    )
    parser.add_argument(
        "--frame-cache-num-workers",
        type=int,
        default=int(os.getenv("FRAME_CACHE_NUM_WORKERS", "1")),
    )
    parser.add_argument("--preview", action="store_true", default=_bool_from_env("PREVIEW"))
    return parser


def run_privacy_blur(args: argparse.Namespace) -> dict:
    video_path, base_dir = _resolve_video_path(args)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    output_path = _default_output_path(args, video_path, base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = blur_video(
        video_path,
        output_path,
        detector_backend="yolo",
        blur_targets=args.blur_targets,
        detection_mode=args.detection_mode,
        frame_sampling_step=args.frame_sampling_step,
        use_frame_cache=args.use_frame_cache,
        frame_cache_num_workers=args.frame_cache_num_workers,
        scale=args.scale,
        face_thresh=args.face_thresh,
        lp_thresh=args.lp_thresh,
        yolo_face_model_path=args.yolo_face_model_path,
        yolo_lp_model_path=args.yolo_lp_model_path,
        yolo_conf_thresh=args.yolo_conf_thresh,
        yolo_input_size=_none_if_zero(args.yolo_input_size),
    )

    preview_path = None
    if args.preview:
        preview_path = output_path.with_name(f"{output_path.stem}_preview{output_path.suffix}")
        generate_preview(video_path, output_path, preview_path)

    summary_path = output_path.with_suffix(output_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "backend": "yolo",
        "video_path": video_path,
        "output_path": output_path,
        "summary_path": summary_path,
        "preview_path": preview_path,
        "result": summary,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_privacy_blur(args)


if __name__ == "__main__":
    main()
