#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from runner import run_privacy_blur
from platform_input import resolve_local_video_inputs


def _load_hand_helpers():
    for candidate in (
        Path("/app/hand_platform_helpers.py"),
        Path(__file__).resolve().parent.parent / "hand_analysis" / "platform_runner.py",
    ):
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("hand_platform_helpers", candidate)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module
    raise RuntimeError("Cannot locate hand_analysis platform helper module")


helpers = _load_hand_helpers()
VIDEO_SUFFIXES = getattr(helpers, "VIDEO_SUFFIXES", {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})


def _pick_runtime_value(hyperparams: dict, env_key: str, *aliases: str, default: str = "") -> str:
    return helpers._pick_runtime_value(hyperparams, env_key, *aliases, default=default)


def _output_name(default_name: str, hyperparams: dict) -> str:
    output_name = _pick_runtime_value(hyperparams, "OUTPUT_NAME", "output_name", default="").strip()
    if not output_name:
        return default_name
    return output_name if Path(output_name).suffix else f"{output_name}.mp4"


def _resolve_platform_video_inputs(artifact: dict, hyperparams: dict) -> list[tuple[str, Path]]:
    videos = resolve_local_video_inputs(
        artifact,
        read_json_file=helpers._read_json_file,
        video_keys=getattr(helpers, "VIDEO_KEYS", ()),
        video_suffixes=VIDEO_SUFFIXES,
        direct_video_names=getattr(helpers, "DIRECT_VIDEO_NAMES", ("rgb.mp4", "cam_head.mp4", "video.mp4")),
    )
    if videos:
        return [("video", path) for path in videos]

    input_kind, input_path = helpers._resolve_runtime_input(artifact, hyperparams)
    return [(input_kind, input_path)]


def _build_args(kind: str, input_path: Path, output_path: Path, hyperparams: dict) -> argparse.Namespace:
    return argparse.Namespace(
        video=input_path if kind == "video" else None,
        episode=input_path if kind == "episode" else None,
        output=output_path,
        detection_mode=_pick_runtime_value(
            hyperparams,
            "DETECTION_MODE",
            "detection_mode",
            default="sampling_expand",
        ).lower(),
        frame_sampling_step=helpers._as_int(
            _pick_runtime_value(hyperparams, "FRAME_SAMPLING_STEP", "frame_sampling_step", "STEP"),
            30,
        ),
        use_frame_cache=helpers._as_bool(
            _pick_runtime_value(hyperparams, "USE_FRAME_CACHE", "use_frame_cache"),
            default=True,
        ),
        frame_cache_num_workers=helpers._as_int(
            _pick_runtime_value(hyperparams, "FRAME_CACHE_NUM_WORKERS", "frame_cache_num_workers"),
            1,
        ),
        blur_targets=_pick_runtime_value(
            hyperparams,
            "BLUR_TARGETS",
            "blur_targets",
            default=os.getenv("BLUR_TARGETS", "face"),
        ).lower(),
        scale=helpers._as_float(_pick_runtime_value(hyperparams, "SCALE", "scale"), 1.0),
        face_thresh=helpers._as_float(
            _pick_runtime_value(hyperparams, "FACE_THRESH", "face_thresh"),
            None,
        ),
        lp_thresh=helpers._as_float(
            _pick_runtime_value(hyperparams, "LP_THRESH", "lp_thresh"),
            None,
        ),
        yolo_face_model_path=_pick_runtime_value(
            hyperparams,
            "YOLO_FACE_MODEL_PATH",
            "yolo_face_model_path",
            default=os.getenv("YOLO_FACE_MODEL_PATH", "/app/weights/yolo26s_face.pt"),
        ),
        yolo_lp_model_path=_pick_runtime_value(
            hyperparams,
            "YOLO_LP_MODEL_PATH",
            "yolo_lp_model_path",
            default=os.getenv("YOLO_LP_MODEL_PATH", "/app/weights/yolo_lp.pt"),
        ),
        yolo_conf_thresh=helpers._as_float(
            _pick_runtime_value(hyperparams, "YOLO_CONF_THRESH", "yolo_conf_thresh"),
            0.25,
        ),
        yolo_input_size=helpers._as_int(
            _pick_runtime_value(hyperparams, "YOLO_INPUT_SIZE", "yolo_input_size"),
            960,
        ),
        preview=helpers._as_bool(_pick_runtime_value(hyperparams, "PREVIEW", "preview"), default=False),
    )


def _default_blurred_name(input_path: Path, index: int, total: int) -> str:
    if total == 1:
        return "rgb_blurred.mp4"
    stem = input_path.stem
    suffix = input_path.suffix or ".mp4"
    return f"{stem}_blurred{suffix}"


def _log_runtime_args(args: argparse.Namespace) -> None:
    print(
        "[egox] resolved runtime args "
        "detector_backend=yolo "
        f"blur_targets={args.blur_targets} "
        f"detection_mode={args.detection_mode} "
        f"frame_sampling_step={args.frame_sampling_step} "
        f"scale={args.scale} "
        f"use_frame_cache={args.use_frame_cache}",
        file=sys.stderr,
        flush=True,
    )


def _runtime_args_summary(args: argparse.Namespace) -> dict:
    return {
        "detectorBackend": "yolo",
        "blurTargets": args.blur_targets,
        "detectionMode": args.detection_mode,
        "frameSamplingStep": args.frame_sampling_step,
        "scale": args.scale,
        "useFrameCache": args.use_frame_cache,
        "frameCacheNumWorkers": args.frame_cache_num_workers,
        "preview": args.preview,
        "faceThresh": args.face_thresh,
        "lpThresh": args.lp_thresh,
        "yoloFaceModelPath": args.yolo_face_model_path,
        "yoloLpModelPath": args.yolo_lp_model_path,
        "yoloConfThresh": args.yolo_conf_thresh,
        "yoloInputSize": args.yolo_input_size,
    }


def main() -> None:
    output_dir = Path(os.environ["OUTPUT_DIR"])
    result_output_path = Path(os.environ["RESULT_OUTPUT_PATH"])
    stage_manifest_path = Path(os.environ["STAGE_MANIFEST_PATH"])
    node_label = os.getenv("NODE_LABEL", "privacy_blur")

    output_dir.mkdir(parents=True, exist_ok=True)
    node_data = helpers._load_node_data()
    hyperparams = node_data.get("hyperparams") if isinstance(node_data.get("hyperparams"), dict) else {}

    artifacts = helpers._load_input_artifacts()
    source_artifact = helpers._select_primary_artifact(artifacts)
    video_inputs = _resolve_platform_video_inputs(source_artifact, hyperparams)
    if not video_inputs:
        raise RuntimeError("privacy_blur requires at least one input video")

    helpers._write_json(stage_manifest_path, {"dataRefs": []})

    output_artifacts = []
    summaries = []
    first_runtime_args = None
    for index, (input_kind, input_path) in enumerate(video_inputs, start=1):
        default_name = _default_blurred_name(input_path, index, len(video_inputs))
        artifact_path = output_dir / _output_name(default_name, hyperparams if len(video_inputs) == 1 else {})
        args = _build_args(input_kind, input_path, artifact_path, hyperparams)
        if index == 1:
            _log_runtime_args(args)
            first_runtime_args = _runtime_args_summary(args)

        runtime_result = run_privacy_blur(args)
        summary = runtime_result["result"]
        runtime_args = _runtime_args_summary(args)
        summaries.append(summary)

        output_artifacts.append({
            "name": runtime_result["output_path"].name,
            "type": "video",
            "path": str(runtime_result["output_path"]),
            "portId": "out-1",
            "portName": "video",
            "data": summary,
            "metadata": {
                "detectorBackend": runtime_result["backend"],
                "runtimeArgs": runtime_args,
                "inputKind": input_kind,
                "inputPath": str(input_path),
                "inputArtifactName": source_artifact.get("name"),
                "inputPortId": source_artifact.get("portId"),
                "inputPortName": source_artifact.get("portName"),
                "inputIndex": index,
                "inputCount": len(video_inputs),
            },
        })
        output_artifacts.append({
            "name": runtime_result["summary_path"].name,
            "type": "json",
            "path": str(runtime_result["summary_path"]),
            "portId": "out-2",
            "portName": "summary",
            "data": summary,
            "metadata": {
                "detectorBackend": runtime_result["backend"],
                "runtimeArgs": runtime_args,
                "inputKind": input_kind,
                "inputPath": str(input_path),
                "inputIndex": index,
                "inputCount": len(video_inputs),
            },
        })
        if runtime_result.get("preview_path") is not None:
            output_artifacts.append({
                "name": runtime_result["preview_path"].name,
                "type": "video",
                "path": str(runtime_result["preview_path"]),
                "portId": "out-3",
                "portName": "preview",
                "data": {"previewFor": str(runtime_result["output_path"])},
                "metadata": {
                    "detectorBackend": runtime_result["backend"],
                    "inputIndex": index,
                    "inputCount": len(video_inputs),
                },
            })

    aggregate_data = {
        "inputCount": len(video_inputs),
        "outputVideoCount": len([artifact for artifact in output_artifacts if artifact.get("portId") == "out-1"]),
        "summaries": summaries,
    }
    aggregate_path = output_dir / "privacy_blur_manifest.json"
    helpers._write_json(aggregate_path, aggregate_data)
    output_artifacts.append({
        "name": aggregate_path.name,
        "type": "json",
        "path": str(aggregate_path),
        "portId": "out-2",
        "portName": "summary",
        "data": aggregate_data,
        "metadata": {
            "detectorBackend": "yolo",
            "runtimeArgs": first_runtime_args or {},
            "inputCount": len(video_inputs),
        },
    })

    result_output = {
        "logs": [
            {
                "timestamp": helpers._now_iso(),
                "level": "INFO",
                "message": (
                    "privacy_blur resolved runtime args: "
                    f"detector_backend=yolo, blur_targets={first_runtime_args.get('blurTargets') if first_runtime_args else '<empty>'}, "
                    f"detection_mode={first_runtime_args.get('detectionMode') if first_runtime_args else '<empty>'}, "
                    f"frame_sampling_step={first_runtime_args.get('frameSamplingStep') if first_runtime_args else '<empty>'}, "
                    f"input_count={len(video_inputs)}"
                ),
                "operator": node_label,
            },
            {
                "timestamp": helpers._now_iso(),
                "level": "INFO",
                "message": "privacy_blur custom operator completed successfully",
                "operator": node_label,
            },
        ],
        "artifacts": output_artifacts,
        "resourceSeries": [],
    }
    helpers._write_json(result_output_path, result_output)

    print(f"DROBOTICFLOW_RESULT_PATH={result_output_path}")


if __name__ == "__main__":
    main()
