#!/usr/bin/env python3
"""
Hand-in-frame detection operator using YOLO.

Detects whether left/right hands are present in each frame of a video.

Usage:
  python detect_hand_in_frame.py --video path/to/rgb.mp4
  python detect_hand_in_frame.py --video path/to/rgb.mp4 --output hand_analysis.json
  python detect_hand_in_frame.py --video path/to/rgb.mp4 --conf 0.3 --step 1
  python detect_hand_in_frame.py --episode path/to/episode_dir    # reads configured input video, writes hand_analysis.json
  python detect_hand_in_frame.py --video path/to/rgb.mp4 --preview   # also generate preview video
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from ..video_path import resolve_episode_video_path
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from video_path import resolve_episode_video_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "detector.pt"

# Class index from HaWoR: 0 = left hand, >0 = right hand
_HAND_LABELS = {0: "left", 1: "right"}


def load_model(weights: Path = WEIGHTS_PATH) -> YOLO:
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    model = YOLO(str(weights))
    log.info(f"Loaded YOLO hand detector from {weights}")
    return model


def detect_hands_in_video(
    video_path: Path,
    conf_thresh: float = 0.3,
    frame_step: int = 1,
    weights: Path = WEIGHTS_PATH,
    input_height: int | None = 720,
    model: YOLO | None = None,
) -> dict:
    """
    Run YOLO hand detection on every `frame_step`-th frame of the video.

    Args:
        input_height: If set, resize frames to this height (keeping aspect ratio) before detection.
                      Bounding boxes are mapped back to original resolution in the output.

    Returns a dict with:
      - frame_results: list of per-frame detection dicts
      - summary: high-level stats
    """
    if model is None:
        model = load_model(weights)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if input_height and input_height < orig_h:
        scale_down = input_height / orig_h
        det_w = int(orig_w * scale_down)
        det_h = input_height
        scale_back = orig_h / input_height  # to map bbox back to original
        log.info(f"Video: {video_path.name}, {total_frames} frames @ {fps:.2f} fps, "
                 f"detect at {det_w}x{det_h} (original {orig_w}x{orig_h})")
    else:
        scale_down = None
        scale_back = 1.0
        det_w, det_h = orig_w, orig_h
        log.info(f"Video: {video_path.name}, {total_frames} frames @ {fps:.2f} fps, "
                 f"detect at {orig_w}x{orig_h}")

    frame_results = []
    frame_idx = 0
    t_start = time.time()

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            if scale_down is not None:
                det_frame = cv2.resize(frame_bgr, (det_w, det_h))
            else:
                det_frame = frame_bgr

            results = model.track(det_frame, conf=conf_thresh, persist=True, verbose=False)
            boxes_result = results[0].boxes

            left_detections = []
            right_detections = []

            if boxes_result is not None and len(boxes_result) > 0:
                xyxy = boxes_result.xyxy.cpu().numpy()
                confs = boxes_result.conf.cpu().numpy()
                classes = boxes_result.cls.cpu().numpy()
                track_ids = (
                    boxes_result.id.cpu().numpy()
                    if boxes_result.id is not None
                    else [-1] * len(xyxy)
                )

                # Deduplicate: keep at most one left and one right (highest conf)
                left_best_conf = -1
                right_best_conf = -1

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = [v * scale_back for v in xyxy[i].tolist()]
                    conf = float(confs[i])
                    hand_cls = int(classes[i])
                    track_id = int(track_ids[i]) if track_ids[i] != -1 else None
                    hand_side = "left" if hand_cls == 0 else "right"

                    det = {
                        "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "conf": round(conf, 4),
                        "track_id": track_id,
                    }

                    if hand_side == "left" and conf > left_best_conf:
                        left_detections = [det]
                        left_best_conf = conf
                    elif hand_side == "right" and conf > right_best_conf:
                        right_detections = [det]
                        right_best_conf = conf

            frame_results.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / fps, 4) if fps > 0 else None,
                "left_hand": left_detections[0] if left_detections else None,
                "right_hand": right_detections[0] if right_detections else None,
                "has_left": len(left_detections) > 0,
                "has_right": len(right_detections) > 0,
                "has_any_hand": len(left_detections) > 0 or len(right_detections) > 0,
            })

            frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start

    # Summary stats
    n = len(frame_results)
    n_left = sum(1 for f in frame_results if f["has_left"])
    n_right = sum(1 for f in frame_results if f["has_right"])
    n_any = sum(1 for f in frame_results if f["has_any_hand"])
    n_both = sum(1 for f in frame_results if f["has_left"] and f["has_right"])
    n_none = n - n_any

    summary = {
        "video": str(video_path),
        "total_frames": total_frames,
        "total_frames_processed": n,
        "processed_frames": n,
        "fps": round(fps, 4),
        "original_resolution": f"{orig_w}x{orig_h}",
        "detection_resolution": f"{det_w}x{det_h}",
        "frame_step": frame_step,
        "conf_thresh": conf_thresh,
        "elapsed_sec": round(elapsed, 3),
        "fps_throughput": round(n / elapsed, 2) if elapsed > 0 else 0,
        "frames_with_left_hand": n_left,
        "frames_with_right_hand": n_right,
        "frames_with_any_hand": n_any,
        "frames_with_at_least_one_hand": n_any,
        "frames_with_hands": n_any,
        "frames_with_both_hands": n_both,
        "frames_with_no_hands": n_none,
        "hands_detected": n_left + n_right,
        "max_hands_in_a_frame": 2 if n_both > 0 else 1 if n_any > 0 else 0,
        "left_hand_ratio": round(n_left / n, 4) if n > 0 else 0.0,
        "right_hand_ratio": round(n_right / n, 4) if n > 0 else 0.0,
        "any_hand_ratio": round(n_any / n, 4) if n > 0 else 0.0,
        "at_least_one_hand_ratio": round(n_any / n, 4) if n > 0 else 0.0,
        "both_hands_ratio": round(n_both / n, 4) if n > 0 else 0.0,
        "no_hands_ratio": round(n_none / n, 4) if n > 0 else 0.0,
    }

    log.info(
        f"Detection complete: {n} frames processed, "
        f"left={n_left} ({summary['left_hand_ratio']:.1%}), "
        f"right={n_right} ({summary['right_hand_ratio']:.1%})"
    )

    return {"summary": summary, "frame_results": frame_results}


def generate_preview(
    video_path: Path,
    detection_result: dict,
    output_path: Path,
    preview_height: int = 640,
):
    """Draw detection boxes on frames and write a 640p preview video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    scale = preview_height / orig_h
    preview_w = int(orig_w * scale)
    # ensure even dimensions for codec
    preview_w = preview_w if preview_w % 2 == 0 else preview_w + 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (preview_w, preview_height))

    # Build frame index lookup for fast access
    frame_lookup = {fr["frame"]: fr for fr in detection_result["frame_results"]}

    LEFT_COLOR = (255, 144, 30)   # blue-ish (BGR)
    RIGHT_COLOR = (0, 200, 0)     # green (BGR)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame_bgr, (preview_w, preview_height))
        det = frame_lookup.get(frame_idx)

        if det:
            for side, color in [("left_hand", LEFT_COLOR), ("right_hand", RIGHT_COLOR)]:
                info = det.get(side)
                if info is None:
                    continue
                x1, y1, x2, y2 = info["bbox_xyxy"]
                # scale bbox to preview resolution
                sx1, sy1 = int(x1 * scale), int(y1 * scale)
                sx2, sy2 = int(x2 * scale), int(y2 * scale)
                cv2.rectangle(frame_resized, (sx1, sy1), (sx2, sy2), color, 2)
                label = f"{side.split('_')[0][0].upper()} {info['conf']:.2f}"
                cv2.putText(frame_resized, label, (sx1, max(sy1 - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # status bar
            status_parts = []
            if det["has_left"]:
                status_parts.append("L")
            if det["has_right"]:
                status_parts.append("R")
            status = f"Frame {frame_idx}  Hand: {'+'.join(status_parts) if status_parts else 'NONE'}"
            cv2.putText(frame_resized, status, (10, preview_height - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame_resized)
        frame_idx += 1

    cap.release()
    writer.release()
    log.info(f"Preview saved to {output_path} ({preview_w}x{preview_height})")


def process_episode(episode_dir: Path, conf_thresh: float = 0.3, frame_step: int = 1) -> Path:
    """Process a single episode directory (reads configured input video, writes hand_analysis.json)."""
    video_path = resolve_episode_video_path(episode_dir)
    output_path = episode_dir / "hand_analysis.json"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    result = detect_hands_in_video(video_path, conf_thresh=conf_thresh, frame_step=frame_step)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"Saved detection results to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="YOLO hand-in-frame detection operator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to video file")
    group.add_argument("--episode", type=Path, help="Episode directory (reads configured input video)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (--video mode only)")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold (default: 0.3)")
    parser.add_argument("--step", type=int, default=1, help="Process every N-th frame (default: 1)")
    parser.add_argument("--preview", action="store_true", help="Generate preview video with detection boxes")
    parser.add_argument("--resize", type=int, default=720, help="Resize input to this height before detection (default: 720)")
    args = parser.parse_args()

    if args.episode:
        process_episode(args.episode, conf_thresh=args.conf, frame_step=args.step)
    else:
        result = detect_hands_in_video(
            args.video, conf_thresh=args.conf, frame_step=args.step, input_height=args.resize,
        )
        out = args.output or args.video.with_name("hand_analysis.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"Saved to {out}")

        if args.preview:
            preview_path = out.with_name(out.stem + "_preview.mp4")
            generate_preview(args.video, result, preview_path)



if __name__ == "__main__":
    main()
