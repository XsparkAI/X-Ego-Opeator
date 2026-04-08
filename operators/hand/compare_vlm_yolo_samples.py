#!/usr/bin/env python3
"""Compare VLM hand-sample labels against YOLO detections on the same frames.

This script is intentionally standalone and does not modify pipeline behavior.

Typical usage:
  python -m operators.hand.compare_vlm_yolo_samples \
    --episode /path/to/episode_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from .detect_hand_in_frame import WEIGHTS_PATH, load_model
from ..video_utils import apply_rotation, get_manual_rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LABEL_TO_COUNT = {
    "无手": 0,
    "单手": 1,
    "双手": 2,
    "fail": -1,
}

FILENAME_RE = re.compile(
    r"^(?P<segment>seg_\d{3})_f(?P<frame>\d+)_(?P<time>[0-9.]+)s_(?P<label>.+)\.jpg$"
)


def parse_sample_filename(path: Path) -> dict | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    label = match.group("label")
    return {
        "sample_path": str(path),
        "sample_name": path.name,
        "segment": match.group("segment"),
        "frame_idx": int(match.group("frame")),
        "time_sec": float(match.group("time")),
        "vlm_label": label,
        "vlm_count": LABEL_TO_COUNT.get(label, -1),
    }


def iter_samples(hand_samples_dir: Path) -> list[dict]:
    samples = []
    for path in sorted(hand_samples_dir.glob("*.jpg")):
        parsed = parse_sample_filename(path)
        if parsed is None:
            log.warning(f"Skipping unrecognized sample name: {path.name}")
            continue
        samples.append(parsed)
    return samples


def extract_frame(video_path: Path, frame_idx: int):
    rotation = get_manual_rotation(str(video_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return apply_rotation(frame, rotation)


def detect_yolo_on_frame(
    frame_bgr,
    model,
    conf_thresh: float,
    input_height: int,
) -> dict:
    orig_h, orig_w = frame_bgr.shape[:2]
    if input_height and input_height < orig_h:
        scale_down = input_height / orig_h
        det_w = int(orig_w * scale_down)
        det_h = input_height
        scale_back = orig_h / input_height
        det_frame = cv2.resize(frame_bgr, (det_w, det_h))
    else:
        det_w, det_h = orig_w, orig_h
        scale_back = 1.0
        det_frame = frame_bgr

    with torch.no_grad():
        results = model.predict(det_frame, conf=conf_thresh, verbose=False)
    boxes_result = results[0].boxes

    left_best = None
    right_best = None
    left_best_conf = -1.0
    right_best_conf = -1.0

    if boxes_result is not None and len(boxes_result) > 0:
        xyxy = boxes_result.xyxy.cpu().numpy()
        confs = boxes_result.conf.cpu().numpy()
        classes = boxes_result.cls.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [v * scale_back for v in xyxy[i].tolist()]
            conf = float(confs[i])
            hand_cls = int(classes[i])
            det = {
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "conf": round(conf, 4),
            }
            if hand_cls == 0 and conf > left_best_conf:
                left_best = det
                left_best_conf = conf
            elif hand_cls != 0 and conf > right_best_conf:
                right_best = det
                right_best_conf = conf

    yolo_count = int(left_best is not None) + int(right_best is not None)
    return {
        "yolo_count": yolo_count,
        "has_left": left_best is not None,
        "has_right": right_best is not None,
        "left_hand": left_best,
        "right_hand": right_best,
        "detection_resolution": f"{det_w}x{det_h}",
    }


def draw_overlay(frame_bgr, comparison: dict):
    out = frame_bgr.copy()
    for side, color in [("left_hand", (255, 144, 30)), ("right_hand", (0, 200, 0))]:
        det = comparison.get(side)
        if not det:
            continue
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{side[0].upper()} {det['conf']:.2f}"
        cv2.putText(out, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    text = f"VLM {comparison['vlm_count']}  YOLO {comparison['yolo_count']}"
    status = "MATCH" if comparison["match"] else "MISMATCH"
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(out, status, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 220, 0) if comparison["match"] else (0, 0, 255), 2)
    return out


def _fit_with_padding(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image to fit inside target box with black padding."""
    h, w = image.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


def generate_preview_sheets(comparisons: list[dict], overlay_dir: Path, preview_dir: Path) -> list[str]:
    """Group overlay images into 3x3 preview sheets with labels below each tile."""
    preview_dir.mkdir(exist_ok=True)

    cols = 3
    rows = 3
    page_size = cols * rows
    tile_w = 420
    tile_h = 300
    label_h = 70
    cell_h = tile_h + label_h
    canvas_w = cols * tile_w
    canvas_h = rows * cell_h
    output_paths: list[str] = []

    for page_idx in range(0, len(comparisons), page_size):
        batch = comparisons[page_idx:page_idx + page_size]
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for i, comparison in enumerate(batch):
            r = i // cols
            c = i % cols
            x0 = c * tile_w
            y0 = r * cell_h

            image_path = overlay_dir / comparison["sample_name"]
            img = cv2.imread(str(image_path))
            if img is None:
                continue

            fitted = _fit_with_padding(img, tile_w, tile_h)
            canvas[y0:y0 + tile_h, x0:x0 + tile_w] = fitted

            line1 = f"{comparison['segment']}  f{comparison['frame_idx']}"
            line2 = f"VLM {comparison['vlm_count']} | YOLO {comparison['yolo_count']}"
            line3 = "MATCH" if comparison["match"] else "MISMATCH"
            color = (0, 220, 0) if comparison["match"] else (0, 0, 255)

            cv2.putText(canvas, line1, (x0 + 12, y0 + tile_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, line2, (x0 + 12, y0 + tile_h + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, line3, (x0 + 12, y0 + tile_h + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        out_path = preview_dir / f"preview_{page_idx // page_size:03d}.jpg"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        output_paths.append(str(out_path))

    return output_paths


def compare_episode(
    episode_dir: Path,
    conf_thresh: float,
    input_height: int,
    save_overlays: bool,
) -> dict:
    hand_samples_dir = episode_dir / "hand_samples"
    segments_dir = episode_dir / "segments"
    if not hand_samples_dir.exists():
        raise FileNotFoundError(f"hand_samples not found: {hand_samples_dir}")
    if not segments_dir.exists():
        raise FileNotFoundError(f"segments not found: {segments_dir}")

    samples = iter_samples(hand_samples_dir)
    if not samples:
        raise RuntimeError(f"No valid hand sample images found in {hand_samples_dir}")

    model = load_model(WEIGHTS_PATH)
    overlay_dir = episode_dir / "hand_samples_yolo_compare"
    preview_dir = episode_dir / "hand_samples_yolo_preview"
    if save_overlays:
        overlay_dir.mkdir(exist_ok=True)

    comparisons = []
    per_segment: dict[str, dict[str, int]] = {}

    for sample in samples:
        segment = sample["segment"]
        video_path = segments_dir / segment / "rgb.mp4"
        frame = extract_frame(video_path, sample["frame_idx"])
        yolo = detect_yolo_on_frame(frame, model, conf_thresh=conf_thresh, input_height=input_height)

        comparison = {
            **sample,
            "segment_video": str(video_path),
            **yolo,
            "match": sample["vlm_count"] == yolo["yolo_count"],
        }
        comparisons.append(comparison)

        stats = per_segment.setdefault(segment, {"total": 0, "match": 0, "mismatch": 0})
        stats["total"] += 1
        if comparison["match"]:
            stats["match"] += 1
        else:
            stats["mismatch"] += 1

        if save_overlays:
            overlay = draw_overlay(frame, comparison)
            out_path = overlay_dir / sample["sample_name"]
            cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

    total = len(comparisons)
    matches = sum(1 for c in comparisons if c["match"])
    mismatches = total - matches
    preview_paths = generate_preview_sheets(comparisons, overlay_dir, preview_dir) if save_overlays else []

    return {
        "episode_dir": str(episode_dir),
        "hand_samples_dir": str(hand_samples_dir),
        "segments_dir": str(segments_dir),
        "weights": str(WEIGHTS_PATH),
        "config": {
            "conf_thresh": conf_thresh,
            "input_height": input_height,
            "save_overlays": save_overlays,
        },
        "summary": {
            "total_samples": total,
            "matches": matches,
            "mismatches": mismatches,
            "match_ratio": round(matches / total, 4) if total else 0.0,
        },
        "preview_sheets": preview_paths,
        "per_segment": per_segment,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare VLM hand samples with YOLO on the same frames")
    parser.add_argument("--episode", type=Path, required=True, help="Episode directory containing hand_samples/ and segments/")
    parser.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold")
    parser.add_argument("--input-height", type=int, default=720, help="YOLO input resize height")
    parser.add_argument("--no-overlays", action="store_true", help="Do not save overlay images")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    result = compare_episode(
        args.episode.resolve(),
        conf_thresh=args.conf,
        input_height=args.input_height,
        save_overlays=not args.no_overlays,
    )

    output_path = args.output or (args.episode / "hand_yolo_compare.json")
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = result["summary"]
    log.info(
        f"Compared {summary['total_samples']} samples: "
        f"match={summary['matches']}, mismatch={summary['mismatches']}, "
        f"ratio={summary['match_ratio']:.1%}"
    )
    log.info(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
