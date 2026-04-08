#!/usr/bin/env python3
"""
Privacy blur operator using EgoBlur Gen2.

Detects and blurs faces and license plates in egocentric video.

Usage:
  python blur_privacy.py --video path/to/rgb.mp4
  python blur_privacy.py --video path/to/rgb.mp4 --output blurred.mp4
  python blur_privacy.py --episode path/to/episode_dir          # reads configured input video, writes rgb_blurred.mp4
  python blur_privacy.py --video path/to/rgb.mp4 --scale 1.3    # enlarge blur region by 30%
  python blur_privacy.py --video path/to/rgb.mp4 --preview      # 480p side-by-side comparison
"""

import argparse
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from ..video_path import resolve_episode_video_path
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from video_path import resolve_episode_video_path

from ego_blur import ClassID, EgoblurDetector
from gen2.script.constants import (
    FACE_THRESHOLDS_GEN2,
    LP_THRESHOLDS_GEN2,
    RESIZE_MIN_GEN2,
    RESIZE_MAX_GEN2,
)
from gen2.script.predictor import PATCH_INSTANCES_FIELDS
from gen2.script.detectron2.export.torchscript_patch import patch_instances
from gen2.script.detectron2.utils.utils import ResizeShortestEdge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
FACE_MODEL = PROJECT_ROOT / "weights" / "ego_blur_face_gen2.jit"
LP_MODEL = PROJECT_ROOT / "weights" / "ego_blur_lp_gen2.jit"


# ── detector management ─────────────────────────────────────────────


def _get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _compute_egoblur_target(orig_h: int, orig_w: int) -> tuple[int, int]:
    """Compute the exact (H, W) that EgoBlur's internal ResizeShortestEdge would produce."""
    return ResizeShortestEdge.get_output_shape(
        orig_h, orig_w, RESIZE_MIN_GEN2, RESIZE_MAX_GEN2,
    )


def _build_detectors(
    device: str,
    face: bool = True,
    lp: bool = True,
    face_thresh: float | None = None,
    lp_thresh: float | None = None,
    nms_iou: float = 0.5,
) -> tuple[EgoblurDetector | None, EgoblurDetector | None]:
    """Initialise face and/or license-plate detectors.

    Detectors are built WITHOUT resize_aug so that the caller can
    externally resize frames to the exact EgoBlur target resolution,
    bypassing the expensive numpy-based internal resize.
    """
    face_det = None
    lp_det = None

    if face:
        ft = face_thresh if face_thresh is not None else FACE_THRESHOLDS_GEN2["camera-rgb"]
        face_det = EgoblurDetector(
            model_path=str(FACE_MODEL),
            device=device,
            detection_class=ClassID.FACE,
            score_threshold=ft,
            nms_iou_threshold=nms_iou,
            resize_aug=None,
        )
        log.info(f"Face detector loaded  (thresh={ft:.4f}, device={device})")

    if lp:
        lt = lp_thresh if lp_thresh is not None else LP_THRESHOLDS_GEN2["camera-rgb"]
        lp_det = EgoblurDetector(
            model_path=str(LP_MODEL),
            device=device,
            detection_class=ClassID.LICENSE_PLATE,
            score_threshold=lt,
            nms_iou_threshold=nms_iou,
            resize_aug=None,
        )
        log.info(f"LP detector loaded    (thresh={lt:.4f}, device={device})")

    return face_det, lp_det


# ── blur helpers ─────────────────────────────────────────────────────


def _apply_blur(image: np.ndarray, boxes: list[list[float]], scale: float = 1.0) -> np.ndarray:
    """Gaussian-blur elliptical regions for each detection box."""
    out = image.copy()
    h, w = out.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = box
        # scale around centre
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
        x1 = max(int(cx - bw / 2), 0)
        y1 = max(int(cy - bh / 2), 0)
        x2 = min(int(cx + bw / 2), w)
        y2 = min(int(cy + bh / 2), h)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]
        ksize = max(((min(roi.shape[:2]) // 2) | 1), 31)  # odd kernel >= 31
        blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)

        # elliptical mask for natural look
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        center = (roi.shape[1] // 2, roi.shape[0] // 2)
        axes = (roi.shape[1] // 2, roi.shape[0] // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        roi_out = np.where(mask[..., None] > 0, blurred_roi, roi)
        out[y1:y2, x1:x2] = roi_out

    return out


def _extract_boxes(result: list) -> list[list[float]]:
    """Safely extract boxes from EgoblurDetector.run() output.

    Returns [] when empty, or flattened list of [x1,y1,x2,y2] boxes.
    run() returns [] on no detection, or [[[x1,y1,x2,y2], ...]] otherwise.
    """
    if not result:
        return []
    return result[0]


def _detect_frame(
    bgr: np.ndarray,
    face_det: EgoblurDetector | None,
    lp_det: EgoblurDetector | None,
    device: str,
) -> list[list[float]]:
    """Run detectors on a single BGR frame, return merged box list."""
    tensor = torch.from_numpy(np.transpose(bgr, (2, 0, 1))).to(device)
    boxes: list[list[float]] = []
    if face_det:
        boxes.extend(_extract_boxes(face_det.run(tensor)))
    if lp_det:
        boxes.extend(_extract_boxes(lp_det.run(tensor)))
    return boxes


# ── video probe ─────────────────────────────────────────────────────


# Source codec → preferred encoder.
# GPU (NVENC) is tried first; if unavailable, falls back to CPU encoder.
_ENCODER_MAP_GPU = {
    "h264": "h264_nvenc",
    "hevc": "hevc_nvenc",
    "h265": "hevc_nvenc",
}
_ENCODER_MAP_CPU = {
    "h264": "libx264",
    "hevc": "libx265",
    "h265": "libx265",
    "vp9": "libvpx-vp9",
    "av1": "libsvtav1",
}

def _nvenc_available() -> bool:
    """Quick check: probe a 1-frame null encode with hevc_nvenc.

    NVENC requires minimum ~145x97 resolution; use 256x256 for the probe.
    """
    import subprocess as _sp
    try:
        r = _sp.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
             "-frames:v", "1", "-c:v", "hevc_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False

_USE_NVENC: bool | None = None  # lazy init


def _probe_video(path: str) -> dict:
    """Probe original video encoding parameters via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    br = int(video.get("bit_rate", 0))
    if br == 0:
        br = int(info.get("format", {}).get("bit_rate", 0))

    return {
        "codec": video["codec_name"],
        "pix_fmt": video.get("pix_fmt", "yuv420p"),
        "bitrate": br,
        "profile": video.get("profile", ""),
    }


def _build_ffmpeg_cmd(
    output_path: str,
    frame_w: int,
    frame_h: int,
    fps: float,
    src: dict,
) -> list[str]:
    """Build ffmpeg command that matches original video encoding config.

    Prefers NVENC GPU encoding when available; falls back to CPU encoders.
    """
    global _USE_NVENC
    if _USE_NVENC is None:
        _USE_NVENC = _nvenc_available()
        import logging as _log
        _log.getLogger(__name__).info(
            f"NVENC GPU encoding: {'enabled' if _USE_NVENC else 'unavailable, using CPU'}"
        )

    codec = src["codec"]
    if _USE_NVENC and codec in _ENCODER_MAP_GPU:
        encoder = _ENCODER_MAP_GPU[codec]
        use_gpu = True
    else:
        encoder = _ENCODER_MAP_CPU.get(codec, "libx264")
        use_gpu = False

    pix_fmt = src["pix_fmt"] or "yuv420p"
    bitrate = src["bitrate"]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", encoder,
        "-pix_fmt", pix_fmt,
    ]

    if use_gpu:
        # NVENC: use CBR with target bitrate, or high-quality VBR
        if bitrate > 0:
            cmd += ["-b:v", str(bitrate), "-maxrate:v", str(int(bitrate * 1.5)),
                    "-bufsize:v", str(bitrate * 2)]
        else:
            cmd += ["-cq", "18", "-preset", "p4"]  # NVENC quality preset
    else:
        # CPU encoders
        if bitrate > 0:
            cmd += ["-b:v", str(bitrate)]
            if encoder == "libx264":
                cmd += ["-preset", "veryfast", "-profile:v", "high"]
            elif encoder == "libx265":
                cmd += ["-preset", "veryfast", "-profile:v", "main"]
        else:
            if encoder == "libx264":
                cmd += ["-crf", "17", "-preset", "veryfast"]
            elif encoder == "libx265":
                cmd += ["-crf", "18", "-preset", "veryfast"]

    cmd.append(output_path)
    return cmd


# ── main video processing ───────────────────────────────────────────


def blur_video(
    video_path: Path,
    output_path: Path,
    face: bool = True,
    lp: bool = True,
    scale: float = 1.0,
    face_thresh: float | None = None,
    lp_thresh: float | None = None,
    resize: int | None = None,
    detectors: tuple | None = None,
) -> dict:
    """
    Process a video: detect faces/LPs and write blurred output.

    Frames are externally resized to EgoBlur's exact internal target
    resolution (computed via ResizeShortestEdge), and detectors run with
    resize_aug=None to skip the expensive numpy-based internal resize.
    Blur is applied on the original resolution frame.

    Args:
        detectors: Optional pre-loaded (face_det, lp_det) tuple to avoid
                   reloading models on every call.  Must be built with
                   resize_aug=None.

    Returns a summary dict with detection stats.
    """
    device = _get_device()
    if detectors is not None:
        face_det, lp_det = detectors
    else:
        face_det, lp_det = _build_detectors(
            device, face=face, lp=lp, face_thresh=face_thresh, lp_thresh=lp_thresh,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute EgoBlur's exact target resolution and resize externally
    det_h, det_w = _compute_egoblur_target(frame_h, frame_w)
    if det_h == frame_h and det_w == frame_w:
        needs_resize = False
        scale_back_h = scale_back_w = 1.0
    else:
        needs_resize = True
        scale_back_h = frame_h / det_h
        scale_back_w = frame_w / det_w

    src = _probe_video(str(video_path))
    ffmpeg_cmd = _build_ffmpeg_cmd(str(output_path), frame_w, frame_h, fps, src)
    log.info(f"Source encoding: {src['codec']}, pix_fmt={src['pix_fmt']}, bitrate={src['bitrate'] // 1000}kbps")
    stderr_tmp = tempfile.TemporaryFile()
    writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=stderr_tmp)

    log.info(
        f"Processing {video_path.name}: {total_frames} frames, "
        f"{frame_w}x{frame_h} @ {fps:.1f} fps, "
        f"detect at {det_w}x{det_h} (exact EgoBlur target, internal resize bypassed)"
    )

    frame_idx = 0
    total_detections = 0
    frames_with_detections = 0
    t_start = time.time()

    with torch.no_grad(), patch_instances(fields=PATCH_INSTANCES_FIELDS):
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            # resize to exact EgoBlur target via OpenCV (fast, C++ optimised)
            if needs_resize:
                det_frame = cv2.resize(bgr, (det_w, det_h))
            else:
                det_frame = bgr

            boxes = _detect_frame(det_frame, face_det, lp_det, device)

            if boxes:
                # map boxes back to original resolution
                if needs_resize:
                    boxes = [
                        [b[0] * scale_back_w, b[1] * scale_back_h,
                         b[2] * scale_back_w, b[3] * scale_back_h]
                        for b in boxes
                    ]
                bgr = _apply_blur(bgr, boxes, scale)
                total_detections += len(boxes)
                frames_with_detections += 1

            writer.stdin.write(bgr.tobytes())
            frame_idx += 1

            if frame_idx % 500 == 0:
                elapsed = time.time() - t_start
                log.info(
                    f"  frame {frame_idx}/{total_frames} "
                    f"({frame_idx / total_frames * 100:.0f}%), "
                    f"{frame_idx / elapsed:.1f} fps"
                )

    cap.release()
    writer.stdin.close()
    writer.wait()
    stderr_tmp.seek(0)
    ffmpeg_stderr = stderr_tmp.read()
    stderr_tmp.close()
    if writer.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{ffmpeg_stderr.decode()[-2000:]}")
    elapsed = time.time() - t_start

    summary = {
        "input": str(video_path),
        "output": str(output_path),
        "total_frames": frame_idx,
        "output_resolution": f"{frame_w}x{frame_h}",
        "detect_resolution": f"{det_w}x{det_h}",
        "fps": round(fps, 2),
        "elapsed_sec": round(elapsed, 2),
        "throughput_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
        "detections_total": total_detections,
        "frames_with_detections": frames_with_detections,
        "blur_face": face,
        "blur_lp": lp,
        "scale_factor": scale,
    }

    log.info(
        f"Done: {frame_idx} frames in {elapsed:.1f}s "
        f"({summary['throughput_fps']:.1f} fps), saved to {output_path}"
    )
    return summary


def generate_preview(
    original: Path,
    blurred: Path,
    output: Path,
    preview_height: int = 480,
):
    """Create a side-by-side comparison video (original | blurred)."""
    cap_orig = cv2.VideoCapture(str(original))
    cap_blur = cv2.VideoCapture(str(blurred))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = preview_height / orig_h
    pw = int(orig_w * scale)
    pw = pw if pw % 2 == 0 else pw + 1  # even width for codec

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (pw * 2, preview_height))

    while True:
        ret1, f1 = cap_orig.read()
        ret2, f2 = cap_blur.read()
        if not ret1 or not ret2:
            break
        f1 = cv2.resize(f1, (pw, preview_height))
        f2 = cv2.resize(f2, (pw, preview_height))

        # labels
        cv2.putText(f1, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(f2, "Blurred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        combined = np.hstack([f1, f2])
        writer.write(combined)

    cap_orig.release()
    cap_blur.release()
    writer.release()
    log.info(f"Preview saved to {output}")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Privacy blur operator (EgoBlur Gen2)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to input video")
    group.add_argument("--episode", type=Path, help="Episode directory (reads configured input video)")
    parser.add_argument("--output", type=Path, default=None, help="Output video path (--video mode)")
    parser.add_argument("--scale", type=float, default=1.0, help="Blur region scale factor (default: 1.0)")
    parser.add_argument("--face-thresh", type=float, default=None, help="Face detection threshold")
    parser.add_argument("--lp-thresh", type=float, default=None, help="LP detection threshold")
    parser.add_argument("--preview", action="store_true", help="Generate side-by-side preview")
    args = parser.parse_args()

    if args.episode:
        video_path = resolve_episode_video_path(args.episode)
        output_path = args.episode / "rgb_blurred.mp4"
    else:
        video_path = args.video
        output_path = args.output or video_path.with_stem(video_path.stem + "_blurred")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    summary = blur_video(
        video_path, output_path, scale=args.scale,
        face_thresh=args.face_thresh, lp_thresh=args.lp_thresh,
    )

    if args.preview:
        preview_path = output_path.with_stem(output_path.stem + "_preview")
        generate_preview(video_path, output_path, preview_path)

    log.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()
