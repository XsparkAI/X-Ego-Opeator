"""Adapter wrapping process_video() into the Operator protocol."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from ..operator_base import OperatorResult

log = logging.getLogger(__name__)


def _episode_root(work_dir: Path) -> Path:
    """Return episode-level dir even when work_dir is a segment subdir."""
    if work_dir.parent.name == "segments":
        return work_dir.parent.parent
    return work_dir


def _seg_prefix(work_dir: Path) -> str:
    """Return filename prefix like 'seg_000_' or '' for direct episodes."""
    if work_dir.parent.name == "segments":
        return work_dir.name + "_"
    return ""


def _extract_quality_frames(
    video_path: Path,
    result: dict,
    episode_root: Path,
    seg_prefix: str,
) -> None:
    """Extract flagged frames into episode-level quality dirs.

    Directory layout (4 dirs total, same level as hand_samples/):
      quality_blurry/     — blurry frames
      quality_jitter/     — one representative frame per jitter event
      quality_exposure/   — overexposed (oe_) + underexposed (ue_) frames
      hand_samples/       — (written by hand operator)
    """
    from ..video_utils import get_manual_rotation, apply_rotation

    blurry = result.get("quality", {}).get("blurry_frames", [])
    jitter = result.get("stability", {}).get("jitter_frames", [])
    overexposed = result.get("exposure", {}).get("overexposed_frames", [])
    underexposed = result.get("exposure", {}).get("underexposed_frames", [])

    # frame_idx → (subdir, filename_label)
    frame_map: dict[int, tuple[str, str]] = {}

    for f in blurry:
        frame_map[f["frame_idx"]] = ("quality_blurry", f"lap{f['laplacian_var']:.0f}")

    # Jitter: take the middle frame of each jitter interval
    for f in jitter:
        mid = (f["frame_idx_from"] + f["frame_idx_to"]) // 2
        trans = f.get("translation", 0)
        label = f"trans{trans:.0f}px"
        frame_map.setdefault(mid, ("quality_jitter", label))  # don't override blurry

    for f in overexposed:
        frame_map[f["frame_idx"]] = ("quality_exposure", f"oe_{f['overexposure_ratio']:.0%}")
    for f in underexposed:
        frame_map[f["frame_idx"]] = ("quality_exposure", f"ue_{f['underexposure_ratio']:.0%}")

    if not frame_map:
        return

    needed_dirs = {subdir for subdir, _ in frame_map.values()}
    for d in needed_dirs:
        (episode_root / d).mkdir(exist_ok=True)

    rotation = get_manual_rotation(str(video_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for frame_idx, (subdir, label) in sorted(frame_map.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = apply_rotation(frame, rotation)
        h, w = frame.shape[:2]
        scale = min(1.0, 1280 / w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        time_sec = round(frame_idx / fps, 2)
        fname = f"{seg_prefix}f{frame_idx:06d}_{time_sec}s_{label}.jpg"
        cv2.imwrite(str(episode_root / subdir / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    cap.release()


def _update_episode_quality_report(
    episode_root: Path,
    seg_name: str,
    result: dict,
) -> None:
    """Write/update quality_report.json at episode root.

    Keyed by segment name (or 'episode' for non-fan-out runs).
    Sequential segments write safely; parallel episodes write to different roots.
    """
    report_path = episode_root / "quality_report.json"
    try:
        existing = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    except Exception:
        existing = {}
    existing[seg_name] = result
    report_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class VideoQualityConfig:
    sample_fps: float | None = None
    check_quality: bool = True
    check_stability: bool = True
    check_exposure: bool = True


class VideoQualityOperator:
    name = "video_quality"

    def __init__(self, config: VideoQualityConfig | None = None):
        self.config = config or VideoQualityConfig()

    def run(self, episode_dir: Path, **kwargs) -> OperatorResult:
        from .assess import process_video

        video_path = episode_dir / "rgb.mp4"
        output_path = episode_dir / "quality_report.json"

        if not video_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"Video not found: {video_path}"],
            )

        try:
            result = process_video(
                str(video_path),
                sample_fps=self.config.sample_fps,
                check_quality=self.config.check_quality,
                check_stability=self.config.check_stability,
                check_exposure=self.config.check_exposure,
            )
            output_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            quality = result.get("quality", {})
            stability = result.get("stability", {})
            exposure = result.get("exposure", {})
            passed = result.get("pass", True)

            errors = []

            blurry = quality.get("blurry_frames", [])
            if blurry:
                frames_str = ", ".join(
                    f"#{f['frame_idx']}({f['time_sec']}s, lap={f['laplacian_var']:.1f})"
                    for f in blurry
                )
                errors.append(f"模糊帧 [{len(blurry)}帧]: {frames_str}")

            jitter = stability.get("jitter_frames", [])
            if jitter:
                frames_str = ", ".join(
                    f"#{f['frame_idx_from']}-{f['frame_idx_to']}({f['time_sec']}s, "
                    f"trans={f['translation']:.1f}px, rot={f['rotation']:.4f}rad)"
                    for f in jitter
                )
                errors.append(f"剧烈抖动 [{len(jitter)}处]: {frames_str}")

            overexposed = exposure.get("overexposed_frames", [])
            underexposed = exposure.get("underexposed_frames", [])
            if overexposed:
                frames_str = ", ".join(
                    f"#{f['frame_idx']}({f['time_sec']}s, {f['overexposure_ratio']:.1%})"
                    for f in overexposed
                )
                errors.append(f"过曝帧 [{len(overexposed)}帧]: {frames_str}")
            if underexposed:
                frames_str = ", ".join(
                    f"#{f['frame_idx']}({f['time_sec']}s, {f['underexposure_ratio']:.1%})"
                    for f in underexposed
                )
                errors.append(f"欠曝帧 [{len(underexposed)}帧]: {frames_str}")

            ep_root = _episode_root(episode_dir)
            seg_pfx = _seg_prefix(episode_dir)
            # "episode" key for non-fan-out, segment name for fan-out
            seg_name = episode_dir.name if episode_dir != ep_root else "episode"

            try:
                _extract_quality_frames(video_path, result, ep_root, seg_pfx)
            except Exception as e:
                log.warning(f"quality frame extraction failed (non-fatal): {e}")

            try:
                _update_episode_quality_report(ep_root, seg_name, result)
            except Exception as e:
                log.warning(f"episode quality report update failed (non-fatal): {e}")

            return OperatorResult(
                status="ok" if passed else "fail",
                operator=self.name,
                output_files=[str(output_path)],
                metrics={
                    "mean_laplacian": quality.get("mean_laplacian"),
                    "translation_std": stability.get("translation_std"),
                    "blur_ratio": quality.get("blur_ratio"),
                    "quality_pass": quality.get("pass"),
                    "stability_pass": stability.get("pass"),
                    "exposure_pass": exposure.get("pass"),
                    "blurry_frame_count": len(blurry),
                    "jitter_count": len(jitter),
                    "overexposed_frame_count": len(overexposed),
                    "underexposed_frame_count": len(underexposed),
                },
                errors=errors,
            )
        except Exception as e:
            log.exception("video_quality failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )
