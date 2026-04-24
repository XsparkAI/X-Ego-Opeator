"""Unified hand analysis operator — dispatches to YOLO or VLM backend."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..operator_base import OperatorResult
from ..video_path import resolve_episode_video_path

log = logging.getLogger(__name__)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COUNT_LABELS = {0: "no_hands", 1: "one_hand", 2: "two_hands"}
COUNT_COLORS_RGB = {0: (220, 50, 50), 1: (30, 180, 60), 2: (50, 100, 220)}


def _episode_root(work_dir: Path) -> Path:
    if work_dir.parent.name == "segments":
        return work_dir.parent.parent
    return work_dir


def _seg_prefix(work_dir: Path) -> str:
    if work_dir.parent.name == "segments":
        return work_dir.name + "_"
    return ""


def _build_hand_summary_lines(summary: dict) -> list[str]:
    lines: list[str] = []

    any_hand_ratio = summary.get("any_hand_ratio")
    if any_hand_ratio is not None:
        left_ratio = float(summary.get("left_hand_ratio", 0) or 0)
        right_ratio = float(summary.get("right_hand_ratio", 0) or 0)
        both_hands_ratio = float(summary.get("both_hands_ratio", 0) or 0)
        lines.append(
            f"Hands visible in {float(any_hand_ratio) * 100:.0f}% of sampled frames "
            f"(left {left_ratio * 100:.0f}%, right {right_ratio * 100:.0f}%, both {both_hands_ratio * 100:.0f}%)"
        )
        return lines

    no_hands_ratio = float(summary.get("ego_0_hands_ratio", 0) or 0)
    one_hand_ratio = float(summary.get("ego_1_hand_ratio", 0) or 0)
    two_hands_ratio = float(summary.get("ego_2_hands_ratio", 0) or 0)
    hands_visible_ratio = max(0.0, 1.0 - no_hands_ratio)
    lines.append(
        f"Ego hands visible in {hands_visible_ratio * 100:.0f}% of sampled frames "
        f"(1 hand {one_hand_ratio * 100:.0f}%, 2 hands {two_hands_ratio * 100:.0f}%)"
    )

    active_ratio = float(summary.get("active_manipulation_ratio", 0) or 0)
    if active_ratio > 0:
        lines.append(
            f"Active manipulation detected in {active_ratio * 100:.0f}% of sampled frames"
        )
    single_operator_ratio = float(summary.get("single_person_operation_ratio", 0) or 0)
    lines.append(
        f"Single-person operation in {single_operator_ratio * 100:.0f}% of sampled frames"
    )
    return lines


def _annotate_hand_frame(frame_bgr: np.ndarray, frame_result: dict, fps: float) -> np.ndarray:
    """Draw hand count annotation banner on a frame."""
    cnt = frame_result.get("ego_hand_count", -1)
    active_manipulation = frame_result.get("active_manipulation")
    single_person_operation = frame_result.get("single_person_operation")
    success = frame_result.get("success", False)
    frame_idx = frame_result.get("global_frame", 0)
    time_sec = frame_result.get("time_sec", frame_idx / max(fps, 1))

    h, w = frame_bgr.shape[:2]
    scale = max(h / 960, 0.5)
    font_size = max(int(32 * scale), 18)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
        font_s = ImageFont.truetype(FONT_PATH, max(int(22 * scale), 14))
    except Exception:
        font = font_s = ImageFont.load_default()

    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # Top banner: hand count
    if not success or cnt < 0:
        label = "Detection failed"
        bg = (130, 130, 130)
    else:
        label = f"Hands: {cnt} ({COUNT_LABELS.get(cnt, str(cnt))})"
        bg = COUNT_COLORS_RGB.get(cnt, (130, 130, 130))

    pad_x, pad_y = 18, 10
    bbox = font.getbbox(label)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bw, bh = tw + pad_x * 2, th + pad_y * 2
    overlay = Image.new("RGBA", (bw, bh), (*bg, 180))
    pil.paste(
        Image.alpha_composite(Image.new("RGBA", overlay.size, (0, 0, 0, 0)), overlay).convert("RGB"),
        (0, 0), overlay,
    )
    draw = ImageDraw.Draw(pil)
    draw.text((pad_x, pad_y), label, font=font, fill=(255, 255, 255))

    # Bottom info bar: frame idx + time + VLM labels
    if active_manipulation is True:
        manipulation_text = "active=yes"
    elif active_manipulation is False:
        manipulation_text = "active=no"
    else:
        manipulation_text = "active=unknown"
    if single_person_operation is True:
        solo_text = "solo=yes"
    elif single_person_operation is False:
        solo_text = "solo=no"
    else:
        solo_text = "solo=unknown"
    info = f"#{frame_idx}  {time_sec:.2f}s  {manipulation_text}  {solo_text}"
    info_bbox = font_s.getbbox(info)
    iw, ih = info_bbox[2] - info_bbox[0], info_bbox[3] - info_bbox[1]
    ibw, ibh = iw + 16, ih + 12
    iy = h - ibh
    info_overlay = Image.new("RGBA", (ibw, ibh), (0, 0, 0, 150))
    pil.paste(
        Image.alpha_composite(Image.new("RGBA", info_overlay.size, (0, 0, 0, 0)), info_overlay).convert("RGB"),
        (0, iy), info_overlay,
    )
    draw = ImageDraw.Draw(pil)
    draw.text((8, iy + 6), info, font=font_s, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _dump_hand_samples(
    video_path: Path,
    frame_results: list[dict],
    episode_root: Path,
    seg_prefix: str,
) -> None:
    """Extract VLM-sampled frames, annotate with hand count, save to hand_samples/."""
    from ..video_utils import get_manual_rotation, apply_rotation

    out_dir = episode_root / "hand_samples"
    out_dir.mkdir(exist_ok=True)

    rotation = get_manual_rotation(str(video_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for fr in frame_results:
        gf = fr.get("global_frame", 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, gf)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = apply_rotation(frame, rotation)
        h, w = frame.shape[:2]
        scale = min(1.0, 1280 / w)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        annotated = _annotate_hand_frame(frame, fr, fps)
        cnt = fr.get("ego_hand_count", -1)
        cnt_label = COUNT_LABELS.get(cnt, "fail")
        fname = f"{seg_prefix}f{gf:06d}_{fr.get('time_sec', 0):.2f}s_{cnt_label}.jpg"
        cv2.imwrite(str(out_dir / fname), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    cap.release()


def _compact_vlm_audit_output(result: dict) -> dict:
    summary = result.get("summary", {})
    frame_results = result.get("frame_results", [])
    return {
        "summary": {
            "total_frames_sampled": summary.get("total_frames_sampled", 0),
            "avg_ego_hand_count": summary.get("avg_ego_hand_count", 0),
            "ego_0_hands_ratio": summary.get("ego_0_hands_ratio", 0),
            "ego_1_hand_ratio": summary.get("ego_1_hand_ratio", 0),
            "ego_2_hands_ratio": summary.get("ego_2_hands_ratio", 0),
            "active_manipulation_ratio": summary.get("active_manipulation_ratio", 0),
            "single_person_operation_ratio": summary.get("single_person_operation_ratio", 0),
        },
        "sampled_hand_counts": [
            {
                "global_frame": item.get("global_frame", 0),
                "time_sec": item.get("time_sec", 0),
                "ego_hand_count": item.get("ego_hand_count", 0),
                "active_manipulation": item.get("active_manipulation"),
                "single_person_operation": item.get("single_person_operation"),
            }
            for item in frame_results
        ],
    }


@dataclass
class HandAnalysisConfig:
    method: str = "yolo"           # "yolo" | "vlm"
    conf_thresh: float = 0.3       # YOLO detection confidence threshold
    input_height: int = 720        # resize height before detection
    yolo_frame_step: int = 1       # process every N-th frame in YOLO mode
    vlm_sample_frame_step: int = 120  # shared sampling step for VLM hand/person/activity labels
    max_workers: int = 4           # kept only for compatibility with older VLM configs
    batch_enabled: bool = True     # whether VLM mode submits one batch job or sends per-frame requests
    save_hand_samples: bool = False


class HandAnalysisOperator:
    name = "hand_analysis"

    def __init__(self, config: HandAnalysisConfig | None = None):
        self.config = config or HandAnalysisConfig()
        self._yolo_model = None

        method = self.config.method.lower()
        if method not in ("yolo", "vlm"):
            raise ValueError(
                f"hand_analysis.method must be 'yolo' or 'vlm', got {method!r}"
            )

    # ── YOLO backend ───────────────────────────────────────────────────

    def _ensure_yolo_model(self):
        if self._yolo_model is None:
            from .detect_hand_in_frame import load_model
            self._yolo_model = load_model()
            log.info("YOLO hand model loaded (will reuse across episodes)")

    def _run_yolo(self, episode_dir: Path) -> OperatorResult:
        from .detect_hand_in_frame import detect_hands_in_video

        video_path = resolve_episode_video_path(episode_dir)
        output_path = episode_dir / "hand_analysis.json"

        self._ensure_yolo_model()

        result = detect_hands_in_video(
            video_path,
            conf_thresh=self.config.conf_thresh,
            frame_step=self.config.yolo_frame_step,
            input_height=self.config.input_height,
            model=self._yolo_model,
        )
        output_path.write_text(
            json.dumps(result["summary"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return OperatorResult(
            status="ok", operator=self.name,
            output_files=[str(output_path)],
            metrics=result["summary"],
            errors=_build_hand_summary_lines(result["summary"]),
        )

    # ── VLM backend ────────────────────────────────────────────────────

    def _run_vlm(self, episode_dir: Path) -> OperatorResult:
        from .vlm_hand_audit import audit_video

        video_path = resolve_episode_video_path(episode_dir)
        output_path = episode_dir / "hand_analysis.json"

        result = audit_video(
            video_path,
            frame_step=self.config.vlm_sample_frame_step,
            max_workers=self.config.max_workers,
            batch_enabled=self.config.batch_enabled,
        )

        output_path.write_text(
            json.dumps(_compact_vlm_audit_output(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary = result.get("summary", {})
        warnings = _build_hand_summary_lines(summary)

        # Export annotated sampled frames into the episode-level hand_samples/ directory.
        ep_root = _episode_root(episode_dir)
        seg_pfx = _seg_prefix(episode_dir)
        if self.config.save_hand_samples:
            try:
                _dump_hand_samples(video_path, result.get("frame_results", []), ep_root, seg_pfx)
            except Exception as e:
                log.warning(f"hand sample dump failed (non-fatal): {e}")

        return OperatorResult(
            status="ok", operator=self.name,
            output_files=[str(output_path)],
            metrics=summary,
            errors=warnings,
        )

    # ── Dispatch ───────────────────────────────────────────────────────

    def run(self, episode_dir: Path, **kwargs) -> OperatorResult:
        video_path = resolve_episode_video_path(episode_dir)
        if not video_path.exists():
            return OperatorResult(
                status="error", operator=self.name,
                errors=[f"Video not found: {video_path}"],
            )

        method = self.config.method.lower()
        try:
            if method == "yolo":
                return self._run_yolo(episode_dir)
            else:
                return self._run_vlm(episode_dir)
        except Exception as e:
            log.exception(f"hand_analysis ({method}) failed")
            return OperatorResult(
                status="error", operator=self.name,
                errors=[str(e)],
            )
