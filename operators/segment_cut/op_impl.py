"""Cut caption segments into video clips.

The standalone adapter still exists for local pipeline compatibility, but the
main integration point is the caption Docker optional segment-cut step. It
reads the unified ``caption.json`` schema and writes per-segment videos plus a
small manifest.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..operator_base import OperatorResult
from ..video_path import resolve_episode_video_path
from ..vlm_limit import cpu_task_slot

log = logging.getLogger(__name__)

_MIN_FILE_BYTES = 0


@dataclass
class SegmentCutConfig:
    granularity: str = "task"
    caption_filename: str = "caption.json"
    output_dir_name: str = "segments"


class SegmentCutOperator:
    name = "segment_cut"

    def __init__(self, config: SegmentCutConfig | None = None):
        self.config = config or SegmentCutConfig()

    def run(self, episode_dir: Path, **kwargs: Any) -> OperatorResult:
        with cpu_task_slot():
            caption_path = Path(kwargs.get("caption_path") or episode_dir / self.config.caption_filename)
            video_path = resolve_episode_video_path(episode_dir)
            output_dir = Path(kwargs.get("output_dir") or episode_dir / self.config.output_dir_name)

            if not caption_path.exists():
                return OperatorResult(
                    status="error",
                    operator=self.name,
                    errors=[f"{caption_path.name} not found in {episode_dir}"],
                )
            if not video_path.exists():
                return OperatorResult(
                    status="error",
                    operator=self.name,
                    errors=[f"Input video not found in {episode_dir}"],
                )

            return cut_caption_segments(
                caption_path=caption_path,
                video_path=video_path,
                output_dir=output_dir,
                granularity=self.config.granularity,
                operator=self.name,
            )

    @staticmethod
    def _write_info(seg_dir: Path, seg: dict, fps: float) -> None:
        fi = seg["frame_interval"]
        info = {
            "start_frame": int(fi[0]),
            "end_frame": int(fi[1]),
            "fps": fps,
            "duration_sec": round((int(fi[1]) - int(fi[0])) / fps, 4),
            "instruction": seg.get("instruction", ""),
        }
        if "task_instruction" in seg:
            info["task_instruction"] = seg["task_instruction"]
        if "action_name" in seg:
            info["action_name"] = seg["action_name"]
        (seg_dir / "segment_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _manifest_entry(seg_dir: Path, seg_video: Path, seg: dict, fps: float) -> dict:
        fi = seg["frame_interval"]
        start_frame = int(fi[0])
        end_frame = int(fi[1])
        return {
            "segmentDir": str(seg_dir),
            "videoPath": str(seg_video),
            "infoPath": str(seg_dir / "segment_info.json"),
            "frameInterval": [start_frame, end_frame],
            "startSec": round(start_frame / fps, 6),
            "durationSec": round((end_frame - start_frame) / fps, 6),
            "instruction": seg.get("instruction", ""),
            "taskInstruction": seg.get("task_instruction", ""),
            "actionName": seg.get("action_name", ""),
        }

    @staticmethod
    def _failure_entry(
        *,
        segment_id: str,
        seg: dict,
        reason: str,
        message: str,
        fps: float | None = None,
    ) -> dict:
        fi = seg.get("frame_interval", [])
        entry = {
            "segmentId": segment_id,
            "reason": reason,
            "message": message,
            "frameInterval": fi,
            "instruction": seg.get("instruction", ""),
            "taskInstruction": seg.get("task_instruction", ""),
            "actionName": seg.get("action_name", ""),
        }
        if fps and len(fi) >= 2:
            start_frame = int(fi[0])
            end_frame = int(fi[1])
            entry["startSec"] = round(start_frame / fps, 6)
            entry["durationSec"] = round((end_frame - start_frame) / fps, 6)
        return entry

    @staticmethod
    def _tasks_to_segments(tasks: list[dict]) -> list[dict]:
        segments = []
        for task in tasks:
            fi = task.get("frame_interval", [])
            if len(fi) < 2:
                continue
            task_caption = task.get("caption") or task.get("instruction") or ""
            segments.append(
                {
                    "frame_interval": [int(fi[0]), int(fi[1])],
                    "instruction": task_caption,
                    "task_instruction": task_caption,
                }
            )
        return segments

    @staticmethod
    def _actions_to_segments(tasks: list[dict]) -> list[dict]:
        segments = []
        for task in tasks:
            task_instruction = task.get("caption") or task.get("instruction", "")
            for action in task.get("atomic_actions", []):
                fi = action.get("frame_interval", [])
                if len(fi) < 2:
                    continue
                action_name = action.get("caption", "")
                segments.append(
                    {
                        "frame_interval": [int(fi[0]), int(fi[1])],
                        "instruction": action_name or task_instruction,
                        "task_instruction": task_instruction,
                        "action_name": action_name,
                    }
                )
        return segments

    @staticmethod
    def _select_segments(caption: dict, granularity: str) -> list[dict]:
        tasks = caption.get("tasks", [])
        if not tasks:
            return []
        if granularity == "task":
            return SegmentCutOperator._tasks_to_segments(tasks)
        if granularity == "atomic_action":
            if not any(task.get("atomic_actions") for task in tasks):
                raise ValueError(
                    "segment_cut.granularity=atomic_action requires caption.json tasks with atomic_actions"
                )
            return SegmentCutOperator._actions_to_segments(tasks)
        raise ValueError(
            f"segment_cut.granularity must be 'atomic_action' or 'task', got {granularity!r}"
        )

    @staticmethod
    def _probe_fps(video_path: Path, probe: dict | None = None) -> float | None:
        try:
            if probe is None:
                probe = SegmentCutOperator._probe_media(video_path)
            video_stream = SegmentCutOperator._first_stream(probe, "video")
            for key in ("avg_frame_rate", "r_frame_rate"):
                fps = SegmentCutOperator._parse_rate(str(video_stream.get(key, "")))
                if fps:
                    return fps
        except Exception:
            return None
        return None

    @staticmethod
    def _probe_media(video_path: Path) -> dict:
        try:
            out = subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    str(video_path),
                ],
                text=True,
                timeout=10,
            )
            return json.loads(out)
        except Exception as e:
            log.warning(f"ffprobe failed for {video_path}: {e}")
            return {}

    @staticmethod
    def _first_stream(probe: dict, codec_type: str) -> dict:
        for stream in probe.get("streams", []):
            if stream.get("codec_type") == codec_type:
                return stream
        return {}

    @staticmethod
    def _parse_rate(value: str) -> float | None:
        if not value or value == "0/0":
            return None
        try:
            if "/" in value:
                num, den = value.split("/", 1)
                den_f = float(den)
                return float(num) / den_f if den_f else None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_ffmpeg_cut_cmd(
        *,
        video_path: Path,
        output_path: Path,
        start_sec: float,
        duration_sec: float,
        probe: dict,
    ) -> list[str]:
        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-ss", f"{start_sec:.6f}",
            "-i", str(video_path),
            "-t", f"{duration_sec:.6f}",
            "-map", "0:v:0",
            "-map", "0:a?",
        ]
        cmd.extend(SegmentCutOperator._video_encode_args(probe))
        cmd.extend(SegmentCutOperator._audio_encode_args(probe))
        cmd.append(str(output_path))
        return cmd

    @staticmethod
    def _video_encode_args(probe: dict) -> list[str]:
        stream = SegmentCutOperator._first_stream(probe, "video")
        codec = str(stream.get("codec_name") or "").lower()
        encoder_by_codec = {
            "h264": "libx264",
            "hevc": "libx265",
            "h265": "libx265",
            "mpeg4": "mpeg4",
            "vp8": "libvpx",
            "vp9": "libvpx-vp9",
        }
        args = ["-c:v", encoder_by_codec.get(codec, "libx264")]

        if codec in {"h264", "hevc", "h265"}:
            args.extend(["-preset", "veryfast"])
        if codec in {"hevc", "h265"}:
            args.extend(["-tag:v", "hvc1"])

        bit_rate = str(stream.get("bit_rate") or "").strip()
        if bit_rate.isdigit() and int(bit_rate) > 0:
            args.extend(["-b:v", bit_rate])
        elif codec in {"h264", "hevc", "h265"}:
            args.extend(["-crf", "18"])

        pix_fmt = str(stream.get("pix_fmt") or "").strip()
        if pix_fmt:
            args.extend(["-pix_fmt", pix_fmt])

        fps = SegmentCutOperator._parse_rate(
            str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")
        )
        if fps:
            args.extend(["-r", f"{fps:.6f}"])

        return args

    @staticmethod
    def _build_ffmpeg_copy_cmd(
        *,
        video_path: Path,
        output_path: Path,
        start_sec: float,
        duration_sec: float,
        probe: dict,
    ) -> list[str]:
        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-ss", f"{start_sec:.6f}",
            "-i", str(video_path),
            "-t", f"{duration_sec:.6f}",
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
        ]
        video_stream = SegmentCutOperator._first_stream(probe, "video")
        codec = str(video_stream.get("codec_name") or "").lower()
        if codec in {"hevc", "h265"}:
            cmd.extend(["-tag:v", "hvc1"])
        cmd.append(str(output_path))
        return cmd

    @staticmethod
    def _audio_encode_args(probe: dict) -> list[str]:
        stream = SegmentCutOperator._first_stream(probe, "audio")
        if not stream:
            return []

        codec = str(stream.get("codec_name") or "").lower()
        encoder_by_codec = {
            "aac": "aac",
            "mp3": "libmp3lame",
            "opus": "libopus",
            "vorbis": "libvorbis",
            # The segment output is rgb.mp4. Lossless/PCM audio codecs are not
            # broadly MP4-compatible, so keep the audio stream playable by
            # transcoding those cases to AAC instead of emitting invalid MP4.
            "flac": "aac",
            "pcm_s16le": "aac",
        }
        args = ["-c:a", encoder_by_codec.get(codec, "aac")]

        bit_rate = str(stream.get("bit_rate") or "").strip()
        if bit_rate.isdigit() and int(bit_rate) > 0 and codec not in {"flac", "pcm_s16le"}:
            args.extend(["-b:a", bit_rate])

        sample_rate = str(stream.get("sample_rate") or "").strip()
        if sample_rate.isdigit() and int(sample_rate) > 0:
            args.extend(["-ar", sample_rate])

        channels = str(stream.get("channels") or "").strip()
        if channels.isdigit() and int(channels) > 0:
            args.extend(["-ac", channels])

        return args


def _truncate_text(value: str, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    return "<truncated>..." + value[-limit:]


def _ffmpeg_error_text(error: subprocess.CalledProcessError) -> str:
    stderr = (error.stderr or b"").decode("utf-8", errors="replace").strip()
    stdout = (error.stdout or b"").decode("utf-8", errors="replace").strip()
    return stderr or stdout or str(error)


def cut_caption_segments(
    *,
    caption_path: Path,
    video_path: Path,
    output_dir: Path,
    granularity: str = "task",
    operator: str = "segment_cut",
) -> OperatorResult:
    caption = json.loads(caption_path.read_text(encoding="utf-8"))
    granularity = granularity.lower()
    probe = SegmentCutOperator._probe_media(video_path)
    fps = caption.get("fps") or SegmentCutOperator._probe_fps(video_path, probe)

    if fps is None or fps <= 0:
        return OperatorResult(
            status="error",
            operator=operator,
            errors=["Cannot determine video fps"],
        )

    segments = SegmentCutOperator._select_segments(caption, granularity)
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_dirs: list[str] = []
    manifest_segments: list[dict] = []
    failed_segments: list[dict] = []
    cut_count = 0
    cached_count = 0
    copy_count = 0
    reencode_count = 0

    for i, seg in enumerate(segments):
        segment_id = f"seg_{i:03d}"
        fi = seg.get("frame_interval", [])
        if len(fi) < 2:
            failure = SegmentCutOperator._failure_entry(
                segment_id=segment_id,
                seg=seg,
                reason="invalid_frame_interval",
                message="frame_interval must contain start and end frame",
            )
            failed_segments.append(failure)
            log.warning("segment_cut failed %s: %s", segment_id, failure["message"])
            continue

        start_frame, end_frame = int(fi[0]), int(fi[1])
        start_sec = start_frame / fps
        duration_sec = (end_frame - start_frame) / fps
        if end_frame <= start_frame:
            failure = SegmentCutOperator._failure_entry(
                segment_id=segment_id,
                seg=seg,
                reason="non_positive_duration",
                message=f"end_frame must be greater than start_frame: {start_frame}->{end_frame}",
                fps=fps,
            )
            failed_segments.append(failure)
            log.warning("segment_cut failed %s: %s", segment_id, failure["message"])
            continue

        seg_dir = output_dir / segment_id
        seg_dir.mkdir(exist_ok=True)
        seg_video = seg_dir / "rgb.mp4"

        if seg_video.exists() and seg_video.stat().st_size > _MIN_FILE_BYTES:
            SegmentCutOperator._write_info(seg_dir, seg, fps)
            segment_dirs.append(str(seg_dir))
            manifest_segments.append(SegmentCutOperator._manifest_entry(seg_dir, seg_video, seg, fps))
            cached_count += 1
            log.info(
                "segment_cut cached %s: frames=%s-%s duration=%.3fs path=%s",
                segment_id, start_frame, end_frame, duration_sec, seg_video,
            )
            continue

        copy_cmd = SegmentCutOperator._build_ffmpeg_copy_cmd(
            video_path=video_path,
            output_path=seg_video,
            start_sec=start_sec,
            duration_sec=duration_sec,
            probe=probe,
        )
        copy_error = ""
        try:
            subprocess.run(
                copy_cmd,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            copy_error = _ffmpeg_error_text(e)
            log.warning(
                "segment_cut stream-copy failed %s, falling back to re-encode: %s",
                segment_id, _truncate_text(copy_error, limit=800),
            )

        if seg_video.exists() and seg_video.stat().st_size > _MIN_FILE_BYTES:
            SegmentCutOperator._write_info(seg_dir, seg, fps)
            segment_dirs.append(str(seg_dir))
            manifest_segments.append(SegmentCutOperator._manifest_entry(seg_dir, seg_video, seg, fps))
            cut_count += 1
            copy_count += 1
            log.info(
                "segment_cut succeeded %s via stream-copy: frames=%s-%s duration=%.3fs path=%s",
                segment_id, start_frame, end_frame, duration_sec, seg_video,
            )
            continue

        cmd = SegmentCutOperator._build_ffmpeg_cut_cmd(
            video_path=video_path,
            output_path=seg_video,
            start_sec=start_sec,
            duration_sec=duration_sec,
            probe=probe,
        )
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            detail = _ffmpeg_error_text(e)
            if copy_error:
                detail = f"stream-copy failed first:\n{copy_error}\n\nre-encode failed:\n{detail}"
            failure = SegmentCutOperator._failure_entry(
                segment_id=segment_id,
                seg=seg,
                reason="ffmpeg_failed",
                message=_truncate_text(detail),
                fps=fps,
            )
            failed_segments.append(failure)
            log.warning("segment_cut failed %s: ffmpeg_failed: %s", segment_id, failure["message"])
            continue

        if seg_video.exists() and seg_video.stat().st_size > _MIN_FILE_BYTES:
            SegmentCutOperator._write_info(seg_dir, seg, fps)
            segment_dirs.append(str(seg_dir))
            manifest_segments.append(SegmentCutOperator._manifest_entry(seg_dir, seg_video, seg, fps))
            cut_count += 1
            reencode_count += 1
            log.info(
                "segment_cut succeeded %s via re-encode: frames=%s-%s duration=%.3fs path=%s",
                segment_id, start_frame, end_frame, duration_sec, seg_video,
            )
        else:
            size = seg_video.stat().st_size if seg_video.exists() else 0
            failure = SegmentCutOperator._failure_entry(
                segment_id=segment_id,
                seg=seg,
                reason="output_too_small",
                message=f"output file missing or too small: {size} bytes",
                fps=fps,
            )
            failed_segments.append(failure)
            log.warning("segment_cut failed %s: %s", segment_id, failure["message"])

    manifest_path = output_dir / "segments_manifest.json"
    manifest = {
        "sourceVideoPath": str(video_path),
        "captionPath": str(caption_path),
        "granularity": granularity,
        "fps": fps,
        "requestedSegments": len(segments),
        "totalSegments": len(manifest_segments),
        "successfulSegments": len(manifest_segments),
        "failedSegments": len(failed_segments),
        "cutSegments": cut_count,
        "cachedSegments": cached_count,
        "copySegments": copy_count,
        "reencodedSegments": reencode_count,
        "segments": manifest_segments,
        "failures": failed_segments,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(
        "segment_cut summary: requested=%s success=%s failed=%s cut=%s cached=%s manifest=%s",
        len(segments), len(segment_dirs), len(failed_segments), cut_count, cached_count, manifest_path,
    )

    return OperatorResult(
        status="ok",
        operator=operator,
        output_files=[str(output_dir), str(manifest_path)],
        metrics={
            "segment_dirs": segment_dirs,
            "manifest_path": str(manifest_path),
            "requested_segments": len(segments),
            "total_segments": len(segment_dirs),
            "failed_segments": len(failed_segments),
            "cut_count": cut_count,
            "cached_count": cached_count,
            "copy_count": copy_count,
            "reencode_count": reencode_count,
            "failures": failed_segments,
            "granularity": granularity,
        },
    )
