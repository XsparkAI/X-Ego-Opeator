#!/usr/bin/env python3
"""
视频格式转换算子 — 无损转码
将视频统一为采购方要求的编码格式、容器格式、分辨率、码率。

无损策略：
  1. 仅换封装（容器）→ stream copy，比特级无损
  2. 需换编码 → 使用目标编码器的数学无损模式（CRF 0 / lossless flag）
  3. 需换分辨率 → 高质量缩放 + 无损编码（像素级不可逆，但编码环节零损失）

用法:
    python -m transcode.transcode <input> -o <output> [OPTIONS]
    python -m transcode.transcode <input_dir> -o <output_dir> --batch [OPTIONS]

示例:
    # 仅换容器（比特无损）
    python -m transcode.transcode input.avi -o output.mp4

    # 转为 H.265 无损
    python -m transcode.transcode input.mp4 -o output.mkv --codec h265

    # 按采购方规格批量转换
    python -m transcode.transcode ./episodes -o ./delivered --batch \\
        --codec h264 --container mp4 --resolution 1920x1080 --bitrate 50M

    # 预览（不执行）
    python -m transcode.transcode input.mp4 -o output.mkv --codec h265 --dry-run
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
log = logging.getLogger(__name__)

# ──────────────────────────── 可选项定义 ────────────────────────────

# 支持的视频编码器（值 = ffmpeg encoder name）
CODECS = {
    "h264":    "libx264",
    "h265":    "libx265",
    "hevc":    "libx265",     # alias
    "ffv1":    "ffv1",        # 开源数学无损
    "prores":  "prores_ks",   # Apple ProRes（专业后期）
    "vp9":     "libvpx-vp9",
    "av1":     "libsvtav1",
    "copy":    "copy",        # stream copy, 不重编码
}

# 各编码器的无损参数
LOSSLESS_PARAMS = {
    "libx264":    ["-crf", "0", "-preset", "veryslow"],
    "libx265":    ["-x265-params", "lossless=1", "-preset", "veryslow"],
    "ffv1":       ["-level", "3", "-slicecrc", "1"],  # FFV1 天然无损
    "prores_ks":  ["-profile:v", "4", "-qscale:v", "0"],  # ProRes 4444 XQ
    "libvpx-vp9": ["-lossless", "1"],
    "libsvtav1":  [],  # SVT-AV1 暂不支持完全无损; 使用最高质量
}

# 各编码器的高质量（视觉无损）参数 — 当指定码率时使用
QUALITY_PARAMS = {
    "libx264":    ["-preset", "veryslow", "-profile:v", "high"],
    "libx265":    ["-preset", "veryslow", "-profile:v", "main"],
    "libvpx-vp9": ["-quality", "best", "-speed", "0"],
    "libsvtav1":  ["-preset", "2"],
    "prores_ks":  ["-profile:v", "4"],
    "ffv1":       ["-level", "3", "-slicecrc", "1"],
}

# 支持的容器格式
CONTAINERS = {
    "mp4":  ".mp4",
    "mkv":  ".mkv",
    "mov":  ".mov",
    "avi":  ".avi",
    "webm": ".webm",
    "mxf":  ".mxf",
}

# 容器与编码兼容性
CONTAINER_CODEC_COMPAT = {
    "mp4":  {"libx264", "libx265", "libsvtav1", "copy"},
    "mkv":  {"libx264", "libx265", "ffv1", "libvpx-vp9", "libsvtav1", "copy"},
    "mov":  {"libx264", "libx265", "prores_ks", "copy"},
    "avi":  {"libx264", "ffv1", "copy"},
    "webm": {"libvpx-vp9", "libsvtav1", "copy"},
    "mxf":  {"libx264", "libx265", "prores_ks", "copy"},
}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".mxf", ".ts", ".flv", ".wmv"}


# ──────────────────────────── 探测 ────────────────────────────

@dataclass
class ProbeResult:
    path: str
    codec: str           # e.g. "h264", "hevc"
    width: int
    height: int
    fps: float
    bitrate_kbps: int    # 视频流码率 (kbps)
    duration_sec: float
    pix_fmt: str         # e.g. "yuv420p"
    container: str       # e.g. "mp4"
    audio_codec: str     # e.g. "aac", "" if none
    rotation: int        # 旋转角度


def probe(path: str) -> ProbeResult:
    """用 ffprobe 获取视频元信息"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    video_stream = next(
        (s for s in info["streams"] if s["codec_type"] == "video"), None
    )
    if not video_stream:
        raise ValueError(f"No video stream found in {path}")

    audio_stream = next(
        (s for s in info["streams"] if s["codec_type"] == "audio"), None
    )

    # 码率：优先取流级别，回退到格式级别
    br = int(video_stream.get("bit_rate", 0))
    if br == 0:
        br = int(info.get("format", {}).get("bit_rate", 0))

    # FPS
    r_fps = video_stream.get("r_frame_rate", "30/1")
    num, den = map(int, r_fps.split("/"))
    fps = num / den if den else 30.0

    # 旋转
    rotation = 0
    tags = video_stream.get("tags", {})
    if "rotate" in tags:
        rotation = int(tags["rotate"])
    side_data = video_stream.get("side_data_list", [])
    for sd in side_data:
        if sd.get("side_data_type") == "Display Matrix" and "rotation" in sd:
            rotation = abs(int(sd["rotation"]))

    ext = Path(path).suffix.lstrip(".").lower()

    return ProbeResult(
        path=str(path),
        codec=video_stream["codec_name"],
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=round(fps, 3),
        bitrate_kbps=br // 1000,
        duration_sec=float(info["format"].get("duration", 0)),
        pix_fmt=video_stream.get("pix_fmt", "unknown"),
        container=ext,
        audio_codec=audio_stream["codec_name"] if audio_stream else "",
        rotation=rotation,
    )


# ──────────────────────────── 转码决策 ────────────────────────────

@dataclass
class TranscodeSpec:
    """转码规格"""
    codec: Optional[str] = None        # 目标编码 (CODECS key)
    container: Optional[str] = None    # 目标容器 (CONTAINERS key)
    resolution: Optional[str] = None   # "WxH", e.g. "1920x1080"
    bitrate: Optional[str] = None      # e.g. "50M", "8000k"
    fps: Optional[float] = None        # 目标帧率
    pix_fmt: Optional[str] = None      # 像素格式, e.g. "yuv420p"


@dataclass
class TranscodePlan:
    """单个文件的转码计划"""
    input_path: str
    output_path: str
    needs_video_reencode: bool
    needs_resolution_change: bool
    ffmpeg_cmd: List[str]
    warnings: List[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


def _parse_resolution(res: str) -> tuple:
    parts = res.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resolution format: {res}. Expected WxH, e.g. 1920x1080")
    return int(parts[0]), int(parts[1])


def plan_transcode(
    input_path: str,
    output_path: str,
    spec: TranscodeSpec,
) -> TranscodePlan:
    """根据源文件信息和目标规格，生成 ffmpeg 命令"""
    info = probe(input_path)
    warnings = []

    # 确定目标容器
    out_ext = Path(output_path).suffix.lstrip(".").lower()
    target_container = spec.container or out_ext or info.container

    # 确定目标编码器
    if spec.codec:
        encoder = CODECS.get(spec.codec)
        if not encoder:
            raise ValueError(f"Unknown codec: {spec.codec}. Supported: {list(CODECS.keys())}")
    else:
        encoder = None  # will decide below

    # 判断是否需要重编码
    needs_resolution = False
    target_w, target_h = info.width, info.height
    if spec.resolution:
        target_w, target_h = _parse_resolution(spec.resolution)
        if target_w != info.width or target_h != info.height:
            needs_resolution = True
            warnings.append(
                f"分辨率变更 {info.width}x{info.height} → {target_w}x{target_h}，"
                f"像素级不可逆。编码环节仍使用无损模式，缩放使用 Lanczos 算法。"
            )

    needs_codec_change = False
    if encoder and encoder != "copy":
        # 检查源编码是否已匹配
        source_codec_map = {"h264": "libx264", "hevc": "libx265", "vp9": "libvpx-vp9",
                            "av1": "libsvtav1", "ffv1": "ffv1", "prores": "prores_ks"}
        source_encoder = source_codec_map.get(info.codec, "")
        if source_encoder != encoder:
            needs_codec_change = True

    needs_pix_fmt = spec.pix_fmt and spec.pix_fmt != info.pix_fmt
    needs_fps = spec.fps and abs(spec.fps - info.fps) > 0.01

    needs_reencode = needs_resolution or needs_codec_change or needs_pix_fmt or needs_fps or spec.bitrate

    # 如果不需要重编码 → stream copy
    if not needs_reencode:
        if encoder and encoder != "copy":
            # 源编码已匹配目标，降级为 copy
            log.info(f"源编码已是 {info.codec}，无需重编码，使用 stream copy")
        encoder = "copy"

    # 如果显式指定 copy 但又需要改分辨率/帧率，报错
    if encoder == "copy" and (needs_resolution or needs_fps or needs_pix_fmt):
        raise ValueError(
            "stream copy (--codec copy) 不能与分辨率/帧率/像素格式变更同时使用。"
            "请指定目标编码器。"
        )

    if not encoder:
        encoder = "copy"

    # 检查容器兼容性
    if target_container in CONTAINER_CODEC_COMPAT:
        if encoder not in CONTAINER_CODEC_COMPAT[target_container] and encoder != "copy":
            raise ValueError(
                f"编码器 {encoder} 与容器 {target_container} 不兼容。"
                f"兼容编码器: {CONTAINER_CODEC_COMPAT[target_container]}"
            )

    # ── 构建 ffmpeg 命令 ──
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    if encoder == "copy":
        cmd += ["-c:v", "copy"]
    else:
        cmd += ["-c:v", encoder]

        # 缩放滤镜
        vf_filters = []
        if needs_resolution:
            vf_filters.append(f"scale={target_w}:{target_h}:flags=lanczos")
        if needs_fps:
            vf_filters.append(f"fps={spec.fps}")
        if vf_filters:
            cmd += ["-vf", ",".join(vf_filters)]

        # 像素格式
        if spec.pix_fmt:
            cmd += ["-pix_fmt", spec.pix_fmt]

        # 码率 or 无损
        if spec.bitrate:
            cmd += ["-b:v", spec.bitrate]
            # 使用高质量参数
            extra = QUALITY_PARAMS.get(encoder, [])
            cmd += extra
            warnings.append(
                f"指定码率 {spec.bitrate}，使用有损模式（视觉无损级别）。"
                f"如需数学无损，请去掉 --bitrate 参数。"
            )
        else:
            # 无损模式
            extra = LOSSLESS_PARAMS.get(encoder, [])
            cmd += extra
            if encoder == "libsvtav1":
                warnings.append(
                    "SVT-AV1 目前不支持数学无损模式，将使用最高质量设置 (CRF 0)。"
                )
                cmd += ["-crf", "0"]

    # 音频：尽量 copy
    if info.audio_codec:
        # 检查音频编码与目标容器兼容性
        audio_copy_ok = True
        if target_container == "webm" and info.audio_codec not in ("opus", "vorbis"):
            audio_copy_ok = False
        if audio_copy_ok:
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "libopus", "-b:a", "192k"]
            warnings.append(f"音频 {info.audio_codec} 与 {target_container} 不兼容，转为 opus 192k。")
    else:
        cmd += ["-an"]

    # 元数据保留
    cmd += ["-map_metadata", "0"]

    cmd.append(str(output_path))

    return TranscodePlan(
        input_path=str(input_path),
        output_path=str(output_path),
        needs_video_reencode=(encoder != "copy"),
        needs_resolution_change=needs_resolution,
        ffmpeg_cmd=cmd,
        warnings=warnings,
    )


# ──────────────────────────── 执行 ────────────────────────────

def execute(plan: TranscodePlan, dry_run: bool = False) -> dict:
    """执行转码计划，返回结果摘要"""
    if plan.skipped:
        log.info(f"SKIP {plan.input_path}: {plan.skip_reason}")
        return {"status": "skipped", "reason": plan.skip_reason}

    cmd_str = " ".join(plan.ffmpeg_cmd)

    if dry_run:
        log.info(f"[DRY-RUN] {cmd_str}")
        return {"status": "dry_run", "command": cmd_str}

    for w in plan.warnings:
        log.warning(w)

    log.info(f"Transcoding: {plan.input_path} → {plan.output_path}")
    log.info(f"Command: {cmd_str}")

    # 确保输出目录存在
    Path(plan.output_path).parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        plan.ffmpeg_cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log.error(f"ffmpeg failed:\n{result.stderr[-2000:]}")
        return {"status": "error", "stderr": result.stderr[-2000:]}

    # 验证输出
    out_info = probe(plan.output_path)
    in_info = probe(plan.input_path)

    summary = {
        "status": "ok",
        "input": plan.input_path,
        "output": plan.output_path,
        "reencode": plan.needs_video_reencode,
        "input_size_mb": round(Path(plan.input_path).stat().st_size / 1024 / 1024, 2),
        "output_size_mb": round(Path(plan.output_path).stat().st_size / 1024 / 1024, 2),
        "input_codec": in_info.codec,
        "output_codec": out_info.codec,
        "input_resolution": f"{in_info.width}x{in_info.height}",
        "output_resolution": f"{out_info.width}x{out_info.height}",
        "duration_diff_sec": round(abs(out_info.duration_sec - in_info.duration_sec), 3),
    }

    if summary["duration_diff_sec"] > 0.5:
        log.warning(
            f"时长偏差 {summary['duration_diff_sec']}s，请检查输出文件完整性。"
        )

    log.info(
        f"Done: {summary['input_size_mb']}MB → {summary['output_size_mb']}MB "
        f"({out_info.codec}, {out_info.width}x{out_info.height})"
    )
    return summary


# ──────────────────────────── 批量处理 ────────────────────────────

def batch_process(
    input_dir: str,
    output_dir: str,
    spec: TranscodeSpec,
    dry_run: bool = False,
) -> List[dict]:
    """批量转码目录下所有视频文件"""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # 收集视频文件
    files = sorted(
        f for f in in_dir.rglob("*")
        if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file()
    )
    if not files:
        log.warning(f"No video files found in {input_dir}")
        return []

    log.info(f"Found {len(files)} video files in {input_dir}")

    target_ext = CONTAINERS.get(spec.container, "") if spec.container else ""
    results = []

    for f in files:
        rel = f.relative_to(in_dir)
        if target_ext:
            out_path = out_dir / rel.with_suffix(target_ext)
        else:
            out_path = out_dir / rel

        try:
            plan = plan_transcode(str(f), str(out_path), spec)
            result = execute(plan, dry_run=dry_run)
        except Exception as e:
            log.error(f"Failed {f}: {e}")
            result = {"status": "error", "input": str(f), "error": str(e)}

        results.append(result)

    return results


# ──────────────────────────── CLI ────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="视频格式转换算子 — 无损转码",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可选编码格式:
  h264     H.264/AVC  (兼容性最佳, 无损模式 CRF=0)
  h265     H.265/HEVC (压缩率高, 无损模式 lossless=1)
  ffv1     FFV1       (开源数学无损, 推荐存档)
  prores   ProRes     (专业后期, 近无损)
  vp9      VP9        (Web 友好, 无损模式)
  av1      AV1/SVT    (下一代, 高压缩率)
  copy     Stream Copy(仅换封装, 比特级无损, 最快)

可选容器格式:
  mp4      最广泛兼容
  mkv      功能最全, 支持所有编码
  mov      Apple 生态 / ProRes
  avi      传统格式
  webm     Web 专用 (VP9/AV1)
  mxf      广播级

常见采购方规格组合:
  高兼容交付:  --codec h264 --container mp4 --resolution 1920x1080
  存档保存:    --codec ffv1 --container mkv
  专业后期:    --codec prores --container mov
  Web 发布:    --codec vp9 --container webm --resolution 1280x720
""",
    )
    parser.add_argument("input", help="输入视频文件或目录")
    parser.add_argument("-o", "--output", required=True, help="输出文件或目录")
    parser.add_argument("--batch", action="store_true", help="批量模式（输入为目录）")
    parser.add_argument(
        "--codec", choices=list(CODECS.keys()), default=None,
        help="目标视频编码 (默认: 自动判断, 尽量 stream copy)",
    )
    parser.add_argument(
        "--container", choices=list(CONTAINERS.keys()), default=None,
        help="目标容器格式 (默认: 由输出文件扩展名决定)",
    )
    parser.add_argument(
        "--resolution", default=None,
        help="目标分辨率, 格式 WxH, 如 1920x1080 (默认: 保持原始)",
    )
    parser.add_argument(
        "--bitrate", default=None,
        help="目标视频码率, 如 50M / 8000k (默认: 不限, 使用无损模式). "
             "注意: 指定码率将切换为有损模式。",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="目标帧率 (默认: 保持原始)",
    )
    parser.add_argument(
        "--pix-fmt", default=None,
        help="目标像素格式, 如 yuv420p / yuv444p (默认: 保持原始)",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅预览命令，不执行")
    parser.add_argument(
        "--report", default=None,
        help="输出 JSON 报告路径",
    )
    args = parser.parse_args()

    spec = TranscodeSpec(
        codec=args.codec,
        container=args.container,
        resolution=args.resolution,
        bitrate=args.bitrate,
        fps=args.fps,
        pix_fmt=args.pix_fmt,
    )

    if args.batch:
        results = batch_process(args.input, args.output, spec, dry_run=args.dry_run)
    else:
        plan = plan_transcode(args.input, args.output, spec)
        result = execute(plan, dry_run=args.dry_run)
        results = [result]

    if args.report:
        Path(args.report).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info(f"Report saved to {args.report}")

    # 统计
    ok = sum(1 for r in results if r.get("status") == "ok")
    err = sum(1 for r in results if r.get("status") == "error")
    skip = sum(1 for r in results if r.get("status") == "skipped")
    dry = sum(1 for r in results if r.get("status") == "dry_run")

    log.info(f"Summary: {ok} ok, {err} error, {skip} skipped, {dry} dry-run")

    if err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
