#!/usr/bin/env python3
"""
根据质量报告，在视频帧上叠加中文异常标记，生成 preview 视频。

用法:
    python -m operators.video_quality.render_preview <video> <report.json> [-o preview.mp4]

标记方式：
  - 过曝帧：红色横幅 + 红色边框
  - 欠曝帧：蓝色横幅 + 蓝色边框
  - 模糊帧：黄色横幅 + 黄色边框
  - 抖动帧：橙色横幅 + 橙色边框
  - 正常帧：绿色小标签
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 颜色 (BGR / RGB)
RED_BGR, RED_RGB = (0, 0, 230), (230, 0, 0)
BLUE_BGR, BLUE_RGB = (230, 80, 0), (0, 80, 230)
YELLOW_BGR, YELLOW_RGB = (0, 220, 255), (255, 220, 0)
ORANGE_BGR, ORANGE_RGB = (0, 140, 255), (255, 140, 0)
GREEN_RGB = (0, 200, 0)
WHITE_RGB = (255, 255, 255)

BORDER_W = 8
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def _load_fonts(frame_h: int):
    """根据输出分辨率自适应字号"""
    scale = max(frame_h / 960, 0.5)
    return (
        ImageFont.truetype(FONT_PATH, max(int(32 * scale), 18)),  # large
        ImageFont.truetype(FONT_PATH, max(int(26 * scale), 14)),  # small
    )


def draw_banner(img, text, position, bg_color, font):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 20, 10

    if position == "top_left":
        x1, y1 = 0, 0
    elif position == "top_right":
        x1, y1 = w - tw - pad_x * 2, 0
    elif position == "bottom_center":
        x1 = (w - tw - pad_x * 2) // 2
        y1 = h - th - pad_y * 2 - 60
    else:
        return

    x2, y2 = x1 + tw + pad_x * 2, y1 + th + pad_y * 2
    overlay = Image.new("RGBA", (x2 - x1, y2 - y1), (*bg_color, 160))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", overlay.size, (0, 0, 0, 0)), overlay).convert("RGB"),
        (x1, y1), overlay,
    )
    draw = ImageDraw.Draw(img)
    draw.text((x1 + pad_x, y1 + pad_y), text, font=font, fill=WHITE_RGB)


def draw_info_bar(img, frame_idx, fps, font):
    w, h = img.size
    text = f"帧 {frame_idx}  |  {frame_idx / fps:.2f}s"
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 10
    y1 = h - th - pad * 2
    overlay = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 130))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", overlay.size, (0, 0, 0, 0)), overlay).convert("RGB"),
        (0, y1), overlay,
    )
    ImageDraw.Draw(img).text((pad, y1 + pad), text, font=font, fill=WHITE_RGB)


def build_frame_sets(report: dict, sample_step: int):
    """从报告构建异常帧集合和详情查找表"""

    def expand_ranges(sampled, key="frame_idx"):
        if not sampled:
            return set()
        indices = sorted(f[key] for f in sampled)
        result = set()
        for i, idx in enumerate(indices):
            lo = idx - sample_step // 2 if i == 0 else (indices[i - 1] + idx) // 2
            hi = idx + sample_step // 2 if i == len(indices) - 1 else (idx + indices[i + 1]) // 2
            if i > 0 and idx - indices[i - 1] == sample_step:
                lo = indices[i - 1]
            if i < len(indices) - 1 and indices[i + 1] - idx == sample_step:
                hi = indices[i + 1]
            for fi in range(max(0, lo), hi + 1):
                result.add(fi)
        return result

    def build_detail(sampled, frame_set, key="frame_idx"):
        detail = {}
        for f in sampled:
            for fi in range(max(0, f[key] - sample_step), f[key] + sample_step + 1):
                if fi in frame_set and fi not in detail:
                    detail[fi] = f
        return detail

    over_sampled = sorted(report.get("exposure", {}).get("overexposed_frames", []), key=lambda f: f["frame_idx"])
    under_sampled = sorted(report.get("exposure", {}).get("underexposed_frames", []), key=lambda f: f["frame_idx"])
    blur_sampled = sorted(report.get("quality", {}).get("blurry_frames", []), key=lambda f: f["frame_idx"])
    jitter_sampled = sorted(report.get("stability", {}).get("jitter_frames", []), key=lambda f: f.get("frame_idx_from", 0))

    overexposed = expand_ranges(over_sampled)
    underexposed = expand_ranges(under_sampled)
    blurry = {f["frame_idx"] for f in blur_sampled}

    jitter_frames = set()
    for j in jitter_sampled:
        for fi in range(j["frame_idx_from"], j["frame_idx_to"] + 1):
            jitter_frames.add(fi)

    return {
        "overexposed": overexposed,
        "underexposed": underexposed,
        "blurry": blurry,
        "jitter": jitter_frames,
        "over_detail": build_detail(over_sampled, overexposed),
        "under_detail": build_detail(under_sampled, underexposed),
        "blur_detail": {f["frame_idx"]: f for f in blur_sampled},
    }


def render(video_path: str, report_path: str, output_path: str):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    fps = report["fps"]
    n_frames = report.get("num_frames", 0)

    # 推断采样步长
    total_frames_vid = None
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vfps = cap.get(cv2.CAP_PROP_FPS)

    sample_step = max(1, total_frames_vid // n_frames) if n_frames else 15

    # 输出宽度：不超过 1280
    scale = min(1.0, 1280 / w)
    out_w, out_h = int(w * scale), int(h * scale)

    font_l, font_s = _load_fonts(out_h)

    sets = build_frame_sets(report, sample_step)

    print(f"视频: {w}x{h} @ {vfps:.1f}fps, {total_frames_vid} 帧")
    print(f"输出: {out_w}x{out_h}, 采样步长={sample_step}")
    n_flagged = len(sets["overexposed"] | sets["underexposed"] | sets["blurry"] | sets["jitter"])
    print(f"异常帧: 过曝={len(sets['overexposed'])}, 欠曝={len(sets['underexposed'])}, "
          f"模糊={len(sets['blurry'])}, 抖动={len(sets['jitter'])} (合计覆盖 {n_flagged} 帧)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, vfps, (out_w, out_h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale < 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        is_over = idx in sets["overexposed"]
        is_under = idx in sets["underexposed"]
        is_blur = idx in sets["blurry"]
        is_jitter = idx in sets["jitter"]

        # OpenCV 边框
        if is_over:
            cv2.rectangle(frame, (0, 0), (out_w - 1, out_h - 1), RED_BGR, BORDER_W)
        elif is_under:
            cv2.rectangle(frame, (0, 0), (out_w - 1, out_h - 1), BLUE_BGR, BORDER_W)
        elif is_blur:
            cv2.rectangle(frame, (0, 0), (out_w - 1, out_h - 1), YELLOW_BGR, BORDER_W)
        elif is_jitter:
            cv2.rectangle(frame, (0, 0), (out_w - 1, out_h - 1), ORANGE_BGR, BORDER_W)

        # PIL 中文标注
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if is_over:
            d = sets["over_detail"].get(idx, {})
            draw_banner(pil_img, f"过曝  {d.get('overexposure_ratio', 0):.1%}", "top_left", RED_RGB, font_l)
        if is_under:
            d = sets["under_detail"].get(idx, {})
            draw_banner(pil_img, f"欠曝  {d.get('underexposure_ratio', 0):.1%}", "top_left", BLUE_RGB, font_l)
        if is_blur:
            d = sets["blur_detail"].get(idx, {})
            draw_banner(pil_img, f"模糊  lap={d.get('laplacian_var', 0):.0f}", "top_right", YELLOW_RGB, font_l)
        if is_jitter:
            draw_banner(pil_img, "抖动", "bottom_center", ORANGE_RGB, font_l)
        if not any([is_over, is_under, is_blur, is_jitter]):
            draw_banner(pil_img, "正常", "top_left", GREEN_RGB, font_s)

        draw_info_bar(pil_img, idx, fps, font_s)

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        writer.write(frame)
        idx += 1

        if idx % 100 == 0:
            print(f"  已渲染 {idx}/{total_frames_vid} 帧...")

    cap.release()
    writer.release()

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\n完成！预览视频: {output_path}")
    print(f"分辨率: {out_w}x{out_h}, 大小: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="视频质量检测预览")
    parser.add_argument("video", help="原始视频路径")
    parser.add_argument("report", help="quality_report.json 路径")
    parser.add_argument("-o", "--output", default=None, help="输出预览视频路径")
    args = parser.parse_args()

    output = args.output or str(Path(args.video).parent / "quality_preview.mp4")
    render(args.video, args.report, output)


if __name__ == "__main__":
    main()
