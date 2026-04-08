#!/usr/bin/env python3
"""
将 vlm_hand_audit.json 的逐帧手部计数叠加到视频上，生成预览视频。

每帧叠加内容：
  - 左上角彩色横幅：手部数量（0/1/2只）+ 置信区间
  - 颜色编码：0只=红色（无手操作）  1只=绿色  2只=蓝色（双手操作）
  - 右下角时间信息栏
  - 底部采样指示器：标记实际被 VLM 审计的帧位置

用法:
    python -m operators.hand.render_preview <video> <audit.json> [-o preview.mp4]
    # 或由 op_impl.py 自动调用
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 颜色 (RGB)
COLOR_0 = (220, 50, 50)    # 红：无手
COLOR_1 = (30, 180, 60)    # 绿：单手
COLOR_2 = (50, 100, 220)   # 蓝：双手
COLOR_FAIL = (150, 150, 150)  # 灰：VLM 失败
WHITE = (255, 255, 255)
BLACK_ALPHA = (0, 0, 0, 150)

BORDER_W = 8
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

COUNT_LABELS = {0: "无手", 1: "单手", 2: "双手"}
COUNT_COLORS = {0: COLOR_0, 1: COLOR_1, 2: COLOR_2}

# BGR 边框颜色
BORDER_BGR = {0: (50, 50, 220), 1: (60, 180, 30), 2: (220, 100, 50)}


def _load_fonts(frame_h: int):
    scale = max(frame_h / 960, 0.5)
    return (
        ImageFont.truetype(FONT_PATH, max(int(36 * scale), 20)),  # large
        ImageFont.truetype(FONT_PATH, max(int(24 * scale), 14)),  # small
    )


def _draw_banner(img: Image.Image, text: str, position: str, bg_color: tuple, font):
    """半透明横幅 + 白色文字。position: top_left | top_right | bottom_right"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 18, 10

    bw, bh = tw + pad_x * 2, th + pad_y * 2
    if position == "top_left":
        x1, y1 = 0, 0
    elif position == "top_right":
        x1, y1 = w - bw, 0
    elif position == "bottom_right":
        x1, y1 = w - bw, h - bh
    else:
        x1, y1 = 0, 0

    overlay = Image.new("RGBA", (bw, bh), (*bg_color, 170))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", overlay.size, (0, 0, 0, 0)), overlay).convert("RGB"),
        (x1, y1), overlay,
    )
    ImageDraw.Draw(img).text((x1 + pad_x, y1 + pad_y), text, font=font, fill=WHITE)


def _draw_info_bar(img: Image.Image, frame_idx: int, fps: float, font):
    """底部左侧：帧号 + 时间。"""
    w, h = img.size
    text = f"#{frame_idx}  {frame_idx / fps:.2f}s"
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 8
    bw, bh = tw + pad * 2, th + pad * 2
    y1 = h - bh
    overlay = Image.new("RGBA", (bw, bh), (0, 0, 0, 140))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", overlay.size, (0, 0, 0, 0)), overlay).convert("RGB"),
        (0, y1), overlay,
    )
    ImageDraw.Draw(img).text((pad, y1 + pad), text, font=font, fill=WHITE)


def _draw_timeline(
    img: Image.Image,
    frame_idx: int,
    total_frames: int,
    audit_frames: list[dict],
    font,
):
    """底部时间轴：显示 VLM 采样点的手部数量，当前位置标记红线。"""
    w, h = img.size
    bar_h = max(int(h * 0.025), 12)
    bar_y = h - bar_h
    bar_w = w

    overlay = Image.new("RGBA", (bar_w, bar_h), (30, 30, 30, 180))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", overlay.size, (0, 0, 0, 0)), overlay).convert("RGB"),
        (0, bar_y), overlay,
    )
    draw = ImageDraw.Draw(img)

    # 采样点小方块
    for af in audit_frames:
        gf = af["global_frame"]
        cnt = af.get("ego_hand_count", -1)
        x = int(gf / max(total_frames, 1) * bar_w)
        color = COUNT_COLORS.get(cnt, (150, 150, 150))
        dot_w = max(bar_w // 200, 3)
        draw.rectangle([x - dot_w, bar_y, x + dot_w, bar_y + bar_h], fill=color)

    # 当前位置红线
    cx = int(frame_idx / max(total_frames, 1) * bar_w)
    draw.line([(cx, bar_y), (cx, bar_y + bar_h)], fill=(255, 60, 60), width=2)


def build_lookup(audit_frames: list[dict]) -> dict[int, dict]:
    """global_frame → result dict"""
    return {af["global_frame"]: af for af in audit_frames}


def nearest_audit(frame_idx: int, sorted_gframes: list[int]) -> int | None:
    """二分查找最近采样帧。"""
    if not sorted_gframes:
        return None
    lo, hi = 0, len(sorted_gframes) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_gframes[mid] < frame_idx:
            lo = mid + 1
        else:
            hi = mid
    # lo is first >= frame_idx
    candidates = []
    if lo < len(sorted_gframes):
        candidates.append(sorted_gframes[lo])
    if lo > 0:
        candidates.append(sorted_gframes[lo - 1])
    return min(candidates, key=lambda x: abs(x - frame_idx))


def render(video_path: str, audit_json_path: str, output_path: str):
    audit = json.loads(Path(audit_json_path).read_text(encoding="utf-8"))
    frame_results: list[dict] = audit.get("frame_results", [])
    summary = audit.get("summary", {})
    fps = summary.get("fps", 30.0)
    total_frames = summary.get("total_frames_sampled", 0)

    lookup = build_lookup(frame_results)
    sorted_gframes = sorted(lookup.keys())

    cap = cv2.VideoCapture(video_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = min(1.0, 1280 / w)
    out_w, out_h = int(w * scale), int(h * scale)
    font_l, font_s = _load_fonts(out_h)

    print(f"视频: {w}x{h} @ {vid_fps:.1f}fps, {vid_total}帧")
    print(f"输出: {out_w}x{out_h}")
    print(f"VLM 审计帧数: {len(frame_results)} (avg_ego_hand={summary.get('avg_ego_hand_count', 0):.2f})")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, vid_fps, (out_w, out_h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale < 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # 查找最近采样帧的手部结果
        nearest_gf = nearest_audit(idx, sorted_gframes)
        af = lookup.get(nearest_gf) if nearest_gf is not None else None
        cnt = af.get("ego_hand_count", -1) if af else -1
        success = af.get("success", False) if af else False

        # 有效 VLM 结果：画彩色边框
        if success and cnt >= 0:
            cv2.rectangle(frame, (0, 0), (out_w - 1, out_h - 1), BORDER_BGR.get(cnt, (128, 128, 128)), BORDER_W)

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 左上角横幅
        if not success or cnt < 0:
            _draw_banner(pil, "检测失败", "top_left", COLOR_FAIL, font_l)
        else:
            label = COUNT_LABELS.get(cnt, str(cnt))
            color = COUNT_COLORS.get(cnt, COLOR_FAIL)
            # 标注是否为实际采样帧
            is_sampled = (nearest_gf == idx)
            suffix = " ●" if is_sampled else f" ←{abs(idx - nearest_gf)}f"
            _draw_banner(pil, f"手部: {cnt}只 ({label}){suffix}", "top_left", color, font_l)

        # 右下角：采样点统计
        stats_text = f"avg={summary.get('avg_ego_hand_count', 0):.2f}  0:{summary.get('ego_0_hands_ratio', 0):.0%}  1:{summary.get('ego_1_hand_ratio', 0):.0%}  2:{summary.get('ego_2_hands_ratio', 0):.0%}"
        _draw_banner(pil, stats_text, "bottom_right", (40, 40, 40), font_s)

        # 时间信息栏 + 时间轴
        _draw_timeline(pil, idx, vid_total, frame_results, font_s)
        _draw_info_bar(pil, idx, vid_fps, font_s)

        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        writer.write(frame)
        idx += 1

        if idx % 200 == 0:
            print(f"  已渲染 {idx}/{vid_total} 帧...")

    cap.release()
    writer.release()

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\n完成！手部分析预览: {output_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="手部分析预览视频生成")
    parser.add_argument("video", help="原始视频路径")
    parser.add_argument("audit", help="vlm_hand_audit.json 路径")
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    output = args.output or str(Path(args.video).parent / "vlm_hand_preview.mp4")
    render(args.video, args.audit, output)


if __name__ == "__main__":
    main()
