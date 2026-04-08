#!/usr/bin/env python3
"""
视频质量检测可靠性验证实验
用合成图像（已知 ground truth）+ 真实视频端到端测试，验证三个算子的检测准确性。

用法:
    cd Ego-X_Operator
    python -m operators.video_quality.test_verify [--video path/to/video.mp4]
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

from . import op_quality, op_stability, op_exposure

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def make_textured(h=480, w=640, seed=42):
    """生成高纹理清晰图像（棋盘格 + 随机噪声）"""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w), dtype=np.uint8)
    block = 32
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((y // block) + (x // block)) % 2 == 0:
                img[y:y+block, x:x+block] = 200
            else:
                img[y:y+block, x:x+block] = 55
    noise = rng.randint(0, 20, (h, w), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def make_flat(h=480, w=640, value=128):
    """生成纯色平坦图像（无纹理，极度模糊）"""
    return np.full((h, w), value, dtype=np.uint8)


def make_gradient(h=480, w=640):
    """生成水平渐变图像（中等纹理）"""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    return np.tile(row, (h, 1))


PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
WARN = "\033[93m WARN \033[0m"

passed_count = 0
failed_count = 0


def check(condition, desc):
    global passed_count, failed_count
    if condition:
        passed_count += 1
        print(f"  {PASS} {desc}")
    else:
        failed_count += 1
        print(f"  {FAIL} {desc}")


# ──────────────────────────────────────────────
# 实验1: 画面质量算子 (op_quality)
# ──────────────────────────────────────────────

def test_quality():
    print("\n" + "=" * 60)
    print("实验1: 画面质量检测 (op_quality)")
    print("=" * 60)

    # 1a. 清晰图 vs 模糊图：Laplacian 应有显著差异
    sharp = make_textured()
    blurred = cv2.GaussianBlur(sharp, (31, 31), sigmaX=10)
    flat = make_flat()

    r_sharp = op_quality.assess_frame(sharp, 0)
    r_blur = op_quality.assess_frame(blurred, 1)
    r_flat = op_quality.assess_frame(flat, 2)

    print(f"\n  清晰图: laplacian={r_sharp.laplacian_var:.1f}, tenengrad={r_sharp.tenengrad:.1f}, blurry={r_sharp.is_blurry}")
    print(f"  模糊图: laplacian={r_blur.laplacian_var:.1f}, tenengrad={r_blur.tenengrad:.1f}, blurry={r_blur.is_blurry}")
    print(f"  纯色图: laplacian={r_flat.laplacian_var:.1f}, tenengrad={r_flat.tenengrad:.1f}, blurry={r_flat.is_blurry}")
    print()

    check(r_sharp.laplacian_var > r_blur.laplacian_var * 5,
          f"清晰图 laplacian ({r_sharp.laplacian_var:.0f}) >> 模糊图 ({r_blur.laplacian_var:.0f})")
    check(r_sharp.tenengrad > r_blur.tenengrad * 5,
          f"清晰图 tenengrad ({r_sharp.tenengrad:.0f}) >> 模糊图 ({r_blur.tenengrad:.0f})")
    check(not r_sharp.is_blurry, "清晰图不应被标记为模糊")
    check(r_blur.is_blurry, "高斯模糊图应被标记为模糊")
    check(r_flat.is_blurry, "纯色图应被标记为模糊")
    check(r_flat.laplacian_var < 1.0, f"纯色图 laplacian ({r_flat.laplacian_var:.2f}) 应接近 0")

    # 1b. 渐进模糊：Laplacian 应单调递减
    print(f"\n  渐进模糊测试（sigma 递增）:")
    base = make_textured()
    prev_lap = float("inf")
    monotonic = True
    for sigma in [0, 2, 5, 10, 20, 40]:
        if sigma == 0:
            img = base
        else:
            img = cv2.GaussianBlur(base, (0, 0), sigmaX=sigma)
        r = op_quality.assess_frame(img, 0)
        ok = r.laplacian_var <= prev_lap
        if not ok:
            monotonic = False
        tag = "  " if ok else "✗ "
        print(f"    {tag}sigma={sigma:2d} → lap={r.laplacian_var:8.1f}  ten={r.tenengrad:10.1f}  blurry={r.is_blurry}")
        prev_lap = r.laplacian_var
    check(monotonic, "Laplacian 随模糊程度单调递减")

    # 1c. summarize 合格判定
    results_good = [op_quality.assess_frame(make_textured(seed=i), i) for i in range(10)]
    results_bad = [op_quality.assess_frame(make_flat(value=100 + i * 5), i) for i in range(10)]
    results_mix = results_good[:7] + results_bad[:3]  # 30% 模糊

    s_good = op_quality.summarize(results_good)
    s_bad = op_quality.summarize(results_bad)
    s_mix = op_quality.summarize(results_mix)

    print(f"\n  全清晰: blur_ratio={s_good['blur_ratio']}, pass={s_good['pass']}")
    print(f"  全模糊: blur_ratio={s_bad['blur_ratio']}, pass={s_bad['pass']}")
    print(f"  混合(30%模糊): blur_ratio={s_mix['blur_ratio']}, pass={s_mix['pass']}")

    check(s_good["pass"], "全清晰帧应通过")
    check(not s_bad["pass"], "全模糊帧不应通过")
    check(s_mix["pass"], "30% 模糊帧应通过（阈值 50%）")


# ──────────────────────────────────────────────
# 实验2: 曝光检测算子 (op_exposure)
# ──────────────────────────────────────────────

def test_exposure():
    print("\n" + "=" * 60)
    print("实验2: 曝光与光照检测 (op_exposure)")
    print("=" * 60)

    # 2a. 正常 / 过曝 / 欠曝合成图
    normal = make_gradient()  # 0-255 渐变，正常曝光
    overexposed = np.full((480, 640), 240, dtype=np.uint8)   # 全白，均值高 + 亮像素多
    underexposed = np.full((480, 640), 15, dtype=np.uint8)   # 全黑，均值低 + 暗像素多

    # 深色物体（大量暗像素但不是欠曝 —— 平均亮度在门控值之上）
    # 场景：30% 深色物体(像素值10) + 70% 正常背景(像素值140)
    # 均值 ≈ (0.3*10 + 0.7*140)/255 ≈ 0.396 > 0.35 门控值
    # underexposure_ratio = 0.3 > 0.20 像素条件满足，但亮度门控阻断
    dark_object = np.full((480, 640), 140, dtype=np.uint8)
    dark_object[:144, :] = 10  # 上 30% 为深色物体

    r_normal = op_exposure.assess_frame(normal, 0)
    r_over = op_exposure.assess_frame(overexposed, 1)
    r_under = op_exposure.assess_frame(underexposed, 2)
    r_dark_obj = op_exposure.assess_frame(dark_object, 3)

    print(f"\n  正常图:   brightness={r_normal.mean_brightness:.3f}, over={r_normal.overexposure_ratio:.3f}, under={r_normal.underexposure_ratio:.3f}, DR={r_normal.dynamic_range:.3f}")
    print(f"  过曝图:   brightness={r_over.mean_brightness:.3f}, over={r_over.overexposure_ratio:.3f}, under={r_over.underexposure_ratio:.3f}, DR={r_over.dynamic_range:.3f}")
    print(f"  欠曝图:   brightness={r_under.mean_brightness:.3f}, over={r_under.overexposure_ratio:.3f}, under={r_under.underexposure_ratio:.3f}, DR={r_under.dynamic_range:.3f}")
    print(f"  深色物体: brightness={r_dark_obj.mean_brightness:.3f}, over={r_dark_obj.overexposure_ratio:.3f}, under={r_dark_obj.underexposure_ratio:.3f}, DR={r_dark_obj.dynamic_range:.3f}")
    print()

    check(r_over.overexposure_ratio > 0.9, f"过曝图 overexposure_ratio ({r_over.overexposure_ratio:.2f}) 应 > 0.9")
    check(r_under.underexposure_ratio > 0.9, f"欠曝图 underexposure_ratio ({r_under.underexposure_ratio:.2f}) 应 > 0.9")
    check(r_normal.dynamic_range > 0.8, f"渐变图 dynamic_range ({r_normal.dynamic_range:.2f}) 应 > 0.8")
    check(r_over.dynamic_range < 0.1, f"全白图 dynamic_range ({r_over.dynamic_range:.2f}) 应 < 0.1")

    # 2b. 双条件门控验证
    print(f"\n  双条件门控验证:")

    # summarize 中的判定逻辑
    results_over = [op_exposure.assess_frame(overexposed, i) for i in range(5)]
    results_under = [op_exposure.assess_frame(underexposed, i) for i in range(5)]
    results_dark_obj = [op_exposure.assess_frame(dark_object, i) for i in range(5)]
    results_normal = [op_exposure.assess_frame(normal, i) for i in range(5)]

    s_over = op_exposure.summarize(results_over)
    s_under = op_exposure.summarize(results_under)
    s_dark_obj = op_exposure.summarize(results_dark_obj)
    s_normal = op_exposure.summarize(results_normal)

    print(f"  过曝图:   pass={s_over['pass']}, overexposed_frames={len(s_over['overexposed_frames'])}")
    print(f"  欠曝图:   pass={s_under['pass']}, underexposed_frames={len(s_under['underexposed_frames'])}")
    print(f"  深色物体: pass={s_dark_obj['pass']}, underexposed_frames={len(s_dark_obj['underexposed_frames'])}")
    print(f"  正常图:   pass={s_normal['pass']}")

    check(not s_over["pass"], "全过曝应不通过")
    check(not s_under["pass"], "全欠曝应不通过")
    check(s_dark_obj["pass"], "深色物体不应误判为欠曝（门控生效）")
    check(s_normal["pass"], "正常曝光应通过")

    # 深色物体详细验证：上半暗下半正常，均值 ~0.27
    mean_b = r_dark_obj.mean_brightness
    under_r = r_dark_obj.underexposure_ratio
    print(f"\n  深色物体门控详情: mean_brightness={mean_b:.3f}, underexposure_ratio={under_r:.3f}")
    print(f"    像素条件 ({under_r:.3f} > {op_exposure.FRAME_UNDEREXPOSURE_LIMIT}): {under_r > op_exposure.FRAME_UNDEREXPOSURE_LIMIT}")
    print(f"    亮度门控 ({mean_b:.3f} < {op_exposure.UNDEREXPOSURE_BRIGHTNESS_GATE}): {mean_b < op_exposure.UNDEREXPOSURE_BRIGHTNESS_GATE}")

    # 2c. histogram_entropy 区分度
    check(r_normal.histogram_entropy > r_over.histogram_entropy,
          f"渐变图 entropy ({r_normal.histogram_entropy:.2f}) > 全白图 ({r_over.histogram_entropy:.2f})")


# ──────────────────────────────────────────────
# 实验3: 稳定性检测算子 (op_stability)
# ──────────────────────────────────────────────

def test_stability():
    print("\n" + "=" * 60)
    print("实验3: 视频稳定性检测 (op_stability)")
    print("=" * 60)

    base = make_textured(480, 640, seed=0)

    # 3a. 完全静止序列
    static_frames = [base.copy() for _ in range(20)]
    r_static = op_stability.assess(static_frames, fps=30.0)
    print(f"\n  静止序列: trans_std={r_static.translation_std:.4f}, rot_std={r_static.rotation_std:.6f}, matched={r_static.mean_matched_points:.0f}")
    check(r_static.translation_std < 0.5, f"静止序列 translation_std ({r_static.translation_std:.4f}) 应 < 0.5")
    check(r_static.rotation_std < 0.001, f"静止序列 rotation_std ({r_static.rotation_std:.6f}) 应 < 0.001")

    # 3b. 已知平移序列
    shift_frames = []
    shifts = []
    for i in range(20):
        dx = int(i * 3)  # 每帧右移 3 像素
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        shifted = cv2.warpAffine(base, M, (640, 480), borderMode=cv2.BORDER_REFLECT)
        shift_frames.append(shifted)
        shifts.append(dx)

    r_shift = op_stability.assess(shift_frames, fps=30.0)
    print(f"\n  匀速平移 (3px/帧): trans_std={r_shift.translation_std:.2f}, rot_std={r_shift.rotation_std:.6f}")

    # 匀速平移 → translation 标准差应很小（因为每帧位移一致）
    check(r_shift.translation_std < 1.0,
          f"匀速平移 translation_std ({r_shift.translation_std:.2f}) 应较小（位移一致）")

    # 验证检测到的平移量接近 3px/帧
    if r_shift.pair_motions:
        mean_trans = np.mean([pm.translation for pm in r_shift.pair_motions])
        print(f"  平均帧间平移量: {mean_trans:.2f} (期望 ~3.0)")
        check(abs(mean_trans - 3.0) < 1.0, f"检测到的平均平移 ({mean_trans:.2f}) 接近真值 3.0")

    # 3c. 随机抖动序列
    rng = np.random.RandomState(123)
    jitter_frames = []
    for i in range(20):
        dx = rng.uniform(-15, 15)
        dy = rng.uniform(-15, 15)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        jittered = cv2.warpAffine(base, M, (640, 480), borderMode=cv2.BORDER_REFLECT)
        jitter_frames.append(jittered)

    r_jitter = op_stability.assess(jitter_frames, fps=30.0)
    print(f"\n  随机抖动 (±15px): trans_std={r_jitter.translation_std:.2f}, rot_std={r_jitter.rotation_std:.6f}")
    check(r_jitter.translation_std > r_static.translation_std * 5,
          f"随机抖动 trans_std ({r_jitter.translation_std:.2f}) >> 静止 ({r_static.translation_std:.4f})")

    # 3d. 旋转序列
    rot_frames = []
    for i in range(20):
        angle = i * 2  # 每帧旋转 2 度
        M = cv2.getRotationMatrix2D((320, 240), angle, 1.0)
        rotated = cv2.warpAffine(base, M, (640, 480), borderMode=cv2.BORDER_REFLECT)
        rot_frames.append(rotated)

    r_rot = op_stability.assess(rot_frames, fps=30.0)
    print(f"\n  匀速旋转 (2°/帧): trans_std={r_rot.translation_std:.2f}, rot_std={r_rot.rotation_std:.6f}")

    if r_rot.pair_motions:
        mean_rot = np.mean([pm.rotation for pm in r_rot.pair_motions])
        expected_rad = np.deg2rad(2.0)
        print(f"  平均帧间旋转量: {np.rad2deg(mean_rot):.2f}° (期望 ~2.0°)")
        check(abs(mean_rot - expected_rad) < np.deg2rad(1.0),
              f"检测到的旋转 ({np.rad2deg(mean_rot):.2f}°) 接近真值 2.0°")

    # 3e. summarize 抖动检测
    s_static = op_stability.summarize(r_static, fps=30.0)
    s_jitter = op_stability.summarize(r_jitter, fps=30.0)
    print(f"\n  静止序列: pass={s_static['pass']}, jitter_frames={len(s_static['jitter_frames'])}")
    print(f"  随机抖动: pass={s_jitter['pass']}, jitter_frames={len(s_jitter['jitter_frames'])}")
    check(s_static["pass"], "静止序列应通过稳定性检测")


# ──────────────────────────────────────────────
# 实验4: 真实视频端到端测试
# ──────────────────────────────────────────────

def test_real_video(video_path: str):
    print("\n" + "=" * 60)
    print(f"实验4: 真实视频端到端测试")
    print(f"  视频: {video_path}")
    print("=" * 60)

    from .assess import process_video

    result = process_video(video_path, sample_fps=2.0)

    print(f"\n  分辨率: {result['resolution']}")
    print(f"  采样帧数: {result['num_frames']}")
    print(f"  处理耗时: {result['processing_time_sec']:.2f}s")
    print(f"  整体通过: {result['pass']}")

    if "quality" in result:
        q = result["quality"]
        print(f"\n  [画面质量]")
        print(f"    mean_laplacian: {q['mean_laplacian']}")
        print(f"    mean_tenengrad: {q['mean_tenengrad']}")
        print(f"    blur_ratio: {q['blur_ratio']}")
        print(f"    pass: {q['pass']}")
        if q["blurry_frames"]:
            print(f"    模糊帧 ({len(q['blurry_frames'])} 帧):")
            for f in q["blurry_frames"][:5]:
                print(f"      frame={f['frame_idx']} ({f['time_sec']}s) lap={f['laplacian_var']}")

    if "stability" in result:
        s = result["stability"]
        print(f"\n  [稳定性]")
        print(f"    translation_std: {s['translation_std']:.2f}")
        print(f"    rotation_std: {s['rotation_std']:.6f}")
        print(f"    mean_matched_points: {s['mean_matched_points']:.0f}")
        print(f"    pass: {s['pass']}")
        if s["jitter_frames"]:
            print(f"    抖动帧 ({len(s['jitter_frames'])} 帧):")
            for f in s["jitter_frames"][:5]:
                print(f"      frame={f['frame_idx_from']}-{f['frame_idx_to']} ({f['time_sec']}s) "
                      f"trans_speed={f['translation_speed']}")

    if "exposure" in result:
        e = result["exposure"]
        print(f"\n  [曝光]")
        print(f"    mean_overexposure: {e['mean_overexposure']}")
        print(f"    mean_underexposure: {e['mean_underexposure']}")
        print(f"    mean_dynamic_range: {e['mean_dynamic_range']}")
        print(f"    mean_entropy: {e['mean_entropy']}")
        print(f"    pass: {e['pass']}")
        if e["overexposed_frames"]:
            print(f"    过曝帧: {len(e['overexposed_frames'])}")
        if e["underexposed_frames"]:
            print(f"    欠曝帧: {len(e['underexposed_frames'])}")

    # 基本健全性检查
    print()
    check(result["num_frames"] > 0, "帧数 > 0")
    check(result["processing_time_sec"] > 0, "处理耗时 > 0")
    if "quality" in result:
        check(result["quality"]["mean_laplacian"] > 0, "mean_laplacian > 0")
        check(0 <= result["quality"]["blur_ratio"] <= 1, "blur_ratio 在 [0,1] 范围")
    if "stability" in result:
        check(result["stability"]["mean_matched_points"] > 10, "特征点匹配数 > 10")
    if "exposure" in result:
        check(0 < result["exposure"]["mean_dynamic_range"] <= 1, "dynamic_range 在 (0,1] 范围")
        check(result["exposure"]["mean_entropy"] > 0, "entropy > 0")

    # 保存报告
    report_path = str(Path(video_path).parent / "quality_report_verify.json")
    Path(report_path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  报告已保存: {report_path}")

    return result, report_path


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="视频质量检测可靠性验证实验")
    parser.add_argument("--video", type=str, default=None, help="真实视频路径（可选）")
    args = parser.parse_args()

    print("=" * 60)
    print("  视频质量检测 — 可靠性验证实验")
    print("=" * 60)

    test_quality()
    test_exposure()
    test_stability()

    if args.video:
        test_real_video(args.video)
    else:
        # 尝试找一个默认的测试视频
        candidates = [
            "test_video/rgb_mid10s.mp4",
            "test_video/rgb.mp4",
            "tmp/test_episode/rgb.mp4",
        ]
        for c in candidates:
            if Path(c).exists():
                test_real_video(c)
                break
        else:
            print("\n  [跳过] 未找到测试视频，跳过端到端测试")
            print("  提示: 使用 --video 指定视频路径")

    # 总结
    print("\n" + "=" * 60)
    print(f"  验证结果: {passed_count} 通过, {failed_count} 失败")
    print("=" * 60)

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
