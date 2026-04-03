# Video Quality Assessment Operator (视频质量评估)

## 概述

纯经典视频质量评估算子，包含三个独立子算子：画面质量检测、视频稳定性检测、曝光与光照检测。基于 OpenCV + NumPy，无深度学习依赖，CPU 运行，零显存占用。

**位置**: `video_quality/`

## 技术方案

### 方案选型

调研对比了 TOPIQ-NR、CLIP-IQA+ (ViT-L/14)、DOVER、Q-Align 等深度学习方案，经实测对比后选择纯经典方案：

| 维度 | 经典方案 | DL 方案 (TOPIQ + CLIP-IQA+) |
|------|----------|------------------------------|
| 处理速度 | ~50 ms/帧 | ~350 ms/帧 |
| 显存占用 | 0 | ~2-3 GB |
| 权重文件 | 无 | ~1.1 GB |
| 依赖 | cv2, numpy | + torch, pyiqa, clip |

**选型理由**：在 egocentric 手动任务视频场景下，DL 感知分与经典指标（Laplacian/直方图）高度相关，信息增益有限，额外资源开销不合理。

### 算子1: 画面质量检测

**文件**: `video_quality/op_quality.py`

| 指标 | 方法 | 说明 |
|------|------|------|
| `laplacian_var` | Laplacian 方差 | 核心清晰度指标，越高越清晰 |
| `tenengrad` | Sobel 梯度能量 | 补充清晰度指标（对纹理更敏感） |
| `mean_brightness` | 灰度均值 | 整体亮度水平 (0-255) |
| `is_blurry` | Laplacian < 阈值 | 模糊帧标记（默认阈值 100.0） |

### 算子2: 视频稳定性检测

**文件**: `video_quality/op_stability.py`

```
逐帧对 → goodFeaturesToTrack (Shi-Tomasi) 特征点提取
    → calcOpticalFlowPyrLK (Lucas-Kanade) 光流追踪
    → estimateAffinePartial2D 仿射估计 (dx, dy, da)
    → 帧间变换标准差统计
```

| 指标 | 方法 | 说明 |
|------|------|------|
| `translation_std` | 帧间平移标准差 | 平移抖动幅度（像素），越低越稳 |
| `rotation_std` | 帧间旋转标准差 | 旋转抖动幅度（弧度），越低越稳 |
| `mean_matched_points` | 平均匹配特征点数 | 光流追踪质量指标 |

所有计算均直接使用 OpenCV 成熟接口（`goodFeaturesToTrack`、`calcOpticalFlowPyrLK`、`estimateAffinePartial2D`），统计层仅使用 NumPy `std`。

### 算子3: 曝光与光照检测

**文件**: `video_quality/op_exposure.py`

**帧级指标**：

| 指标 | 方法 | 说明 |
|------|------|------|
| `mean_brightness` | 灰度均值 / 255 | 归一化亮度 (0-1) |
| `overexposure_ratio` | 像素 > 220 的比例 | 过曝区域占比 |
| `underexposure_ratio` | 像素 < 30 的比例 | 欠曝区域占比 |
| `dynamic_range` | (P99 - P1) / 255 | 有效动态范围 (0-1) |
| `histogram_entropy` | 直方图信息熵 | 分布均匀性，越高越好 |

### 架构

``` 
video_quality/
├── __init__.py
├── assess.py            # 整合脚本（读帧 + 调度三个算子 + 输出报告）
├── op_quality.py        # 算子1: 画面质量
├── op_stability.py      # 算子2: 视频稳定性
└── op_exposure.py       # 算子3: 曝光与光照
```

三个算子完全独立，各自暴露统一接口：
- `assess_frame(gray, frame_idx)` / `assess(frames_gray, fps)` — 评估
- `summarize(results)` — 汇总为字典

## 安装

```bash
pip install opencv-python numpy
```

无模型下载，无 GPU 依赖。

## 使用方法

```bash
# 基本用法
python -m video_quality.assess path/to/video.mp4

# 指定采样帧率（推荐，避免全帧处理 4K 视频）
python -m video_quality.assess path/to/video.mp4 --sample-fps 2

# 输出到文件
python -m video_quality.assess path/to/video.mp4 --sample-fps 2 -o report.json
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `video` | (必填) | 视频文件路径 |
| `--sample-fps` | None (全帧) | 采样帧率，设为 2 表示每秒采样 2 帧 |
| `-o, --output` | stdout | 输出 JSON 路径 |

## 输出格式

```json
{
  "video_path": "test_video/test_10s.mp4",
  "num_frames": 22,
  "fps": 29.97,
  "resolution": "3840x2880",
  "processing_time_sec": 9.603,
  "quality": {
    "mean_laplacian": 121.9,
    "mean_tenengrad": 4835.64,
    "mean_brightness": 102.86,
    "blur_ratio": 0.4091
  },
  "stability": {
    "translation_std": 12.35,
    "rotation_std": 0.008,
    "mean_matched_points": 186.5
  },
  "exposure": {
    "mean_overexposure": 0.0155,
    "mean_underexposure": 0.092,
    "mean_dynamic_range": 0.8446,
    "mean_entropy": 7.665
  }
}
```

## Programmatic API

```python
from video_quality import op_quality, op_stability, op_exposure
from video_quality.assess import process_video, read_frames

# 完整评估
result = process_video("path/to/video.mp4", sample_fps=2.0)

# 单独使用某个算子
frames_gray, frame_indices, meta = read_frames("video.mp4", sample_fps=2.0)

quality_results = [op_quality.assess_frame(g, fi) for g, fi in zip(frames_gray, frame_indices)]
stability = op_stability.assess(frames_gray, fps=meta["fps"] / meta["step"])
exposure_results = [op_exposure.assess_frame(g, fi) for g, fi in zip(frames_gray, frame_indices)]
```

## 关键参数调优

| 参数 | 当前值 | 位置 | 说明 |
|------|--------|------|------|
| `BLUR_THRESHOLD` | 100.0 | op_quality.py | Laplacian 模糊阈值，降低可减少误标 |
| `FEATURE_MAX_CORNERS` | 500 | op_stability.py | 特征点数量上限 |
| `FEATURE_QUALITY` | 0.01 | op_stability.py | Shi-Tomasi 角点质量阈值 |
| `FEATURE_MIN_DIST` | 30 | op_stability.py | 特征点最小间距（像素） |
| `OVEREXPOSURE_THRESH` | 220 | op_exposure.py | 过曝像素阈值 (0-255) |
| `UNDEREXPOSURE_THRESH` | 30 | op_exposure.py | 欠曝像素阈值 (0-255) |

## 性能

测试视频：4K (3840x2880) HEVC, 10.9s, 30fps, 采样 2fps (22帧)

| 算子 | 耗时 | 每帧耗时 |
|------|------|----------|
| 画面质量 | 0.47s | 21 ms |
| 稳定性 | 0.44s | 20 ms |
| 曝光与光照 | 0.18s | 8 ms |
| **总计（含读帧）** | **9.6s** | — |

读帧占总耗时 ~89%（4K HEVC 解码），三个算子合计仅 ~1.1s。

## 依赖

```
opencv-python
numpy
```
