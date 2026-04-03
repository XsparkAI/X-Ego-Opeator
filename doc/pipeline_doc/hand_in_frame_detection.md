# Hand-in-Frame Detection Operator

## 概述

基于 YOLO 的手部在画检测算子，用于逐帧判断第一人称视频中左手/右手是否出现在画面中，并输出检测框、置信度和跨帧跟踪 ID。

**位置**: `hand/detect_hand_in_frame.py`

## 技术方案

### 模型选型

使用 HaWoR 项目中训练好的 YOLO 手部检测器（`detector.pt`, 52MB），该模型基于 Ultralytics YOLO 框架，支持：

- 手部检测 + 左右手分类（class 0 = 左手, class > 0 = 右手）
- 内置跟踪器（`model.track()` with `persist=True`），跨帧维持 track ID

### 检测流程

```
输入视频 → 逐帧读取 → resize 至检测分辨率(720p) → YOLO track → 去重(每只手保留最高置信度) → bbox 坐标映射回原分辨率 → 输出 JSON
```

核心设计：

1. **输入降分辨率**：默认将输入帧 resize 到 720p 高度（保持宽高比），检测完成后将 bbox 坐标按比例映射回原始分辨率
2. **去重策略**：每帧最多保留一个左手和一个右手检测，取置信度最高的
3. **跟踪持续性**：使用 `model.track(persist=True)` 保持跨帧 track ID，支持下游的手部轨迹连续性分析

### 权重来源

| 文件 | 来源 | 大小 |
|------|------|------|
| `hand/weights/detector.pt` | HaWoR 项目 (`Ego-X-Pipeline/HaWoR/weights/external/detector.pt`) | 52MB |

权重已独立 copy 至本项目，不依赖外部目录。

## 输出格式

```json
{
  "summary": {
    "video": "path/to/rgb.mp4",
    "total_frames_processed": 300,
    "fps": 29.97,
    "original_resolution": "3840x2880",
    "detection_resolution": "960x720",
    "frame_step": 1,
    "conf_thresh": 0.3,
    "elapsed_sec": 7.24,
    "fps_throughput": 41.5,
    "frames_with_left_hand": 226,
    "frames_with_right_hand": 249,
    "frames_with_any_hand": 298,
    "left_hand_ratio": 0.7533,
    "right_hand_ratio": 0.83,
    "any_hand_ratio": 0.9933
  },
  "frame_results": [
    {
      "frame": 0,
      "time_sec": 0.0,
      "left_hand": null,
      "right_hand": {
        "bbox_xyxy": [1188.6, 44.1, 1609.7, 543.0],
        "conf": 0.6338,
        "track_id": 1
      },
      "has_left": false,
      "has_right": true,
      "has_any_hand": true
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `bbox_xyxy` | `[x1, y1, x2, y2]` | 原始分辨率下的检测框坐标 |
| `conf` | `float` | YOLO 检测置信度 |
| `track_id` | `int \| null` | 跨帧跟踪 ID，丢失时为 null |
| `has_left` / `has_right` | `bool` | 该帧是否检测到左/右手 |
| `has_any_hand` | `bool` | 该帧是否检测到任意手 |

## 使用方法

**环境**: `conda activate hawor`（需要 ultralytics, torch, opencv）

```bash
# 基本用法
python hand/detect_hand_in_frame.py --video path/to/rgb.mp4

# episode 目录模式（自动读 rgb.mp4，输出 hand_detection.json）
python hand/detect_hand_in_frame.py --episode path/to/episode_dir

# 生成可视化 preview 视频（640p，带检测框叠加）
python hand/detect_hand_in_frame.py --video path/to/rgb.mp4 --preview

# 自定义参数
python hand/detect_hand_in_frame.py --video rgb.mp4 --conf 0.4 --step 2 --resize 1080
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | - | 输入视频路径 |
| `--episode` | - | Episode 目录（与 --video 互斥） |
| `--output` | `hand_detection.json` | 输出 JSON 路径 |
| `--conf` | 0.3 | YOLO 置信度阈值 |
| `--step` | 1 | 每 N 帧检测一次 |
| `--resize` | 720 | 检测分辨率高度（0 = 原始分辨率） |
| `--preview` | false | 是否生成可视化 preview 视频 |

### Preview 可视化

`--preview` 生成 640p 视频，叠加：
- 蓝色框 + "L" 标签：左手检测
- 绿色框 + "R" 标签：右手检测
- 底部状态栏：帧号 + 当前手部状态

## 分辨率实验

### 实验设计

在同一段 10 秒 4K 视频（3840x2880, 300 帧）上，测试 6 种输入分辨率对 YOLO 检测速度和准确率的影响。置信度阈值固定 0.3，逐帧检测。

### 实验结果

| 分辨率 | 检测尺寸 | 耗时(s) | 吞吐(FPS) | 加速比 | 左手% | 右手% | 任一手% |
|--------|----------|---------|-----------|--------|-------|-------|---------|
| 2880p (原始) | 3840x2880 | 23.06 | 13.0 | 1.0x | 73.7% | 78.0% | 97.3% |
| 1440p | 1920x1440 | 9.11 | 32.9 | 2.5x | 73.7% | 80.3% | 98.3% |
| 1080p | 1440x1080 | 8.57 | 35.0 | 2.7x | 74.0% | 82.0% | 99.0% |
| **720p** | **960x720** | **7.24** | **41.5** | **3.2x** | **75.3%** | **83.0%** | **99.3%** |
| 480p | 640x480 | 6.71 | 44.7 | 3.4x | 74.3% | 86.0% | 98.3% |
| 360p | 480x360 | 6.47 | 46.4 | 3.6x | 76.7% | 82.3% | 98.3% |

### 分析

1. **速度方面**：原始 4K 输入最慢（13 FPS），降至 1440p 速度提升 2.5 倍，后续递减。720p→360p 的增益已不显著（41.5→46.4 FPS）
2. **检出率方面**：各分辨率的检出率差异很小（97.3%~99.3%）。720p 的 `any_hand_ratio` 达到最高值 99.3%，说明第一人称视角下手部目标相对较大，低分辨率足以有效检测
3. **最佳平衡**：**720p** 兼顾速度（3.2x 加速）和最高检出率（99.3%），已设为默认值

### 测试环境

- GPU: 单卡（CUDA enabled）
- 测试视频: 4K egocentric video, 3840x2880 @ 29.97fps
- 模型: HaWoR YOLO hand detector (52MB)

## 依赖

```
ultralytics
torch
opencv-python
numpy
```
