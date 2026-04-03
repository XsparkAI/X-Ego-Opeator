# Privacy Blur Operator (视频脱敏)

## 概述

基于 Meta EgoBlur 的视频隐私脱敏算子，检测并模糊第一人称视频中的人脸和车牌。上层封装 `egoblur` 包，不修改其源码。

**位置**: `privacy_blur/blur_privacy.py`

## 技术方案

### 模型选型

使用 [EgoBlur](https://github.com/facebookresearch/EgoBlur) Gen2 模型（Faster RCNN + ResNeXt，Detectron2 框架）：

| 模型文件 | 用途 | 大小 | 默认阈值 |
|----------|------|------|----------|
| `ego_blur_face_gen2.jit` | 人脸检测 | 401MB | 0.67 |
| `ego_blur_lp_gen2.jit` | 车牌检测 | 401MB | 0.74 |

**选型理由**：EgoBlur 专为第一人称/可穿戴视频训练（23M 图像 / 790M bbox），与 Ego-X 数据分布高度契合。Apache 2.0 开源。

### 处理流程

```
输入视频 → 逐帧读取 → resize 至检测分辨率(默认720p) → EgoBlur 检测(face+LP)
    → bbox 坐标映射回原分辨率 → 椭圆高斯模糊 → 写出原分辨率脱敏视频
```

核心设计：

1. **输入降分辨率**：默认 resize 到 720p 高度进行检测，bbox 坐标按比例映射回原始分辨率后在原图上做模糊，输出视频保持原始分辨率不降质
2. **椭圆模糊**：在 bbox 区域用椭圆 mask + 高斯模糊（kernel 自适应，≥31），比矩形遮挡更自然
3. **上层封装**：通过 `egoblur` pip 包引入，不修改 EgoBlur 仓库代码

### 架构

```
privacy_blur/
├── blur_privacy.py          # 主算子脚本
└── weights/                 # EgoBlur Gen2 模型文件
    ├── ego_blur_face_gen2.jit
    └── ego_blur_lp_gen2.jit
```

## 安装

### 1. 安装 egoblur

```bash
pip install egoblur   # 需要 Python 3.10-3.12
```

依赖 PyTorch、torchvision、OpenCV、numpy。

### 2. 下载模型

从 [Project Aria Tools - EgoBlur](https://www.projectaria.com/tools/egoblur) 下载 Gen2 模型，放入 `privacy_blur/weights/`：

```bash
# 下载后放置：
#   privacy_blur/weights/ego_blur_face_gen2.jit
#   privacy_blur/weights/ego_blur_lp_gen2.jit
```

## 使用方法

**环境**: Python 3.10（egoblur 安装在该版本下）

```bash
# 基本用法：脱敏单个视频（人脸 + 车牌，720p 检测，原分辨率输出）
python3.10 privacy_blur/blur_privacy.py --video path/to/rgb.mp4

# Episode 目录模式（读 rgb.mp4，输出 rgb_blurred.mp4）
python3.10 privacy_blur/blur_privacy.py --episode path/to/episode_dir

# 自定义输出路径
python3.10 privacy_blur/blur_privacy.py --video path/to/rgb.mp4 --output blurred.mp4

# 放大模糊区域 30%
python3.10 privacy_blur/blur_privacy.py --video path/to/rgb.mp4 --scale 1.3

# 生成左右对比预览视频（用于人工 QA 检查脱敏效果）
python3.10 privacy_blur/blur_privacy.py --video path/to/rgb.mp4 --preview
```

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | - | 输入视频路径 |
| `--episode` | - | Episode 目录（与 --video 互斥） |
| `--output` | `<name>_blurred.mp4` | 输出视频路径（--video 模式） |
| `--scale` | 1.0 | 模糊区域缩放因子 |
| `--face-thresh` | 0.67 | 人脸检测置信度阈值 |
| `--lp-thresh` | 0.74 | 车牌检测置信度阈值 |
| `--preview` | false | 生成左右对比预览视频（用于人工 QA） |

## 输出格式

- Episode 模式：`rgb_blurred.mp4`（与 `rgb.mp4` 同目录）
- 单视频模式：`<name>_blurred.mp4`（或 `--output` 指定路径）
- 运行结束后在终端输出 summary dict，包含帧数、检测数、耗时等统计信息

## 分辨率实验

### 实验设计

在同一段 20 秒 4K 视频（3840x2880, 30fps, 600 帧, HEVC）上，测试 5 种检测分辨率对处理速度、GPU 显存和检测效果的影响。所有输出均保持原始 4K 分辨率。

- **测试视频**: DJI Osmo Nano 拍摄的室外场景，含行人和车辆
- **GPU**: 单卡 CUDA
- **检测目标**: 人脸 + 车牌（默认阈值 face=0.67, LP=0.74）

### 实验结果

| 分辨率 | 检测尺寸 | 耗时(s) | 吞吐(FPS) | 加速比 | GPU峰值(MB) | 检测数 |
|--------|----------|---------|-----------|--------|-------------|--------|
| 2880p (原始) | 3840x2880 | 194.8 | 3.1 | 1.0x | 1230 | 193 |
| 1440p | 1920x1440 | 99.8 | 6.0 | 2.0x | 1210 | 192 |
| 1080p | 1440x1080 | 89.7 | 6.7 | 2.2x | 1206 | 191 |
| **720p** | **960x720** | **82.0** | **7.3** | **2.4x** | **1205** | **189** |
| 480p | 640x480 | 77.6 | 7.7 | 2.5x | 1204 | 191 |

### 分析

1. **速度**：原始 4K → 720p 处理速度提升 **2.4 倍**（3.1 → 7.3 fps）。720p → 480p 仅提升 5%，说明瓶颈已转移到模型推理固定开销（EgoBlur 内部统一 resize 到 1200px 短边）
2. **GPU 显存**：各分辨率差异极小（1204~1230 MB），因 EgoBlur 内部推理分辨率固定为 1200px
3. **检测效果**：各分辨率检测数高度一致（189~193），**720p 检测几乎不损失精度**（仅少 4 个检测，差 2%）
4. **最佳平衡**：**720p** 在速度（2.4x 加速）和检测完整性（98% 保持率）之间取得最优平衡，已设为默认值。1080p 适合对精度要求更高的场景

### 关键发现

- EgoBlur 内部会将输入 resize 到短边 1200px 再推理，因此即使输入分辨率差异很大，实际模型推理开销差异有限
- 主要时间差来自 **输入帧的 decode + resize I/O**，4K 帧的读取和预处理本身耗时显著
- GPU 显存由模型权重和内部固定推理分辨率决定，与输入分辨率关系不大

## 关键参数调优

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `face_thresh` | 0.67 | EgoBlur Gen2 camera-rgb 默认值，降低可提高召回但增加误检 |
| `lp_thresh` | 0.74 | EgoBlur Gen2 camera-rgb 默认值 |
| `scale` | 1.0 | bbox 缩放因子，>1 可扩大模糊区域确保完整覆盖 |
| `resize` | 720 | 检测分辨率，影响速度但对精度影响极小 |

## Programmatic API

```python
from privacy_blur.blur_privacy import blur_video

# 处理单个视频（720p 检测，4K 输出）
summary = blur_video(
    video_path=Path("input_4k.mp4"),
    output_path=Path("output_4k_blurred.mp4"),
    face=True, lp=True,
    resize=720,
)
```

## 依赖

```
egoblur>=2.0.0
torch
opencv-python
numpy
```
