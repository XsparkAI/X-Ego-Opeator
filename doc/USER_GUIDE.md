# Ego-X Operator 用户指南

Ego-X Operator 是一个自我中心视频处理工具箱，包含 5 个独立算子和一个统一的流水线编排器。

## 快速开始

### 环境依赖

```bash
# 基础依赖（所有算子）
pip install opencv-python numpy pyyaml

# GPU 算子（privacy_blur, hand_detection）
pip install torch ultralytics

# 视频分割算子
pip install dashscope pandas

# 转码算子需要 ffmpeg
sudo apt install ffmpeg   # Ubuntu/Debian
```

### 30 秒上手

```bash
# 1. 单个算子独立使用
python -m operators.video_quality.assess path/to/video.mp4

# 2. 流水线处理单个 episode
python pipeline.py --episode path/to/episode_dir

# 3. 流水线批量处理（使用配置文件）
python pipeline.py --config pipeline_config.yaml

# 4. 预览流水线计划（不执行）
python pipeline.py --config pipeline_config.yaml --dry-run
```

## 算子参考

| 算子 | 功能 | GPU | 输入 | 输出 | 独立使用 |
|------|------|-----|------|------|----------|
| **video_quality** | 视频质量评估（清晰度/稳定性/曝光） | 否 | rgb.mp4 | quality_report.json | `python -m operators.video_quality.assess <video>` |
| **hand_detection** | 手部检测（左/右手） | 是 | rgb.mp4 | hand_detection.json | `python -m operators.hand.detect_hand_in_frame --video <video>` |
| **privacy_blur** | 人脸/车牌脱敏 | 是 | rgb.mp4 | rgb_blurred.mp4 | `python -m operators.privacy_blur.blur_privacy --video <video>` |
| **transcode** | 视频格式转换 | 否 | rgb.mp4 或 rgb_blurred.mp4 | *_transcoded.mp4 | `python -m operators.transcode.transcode <input> -o <output>` |
| **video_segmentation** | VLM 视频分割 | 否 | rgb.mp4 + frames.parquet | caption_v2t.json | `python -m operators.caption.segment_v2t --episode <dir>` |

详细文档见 `doc/pipeline_doc/` 目录下各算子文档。

## 流水线使用

### 配置文件

流水线通过 YAML 配置文件控制。默认配置文件：`pipeline_config.yaml`。

```yaml
# 指定要处理的 episode 目录
episodes:
  - FusionX-Multimodal-Sample-Data-V2/waterpour1
  - FusionX-Multimodal-Sample-Data-V2/drill1

# 或自动发现某目录下所有 episode
# episode_root: caption/FusionX-Multimodal-Sample-Data-V2

# 并行执行（无依赖的算子并发运行）
parallel: true

# 错误处理策略
on_error: continue   # continue = 继续执行其余算子 | fail_fast = 立即停止

# 算子配置（enabled: false 跳过该算子）
operators:
  video_quality:
    enabled: true
    sample_fps: 2.0

  hand_detection:
    enabled: true
    conf_thresh: 0.3

  privacy_blur:
    enabled: true

  transcode:
    enabled: false

  video_segmentation:
    enabled: false
```

### 并行执行与依赖

流水线自动分析算子间的依赖关系，将无依赖的算子分组并行执行：

```
┌─ Stage 1（并行）──────────────────────────────┐
│  video_quality · hand_detection · privacy_blur │
│  video_segmentation                            │
└────────────────────────────────────────────────┘
                      ↓
┌─ Stage 2（等 Stage 1 完成）────────────────────┐
│  transcode（依赖 privacy_blur 的输出）          │
└────────────────────────────────────────────────┘
```

- `parallel: true`（默认）— 同一 Stage 内的算子并发执行
- `parallel: false` — 所有算子串行执行

GPU 模型（privacy_blur、hand_detection）在多个 episode 间**只加载一次**，跨 episode 复用。

### CLI 参数

```bash
python pipeline.py [OPTIONS]

  --config FILE    配置文件路径（默认 pipeline_config.yaml）
  --episode DIR    处理单个 episode（覆盖配置文件中的 episodes）
  --report FILE    保存 JSON 报告到指定路径
  --dry-run        预览计划，不执行
```

### 输出报告

```bash
# 在终端打印摘要 + 保存 JSON 报告
python pipeline.py --config pipeline_config.yaml --report results/report.json
```

报告结构：

```json
{
  "total_episodes": 2,
  "operators_ok": 6,
  "operators_error": 0,
  "total_elapsed_sec": 45.2,
  "episodes": {
    "waterpour1": [
      {"status": "ok", "operator": "video_quality", "metrics": {...}},
      {"status": "ok", "operator": "hand_detection", "metrics": {...}},
      {"status": "ok", "operator": "privacy_blur", "metrics": {...}}
    ]
  }
}
```

## 常用场景

### 场景 1：只做质量评估

```yaml
operators:
  video_quality:
    enabled: true
    sample_fps: 2.0
  hand_detection:
    enabled: false
  privacy_blur:
    enabled: false
  transcode:
    enabled: false
  video_segmentation:
    enabled: false
```

### 场景 2：脱敏 + 转码交付

```yaml
operators:
  video_quality:
    enabled: false
  hand_detection:
    enabled: false
  privacy_blur:
    enabled: true
    scale: 1.3
  transcode:
    enabled: true
    codec: h264
    container: mp4
    resolution: 1920x1080
  video_segmentation:
    enabled: false
```

### 场景 3：完整流水线

```yaml
operators:
  video_quality:
    enabled: true
  hand_detection:
    enabled: true
  privacy_blur:
    enabled: true
  transcode:
    enabled: true
    codec: h265
  video_segmentation:
    enabled: true
```

需要设置环境变量：`export DASHSCOPE_API_KEY=your_key`

## Episode 目录结构

每个 episode 目录应包含：

```
episode_name/
├── rgb.mp4              # 输入视频（必需）
├── frames.parquet       # IMU 数据（video_segmentation 可选）
├── rgb_blurred.mp4      # privacy_blur 输出
├── hand_detection.json  # hand_detection 输出
├── quality_report.json  # video_quality 输出
└── caption_v2t.json     # video_segmentation 输出
```

## 编程接口

算子可在 Python 代码中直接使用：

```python
from pathlib import Path
from video_quality.operator import VideoQualityOperator, VideoQualityConfig
from hand.operator import HandDetectionOperator, HandDetectionConfig

# 单个算子
op = VideoQualityOperator(config=VideoQualityConfig(sample_fps=2.0))
result = op.run(Path("path/to/episode"))
print(result.status, result.metrics)

# 自定义流水线
operators = [
    VideoQualityOperator(),
    HandDetectionOperator(config=HandDetectionConfig(conf_thresh=0.5)),
]
for op in operators:
    result = op.run(episode_dir)
    if result.status == "error":
        print(f"{op.name} failed: {result.errors}")
        break
```

## 算子参数速查

### video_quality

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| sample_fps | float | null | 采样帧率，null 表示全帧 |

### hand_detection

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| conf_thresh | float | 0.3 | 检测置信度阈值 |
| frame_step | int | 1 | 每 N 帧处理一次 |
| input_height | int | 720 | 检测分辨率高度 |

### privacy_blur

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| face | bool | true | 检测并模糊人脸 |
| lp | bool | true | 检测并模糊车牌 |
| scale | float | 1.0 | 模糊区域放大系数 |
| face_thresh | float | null | 人脸检测阈值（null 用默认值） |
| lp_thresh | float | null | 车牌检测阈值（null 用默认值） |

### transcode

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| codec | str | null | 目标编码（h264/h265/ffv1/prores/vp9/av1/copy） |
| container | str | null | 目标容器（mp4/mkv/mov/avi/webm/mxf） |
| resolution | str | null | 目标分辨率（如 1920x1080） |
| bitrate | str | null | 目标码率（如 50M），启用有损模式 |
| fps | float | null | 目标帧率 |
| output_suffix | str | _transcoded | 输出文件后缀 |

### video_segmentation

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| window_sec | float | 10.0 | 滑动窗口时长（秒） |
| step_sec | float | 5.0 | 窗口步长（秒） |
| frames_per_window | int | 12 | 每窗口采样帧数 |
| snap_radius | int | 45 | IMU 谷值吸附半径（帧） |
| preview | bool | false | 生成预览视频 |
| dry_run | bool | false | 只显示计划不执行 |

## 输出指标对照表

### video_quality — quality_report.json

#### 清晰度 (quality)

| 字段 | 中文 | 说明 | 优秀阈值 |
|------|------|------|----------|
| mean_laplacian | 平均拉普拉斯方差 | 基于 Laplacian 算子的锐度评分，越高越清晰 | > 100 |
| mean_tenengrad | 平均 Tenengrad 梯度能量 | 基于 Sobel 梯度的锐度评分，越高越清晰 | > 50 |
| mean_brightness | 平均亮度 | 帧均亮度（0-255） | 80 ~ 180 |
| blur_ratio | 模糊帧占比 | 低于模糊阈值的帧比例（0-1），越低越好 | < 0.1 |

#### 稳定性 (stability)

| 字段 | 中文 | 说明 | 优秀阈值 |
|------|------|------|----------|
| translation_std | 平移抖动 | 帧间平移标准差（像素），越低越稳 | < 5.0 |
| rotation_std | 旋转抖动 | 帧间旋转标准差（弧度），越低越稳 | < 0.01 |
| mean_matched_points | 平均匹配点数 | 光流追踪匹配的特征点数，越高追踪越可靠 | > 50 |

#### 曝光 (exposure)

| 字段 | 中文 | 说明 | 优秀阈值 |
|------|------|------|----------|
| mean_overexposure | 平均过曝像素比例 | 亮度 > 220 的像素占比（0-1） | < 0.05 |
| mean_underexposure | 平均欠曝像素比例 | 亮度 < 30 的像素占比（0-1） | < 0.1 |
| mean_dynamic_range | 平均动态范围 | 1%-99% 亮度区间宽度（0-1） | > 0.5 |
| mean_entropy | 平均直方图熵 | 信息熵（bits），越高细节越丰富 | > 6.0 |

### hand_detection — hand_detection.json

| 字段 | 中文 | 说明 | 优秀阈值 |
|------|------|------|----------|
| total_frames_processed | 处理帧数 | 实际检测的帧总数 | — |
| fps_throughput | 检测吞吐量 | 每秒处理帧数（帧/秒） | — |
| frames_with_left_hand | 左手出现帧数 | 检测到左手的帧数量 | — |
| frames_with_right_hand | 右手出现帧数 | 检测到右手的帧数量 | — |
| frames_with_any_hand | 任意手出现帧数 | 检测到任一手的帧数量 | — |
| left_hand_ratio | 左手出现率 | 左手出现帧占比（0-1） | 视任务而定 |
| right_hand_ratio | 右手出现率 | 右手出现帧占比（0-1） | 视任务而定 |
| any_hand_ratio | 手部出现率 | 任一手出现帧占比（0-1），自我中心视频通常应 > 0.5 | > 0.5 |
