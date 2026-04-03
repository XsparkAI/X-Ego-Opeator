# Pipeline Benchmark — 2min 4K Video

**日期**: 2026-04-02  
**视频**: DJI Osmo Nano, 3840x2880, HEVC, 29.97fps, 3597 frames (120s)  
**环境**: conda `sdk`, CUDA GPU, Linux

## 测试配置

```yaml
parallel: true
on_error: continue
operators:
  video_quality:    { sample_fps: 2.0 }
  hand_detection:   { frame_step: 3, input_height: 720 }
  privacy_blur:     { resize: 720 }
  transcode:        { codec: h264, resolution: "1920x1440" }
  video_segmentation: { enabled: false }  # 跳过 VLM caption
```

**执行计划** (pipeline 自动调度):
- Stage 1 (并行): video_quality, hand_detection, privacy_blur
- Stage 2 (串行): transcode (依赖 privacy_blur 输出 `rgb_blurred.mp4`)

## 各步骤用时

| 算子 | 耗时 | 处理帧数 | 吞吐量 | 阶段 |
|------|------|----------|--------|------|
| **video_quality** | 126.9s | 240 (sample_fps=2) | 34.8 ms/frame (quality) + 33.7 ms/frame (stability) + 9.2 ms/frame (exposure) | Stage 1 并行 |
| **hand_detection** | 120.4s | 1,199 (step=3) | 10.1 fps | Stage 1 并行 |
| **privacy_blur** | 499.3s | 3,597 (全帧) | 7.2 fps | Stage 1 并行 |
| **transcode** | 258.7s | 3,597 | — | Stage 2 串行 |
| **总计 (wall clock)** | **757.9s** | — | — | — |

### 时间线分析

```
0s          127s          499s    758s
├─ Stage 1 (并行) ────────┤       │
│  video_quality  [██ 127s]       │
│  hand_detection [█ 120s]        │
│  privacy_blur   [██████ 499s]   │  ← 瓶颈
│                         ├ Stage 2 ─┤
│                         │ transcode [████ 259s]
```

**Stage 1 瓶颈: privacy_blur** — 全帧处理 4K→720p 检测 + 原始分辨率写回，耗时是其他两个算子的 4x。  
**Stage 2**: transcode 必须等 privacy_blur 完成后才能开始（依赖 `rgb_blurred.mp4`）。

## 各算子详细结果

### 1. Video Quality

| 指标 | 值 | 说明 |
|------|-----|------|
| mean_laplacian | 260.95 | 清晰度（>100 为正常） |
| blur_ratio | 14.6% | 模糊帧占比 |
| stability_score | 1.0 | 稳定性（1.0 = 极稳定） |

- 处理了 240 帧（sample_fps=2，每秒采样 2 帧）
- 三个子算子（quality/stability/exposure）串行执行，合计 18.6s 纯计算 + 107.6s 视频读取

### 2. Hand Detection (YOLO)

| 指标 | 值 |
|------|-----|
| 处理帧数 | 1,199 (每 3 帧取 1 帧) |
| 检测分辨率 | 960x720 |
| 左手检出率 | 44.9% |
| 右手检出率 | 58.1% |
| 任意手检出率 | 74.7% |
| 吞吐量 | 10.1 fps |

### 3. Privacy Blur (EgoBlur Gen2)

| 指标 | 值 |
|------|-----|
| 处理帧数 | 3,597 (全帧) |
| 检测分辨率 | 960x720 |
| 输出分辨率 | 3840x2880（原始） |
| 人脸检测 | 开启 (thresh=0.674) |
| 车牌检测 | 开启 (thresh=0.745) |
| 检测到的目标 | 518 次，覆盖 513 帧 |
| 吞吐量 | 7.2 fps |
| 输出文件 | rgb_blurred.mp4 (1245 MB) |

**瓶颈分析**: 全帧处理 + 4K 分辨率写回是主要开销。检测本身在 GPU 上很快，但逐帧读取 4K → resize → 检测 → 原始分辨率 blur → 写回的 I/O 链路很慢。

### 4. Transcode (FFmpeg)

| 指标 | 值 |
|------|-----|
| 输入 | rgb_blurred.mp4 (mpeg4, 3840x2880, 1245 MB) |
| 输出 | rgb_blurred_transcoded.mp4 (h264, 1920x1440, 3348 MB) |
| 缩放算法 | Lanczos |
| 编码模式 | CRF 0 (无损) + preset veryslow |

**注意**: 输出文件 (3348 MB) > 输入文件 (1245 MB)，因为 CRF 0 无损编码产生的 H.264 比 OpenCV 的 mpeg4 编码更大。生产环境应使用有损编码（如 CRF 18-23）控制文件大小。

## 优化建议

| 优化点 | 预估效果 |
|--------|----------|
| privacy_blur 降低输出分辨率（写 1920x1440 而非 4K） | 写回 I/O 减少 ~4x |
| transcode 使用 CRF 18 + preset medium 替代 CRF 0 + veryslow | 速度提升 ~5x，文件大小减少 ~10x |
| hand_detection 已用 frame_step=3，效果可接受 | 无需进一步优化 |
| video_quality 已用 sample_fps=2，效果可接受 | 无需进一步优化 |

## 运行命令

```bash
# 完整运行
conda run -n sdk python pipeline.py \
  --config tmp/test_config.yaml \
  --episode tmp/test_episode \
  --report tmp/test_report.json

# 仅查看执行计划
conda run -n sdk python pipeline.py \
  --config tmp/test_config.yaml \
  --episode tmp/test_episode \
  --dry-run
```
