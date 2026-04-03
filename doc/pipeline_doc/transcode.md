# 视频格式转换算子 (transcode)

将视频统一为采购方要求的编码格式、容器格式、分辨率、码率，确保转换过程无损。

## 依赖

- **ffmpeg** >= 4.0（须在 PATH 中）
- **ffprobe**（ffmpeg 附带）
- Python 3.8+，无额外 pip 依赖

## 快速开始

```bash
# 单文件 — 仅换容器（比特级无损，最快）
python -m transcode.transcode input.avi -o output.mp4

# 单文件 — 转编码（数学无损）
python -m transcode.transcode input.mp4 -o output.mkv --codec ffv1

# 批量 — 整个目录
python -m transcode.transcode ./episodes -o ./delivered --batch --codec h264 --container mp4

# 预览命令（不执行）
python -m transcode.transcode input.mp4 -o output.mkv --codec h265 --dry-run
```

## 无损策略

| 场景 | 策略 | 无损级别 |
|------|------|----------|
| 仅换容器（如 AVI→MP4） | `stream copy` (`-c copy`) | **比特级无损** — 视频流零修改 |
| 换编码（如 H.264→H.265） | 目标编码器的无损模式 | **数学无损** — 解码后像素值相同 |
| 换编码 + 换分辨率 | 高质量 Lanczos 缩放 + 无损编码 | **编码环节无损** — 缩放不可逆但编码零损失 |
| 指定码率 | 视觉无损级别（最高质量预设） | **视觉无损** — 人眼不可辨，非数学等价 |

> 核心原则：不指定 `--bitrate` 时，编码环节**一定是数学无损**的。

## 可选项

### 编码格式 (`--codec`)

| 值 | 编码器 | 无损模式 | 适用场景 |
|----|--------|----------|----------|
| `h264` | libx264 | CRF=0, veryslow | **兼容性最佳**，所有播放器支持 |
| `h265` / `hevc` | libx265 | lossless=1, veryslow | 压缩率比 H.264 高 ~30-50% |
| `ffv1` | FFV1 | 天然无损 | **存档首选**，开源、可验证 |
| `prores` | ProRes 4444 XQ | 近无损 (qscale=0) | 专业后期工作流 |
| `vp9` | libvpx-vp9 | lossless=1 | Web/开放格式 |
| `av1` | SVT-AV1 | CRF=0 (非严格无损) | 下一代高压缩 |
| `copy` | Stream Copy | 比特级无损 | **仅换容器**，速度最快 |

### 容器格式 (`--container`)

| 值 | 说明 | 兼容编码 |
|----|------|----------|
| `mp4` | 最广泛兼容 | H.264, H.265, AV1 |
| `mkv` | 功能最全 | **所有编码** |
| `mov` | Apple 生态 | H.264, H.265, ProRes |
| `avi` | 传统格式 | H.264, FFV1 |
| `webm` | Web 专用 | VP9, AV1 |
| `mxf` | 广播级 | H.264, H.265, ProRes |

### 其他参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--resolution WxH` | 目标分辨率 | `--resolution 1920x1080` |
| `--bitrate RATE` | 视频码率（启用后切为有损） | `--bitrate 50M` |
| `--fps FPS` | 目标帧率 | `--fps 30` |
| `--pix-fmt FMT` | 像素格式 | `--pix-fmt yuv420p` |
| `--dry-run` | 仅预览 ffmpeg 命令 | |
| `--report PATH` | 输出 JSON 报告 | `--report report.json` |

## 常见采购方规格组合

### 1. 高兼容交付（推荐）

```bash
python -m transcode.transcode ./episodes -o ./delivered --batch \
    --codec h264 --container mp4 --resolution 1920x1080 --pix-fmt yuv420p
```

适用：大多数采购方要求，兼容所有主流播放器和平台。

### 2. 无损存档

```bash
python -m transcode.transcode ./episodes -o ./archive --batch \
    --codec ffv1 --container mkv
```

适用：长期保存原始数据，可随时从存档重新转出任意格式。

### 3. 专业后期交付

```bash
python -m transcode.transcode ./episodes -o ./post --batch \
    --codec prores --container mov
```

适用：需要在 DaVinci Resolve、Final Cut Pro 等专业软件中编辑。

### 4. Web 发布

```bash
python -m transcode.transcode ./episodes -o ./web --batch \
    --codec vp9 --container webm --resolution 1280x720
```

### 5. 仅换容器（最快）

```bash
python -m transcode.transcode ./episodes -o ./remuxed --batch \
    --codec copy --container mp4
```

比特级无损，速度接近文件复制。

## 输出结构

批量模式保持目录结构：

```
input_dir/                    output_dir/
  task1/                        task1/
    rgb.mp4          →            rgb.mkv
  task2/                        task2/
    rgb.avi          →            rgb.mkv
```

## JSON 报告

使用 `--report` 生成转码结果报告：

```json
[
  {
    "status": "ok",
    "input": "episodes/task1/rgb.mp4",
    "output": "delivered/task1/rgb.mp4",
    "reencode": true,
    "input_size_mb": 245.3,
    "output_size_mb": 312.1,
    "input_codec": "h264",
    "output_codec": "h264",
    "input_resolution": "3840x2160",
    "output_resolution": "1920x1080",
    "duration_diff_sec": 0.0
  }
]
```

`duration_diff_sec` > 0.5 时会发出警告，提示检查输出完整性。

## 注意事项

1. **无损 ≠ 体积不变**：数学无损编码（如 FFV1）的输出通常比有损源文件更大，这是正常的。
2. **分辨率变更不可逆**：缩放是有损操作。如需保留原始分辨率，请省略 `--resolution`。
3. **指定码率即有损**：`--bitrate` 会关闭无损模式，切换为视觉无损级别。如需数学无损，不要指定码率。
4. **音频默认 copy**：音频流尽量直接复制。仅当音频编码与目标容器不兼容时才转码（如转 webm 时非 opus/vorbis 音频会转为 opus 192k）。
5. **元数据保留**：所有原始元数据（时间戳、旋转信息等）通过 `-map_metadata 0` 保留。
