# Egocentric-10K 技术调研报告

## 1. 数据集概览

**Egocentric-10K** 是目前规模最大的第一人称（egocentric）视频数据集，也是首个完全在真实工厂环境中采集的同类数据集。由 **Build AI** 公司发布，采用 **Apache 2.0** 许可证，允许自由商业使用与修改。

- **创始人**：Eddy Xu（2025年从哥伦比亚大学辍学创办 Build AI）
- **发布渠道**：HuggingFace（无传统学术论文，直接开源发布）
- **核心目标**：解决"物理 AI 瓶颈"——通用机器人训练所需的真实世界操作数据严重不足
- **关联研究**：EgoScale（arXiv:2602.16710, 2026年2月），由 NVIDIA / UC Berkeley / UMD 合作，利用 20,854 小时第一人称视频验证了人类数据规模与机器人操作性能之间的对数线性缩放关系（R² = 0.9983）
- **后续版本**：已扩展为 **Egocentric-100K**（100,000 小时，14,228 名工人）

## 2. 数据规模

| 指标 | 数值 |
|------|------|
| 总时长 | 10,000 小时 |
| 总帧数 | 10.8 亿 |
| 视频片段数 | 192,900 |
| 中位片段时长 | 180 秒 |
| 工人数 | 2,138 |
| 工厂数 | 85 |
| 人均录制时长 | 4.68 小时 |
| 存储总量 | 16.4 TB |

## 3. 数据格式与技术参数

### 3.1 视频参数

| 属性 | 值 |
|------|-----|
| 视频编码 | H.265 / HEVC，MP4 容器 |
| 分辨率 | 1920 × 1080 (1080p) |
| 帧率 | 30 fps |
| 视场角 | 水平 128°，垂直 67° |
| 音频 | 无 |
| 采集设备 | 单目头戴式相机（Build AI Gen 1） |
| 相机模型 | OpenCV fisheye（Kannala-Brandt 等距投影），4 个畸变系数 |
| 数据模态 | 仅 RGB 视频（无 IMU / 深度 / 音频） |

### 3.2 相机内参

每个工人配备独立的 `intrinsics.json`，包含鱼眼相机标定参数：

```json
{
  "model": "fisheye",
  "image_width": 1920,
  "image_height": 1080,
  "fx": 1030.59,
  "fy": 1032.82,
  "cx": 966.69,
  "cy": 539.69,
  "k1": -0.1166,
  "k2": -0.0236,
  "k3": 0.0694,
  "k4": -0.0463
}
```

## 4. 目录结构

采用 **WebDataset** 格式组织，每个 `.tar` 分片不超过 1GB：

```
builddotai/Egocentric-10K/
├── factory_001/ ... factory_085/
│   └── workers/
│       ├── worker_001/
│       │   ├── intrinsics.json              ← 相机内参
│       │   ├── factory001_worker001_part00.tar   ← ≤1GB 分片
│       │   └── factory001_worker001_part01.tar
│       └── worker_NNN/
```

每个 `.tar` 分片内部为配对的视频文件与元数据文件：

```
factory001_worker001_part00.tar
├── factory001_worker001_00001.mp4    ← 视频片段
├── factory001_worker001_00001.json   ← 元数据
├── factory001_worker001_00002.mp4
├── factory001_worker001_00002.json
└── ...
```

## 5. 标注格式

### 5.1 主数据集元数据

数据集本身**不包含动作标签、分割掩码或任务标注**，仅有结构化元数据：

```json
{
  "factory_id": "factory_002",
  "worker_id": "worker_002",
  "video_index": 0,
  "duration_sec": 1200.0,
  "width": 1920,
  "height": 1080,
  "fps": 30.0,
  "size_bytes": 599697350,
  "codec": "h265"
}
```

### 5.2 评估标注集（Egocentric-10K-Evaluation）

独立发布的评估数据集，对 30,000 个采样帧使用 **Gemini 2.5 Flash** 进行自动标注：

| 字段 | 类型 | 描述 |
|------|------|------|
| `frame_id` | UUID 字符串 | 帧唯一标识符 |
| `image` | ImageObject | 提取的帧图像 |
| `source_dataset` | 枚举 | "egocentric-10k" / "ego4d" / "epic-kitchens" |
| `hand_count` | Int32 | 可见手数量（0 / 1 / 2） |
| `active_labor` | 枚举 | 是否在主动操作（"yes" / "no"） |

## 6. 数据质量特征

与主流第一人称数据集对比，Egocentric-10K 在手部可见性和主动操作密度上具有显著优势：

| 数据集 | ≥1 只手可见 | 主动操作帧占比 |
|--------|------------|---------------|
| **Egocentric-10K** | **96.42%** | **91.66%** |
| EPIC-KITCHENS-100 | 90.37% | 85.04% |
| Ego4D | 67.33% | 50.07% |

这是因为工厂环境中工人几乎持续进行手动操作，数据中手-物交互的密度远高于日常生活场景。

## 7. 数据采集与处理流程

- **采集方式**：工人在正常工作期间佩戴头戴式相机，被动采集（无额外指令）
- **采集场景**：零件加工、分拣、装配、包装、质量检测等工厂手工操作
- **预处理**：H.265 编码 → 按工厂/工人组织 → 切分为 ≤1GB 的 WebDataset tar 分片
- **相机标定**：每个工人独立标定鱼眼相机内参
- **评估标注**：通过 Gemini 2.5 Flash 结构化 prompt 自动标注手部数量和操作状态

## 8. 支持的任务与应用场景

| 任务 | 说明 |
|------|------|
| 机器人操作预训练 | 学习手-物交互先验，用于灵巧操作（EgoScale 验证 55%+ 任务完成率提升） |
| 人-机器人技能迁移 | 从人类第一人称视频中学习操作技能并迁移到机器人 |
| 视觉里程计 / 3D 感知 | 利用标定的相机内参进行几何推理 |
| 手部检测与跟踪 | 高密度手部可见帧提供充足训练数据 |
| 工业动作识别 | 工厂场景下的操作分类与理解 |

## 9. 数据加载方式

### 9.1 HuggingFace Streaming（无需完整下载 16.4TB）

```python
from datasets import load_dataset, Features, Value

features = Features({
    'mp4': Value('binary'),
    'json': {
        'factory_id': Value('string'),
        'worker_id': Value('string'),
        'video_index': Value('int64'),
        'duration_sec': Value('float64'),
        'width': Value('int64'),
        'height': Value('int64'),
        'fps': Value('float64'),
        'size_bytes': Value('int64'),
        'codec': Value('string')
    },
    '__key__': Value('string'),
    '__url__': Value('string')
})

dataset = load_dataset(
    "builddotai/Egocentric-10K",
    streaming=True,
    features=features
)
```

### 9.2 FiftyOne 子集快速体验

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# 加载 416 个片段的子集（Factory 51）
dataset = load_from_hub("Voxel51/Egocentric_10K_subset")
```

## 10. 关键资源链接

| 资源 | 地址 |
|------|------|
| 主数据集 | `huggingface.co/datasets/builddotai/Egocentric-10K` |
| 评估集 | `huggingface.co/datasets/builddotai/Egocentric-10K-Evaluation` |
| 100K 扩展版 | `huggingface.co/datasets/builddotai/Egocentric-100K` |
| 快速体验子集 | `huggingface.co/datasets/Voxel51/Egocentric_10K_subset` |
| EgoScale 论文 | `arxiv.org/abs/2602.16710` |

## 11. 与 Ego-X 项目的关联性分析

| 维度 | Egocentric-10K | Ego-X_Operator |
|------|---------------|----------------|
| 场景 | 工厂手工操作 | 实验室手工任务（FusionX） |
| 视角 | 第一人称 | 第一人称 |
| 模态 | 仅 RGB 视频 | RGB 视频 + IMU |
| 标注 | 无动作标注 | SOP 步骤切分 + 原子动作标注 |
| 处理方式 | 原始视频分片存储 | VLM 滑窗检测 + IMU 校准切分 |

**结论**：

- Egocentric-10K 可作为**预训练数据源**，增强 VLM 对手部操作场景的理解能力
- 但**无法直接用于**当前的 `segment_v2t` 流程，因为缺少时序动作标注和 IMU 信号
- 其 WebDataset 分片格式和流式加载方式值得借鉴，适合大规模数据管理
