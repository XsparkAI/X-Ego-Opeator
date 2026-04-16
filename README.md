# Ego-X Operator 安装教程

## 1. 创建 conda 环境

推荐使用 Python 3.10。

```bash
cd /home/pc/Desktop/zijian/ego/Ego-X_Operator
conda create -n ego_operator python=3.10 -y
conda activate ego_operator
```

## 2. 安装基础依赖

```bash
pip install -U pip setuptools wheel
pip install pyyaml numpy opencv-python pillow pandas scipy pyarrow tqdm dashscope openai httpx[socks]
```

## 3. 安装 YOLO / Torch 依赖

如果需要运行手部检测 `hand_analysis(method=yolo)`，或隐私打码的 YOLO 模式 `privacy_blur(detector_backend=yolo)`：

```bash
pip install torch torchvision ultralytics
```

如果你机器有 CUDA，请按你的 CUDA 版本安装对应的 PyTorch 版本。

建议额外检查一下安装是否成功：

```bash
python -c "import torch, ultralytics; print(torch.__version__); print(ultralytics.__version__)"
```

### 下载 YOLO 权重

当前项目默认使用：

```bash
weights/yolo26n.pt
```

你可以手动把权重放到 `weights/` 目录，或者让 `ultralytics` 自动下载一次：

```bash
mkdir -p weights
python -c "from ultralytics import YOLO; YOLO('weights/yolo26n.pt')"
```

第一次运行会自动下载官方权重到 `weights/yolo26n.pt`。

如果你只想验证模型能否正常加载，可以执行：

```bash
python -c "from ultralytics import YOLO; YOLO('weights/yolo26n.pt'); print('ok')"
```

## 4. 安装 Privacy Blur 依赖
```bash
pip install egoblur
unzip ego_blur_face_gen2.zip -d ./operators/privacy_blur/weights
unzip ego_blur_lp_gen2.zip -d ./operators/privacy_blur/weights
```

如果你要把 `yolo26n.pt` 接到打码流程里，建议把权重放在 `weights/` 目录，然后在 `pipeline_config.yaml` 里这样配置：

```yaml
operators:
  privacy_blur:
    enabled: true
    detector_backend: yolo
    blur_targets: both
    yolo_model_path: weights/yolo26n.pt
    yolo_conf_thresh: 0.25
```

现在 `privacy_blur` 统一支持 `blur_targets: face|lp|both`。在 `egoblur` 后端下，这会直接控制人脸/车牌检测器是否启用；在 `yolo` 后端下，会根据模型自带的类别名自动匹配 `face` / `license plate` 一类目标。

注意：官方 `yolo26n.pt` 是通用检测模型，不会直接检测“人脸/车牌”；更适合按 `person`、`car` 等类别整框打码。若你需要精确的人脸/车牌隐私打码，默认的 `egoblur` 后端更合适；若要在 YOLO 模式下使用 `blur_targets: face|lp|both`，请使用带有人脸、车牌类别名称的专用 YOLO 权重。

如果你想先快速试跑 YOLO 打码，可以把 `pipeline_config.yaml` 里的 `privacy_blur` 改成：

```yaml
operators:
  privacy_blur:
    enabled: true
    detector_backend: yolo
    blur_targets: both
    yolo_model_path: weights/yolo26n.pt
```
## 5. 系统依赖

需要系统里已经可用：

```bash
ffmpeg
ffprobe
```


## 6. 环境变量

如果需要运行 `video_segmentation` 或 `hand_analysis(method=vlm)`，请先设置：

```bash
export DASHSCOPE_API_KEY=你的Key
```

### 跑完整流水线

```bash
python pipeline.py --config pipeline_config.yaml
```
