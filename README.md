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

当前 `privacy_blur` 的 YOLO 模式使用两份专用权重：

```bash
weights/yolov8s-face-lindevs.pt
weights/yolo_lp.pt
```

如果你只想验证模型能否正常加载，可以执行：

```bash
python -c "from ultralytics import YOLO; YOLO('weights/yolov8s-face-lindevs.pt'); YOLO('weights/yolo_lp.pt'); print('ok')"
```

## 4. 安装 Privacy Blur 依赖
```bash
pip install egoblur
unzip ego_blur_face_gen2.zip -d ./operators/privacy_blur/weights
unzip ego_blur_lp_gen2.zip -d ./operators/privacy_blur/weights
```

```yaml
operators:
  privacy_blur:
    enabled: true
    detector_backend: yolo
    blur_targets: both
    yolo_face_model_path: weights/yolov8s-face-lindevs.pt
    yolo_lp_model_path: weights/yolo_lp.pt
    yolo_conf_thresh: 0.25
```

### 推荐的人脸 / 车牌 YOLO 权重

可以直接参考下面两个社区模型：

- 人脸检测：`arnabdhar/YOLOv8-Face-Detection`
  模型页：
  `https://huggingface.co/arnabdhar/YOLOv8-Face-Detection`
  权重文件：
  `https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt`
- 车牌检测：`Koushim/yolov8-license-plate-detection`
  模型页：
  `https://huggingface.co/Koushim/yolov8-license-plate-detection`
  权重文件：
  `https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt`

示例下载方式：

```bash
mkdir -p weights
wget -O weights/yolov8s-face-lindevs.pt https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8s-face-lindevs.pt
wget -O weights/yolo_lp.pt https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt
```

下载后可直接在配置里使用：

```yaml
operators:
  privacy_blur:
    enabled: true
    detector_backend: yolo
    blur_targets: both
    yolo_face_model_path: weights/yolov8s-face-lindevs.pt
    yolo_lp_model_path: weights/yolo_lp.pt
```

现在 `privacy_blur` 统一支持 `blur_targets: face|lp|both`。在 `egoblur` 后端下，这会直接控制人脸/车牌检测器是否启用；在 `yolo` 后端下，会分别调用 `yolo_face_model_path` / `yolo_lp_model_path` 对应的专用模型。默认示例里的人脸模型已经切到 `weights/yolov8s-face-lindevs.pt`。

另外，`privacy_blur` 现在支持两种检测模式：

- `detection_mode: legacy_per_frame`
  旧版逻辑，逐帧读取原视频、逐帧检测、逐帧写出。
- `detection_mode: sampling_expand`
  新版逻辑，按 `frame_sampling_step` 抽样检测；抽样帧命中后，再向前/向后逐帧补扫连续出现区间。

默认值是 `sampling_expand`。如果你想保留原来的逐帧 YOLO 行为，可以显式设成：

```yaml
operators:
  privacy_blur:
    detector_backend: yolo
    detection_mode: legacy_per_frame
    frame_sampling_step: 1
```

注意：YOLO 模式不再使用共享的通用检测权重。若你需要 YOLO 打码，请准备专用的人脸模型和车牌模型；如果只想要稳定的人脸+车牌隐私打码，默认的 `egoblur` 后端仍然是更稳妥的选择。

如果你想先快速试跑 YOLO 打码，可以把 `pipeline_config.yaml` 里的 `privacy_blur` 改成：

```yaml
operators:
  privacy_blur:
    enabled: true
    detector_backend: yolo
    blur_targets: both
    yolo_face_model_path: weights/yolov8s-face-lindevs.pt
    yolo_lp_model_path: weights/yolo_lp.pt
```
```
## 5. 系统依赖

需要系统里已经可用：

```bash
ffmpeg
ffprobe
```


## 6. 环境变量

如果需要运行 `video_segmentation` 或 `hand_analysis(method=vlm)`，现在所有 VLM 调用都统一走 `operators/caption/vlm_api.py`。默认 provider 是 DashScope。

DashScope 示例：

```bash
export DASHSCOPE_API_KEY=你的Key
```

如果不想分平台单独配环境变量，也可以统一使用 `VLM_API_KEY`；当前 provider 下若未设置专用环境变量，会回退到它。

如果你想切到火山方舟 `doubao-seed-2-0-lite-260215`，可以改成：

```bash
export VLM_API_PROVIDER=volcengine_ark
export ARK_API_KEY=你的Key
export VLM_DEFAULT_MODEL=doubao-seed-2-0-lite-260215
```

也可以直接在 `pipeline_config.yaml` 里配置：

```yaml
vlm_api_provider: volcengine_ark
vlm_api_key: "你的Key"
vlm_default_model: doubao-seed-2-0-lite-260215
```

说明：

- 业务层统一仍然使用同一个上层接口，切 provider / model 不需要改算子代码。
- 当前项目里的异步 batch 提交仅对 DashScope 兼容模式可用；如果使用方舟，请把相关算子的 `batch_enabled` 设为 `false`。

### 跑完整流水线

```bash
python pipeline.py --config pipeline_config.yaml
```
