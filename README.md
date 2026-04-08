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

如果需要运行手部检测 `hand_analysis(method=yolo)`：

```bash
pip install torch torchvision ultralytics
```

如果你机器有 CUDA，请按你的 CUDA 版本安装对应的 PyTorch 版本。

## 4. 安装 Privacy Blur 依赖
```bash
pip install egoblur
unzip ego_blur_face_gen2.zip -d ./operators/privacy_blur/weights
unzip ego_blur_lp_gen2.zip -d ./operators/privacy_blur/weights
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
