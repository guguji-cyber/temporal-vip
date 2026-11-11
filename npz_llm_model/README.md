# VIP-Net (npz_llm_model)

VIP-Net 是一个用于视频重要人物检测（Video Important Person Detection）的多模态模型实现。
该实现融合了静态视觉特征、动态时序特征和文本语义特征，并支持训练、评估与预测。

## 1. 功能概览

- 多模态输入：视觉 + 文本
- 多分支特征提取：
  - 静态特征：位置/面积/清晰度/中心性
  - 动态特征：动作/说话（可切换光流替代时序）
  - 文本特征：人物描述 + 场景描述
- 融合建模：增强型 Transformer + 时空对齐 + 对比学习
- 任务模式：
  - `train` 训练
  - `predict` 推理

## 2. 当前目录结构

```text
npz_llm_model/
├── README.md
├── main.py
├── .gitignore
├── configs/
├── data_processing/
├── feature_extraction/
├── models/
├── text_processing/
├── train/
└── utils/
```

说明：`docs/` 已在 `.gitignore` 中忽略，不作为本仓库提交内容。

## 3. 环境要求

- Python 3.9+（建议 3.10）
- PyTorch（建议 CUDA 版本）
- 依赖模块（按代码引用）：
  - `torch`, `numpy`, `tqdm`
  - `opencv-python`
  - `mediapipe`
  - `transformers`, `sentence-transformers`
  - 其他训练常用包

建议使用你已有环境（如 `vip` conda 环境）。

### 3.1 使用 environment.yml（推荐）

仓库已提供 `environment.yml`，环境名固定为 `vip`。

首次创建环境：

```bash
conda env create -f environment.yml
conda activate vip
```

后续同步依赖更新：

```bash
conda env update -f environment.yml --prune
conda activate vip
```

## 4. 数据与模型路径说明

当前 `configs/config.py` 里的路径是按你原工程根目录写的：

- `code/data/vip/preprocessed_fixed`
- `code/data/vip/llm_marked_videos_description`
- `code/src/models/huggingface/hub/...`

也就是说，直接单独克隆这个子目录后，默认路径通常不可用。

你有两种方式：

1. 保持与原工程一致的目录层级（推荐）
2. 修改 `configs/config.py` 中的 `base_dir/npz_dir/json_dir/models_base` 到你自己的路径

## 5. 快速开始

### 5.1 训练

```bash
python main.py --mode train
```

常用参数示例：

```bash
python main.py \
  --mode train \
  --batch_size 32 \
  --epochs 50 \
  --learning_rate 5e-5
```

### 5.2 预测

```bash
python main.py --mode predict --model_path <checkpoint_path>
```

带重复统计（mean/std）示例：

```bash
python main.py \
  --mode predict \
  --model_path <checkpoint_path> \
  --predict_repeats 5 \
  --predict_split val
```

## 6. 关键脚本职责

- `main.py`：入口，参数解析，调度 train/predict
- `configs/config.py`：全部配置、路径、超参数、命令行参数
- `data_processing/dataset.py`：读取 NPZ/JSON 样本
- `data_processing/data_loader.py`：DataLoader 与 batch 组装
- `models/enhanced_transformer_model.py`：主模型
- `train/trainer.py`：训练循环、优化器、调度器、checkpoint
- `train/evaluator.py`：验证与指标
- `train/predictor.py`：推理与结果导出

