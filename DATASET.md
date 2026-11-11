# 数据集说明

完整数据集 **87.3 GB**（train 50.8 GB / val 16.7 GB / test 16.7 GB，共 9249 个 `.npz`），
超出 GitHub 仓库与 Git-LFS 的限制，因此托管在 Hugging Face，不随本仓库分发。

数据集与原始代码由论文作者维护：

- 数据集：https://huggingface.co/datasets/yml2002/Temporal-VIP
- 原始代码：https://github.com/yml2002/vip

## 下载

```bash
pip install huggingface_hub
huggingface-cli download yml2002/Temporal-VIP \
    --repo-type dataset \
    --local-dir data/Temporal-VIP
```

## 放置位置

`npz_llm_model/configs/config.py` 用**相对路径**解析数据目录：

```python
self.base_dir = os.getcwd()                                     # 必须从仓库根目录启动
self.npz_dir  = os.path.join(self.base_dir, 'code/data/vip/preprocessed_fixed')
self.json_dir = os.path.join(self.base_dir, 'code/data/vip/llm_marked_videos_description')
```

因此下载后请整理成：

```
<repo-root>/
└── code/
    ├── src/
    │   ├── npz_llm_model/          # 代码
    │   └── models/
    │       └── huggingface/hub/    # BERT / BART / SBERT（另需下载，约 1.5 GB）
    └── data/
        └── vip/
            ├── preprocessed_fixed/
            │   ├── train/          # 5549 clips
            │   ├── val/            # 1850 clips
            │   └── test/           # 1850 clips
            └── llm_marked_videos_description/   # 片段描述的 JSON
```

> 本仓库的代码放在 `npz_llm_model/`，而 `config.py` 期望的是 `code/src/npz_llm_model/`。
> 两种做法任选：
> 1. 克隆后把 `npz_llm_model/` 移到 `code/src/npz_llm_model/`；
> 2. 或保持本仓库结构，在仓库根目录建软链接：
>    ```bash
>    mkdir -p code/src && ln -s ../../npz_llm_model code/src/npz_llm_model
>    ```

## 每个 `.npz` 的字段

| 键 | 形状 | 说明 |
|---|---|---|
| `frames` | `120 × 192 × 336 × 3`，uint8 | 原始帧 |
| `bboxes` | `120 × 20 × 4` | 逐帧人物框（20 个人物槽） |
| `person_mask` | `120 × 20` | 有效人物槽 |
| `frame_mask` | `120` | 有效帧 |
| `target_index` | 标量 | 标注 VIP 的索引 |
| `video_id` / `scene_category` / `original_ids` / `person_ids` | — | 元信息 |

### ⚠️ 帧必须是 uint8（0–255）

`extract_lip_sequence` 内部用 `.astype(np.uint8)` 取像素。若事先把 frames 归一化到
`[0,1]`，会被全部截断为 0，**导致唇动分支静默失效**。验证脚本里踩过这个坑。

## 加速：预计算缓存

MediaPipe FaceMesh 是整条流水线最慢的一环。建议先算一次：

```bash
python code/src/npz_llm_model/baselines/precompute_lip.py     # 唇动关键点 → cache/lip_raw/
python code/src/npz_llm_model/baselines/precompute_static.py  # 静态特征
```

用 `baselines/verify_lip_cache_strict.py` 可以验证缓存与实时路径逐位等价
（action / lip / dynamic / logits 的 `max|diff|` 均为 0）。
