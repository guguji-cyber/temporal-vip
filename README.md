# Temporal-VIP: Video Important Person Identification

Code for **"Mining Multi-Modality Spatio-Temporal Cues for Video Important Person Identification"**.

**VIP-Net** identifies the *Video Important Person* — the single individual who holds social
dominance over a multi-person video clip — from static spatial cues, dynamic temporal cues, and
textual descriptions, and generates a natural-language rationale for its decision.

| Result (Temporal-VIP test) | Rank@1 | Rank@2 | Rank@3 |
|---|---|---|---|
| VIP-Net (Ours) | **67.3** | **83.2** | **94.1** |
| Max Centrality (best heuristic) | 50.7 | 76.8 | 92.0 |
| Best SOTA adaptation | 53.9 | 78.8 | 92.0 |

---

## Authors & attribution

**Paper authors:** Xiao Wang, Weijie Jin, Minglei Yang, Bin Yang, Wenke Huang,
Zheng Wang, Xin Xu, Mang Ye.

The original implementation and dataset are maintained at
[`yml2002/vip`](https://github.com/yml2002/vip) and
[`yml2002/Temporal-VIP`](https://huggingface.co/datasets/yml2002/Temporal-VIP).
This repository redistributes the code with the authors' permission and links to the
original dataset; it is not an independent reimplementation.

---

## Links

| Resource | URL |
|---|---|
| **Dataset** (87.3 GB) | https://huggingface.co/datasets/yml2002/Temporal-VIP |
| **Original code** | https://github.com/yml2002/vip |

> The dataset is **87.3 GB** (train 50.8 GB / val 16.7 GB / test 16.7 GB; 9249 clips).
> It exceeds GitHub's repository and Git-LFS limits and is therefore hosted on Hugging Face
> rather than in this repository. See [DATASET.md](DATASET.md) for the download command,
> directory layout and the `.npz` schema.

---

## Model overview

```
frames ─┬─► Spatial cues   (area, centrality, clarity)  ──┐
        │      K_s = 3, D = 32                            │
        │                                                 │
        ├─► Temporal cues  (action, lip movement)        ├─► Temporal Importance Rectifier ─► Ranking
        │      K_t = 2, D = 512                           │   Eq.(8)  intra-personal gate
        │                                                 │   Eq.(9)  semantic gate (text)
        └─► Text cues      (BERT-base-chinese)           ──┘   Eq.(10) spatial→temporal cross-attn
               per-person, D = 512                             Eq.(11) attention pooling
                                                               Eq.(12) inter-personal Transformer
```

Key components

| File | Role |
|---|---|
| `feature_extraction/static_extractor.py` | area / centrality / clarity, ResNet18 + MediaPipe FaceMesh |
| `feature_extraction/dynamic_extractor.py` | `Resnet3D` (action) + `LipEncoder` (40-d FaceMesh landmarks) |
| `feature_extraction/text_extractor.py` | BERT-base-chinese → 512-d, per-person + scene cross-attention |
| `models/temporal_alignment.py` | `SubFeatureAggregator` (Eq.8), cross-modal attention (Eq.9–10), `AttentionPooling` (Eq.11) |
| `models/enhanced_transformer_model.py` | full pipeline, inter-personal Transformer (Eq.12), loss composition |
| `train/losses.py` | classification + text similarity + contrastive + ranking + L2 |
| `configs/config.py` | every hyper-parameter and CLI override |

---

## Installation

### 1. Environment

```bash
conda env create -f npz_llm_model/environment.yml
conda activate vip
```

Tested with Python 3.10, PyTorch 2.14.0+cu130, transformers 4.57.6, mediapipe 0.10.21.

### 2. Pretrained encoders (required)

`configs/config.py` expects a local Hugging Face cache laid out as:

```
code/src/models/huggingface/hub/
├── models--google-bert--bert-base-chinese/snapshots/<hash>/
├── models--fnlp--bart-base-chinese/snapshots/<hash>/
└── models--sentence-transformers--distiluse-base-multilingual-cased-v1/snapshots/<hash>/
```

Download them with the Hugging Face CLI (≈1.5 GB total):

```bash
export HF_HOME=<repo>/code/src/models/huggingface
huggingface-cli download google-bert/bert-base-chinese
huggingface-cli download fnlp/bart-base-chinese
huggingface-cli download sentence-transformers/distiluse-base-multilingual-cased-v1
```

> `config.py` raises `ValueError: 未找到BART模型` if these are missing. The lookup is relative to
> the **current working directory** (`base_dir = os.getcwd()`), so always launch from the
> repository root.

---

## Dataset

The annotated **Temporal-VIP** dataset lives on Hugging Face:

```bash
huggingface-cli download yml2002/Temporal-VIP --repo-type dataset --local-dir data/Temporal-VIP
```

Expected layout after download and preprocessing:

```
code/data/vip/
├── preprocessed_fixed/
│   ├── train/   # 5549 clips
│   ├── val/     # 1850 clips
│   └── test/    # 1850 clips
└── llm_marked_videos_description/   # LLM-refined clip descriptions (JSON)
```

Each `.npz` clip contains:

| Key | Shape | Meaning |
|---|---|---|
| `frames` | `120 × 192 × 336 × 3` (uint8) | raw frames |
| `bboxes` | `120 × 20 × 4` | per-frame person boxes (20 slots) |
| `person_mask` | `120 × 20` | valid person slots |
| `frame_mask` | `120` | valid frames |
| `target_index` | scalar | index of the annotated VIP |
| `video_id`, `scene_category`, `original_ids`, `person_ids` | — | metadata |

> **Frames must stay in uint8 (0–255) scale.** `extract_lip_sequence` casts with
> `.astype(np.uint8)`, so feeding pre-normalised `[0,1]` tensors silently zeroes out the entire
> lip branch.

### Optional: precompute caches

MediaPipe FaceMesh is the slowest part of the pipeline. Precompute once, then reuse:

```bash
python code/src/npz_llm_model/baselines/precompute_lip.py    # lip landmarks  → cache/lip_raw/
python code/src/npz_llm_model/baselines/precompute_static.py # static features
```

`baselines/verify_lip_cache_strict.py` checks the cache is bit-identical to the live path.

---

## Training

Launch **from the repository root** so `base_dir` resolves correctly:

```bash
export PYTHONPATH=$PWD/code/src
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# single GPU
python code/src/npz_llm_model/main.py \
    --mode train \
    --batch_size 16 \
    --epochs 50 \
    --num_workers 8 \
    --feature_selection static,dynamic,text \
    --fusion_type transformer \
    --lip_cache_dirs "$PWD/cache/lip_raw/train,$PWD/cache/lip_raw/val" \
    --is_ablation

# 2 GPUs
python -m torch.distributed.run --nproc_per_node=2 --standalone \
    code/src/npz_llm_model/main.py --mode train --batch_size 16 --epochs 50 \
    --num_workers 8 --feature_selection static,dynamic,text \
    --fusion_type transformer --is_ablation
```

### Reported hyper-parameters

| Setting | Value | Source |
|---|---|---|
| Optimizer | Adam, lr `5e-5`, wd `1e-4`, grad-clip `1.0` | `configs/config.py` |
| LR schedule | `CosineAnnealingWarmRestarts(T_0=5, T_mult=2, eta_min=lr*0.01)` | `train/trainer.py` |
| Layer-wise LR | backbone `1×`, text `2×`, fusion `5×` | `train/trainer.py` |
| Batch size | 16 | CLI |
| Epochs | 50 | CLI |
| Dropout | 0.3 | `configs/config.py` |
| Fusion width `D` | 256 | `configs/config.py` |
| Attention heads | 8 (rectifier), 4 (cue encoders) | `configs/config.py` |
| Person slots `N` | 20 | `configs/config.py` |
| Stored frames `T` | 120 (84.4 valid on average) | `configs/config.py` |
| Crop size | 32 × 32 | `dynamic_extractor.py` |
| τ_c (classification) | 0.5 | `configs/config.py` |
| τ_cont (contrastive) | 0.07 | `configs/config.py` |
| λ_cls / λ_text / λ_cont / λ_rank / λ_reg | 1.0 / 0.5 / 0.3 / 0.2 / 0.005 | `configs/config.py` |

---

## Evaluation & prediction

```bash
python code/src/npz_llm_model/main.py \
    --mode predict \
    --model_path outputs/<run>/checkpoints/main_model/best.pth \
    --predict_split test \
    --batch_size 8
```

Outputs per-clip predictions plus aggregate Rank@1/2/3 to `logs/preds_logs/`.

---

## Ablations

```bash
# remove one cue
--ablate_feature area        # or centrality | clarity | action | speech
# remove a whole modality
--feature_selection static,dynamic          # w/o Text
--feature_selection dynamic,text            # w/o Spatial
--feature_selection static,text             # w/o Temporal
# fusion variants
--fusion_type concat | mlp | gated | transformer
# replace temporal cues with optical flow
--replace_temporal_with_flow --optical_flow_backend farneback   # or raft
# loss-weight sweeps
--lambda_cont 0 --lambda_text 0 --lambda_reg 0.0005 --temperature_scale 0.5
```

> **Note on `--ablate_feature`.** `feature_names` maps to the emitted cue channels:
> `speech` is an alias of `lip`. Removing a cue drops that channel from the stacked
> `[B,T,N,K,D]` tensor, and the downstream `SubFeatureAggregator` is rebuilt with the
> matching `K`, so the ablation is faithful.

### Baselines

| Script | Purpose |
|---|---|
| `baselines/train_tracklet.py --temporal {transformer,lstm}` | controlled tracklet baselines |
| `baselines/tracklet_baseline.py` | tracklet feature extraction |
| `baselines/eval_tracklet.py` | baseline evaluation |
| `baselines/ablate_eq12.py` | Eq.(12) inter-personal Transformer replay ablation |
| `baselines/fast_clips.py` | fast clip/report utilities |

---

## Repository layout

```
npz_llm_model/
├── main.py                       # train / predict entry point
├── environment.yml
├── configs/config.py             # all hyper-parameters + CLI
├── data_processing/              # dataset, dataloader, splits
├── feature_extraction/
│   ├── static_extractor.py       # area / centrality / clarity
│   ├── dynamic_extractor.py      # Resnet3D (action) + LipEncoder
│   └── text_extractor.py         # BERT-base-chinese
├── models/
│   ├── enhanced_transformer_model.py
│   ├── temporal_alignment.py     # Eq.8-11
│   ├── enhanced_fusion.py
│   └── contrastive_module.py
├── text_processing/              # explanation generation
├── train/                        # trainer, evaluator, predictor, losses
├── utils/                        # metrics, logging, visualization
└── baselines/                    # controlled baselines & cache builders
```

---

## Citation

```bibtex
@article{wang2026temporalvip,
  title   = {Mining Multi-Modality Spatio-Temporal Cues for Video Important Person Identification},
  author  = {Wang, Xiao and Jin, Weijie and Yang, Minglei and Yang, Bin and
             Huang, Wenke and Wang, Zheng and Xu, Xin and Ye, Mang},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026}
}
```

## License

Released for academic research use. Please also respect the terms of the Temporal-VIP dataset
on Hugging Face.
