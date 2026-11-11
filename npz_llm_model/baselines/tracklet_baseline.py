"""受控基线：tracklet 时序分类器（无 TIR）。

这是回复信中 AE-3 / R2-4 承诺的那个基线。设计原则是**只改架构，不改输入**：

复用（与 VIP-Net 逐位相同）
    StaticFeatureExtractor    -> [B, T, N, 3, 32]    (area, centrality, clarity)
    DynamicFeatureExtractor   -> [B, T, N, 2, 512]   (action, speech/lip)
    TextFeatureExtractor      -> [B, N, 512]         (bert-base-chinese，逐人)
    同一批数据、同一批 tracklet、同一套标注、同一个评测口径

去掉的正是 TIR
    ✗ TemporalAwareAlignment  —— 含 Eq.(9) 语义门控、Eq.(10) 人际交叉注意力、Eq.(11) 时序池化
    ✗ 人级 TransformerEncoder —— Eq.(12)，让不同人互相看
    两者都删掉：**每个人都独立打分，人之间没有任何信息交换**

替换为刻意简单的时序模型
    ✓ 每条模态的 K 个子流直接取均值（不做学习式门控 —— 越简单越好，避免把 TIR 的能力偷偷搬进来）
    ✓ 三条模态投影到同一维度后拼接 -> [B, T, N, 3D]
    ✓ 每个人独立跑一个时序编码器（Transformer 或 LSTM）
    ✓ 按有效帧做 masked mean pooling -> [B, N, D]
    ✓ 线性头 -> [B, N] 分数

静态线索缓存（可选）
    StaticFeatureExtractor.compute_static_features 里有一个人脸清晰度的三层循环，
    每片段要跑约 276 次 MediaPipe（实测 1.04 s/片段），一个 epoch 光这一项就 ~20 分钟。
    用 precompute_static.py 把三条线索的**原始量**存成 npz，训练时读盘后按真实 batch
    重做归一化，可把静态分支降到几乎为零，且与实时计算逐位一致（实测 max|diff| = 0）。
    缺失缓存时自动回退到实时计算。

用法
    from baselines.tracklet_baseline import TrackletBaseline
    model = TrackletBaseline(config, temporal="transformer")   # 或 "lstm"
    model = TrackletBaseline(config, temporal="transformer",
                             static_cache_split=["cache/static_features/train",
                                                 "cache/static_features/val"])
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from npz_llm_model.feature_extraction.static_extractor import StaticFeatureExtractor
from npz_llm_model.feature_extraction.dynamic_extractor import DynamicFeatureExtractor
from npz_llm_model.feature_extraction.text_extractor import TextFeatureExtractor
from npz_llm_model.baselines.fast_clips import extract_person_clips_fast


class TrackletBaseline(nn.Module):
    """Tracklet 时序分类器，无 TIR。"""

    def __init__(
        self,
        config: Any,
        temporal: str = "transformer",
        static_cache_split: Optional[Sequence[str]] = None,
        lip_cache_split: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if temporal not in ("transformer", "lstm"):
            raise ValueError(f"temporal must be 'transformer' or 'lstm', got {temporal!r}")
        self.config = config
        self.temporal_kind = temporal

        D = 256                      # 与 VIP-Net 的 fusion 宽度一致，保证是同一量级的容量
        self.fusion_dim = D
        n_mod = 3                    # static / dynamic / text

        # ---- 与 VIP-Net 完全相同的 cue 提取器 ----
        self.static_extractor = StaticFeatureExtractor(config)
        self.dynamic_extractor = DynamicFeatureExtractor(config)
        self.text_extractor = TextFeatureExtractor(config)

        s_dim = int(config.static_feature_dim)      # 32
        d_dim = int(config.dynamic_feature_dim)     # 512
        t_dim = int(config.text_feature_dim)        # 512
        self.s_proj = nn.Linear(s_dim, D)
        self.d_proj = nn.Linear(d_dim, D)
        self.t_proj = nn.Linear(t_dim, D)

        in_dim = n_mod * D
        drop = float(getattr(config, "dropout", 0.3))

        if temporal == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=in_dim, nhead=8, dim_feedforward=in_dim * 2,
                dropout=drop, batch_first=True, norm_first=False,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        else:  # lstm
            self.temporal = nn.LSTM(
                in_dim, in_dim // 2, num_layers=2, batch_first=True,
                bidirectional=True, dropout=drop,
            )

        self.score_head = nn.Linear(in_dim, 1)
        self.out_dim = in_dim

        # ---- 静态线索缓存（只影响速度，不影响数值）----
        self.static_cache_dirs: List[Path] = [Path(p) for p in (static_cache_split or [])]
        for d in self.static_cache_dirs:
            if not d.is_dir():
                raise FileNotFoundError(f"静态缓存目录不存在: {d}")
        self._cache_index: Optional[Dict[str, Path]] = None
        self._cache_miss = 0

        # ---- 唇动关键点缓存（同样是纯加速，逐位等价）----
        self.lip_cache_dirs: List[Path] = [Path(p) for p in (lip_cache_split or [])]
        for d in self.lip_cache_dirs:
            if not d.is_dir():
                raise FileNotFoundError(f"唇动缓存目录不存在: {d}")
        self._lip_index: Optional[Dict[str, Path]] = None
        self._lip_miss = 0

    # ------------------------------------------------------------------
    def _action_only(self, frames: torch.Tensor, bboxes: torch.Tensor,
                     person_masks: torch.Tensor, frame_masks: torch.Tensor) -> torch.Tensor:
        """复刻 dynamic_extractor.forward 的 action 分支（不含唇动、不含光流）。

        源实现（dynamic_extractor.py L522-570）：
            clips  = extract_person_clips(...)          # [B,N,3,T,32,32]
            pv     = any(person_masks, dim=1)           # [B,N]
            action = zeros(B,T,N,D)
            feats  = action_encoder(clips[pv])          # [V,T,D]
            action[b,:,n,:] = feats[idx]  for idx,(b,n) in enumerate(nonzero(pv))
        """
        clips = extract_person_clips_fast(
            frames, bboxes, person_masks, frame_masks)
        pv = torch.any(person_masks, dim=1)
        B, T, N = person_masks.shape
        D = int(self.config.dynamic_feature_dim)
        action = torch.zeros(B, T, N, D, device=frames.device, dtype=clips.dtype)
        if not pv.any():
            return action
        with torch.autocast(device_type="cuda", enabled=frames.device.type == "cuda"):
            feats = self.dynamic_extractor.action_encoder(clips[pv])
        for idx, (b, n) in enumerate(torch.nonzero(pv, as_tuple=False)):
            action[b, :, n, :] = feats[idx]
        return action

    # ------------------------------------------------------------------
    def _build_cache_index(self) -> Dict[str, Path]:
        """扫描所有静态缓存目录，建立 {video_id: npz路径}。文件名在各 split 间不重复。"""
        idx: Dict[str, Path] = {}
        for d in self.static_cache_dirs:
            for f in d.glob("*.npz"):
                idx[f.stem] = f
        return idx

    def _static_raw_from_cache(self, video_ids: Sequence[str]) -> Optional[Dict[str, torch.Tensor]]:
        """按 video_id 读缓存的**原始量**；任一缺失则返回 None（整批回退实时计算）。"""
        if not self.static_cache_dirs:
            return None
        if self._cache_index is None:
            self._cache_index = self._build_cache_index()

        out: Dict[str, list] = {}
        for vid in video_ids:
            path = self._cache_index.get(vid)
            if path is None:
                self._cache_miss += 1
                return None
            with np.load(path) as z:
                for name in self.config.static_extractor["feature_names"]:
                    key = f"{name}_raw"
                    if key not in z:
                        raise KeyError(f"缓存 {path} 缺少 {key}（请用新版 precompute_static.py 重新生成）")
                    out.setdefault(name, []).append(torch.from_numpy(z[key]).float())
        return {k: torch.stack(v, dim=0) for k, v in out.items()}

    @staticmethod
    def _normalize_static(raw: Dict[str, torch.Tensor],
                          masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """逐位复刻 compute_static_features 的归一化后半段。

        log1p -> 无效位置填 batch 均值 -> clamp(0,1.6) -> ÷1.6
        注意填充用的是**整个 batch** 的均值，必须在这里按真实 batch 做，
        不能在离线阶段按 B=1 预算好（那样 clarity 会偏 0.44）。
        """
        valid = masks > 0
        out = {}
        for name, x in raw.items():
            v = torch.log1p(x)
            mean_valid = v[valid].mean() if valid.any() else torch.zeros((), device=v.device)
            v = torch.where(valid, v, mean_valid)
            out[name] = torch.clamp(v, min=0.0, max=1.6) / 1.6
        return out

    # ------------------------------------------------------------------
    def _lip_encode_from_cache(self, video_ids: Sequence[str], person_masks: torch.Tensor
                               ) -> Optional[torch.Tensor]:
        """用缓存的关键点序列重建 dynamic_extractor 的唇动分支 [B,T,N,D]。

        逐位复刻源实现（dynamic_extractor.py L575-596）：
            for (b,n) 有效: lip_seq = extract_lip_sequence(...)
                             if any(lip_seq != 0): 收进 valid_lip_sequences
            mask = frame_masks[b] & person_masks[b,:,n]      -> src_key_padding_mask = ~mask
            全真行修第一个为 False（避免整行被 mask 掉）
            projected = lip_encoder(stack(seqs), mask=...)
            lip[b,:,n,:] = projected[idx]
        缓存里只存了非零序列，顺序按 n 升序，与源实现的遍历顺序一致。
        """
        if not self.lip_cache_dirs:
            return None
        if self._lip_index is None:
            idx: Dict[str, Path] = {}
            for d in self.lip_cache_dirs:
                for f in d.glob("*.npz"):
                    idx[f.stem] = f
            self._lip_index = idx

        frames = person_masks.shape[1]
        device = person_masks.device
        D = int(self.config.dynamic_feature_dim)
        B, T, N = person_masks.shape
        lip = torch.zeros(B, T, N, D, device=device)

        pv = person_masks.any(dim=1)                       # [B,N]
        seqs, indices, masks = [], [], []
        for b, vid in enumerate(video_ids):
            path = self._lip_index.get(vid)
            if path is None:
                self._lip_miss += 1
                return None
            with np.load(path) as z:
                for n in range(N):
                    if not pv[b, n]:
                        continue
                    key = f"lip_{n}"
                    if key not in z:
                        continue                            # 源实现里全零序列会被跳过
                    seq = torch.from_numpy(z[key]).float()
                    if seq.shape[0] != T:
                        raise ValueError(f"{vid} lip_{n} 长度 {seq.shape[0]} != T={T}")
                    seqs.append(seq.to(device))
                    indices.append((b, n))
                    masks.append(person_masks[b, :, n])   # 源实现只用 person_masks，不用 frame_masks

        if not seqs:
            return lip

        stacked = torch.stack(seqs)                        # [V,T,40]
        m = torch.stack(masks)                             # [V,T] bool
        lip_mask = ~m
        all_true = lip_mask.all(dim=1)
        if all_true.any():
            lip_mask[all_true, 0] = False
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            projected = self.dynamic_extractor.lip_encoder(stacked, mask=lip_mask)
        for i, (b, n) in enumerate(indices):
            lip[b, :, n, :] = projected[i]
        return lip

    # ------------------------------------------------------------------
    def _encode_static(self, raw_features: Dict[str, torch.Tensor],
                       person_masks: torch.Tensor) -> torch.Tensor:
        """复刻 StaticFeatureExtractor.forward 中 compute_static_features 之后的部分。

        源实现（static_extractor.py L186-215）：
            for name, v in raw: encoded[name] = self.feature_encoders[name](v)   # [B,T,N,D]
            stacked = torch.stack([...], dim=3)                                  # [B,T,N,Ks,D]
            static  = self.projection(stacked)
            static  = static * torch.any(person_masks, dim=1)[:,None,:,None,None]
        这里逐句照抄，保证与主模型的静态分支完全等价。
        """
        se = self.static_extractor
        enc = {n: se.feature_encoders[n](v) for n, v in raw_features.items()}
        order = [n for n in self.config.static_extractor["feature_names"] if n in enc]
        stacked = torch.stack([enc[n] for n in order], dim=3)     # [B,T,N,Ks,D]
        static = se.projection(stacked)
        person_valid = torch.any(person_masks, dim=1)             # [B,N]
        return static * person_valid[:, None, :, None, None]

    # ------------------------------------------------------------------
    def _substream_mean(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,N,K,d] -> [B,T,N,d]，子流直接取均值（刻意不做学习式门控）。"""
        return x.mean(dim=3)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        frames = batch["frames"]
        bboxes = batch["bboxes"]
        pm = batch["person_masks"]           # [B,T,N] bool
        fm = batch["frame_masks"]            # [B,T]   bool
        B, T, N = pm.shape

        # ---- 1) 同一套 cue ----
        # 静态：优先读缓存（原始量），再按真实 batch 归一化，与 static_extractor.forward
        # 里的 compute_static_features 逐位一致；缓存缺失则整批回退实时计算。
        raw_st = self._static_raw_from_cache(batch["video_ids"])
        if raw_st is not None:
            raw_st = {k: v.to(frames.device) for k, v in raw_st.items()}
            raw_st = self._normalize_static(raw_st, pm)
            st = self._encode_static(raw_st, pm)                             # [B,T,N,Ks,32]
        else:
            st = self.static_extractor(frames, bboxes, pm, fm)[0]             # [B,T,N,Ks,32]

        # 动态：如果唇动缓存可用，就只跑 action 分支（3D ResNet），唇动直接读关键点；
        # 否则整批回退到源实现（含逐帧 MediaPipe，很慢）。
        lip_cached = self._lip_encode_from_cache(batch["video_ids"], pm) if self.lip_cache_dirs else None
        if lip_cached is not None:
            act = self._action_only(frames, bboxes, pm, fm)                     # [B,T,N,512]
            dy = torch.stack([act, lip_cached], dim=-2)                        # [B,T,N,2,512]
            dy = dy.masked_fill(~pm.unsqueeze(-1).unsqueeze(-1).expand_as(dy), 0.0)
        else:
            dy = self.dynamic_extractor(frames, bboxes, pm, fm)[0]             # [B,T,N,Kt,512]
        tx, _dbg_tx = self.text_extractor(
            batch["context_descriptions"], batch["person_descriptions"], batch["original_ids"]
        )                                                                      # [B,N,512]

        # ---- 2) 子流平均 + 投影（无跨模态门控、无跨人注意力）----
        st = self.s_proj(self._substream_mean(st))                             # [B,T,N,D]
        dy = self.d_proj(self._substream_mean(dy))                             # [B,T,N,D]
        tx = self.t_proj(tx).unsqueeze(1).expand(-1, T, -1, -1)                # [B,T,N,D]

        x = torch.cat([st, dy, tx], dim=-1)                                    # [B,T,N,3D]

        # ---- 3) 每个人独立跑时序模型 ----
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)                        # [B*N,T,3D]
        if self.temporal_kind == "transformer":
            h = self.temporal(x)
        else:
            h, _ = self.temporal(x)

        # 有效帧 = person_mask 且 frame_mask
        valid_t = (pm & fm.unsqueeze(-1)).permute(0, 2, 1).reshape(B * N, T)   # [B*N,T]
        m = valid_t.float().unsqueeze(-1)
        h = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)                   # [B*N,3D]

        # ---- 4) 打分 ----
        logits = self.score_head(h).view(B, N)                                 # [B,N]
        person_valid = pm.any(dim=1)                                           # [B,N]
        logits = logits.masked_fill(~person_valid, -1e4)

        feats = F.normalize(h.view(B, N, -1), p=2, dim=-1)                     # 供对比损失
        return {
            "logits": logits,
            "features": feats,
            "scores": logits,
            "person_valid": person_valid,
        }


def build(config: Any, temporal: str = "transformer") -> TrackletBaseline:
    return TrackletBaseline(config, temporal=temporal)
