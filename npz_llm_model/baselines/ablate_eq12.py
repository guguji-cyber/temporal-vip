#!/usr/bin/env python3
"""Eq.(12) 人物间关系建模的推理时探针 —— 两阶段版。

阶段一 (cache)：完整跑一遍 split，把 temporal_alignment 的输出
        fusion_features [B, N, D] 与标签、mask 一起缓存到磁盘。
阶段二 (replay)：从缓存重放，只替换 encoder 这一段，对比不同变体。

为什么只做"推理时"探针
------------------------
位置编码是**可学习参数**。把随机初始化的位置编码加进去，等于往特征里注入
噪声，得到的数字没有意义。因此这里对位置编码只报告 **顺序敏感度**
（打乱人物顺序后输出变化多少），不报告 Rank。

只有 +padding mask 这一项在推理时是公平的：mask 不含任何参数，纯粹是
"不该看的别去看"。它回答的问题是：当前权重有多少输出被 padding 槽位污染。

用法：
  PYTHONPATH=$PWD/code/src ./venv/bin/python \
      code/src/npz_llm_model/baselines/ablate_eq12.py --stage cache
  PYTHONPATH=$PWD/code/src ./venv/bin/python \
      code/src/npz_llm_model/baselines/ablate_eq12.py --stage replay
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE / "code" / "src"))

from npz_llm_model.configs.config import Config                        # noqa: E402
from npz_llm_model.data_processing.dataset import VIPDataset           # noqa: E402
from npz_llm_model.data_processing.data_loader import collate_fn       # noqa: E402
from npz_llm_model.models.enhanced_transformer_model import (
    EnhancedTransformerModel,
)                                                                      # noqa: E402

CKPT = (BASE / "code/src/npz_llm_model/outputs/20261104_193555_133444"
        / "checkpoints/main_model/best.pth")
CACHE_DIR = BASE / "cache/eq12_probe"


def rankk(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, k: int) -> float:
    masked = logits.masked_fill(~valid, float("-inf"))
    topk = masked.topk(min(k, masked.size(1)), dim=1).indices
    return float((topk == target.unsqueeze(1)).any(dim=1).float().mean() * 100.0)


def build_model(device: str):
    cfg = Config()
    cfg.device = device
    cfg.is_ablation = True
    cfg.feature_selection = ["static", "dynamic", "text"]
    cfg.fusion_type = "transformer"
    cfg.dynamic_extractor = dict(cfg.dynamic_extractor)
    cfg.dynamic_extractor["lip_cache_dirs"] = [
        str(BASE / "cache/lip_raw" / s) for s in ("train", "val", "test")
    ]
    model = EnhancedTransformerModel(cfg).to(device)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"载入权重 missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, cfg


# ---------------------------------------------------------------- 阶段一
@torch.no_grad()
def stage_cache(split: str, device: str, bs: int, nw: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model, cfg = build_model(device)

    cap: list = []
    model.temporal_alignment.register_forward_hook(
        lambda mod, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach()))

    ds = VIPDataset(cfg, split=split)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=collate_fn, pin_memory=True, drop_last=False)
    print(f"split={split} 片段数={len(ds)} batches={len(loader)}")

    feats_l, valid_l, tgt_l = [], [], []
    ok = 0
    for i, b in enumerate(loader):
        if b is None:
            continue
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        cap.clear()
        out = model(b, return_explanations=False)
        if out is None or not cap:
            continue
        feats_l.append(cap[-1].float().cpu())
        valid_l.append(b["person_masks"].any(dim=1).cpu())
        tgt_l.append(b["target_indices"].cpu())
        ok += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(loader)}] 已缓存 {ok} batch", flush=True)

    pack = {"feats": torch.cat(feats_l), "valid": torch.cat(valid_l),
            "target": torch.cat(tgt_l)}
    torch.save(pack, CACHE_DIR / f"{split}.pt")
    print(f"已缓存 {ok} batch, 共 {pack['feats'].shape[0]} 片段")
    print(f"  文件: {CACHE_DIR / f'{split}.pt'}")


# ---------------------------------------------------------------- 阶段二
@torch.no_grad()
def stage_replay(split: str, device: str) -> None:
    pack = torch.load(CACHE_DIR / f"{split}.pt", map_location="cpu")
    F = pack["feats"].to(device)
    valid = pack["valid"].to(device)
    tgt = pack["target"].to(device)
    print(f"载入缓存: {F.shape[0]} 片段, F.shape={tuple(F.shape)}")
    print(f"有效人数: 平均 {valid.sum(1).float().mean():.2f} / "
          f"最少 {int(valid.sum(1).min())} / 最多 {int(valid.sum(1).max())}")
    print(f"padding 槽位占比: {(1 - valid.float().mean()) * 100:.1f}%")

    model, cfg = build_model(device)
    enc = model.encoder
    tau = getattr(cfg, "temperature", 1.0)

    def head_plain(x):
        e = torch.nn.functional.normalize(enc(x), p=2, dim=-1)
        return model.classifier(e).squeeze(-1).float() / tau

    def head_masked(x):
        e = enc(x, src_key_padding_mask=~valid)
        e = torch.nn.functional.normalize(e, p=2, dim=-1)
        return model.classifier(e).squeeze(-1).float() / tau

    results: dict = {}
    for key, name, fn in [("A", "A. 原样(无 mask)", head_plain),
                          ("B", "B. +padding mask", head_masked)]:
        logits = fn(F).masked_fill(~valid, float("-inf"))
        r = {f"R@{k}": rankk(logits, tgt, valid, k) for k in (1, 2, 3)}
        results[key] = {"name": name, **{k: round(v, 2) for k, v in r.items()}}
        print(f"{name:<22} R@1={r['R@1']:.2f}  R@2={r['R@2']:.2f}  R@3={r['R@3']:.2f}")

    # ---- 顺序敏感度 ----
    print("\n=== 顺序敏感度（打乱 N 维顺序，比较同一个人 logit）===")
    perm = torch.randperm(F.size(1), device=device)
    inv = torch.argsort(perm)
    for key, name, fn in [("A", "A. 原样(无 mask)", head_plain),
                          ("B", "B. +padding mask", head_masked)]:
        d = (fn(F) - fn(F[:, perm])[:, inv]).abs()[valid]
        results[key]["order_sens_mean"] = float(d.mean())
        results[key]["order_sens_max"] = float(d.max())
        print(f"{name:<22} 平均 {d.mean():.3e}   最大 {d.max():.3e}")

    # ---- 截断等价性：同一批 3 人片段，喂 3 槽 vs 喂 20 槽 ----
    print("\n=== 截断等价性（真实人数 vs 补齐 20 槽）===")
    idx = (valid.sum(1) == 3).nonzero().flatten()[:128]
    if len(idx):
        e_s = torch.nn.functional.normalize(enc(F[idx, :3, :]), p=2, dim=-1)
        l_s = model.classifier(e_s).squeeze(-1).float() / tau
        e_f = torch.nn.functional.normalize(enc(F[idx]), p=2, dim=-1)
        l_f = (model.classifier(e_f).squeeze(-1).float() / tau)[:, :3]
        d = (l_s - l_f).abs()
        agree = (l_s.argmax(1) == l_f.argmax(1)).float().mean()
        results["truncation"] = {"n": int(len(idx)), "logit_diff_mean": float(d.mean()),
                                 "logit_diff_max": float(d.max()),
                                 "argmax_agree": float(agree)}
        print(f"  {len(idx)} 个 3 人片段: logit 平均差={d.mean():.4f}  最大差={d.max():.4f}")
        print(f"  argmax 一致率: {agree * 100:.1f}%")

    print("\n" + "=" * 62)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    out = CACHE_DIR / f"ablate_{split}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已保存 {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cache", "replay"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.stage == "cache":
        stage_cache(args.split, args.device, args.batch_size, args.num_workers)
    else:
        stage_replay(args.split, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
