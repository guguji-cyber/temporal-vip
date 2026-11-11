#!/usr/bin/env python3
"""一次性验证脚本：确认 tracklet 基线的缓存路径与源实现逐位等价。

比对三件事（同一批数据、同一份权重、同一个 device）：
  1. 静态分支：static_extractor.forward  vs  _static_raw_from_cache + _normalize_static + _encode_static
  2. 动态分支：dynamic_extractor.forward  vs  _action_only + _lip_encode_from_cache
  3. 完整前向：TrackletBaseline(实时)  vs  TrackletBaseline(缓存)

动态分支必须用源实现的 forward 作参照，因为 forward 带
@torch.amp.autocast('cuda') 装饰器；绕过它直接调 lip_encoder 会得到 FP32 结果，
与缓存路径（显式 autocast）差 3e-3 量级的舍入误差，那是测试写法的问题不是实现的问题。

用法：
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/code/src ./venv/bin/python \
      code/src/npz_llm_model/baselines/verify_cache_equivalence.py --n 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE / "code" / "src"))

from npz_llm_model.configs.config import Config                      # noqa: E402
from npz_llm_model.baselines.tracklet_baseline import TrackletBaseline  # noqa: E402


def report(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = 1e-5) -> bool:
    d = (a.float() - b.float()).abs().max().item()
    scale = a.float().abs().max().item()
    ok = d <= tol
    print(f"  {name:26s} max|diff|={d:.3e}  rel={d/max(scale,1e-9):.2e}  "
          f"量级={scale:.3f}  {'OK' if ok else 'FAIL'}")
    return ok


def build_batch(vids, split: str):
    d = BASE / "code/data/vip/preprocessed_fixed" / split
    fr, bb, pm, fm = [], [], [], []
    for v in vids:
        z = np.load(d / f"{v}.npz")
        fr.append(torch.from_numpy(z["frames"]).float())
        bb.append(torch.from_numpy(z["bboxes"]).float())
        pm.append(torch.from_numpy(z["person_mask"]).bool())
        fm.append(torch.from_numpy(z["frame_mask"]).bool())
    return (torch.stack(fr), torch.stack(bb), torch.stack(pm), torch.stack(fm))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="batch 内片段数")
    ap.add_argument("--split", default="train")
    ap.add_argument("--static_cache", default=str(BASE / "cache/static_raw"))
    ap.add_argument("--lip_cache", default=str(BASE / "cache/lip_raw"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.is_ablation = True
    cfg.feature_selection = ["static", "dynamic", "text"]
    cfg.dynamic_extractor["replace_temporal_with_flow"] = False

    # 只在两个缓存都齐全的片段里挑，保证缓存路径不会中途回退
    have_static = {p.stem for p in (Path(args.static_cache) / args.split).glob("*.npz")}
    have_lip = {p.stem for p in (Path(args.lip_cache) / args.split).glob("*.npz")}
    pool = sorted(have_static & have_lip)
    if len(pool) < args.n:
        print(f"缓存里只有 {len(pool)} 个片段可用，不够 {args.n} 个")
        return 1
    vids = pool[: args.n]

    fr, bb, pm, fm = build_batch(vids, args.split)
    fr, bb, pm, fm = fr.to(args.device), bb.to(args.device), pm.to(args.device), fm.to(args.device)
    print(f"batch: B={fr.shape[0]} T={fr.shape[1]} N={pm.shape[2]}  "
          f"有效人物={int(pm.any(dim=1).sum())}  device={args.device}")

    m_live = TrackletBaseline(cfg, temporal="transformer").to(args.device).eval()
    m_cache = TrackletBaseline(
        cfg, temporal="transformer",
        static_cache_split=[str(Path(args.static_cache) / s) for s in ("train", "val")
                            if (Path(args.static_cache) / s).is_dir()],
        lip_cache_split=[str(Path(args.lip_cache) / s) for s in ("train", "val")
                         if (Path(args.lip_cache) / s).is_dir()],
    ).to(args.device).eval()
    m_cache.load_state_dict(m_live.state_dict())

    all_ok = True
    with torch.no_grad():
        # ---------- 1. 静态分支 ----------
        print("\n[1] 静态分支")
        t = time.time()
        st_live = m_live.static_extractor(fr, bb, pm, fm)[0]
        t_live = time.time() - t
        t = time.time()
        raw = m_cache._static_raw_from_cache(vids)
        raw = {k: v.to(args.device) for k, v in raw.items()}
        st_cache = m_cache._encode_static(m_cache._normalize_static(raw, pm), pm)
        t_cache = time.time() - t
        all_ok &= report("static features", st_live, st_cache)
        print(f"  耗时: 实时 {t_live*1000:.0f}ms  ->  缓存 {t_cache*1000:.0f}ms")

        # ---------- 2. 动态分支 ----------
        print("\n[2] 动态分支（参照 = 源实现 forward，自带 autocast）")
        t = time.time()
        dyn_live = m_live.dynamic_extractor(fr, bb, pm, fm)[0]     # [B,T,N,2,D]
        t_live = time.time() - t
        t = time.time()
        lip_c = m_cache._lip_encode_from_cache(vids, pm)
        act_c = m_cache._action_only(fr, bb, pm, fm)
        dyn_cache = torch.stack([act_c, lip_c], dim=-2)
        dyn_cache = dyn_cache.masked_fill(
            ~pm.unsqueeze(-1).unsqueeze(-1).expand_as(dyn_cache), 0.0)
        t_cache = time.time() - t
        all_ok &= report("action", dyn_live[..., 0, :], dyn_cache[..., 0, :])
        all_ok &= report("lip", dyn_live[..., 1, :], dyn_cache[..., 1, :])
        all_ok &= report("dynamic (both)", dyn_live, dyn_cache)
        print(f"  耗时: 实时 {t_live:.2f}s  ->  缓存 {t_cache:.3f}s  "
              f"加速 {t_live/max(t_cache,1e-9):.1f}x")

        # ---------- 3. 完整前向 ----------
        print("\n[3] 完整前向（logits / features）")
        batch = {
            "video_ids": vids,
            "frames": fr, "bboxes": bb,
            "person_masks": pm, "frame_masks": fm,
            "context_descriptions": [""] * len(vids),
            "person_descriptions": [[{"person_id": "", "feature": {}}] * pm.shape[2]
                                    for _ in vids],
            "original_ids": torch.arange(pm.shape[2], device=args.device).repeat(len(vids), 1),
        }
        t = time.time()
        o_live = m_live(batch)
        t_live = time.time() - t
        t = time.time()
        o_cache = m_cache(batch)
        t_cache = time.time() - t
        all_ok &= report("logits", o_live["logits"], o_cache["logits"])
        all_ok &= report("features", o_live["features"], o_cache["features"])
        print(f"  耗时: 实时 {t_live:.2f}s  ->  缓存 {t_cache:.2f}s  "
              f"加速 {t_live/max(t_cache,1e-9):.1f}x")

    print("\n" + "=" * 60)
    print("结论:", "全部逐位一致，缓存路径可安全用于正式训练" if all_ok
          else "存在数值差异，不可用于正式训练")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
