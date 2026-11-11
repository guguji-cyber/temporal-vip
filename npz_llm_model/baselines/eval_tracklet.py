#!/usr/bin/env python3
"""在指定 split 上评测 tracklet 基线。

论文的 Rank-1/2/3 是在 **test split** 上报告的（Sec. V-A: "the percentage of test
samples"），而训练过程只在 val 上做模型选择。所以要把基线和论文的 67.3% 对比，
必须用本脚本在 test 上跑一遍。

用法：
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/code/src ./venv/bin/python \
      code/src/npz_llm_model/baselines/eval_tracklet.py \
      --ckpt outputs_tracklet/tracklet_transformer/best.pt --split test
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
from npz_llm_model.baselines.tracklet_baseline import TrackletBaseline  # noqa: E402


def rankk(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, k: int) -> float:
    masked = logits.masked_fill(~valid, float("-inf"))
    topk = masked.topk(min(k, masked.size(1)), dim=1).indices
    return float((topk == target.unsqueeze(1)).any(dim=1).float().mean() * 100.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--static_cache", default=str(BASE / "cache/static_raw"))
    ap.add_argument("--lip_cache", default=str(BASE / "cache/lip_raw"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="结果 json 路径，默认与 ckpt 同目录")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    temporal = saved.get("temporal", "transformer")

    cfg = Config()
    cfg.device = args.device
    cfg.is_ablation = True
    cfg.feature_selection = ["static", "dynamic", "text"]
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.data_ratio = 1.0

    def dirs(root: str):
        return [str(Path(root) / s) for s in ("train", "val", "test")
                if (Path(root) / s).is_dir()]

    model = TrackletBaseline(cfg, temporal=temporal,
                             static_cache_split=dirs(args.static_cache) or None,
                             lip_cache_split=dirs(args.lip_cache) or None).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = VIPDataset(cfg, split=args.split)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False,
    )
    print(f"split={args.split} 片段数={len(ds)} batches={len(loader)} "
          f"temporal={temporal} ckpt_epoch={ckpt.get('epoch')}")

    r_sum = torch.zeros(3)
    r_cnt = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            valid = batch["person_masks"].any(dim=1)
            with torch.autocast(device_type="cuda", enabled=args.device.startswith("cuda")):
                out = model(batch)
            lg = out["logits"].float()
            tg = batch["target_indices"]
            bs = lg.size(0)
            for k in (1, 2, 3):
                r_sum[k - 1] += rankk(lg, tg, valid, k) * bs
            r_cnt += bs
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(loader)}] 已评测 {r_cnt} 个样本", flush=True)

    r1, r2, r3 = (r_sum / max(r_cnt, 1)).tolist()
    res = {"split": args.split, "ckpt": str(args.ckpt), "temporal": temporal,
           "ckpt_epoch": ckpt.get("epoch"), "n_samples": r_cnt,
           "rank1": r1, "rank2": r2, "rank3": r3}
    print(f"\n=== {args.split} 结果 (n={r_cnt}) ===")
    print(f"  R@1={r1:.2f}  R@2={r2:.2f}  R@3={r3:.2f}")

    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  已保存 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
