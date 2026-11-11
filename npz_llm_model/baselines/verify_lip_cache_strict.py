#!/usr/bin/env python3
"""严格验证：把 MediaPipe 换成缓存后，**最终 logits** 是否完全一致。

方法（关键）
    不改任何源文件。只在验证脚本内部替换 DynamicFeatureExtractor.extract_lip_sequence
    这一个方法，其余代码（forward 里的 lip 分支、掩码、LipEncoder、action 分支）
    全部原样执行。

    原实现里 extract_lip_sequence 的调用顺序是确定的：
        for b in range(B):
            for n in range(N):
                if person_valid_BN[b, n]:
                    call(...)
    所以在 patched 函数里用一个计数器就能精确知道当前是第几个 (b, n)，
    从而从缓存读取对应人物的关键点序列。

对比内容
    action / lip / dynamic 整体 / **模型最终 logits**

通过标准
    logits 的 max|diff| 必须严格等于 0，否则不能用缓存跑正式复现。

用法
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/code/src ./venv/bin/python \
      code/src/npz_llm_model/baselines/verify_lip_cache_strict.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BASE / "code" / "src"))

from npz_llm_model.configs.config import Config                      # noqa: E402
from npz_llm_model.data_processing.data_loader import DataLoader     # noqa: E402
from npz_llm_model.models.enhanced_transformer_model import EnhancedTransformerModel  # noqa: E402


def report(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    d = (a.float() - b.float()).abs().max().item()
    ok = d == 0.0
    print(f"  {name:24s} max|diff|={d:.3e}  {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lip_cache", default=str(BASE / "cache/lip_raw"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.is_ablation = True
    cfg.feature_selection = ["static", "dynamic", "text"]
    cfg.batch_size = args.batch
    cfg.num_workers = 2

    dl = DataLoader(cfg)
    batch = next(b for b in dl.get_train_loader() if b is not None)
    batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    vids = list(batch["video_ids"])
    print(f"batch={len(vids)} 片段: {vids}")

    torch.manual_seed(0)
    model = EnhancedTransformerModel(cfg).to(args.device).eval()
    de = model.dynamic_extractor

    # ---------- 建立缓存索引 ----------
    idx: dict[str, Path] = {}
    for sp in ("train", "val", "test"):
        d = Path(args.lip_cache) / sp
        if d.is_dir():
            for f in d.glob("*.npz"):
                idx[f.stem] = f
    print(f"缓存索引: {len(idx)} 个片段\n")

    # ---------- 基准：原始路径（真跑 MediaPipe） ----------
    with torch.no_grad():
        dyn_orig = de(batch["frames"], batch["bboxes"], batch["person_masks"],
                      batch["frame_masks"])[0].clone()
        logits_orig = model(batch)["logits"].clone()
    print(f"[原始] dynamic={tuple(dyn_orig.shape)}  "
          f"action max={dyn_orig[..., 0, :].abs().max().item():.3f}  "
          f"lip max={dyn_orig[..., 1, :].abs().max().item():.3f}")

    # ---------- 替换 extract_lip_sequence ----------
    orig_fn = de.extract_lip_sequence
    B, T, N = batch["person_masks"].shape
    pv = batch["person_masks"].any(dim=1)
    call_order = [(b, n) for b in range(B) for n in range(N) if pv[b, n]]
    state = {"i": 0, "hit": 0, "fallback": 0}

    def patched(frames_b, bboxes_n, person_mask_n, frame_mask_b):
        # 注意：一次 model(batch) 会调用 dynamic_extractor 两次（_extract_features
        # 与后续分支），所以计数器必须对 len(call_order) 取模，才能让每一轮
        # 调用都精确对应同一个 (b, n) 序列。
        i = state["i"] % len(call_order) if call_order else 0
        state["i"] += 1
        if not call_order:
            state["fallback"] += 1
            return orig_fn(frames_b, bboxes_n, person_mask_n, frame_mask_b)
        b, n = call_order[i]
        path = idx.get(str(vids[b]))
        if path is None:
            state["fallback"] += 1
            return orig_fn(frames_b, bboxes_n, person_mask_n, frame_mask_b)
        with np.load(path) as z:
            key = f"lip_{n}"
            state["hit"] += 1
            if key not in z:
                # 源实现此时返回全零张量，随后 any(lip_seq != 0) 为 False
                return torch.zeros(T, 40, device=frames_b.device)
            return torch.from_numpy(z[key]).float().to(frames_b.device)

    de.extract_lip_sequence = patched        # type: ignore[assignment]
    with torch.no_grad():
        dyn_cache = de(batch["frames"], batch["bboxes"], batch["person_masks"],
                       batch["frame_masks"])[0].clone()
        logits_cache = model(batch)["logits"].clone()
    de.extract_lip_sequence = orig_fn        # type: ignore[assignment]

    print(f"[缓存] 调用 {state['i']} 次，命中 {state['hit']}，回退 {state['fallback']}\n")

    # ---------- 对比 ----------
    print("[对比]")
    ok = True
    ok &= report("action", dyn_orig[..., 0, :], dyn_cache[..., 0, :])
    ok &= report("lip", dyn_orig[..., 1, :], dyn_cache[..., 1, :])
    ok &= report("dynamic (整体)", dyn_orig, dyn_cache)
    ok &= report("MODEL logits (最终)", logits_orig, logits_cache)

    print("\n" + "=" * 58)
    if ok:
        print("结论：缓存与实时 MediaPipe 完全等价，logits max|diff| = 0")
        print("      → 可以用缓存跑正式复现")
    else:
        print("结论：存在数值差异，不可用缓存跑正式复现")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
