#!/usr/bin/env python3
"""预计算静态线索（area / centrality / clarity），供 tracklet 基线复用。

为什么要缓存
    StaticFeatureExtractor.compute_static_features 里人脸清晰度是一个
    (b, t, n) 三层循环，每片段约 276 次 MediaPipe 调用，实测 1.04 s/片段。
    一个 epoch 光这一项就要 ~20 分钟，30 个 epoch 不可接受。
    所以先算一次存盘，训练时读 npz（每个 4 KB）。

存什么（关键）
    只存 **未归一化的原始量**：area_raw / centrality_raw / clarity_raw。
    不能存 compute_static_features 的最终返回值，因为它把无效 (t,n) 位置填成了
    **整个 batch 的均值**；离线按 B=1 算会得到单片段均值，与训练时的 batch 值不同
    （实测 clarity 最大偏差 0.44，值域只有 0~1）。而且 clarity 全量有 22 万个点
    卡在 clamp 上界，从归一化值反解是有损的。
    存原始量之后，训练时按真实 batch 重做
        log1p -> 无效位置填 batch 均值 -> clamp(0,1.6) -> ÷1.6
    与主模型 StaticFeatureExtractor 前向里的那条路径逐位等价。

用法：
  PYTHONPATH=$PWD/code/src ./venv/bin/python baselines/precompute_static.py \
      --splits train,val --workers 12
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[4]          # .../vip  (baselines -> npz_llm_model -> src -> code -> vip)
sys.path.insert(0, str(BASE / "code" / "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch


def _init_worker():
    """每个 worker 独立构建一次提取器（MediaPipe 不可跨进程共享）。"""
    global _EXTRACTOR
    from npz_llm_model.configs.config import Config
    from npz_llm_model.feature_extraction.static_extractor import StaticFeatureExtractor

    cfg = Config()
    cfg.device = "cpu"
    _EXTRACTOR = StaticFeatureExtractor(cfg)


def _compute_raw(extractor, frames, bboxes, pmask):
    """复刻 compute_static_features 的前半段，但**不归一化、不填充**。

    与源实现的区别只在最后：源实现接着做 log1p / batch 均值填充 / clamp / ÷1.6，
    那一步依赖 batch 组成，必须留到训练时按真实 batch 做。
    这里返回的 area/centrality/clarity 与源实现中 clarity_proc 之前的中间量一致。
    """
    B, T, H, W, C = frames.shape
    N = bboxes.shape[2]

    x1, y1, x2, y2 = (bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3])
    area = torch.clamp(((x2 - x1) * (y2 - y1)) / (W * H), min=0)

    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    center_dist = torch.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    centrality = 1.0 - torch.clamp(center_dist / 0.5, max=1.0)

    # 唯一的重活：每人每帧一次 MediaPipe。逐位复刻源实现的循环与判断顺序。
    clarity = torch.zeros(B, T, N)
    for b in range(B):
        for t in range(T):
            for n in range(N):
                if not pmask[b, t, n]:
                    continue
                box = bboxes[b, t, n].cpu().numpy()
                frame_np = frames[b, t].cpu().numpy().astype(np.uint8)
                clarity[b, t, n] = extractor.extract_face_clarity(frame_np, box)

    return {"area_raw": area, "centrality_raw": centrality, "clarity_raw": clarity}


def _process_one(args):
    npz_path, out_path = args
    try:
        d = np.load(npz_path)
        frames = torch.from_numpy(d["frames"]).float()
        bboxes = torch.from_numpy(d["bboxes"]).float()
        pmask = torch.from_numpy(d["person_mask"]).float()

        with torch.no_grad():
            feats = _compute_raw(_EXTRACTOR, frames.unsqueeze(0), bboxes.unsqueeze(0),
                                 pmask.unsqueeze(0))

        payload = {k: v.squeeze(0).cpu().numpy().astype(np.float32) for k, v in feats.items()}
        np.savez_compressed(out_path, **payload)
        return (npz_path, None, int(pmask.sum()))
    except Exception as e:  # noqa: BLE001
        return (npz_path, f"{type(e).__name__}: {e}", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data_root",
                    default=str(BASE / "code/data/vip/preprocessed_fixed"))
    ap.add_argument("--cache_root", default=str(BASE / "cache/static_features"))
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    tasks = []
    for split in args.splits.split(","):
        split = split.strip()
        src_dir = Path(args.data_root) / split
        dst_dir = cache_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_dir.glob("*.npz")):
            tasks.append((str(f), str(dst_dir / f.name)))

    print(f"[precompute] splits={args.splits} 片段数={len(tasks)} workers={args.workers}", flush=True)
    t0 = time.time()
    done = 0
    failed = 0
    total_pairs = 0

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(_process_one, t): t for t in tasks}
        for fut in as_completed(futs):
            path, err, npairs = fut.result()
            done += 1
            total_pairs += npairs
            if err:
                failed += 1
                print(f"[FAIL] {Path(path).name}: {err}", flush=True)
            if done % 100 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(tasks) - done)
                print(f"[{done}/{len(tasks)}] elapsed={elapsed/60:.1f}m "
                      f"eta={eta/60:.1f}m failed={failed}", flush=True)

    elapsed = time.time() - t0
    print(f"\n[precompute] 完成 {done} 片段，失败 {failed}，有效(t,n)对 {total_pairs}", flush=True)
    print(f"[precompute] 总耗时 {elapsed/60:.1f} 分钟", flush=True)
    print(f"[precompute] 缓存目录 {cache_root}", flush=True)


if __name__ == "__main__":
    main()
