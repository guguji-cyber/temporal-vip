#!/usr/bin/env python3
"""预计算唇动关键点序列（MediaPipe FaceMesh），供 tracklet 基线复用。

为什么
    DynamicFeatureExtractor.extract_lip_sequence 对**每个人物**逐帧跑 MediaPipe，
    一个 batch=8 有 26 个有效人物，实测 8.43 s/batch，占总前向的 80%。
    一个 epoch 693 batch 就是 1.6 小时，30 epoch 不可接受。

存什么
    extract_lip_sequence 的**原始返回值**：[T, 40]，float32，未经任何变换。
    逐位复刻源实现的循环与判断（person_mask / frame_mask / bbox 裁剪 /
    FaceMesh 61:81 号 landmark / clip(0,1)），因此训练时直接拿来用即可，
    与实时计算完全等价（不需要任何重放/反解）。

缓存布局
    {cache_root}/{split}/{video_id}.npz
        keys: lip_{n}  for n in 0..N-1  -> [T, 40] float32
    只存有效人物（整段至少出现过一次）；无效人物在训练时补零。

用法
  PYTHONPATH=$PWD/code/src ./venv/bin/python baselines/precompute_lip.py \
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
    """每个 worker 独立构建一次 FaceMesh（不可跨进程共享）。"""
    global _EXTRACTOR
    from npz_llm_model.configs.config import Config
    from npz_llm_model.feature_extraction.dynamic_extractor import DynamicFeatureExtractor

    cfg = Config()
    cfg.device = "cpu"
    _EXTRACTOR = DynamicFeatureExtractor(cfg)


def _process_one(args):
    npz_path, out_path = args
    try:
        d = np.load(npz_path)
        frames = torch.from_numpy(d["frames"]).float()
        bboxes = torch.from_numpy(d["bboxes"]).float()
        pmask = torch.from_numpy(d["person_mask"]).bool()
        fmask = torch.from_numpy(d["frame_mask"]).bool()

        T = frames.shape[0]
        N = bboxes.shape[1]
        pv = pmask.any(dim=0)                     # [N] 整段是否出现过

        payload = {}
        npairs = 0
        with torch.no_grad():
            for n in range(N):
                if not pv[n]:
                    continue
                # 逐位复刻 forward 里的调用：frames[b], bboxes[b,:,n], person_masks[b,:,n], frame_masks[b]
                seq = _EXTRACTOR.extract_lip_sequence(frames, bboxes[:, n], pmask[:, n], fmask)
                if torch.any(seq != 0):           # 与源实现一致：全零则不进入编码器
                    payload[f"lip_{n}"] = seq.cpu().numpy().astype(np.float32)
                    npairs += 1

        np.savez_compressed(out_path, **payload)
        return (npz_path, None, npairs)
    except Exception as e:  # noqa: BLE001
        return (npz_path, f"{type(e).__name__}: {e}", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--data_root",
                    default=str(BASE / "code/data/vip/preprocessed_fixed"))
    ap.add_argument("--cache_root", default=str(BASE / "cache/lip_raw"))
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

    print(f"[precompute-lip] splits={args.splits} 片段数={len(tasks)} workers={args.workers}",
          flush=True)
    t0 = time.time()
    done = failed = total = 0

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futs = {ex.submit(_process_one, t): t for t in tasks}
        for fut in as_completed(futs):
            path, err, npairs = fut.result()
            done += 1
            total += npairs
            if err:
                failed += 1
                print(f"[FAIL] {Path(path).name}: {err}", flush=True)
            if done % 100 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(tasks) - done)
                print(f"[{done}/{len(tasks)}] elapsed={elapsed/60:.1f}m "
                      f"eta={eta/60:.1f}m failed={failed}", flush=True)

    elapsed = time.time() - t0
    print(f"\n[precompute-lip] 完成 {done} 片段，失败 {failed}，有效(片段,人物)对 {total}",
          flush=True)
    print(f"[precompute-lip] 总耗时 {elapsed/60:.1f} 分钟", flush=True)
    print(f"[precompute-lip] 缓存目录 {cache_root}", flush=True)


if __name__ == "__main__":
    main()
