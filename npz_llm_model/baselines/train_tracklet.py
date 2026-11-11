#!/usr/bin/env python3
"""训练 / 评测 tracklet 基线（无 TIR）。

用法（在服务器上，cwd = /home/jwj/data/projects/vip）：
    PYTHONPATH=$PWD/code/src \
    ./venv/bin/python code/src/npz_llm_model/baselines/train_tracklet.py \
        --temporal transformer --epochs 30 --batch_size 8 --tag tracklet_transformer

损失与监督刻意与 VIP-Net 的消融设置对齐：
    cls (CE, w=1.0) + contrastive (InfoNCE, w=0.3, tau=0.5) + L2 正则 (w=0.0005)
不含文本相似度损失，因为论文 Table IV 的每一行都是用 --is_ablation 跑的，
那个开关同时把文本损失置零（见 train/losses.py::TotalLoss.forward）。
基线沿用同一口径，才与 Table IV 可比。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[4]          # .../vip
sys.path.insert(0, str(BASE / "code" / "src"))

from npz_llm_model.configs.config import Config                              # noqa: E402
from npz_llm_model.data_processing.data_loader import DataLoader             # noqa: E402
from npz_llm_model.baselines.tracklet_baseline import TrackletBaseline       # noqa: E402


def rankk(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, k: int) -> float:
    masked = logits.masked_fill(~valid, float("-inf"))
    topk = masked.topk(min(k, masked.size(1)), dim=1).indices
    return float((topk == target.unsqueeze(1)).any(dim=1).float().mean() * 100.0)


def parse_args():
    p = argparse.ArgumentParser(description="Tracklet baseline (no TIR)")
    p.add_argument("--temporal", choices=["transformer", "lstm"], default="transformer")
    p.add_argument("--tag", default=None, help="输出子目录名，默认 tracklet_<temporal>")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--out_dir", default=str(BASE / "outputs_tracklet"))
    p.add_argument("--data_ratio", type=float, default=1.0)
    p.add_argument("--limit_batches", type=int, default=0, help=">0 时只跑这么多 batch（冒烟用）")
    p.add_argument("--device", default=None)
    p.add_argument("--static_cache", default=str(BASE / "cache/static_raw"),
                   help="静态线索原始量缓存根目录（含 train/ val/ 子目录）；传空串则实时计算")
    p.add_argument("--lip_cache", default=str(BASE / "cache/lip_raw"),
                   help="唇动关键点缓存根目录（含 train/ val/ 子目录）；传空串则实时跑 MediaPipe")
    p.add_argument("--train_split", default="train", help="训练集 split 名，用于定位缓存子目录")
    p.add_argument("--val_split", default="val", help="验证集 split 名，用于定位缓存子目录")
    return p.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    cfg.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.learning_rate = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.feature_selection = ["static", "dynamic", "text"]   # 与 VIP-Net 相同的 cue
    cfg.is_ablation = True                                   # 跳过 BART 解释生成器
    cfg.debug = False
    cfg.data_ratio = args.data_ratio
    return cfg


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = build_config(args)

    tag = args.tag or f"tracklet_{args.temporal}"
    out = Path(args.out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"=== tracklet 基线 === temporal={args.temporal} seed={args.seed} device={cfg.device}")
    log(f"cue = {cfg.feature_selection}   输出 = {out}")

    # ---------------- 数据 ----------------
    dloader = DataLoader(cfg)
    train_loader = dloader.get_train_loader()
    val_loader = dloader.get_val_loader()
    log(f"train batches={len(train_loader)}  val batches={len(val_loader)}")

    # ---------------- 模型 ----------------
    # 缓存索引：train/val 的 npz 文件名不重叠（实测交集为 0），合并成一个索引即可，
    # 同一个模型实例在训练和验证阶段都能命中。
    def _split_dirs(root_arg: str) -> list:
        if not root_arg:
            return []
        root = Path(root_arg)
        found = [str(root / sp) for sp in (args.train_split, args.val_split)
                 if (root / sp).is_dir()]
        return found

    static_dirs = _split_dirs(args.static_cache)
    lip_dirs = _split_dirs(args.lip_cache)
    if args.static_cache and len(static_dirs) != 2:
        log(f"警告：静态缓存不完整（找到 {static_dirs}），缺失部分将回退实时计算（很慢）")
    if args.lip_cache and len(lip_dirs) != 2:
        log(f"警告：唇动缓存不完整（找到 {lip_dirs}），缺失部分将回退实时计算（很慢）")
    log(f"静态缓存 {static_dirs if static_dirs else '未启用（实时计算，约 1.04 s/片段）'}")
    log(f"唇动缓存 {lip_dirs if lip_dirs else '未启用（实时 MediaPipe，约 8.4 s/batch）'}")

    model = TrackletBaseline(cfg, temporal=args.temporal,
                             static_cache_split=static_dirs or None,
                             lip_cache_split=lip_dirs or None).to(cfg.device)
    n = sum(p.numel() for p in model.parameters())
    log(f"参数量 {n/1e6:.1f}M")

    # ---------------- 损失（与 VIP-Net 消融口径一致）----------------
    from npz_llm_model.train.losses import ClassificationLoss, ContrastiveLoss, WeightRegularizationLoss
    cls_loss = ClassificationLoss(weight=1.0, config=cfg)
    con_loss = ContrastiveLoss(temperature=0.5, weight=0.3, config=cfg)
    reg_loss = WeightRegularizationLoss(weight=0.0005)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.device.startswith("cuda"))

    best = {"rank@1": 0.0, "epoch": 0}
    hist = []

    # ---------------- 训练 ----------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run = 0.0
        nb = 0
        for it, batch in enumerate(train_loader):
            if batch is None:
                continue
            if args.limit_batches and it >= args.limit_batches:
                break
            batch = {k: (v.to(cfg.device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            tgt = batch["target_indices"]
            # ClassificationLoss 需要 [B,N] 的逐人掩码；ContrastiveLoss 需要 [B] 的
            # "该样本的 VIP 是否有效" 掩码（与 TotalLoss.forward 中的 batch_valid 一致）
            valid_pn = batch["person_masks"].any(dim=1)
            valid_b = valid_pn.gather(1, tgt.unsqueeze(1)).squeeze(1)

            with torch.autocast(device_type="cuda", enabled=cfg.device.startswith("cuda")):
                out_d = model(batch)
                l_cls = cls_loss(out_d["logits"], tgt, valid_pn)
                l_con = con_loss(out_d["features"], tgt, valid_b)
                loss = l_cls + l_con

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(opt)
            scaler.update()

            run += float(loss.detach())
            nb += 1
            if nb % 50 == 0:
                log(f"  ep{epoch} it{nb}/{len(train_loader)} loss={run/nb:.4f}")

        l_reg = float(reg_loss(model).detach())
        sched.step()

        # ---------------- 验证 ----------------
        # 按样本数加权，避免最后一个不满的 batch 权重偏大
        model.eval()
        r_sum = torch.zeros(3)
        r_cnt = 0
        with torch.no_grad():
            for it, batch in enumerate(val_loader):
                if batch is None:
                    continue
                if args.limit_batches and it >= args.limit_batches:
                    break
                batch = {k: (v.to(cfg.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                valid = batch["person_masks"].any(dim=1)
                with torch.autocast(device_type="cuda", enabled=cfg.device.startswith("cuda")):
                    out_d = model(batch)
                lg = out_d["logits"].float()
                tg = batch["target_indices"]
                bs = lg.size(0)
                for k in (1, 2, 3):
                    r_sum[k - 1] += rankk(lg, tg, valid, k) * bs
                r_cnt += bs
        r1, r2, r3 = (r_sum / max(r_cnt, 1)).tolist()
        log(f"epoch {epoch}: train_loss={run/max(nb,1):.4f} reg={l_reg:.5f} "
            f"| val R@1={r1:.2f} R@2={r2:.2f} R@3={r3:.2f} | {time.time()-t0:.0f}s")
        hist.append({"epoch": epoch, "train_loss": run / max(nb, 1),
                     "rank@1": r1, "rank@2": r2, "rank@3": r3})

        if r1 > best["rank@1"]:
            best = {"rank@1": r1, "rank@2": r2, "rank@3": r3, "epoch": epoch}
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "args": vars(args), "best": best}, out / "best.pt")

    (out / "history.json").write_text(json.dumps({"best": best, "history": hist},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"=== 完成  best: R@1={best['rank@1']:.2f} R@2={best.get('rank@2',0):.2f} "
        f"R@3={best.get('rank@3',0):.2f} @ep{best['epoch']} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
