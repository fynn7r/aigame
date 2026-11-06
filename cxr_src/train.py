# -*- coding: utf-8 -*-
"""训练主脚本（从零训练，无预训练权重）。

示例:
  python train.py --arch convnext_tiny --size 384 --fold 0 --epochs 40 \
      --bs 32 --lr 3e-4 --out /data/cxr_first/runs/cnxt384_f0
"""
import os, sys, time, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_meta, ImageCache, CXRDataset, build_model, EMA, macro_auc,
                    GroupKFold, seed_all, study_aggregate)


class AsymmetricLoss(nn.Module):
    """ASL：多标签长尾任务的标准损失，压制易分负样本、聚焦稀有病种。

    CE 形式: L = -y*(1-p)^γ+ * log(p) - (1-y)*p_clip^γ- * log(1-p_clip)
    """

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=self.eps) if self.clip > 0 else p
        los_pos = targets * torch.log(p.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log((1 - p_neg).clamp(min=self.eps))
        if self.gp > 0:
            los_pos = los_pos * torch.pow(1 - p, self.gp)
        if self.gn > 0:
            los_neg = los_neg * torch.pow(p_neg, self.gn)
        return -(los_pos + los_neg).mean()


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="convnext_tiny")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--fold", type=int, default=0, help="-1 表示用全部 train，验证用 valid 集")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--pos_weight", type=float, default=0.0, help=">0 时使用固定正样本权重")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--resume", default="")
    p.add_argument("--init", default="", help="SSL 预训练权重（backbone state_dict）")
    p.add_argument("--weights", default="", help="ImageNet 预训练权重")
    p.add_argument("--pw_cap", type=float, default=20.0, help="pos_weight 上限")
    p.add_argument("--clahe", action="store_true", help="用 CLAHE 增强局部对比度")
    p.add_argument("--mixup", type=float, default=0.0, help="mixup alpha, 0=关闭")
    p.add_argument("--loss", default="bce", choices=["bce", "asl"], help="损失函数")
    p.add_argument("--asl_gn", type=float, default=4.0, help="ASL 负样本聚焦参数")
    p.add_argument("--asl_clip", type=float, default=0.05, help="ASL 概率下限裁剪")
    return p.parse_args()


def run_eval(model, loader, meta_df, device, use_amp=True):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for img, y in loader:
            img = img.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                out = model(img)
            probs.append(torch.sigmoid(out.float()).cpu().numpy())
            ys.append(y.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    img_auc, per = macro_auc(ys, probs)
    sid = meta_df["Study_id"].values
    agg = study_aggregate(meta_df, probs, sid, "mean")
    s_y = meta_df.groupby("Study_id")[[f"c{i}" for i in range(1, 11)]].first()
    s_y = s_y.loc[agg.index].values.astype(np.float32)
    st_auc, per_s = macro_auc(s_y, agg.values)
    return dict(img_auc=img_auc, study_auc=st_auc, per_class=per_s,
                study_per_class=per_s), probs


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    seed_all(a.seed)
    device = "cuda"
    meta = load_meta()
    cache = ImageCache()

    tr_idx = np.where(meta["split"] == "train")[0]
    va_idx = np.where(meta["split"] == "valid")[0]
    if a.fold >= 0:
        gkf = GroupKFold(meta.iloc[tr_idx].reset_index(drop=True), 5, seed=a.seed)
        tr_local, va_local = gkf.split(a.fold)
        tr_fit = tr_idx[tr_local]
        va_fold = tr_idx[va_local]
    else:
        tr_fit = tr_idx
        va_fold = va_idx

    print(f"fit {len(tr_fit)} imgs | fold-val {len(va_fold)} | valid {len(va_idx)}", flush=True)

    ds_tr = CXRDataset(meta, cache, tr_fit, train=True, img_size=a.size, clahe=a.clahe)
    ds_va = CXRDataset(meta, cache, va_fold, train=False, img_size=a.size, clahe=a.clahe)
    ds_valid = CXRDataset(meta, cache, va_idx, train=False, img_size=a.size, clahe=a.clahe)

    dl_tr = DataLoader(ds_tr, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                       pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=a.bs * 2, shuffle=False, num_workers=max(2, a.workers // 2),
                       pin_memory=True)
    dl_valid = DataLoader(ds_valid, batch_size=a.bs * 2, shuffle=False, num_workers=max(2, a.workers // 2),
                          pin_memory=True)

    builder = "tv" if (a.weights and a.weights.endswith(".pth")) else "timm"
    model = build_model(a.arch, drop=a.drop, drop_path=a.drop_path, weights=a.weights,
                        builder=builder).to(device)
    if a.init:
        st = torch.load(a.init, map_location="cpu")
        sd = st.get("backbone", st.get("model", st))
        sd = {k: v for k, v in sd.items() if k in model.state_dict()
              and model.state_dict()[k].shape == v.shape}
        miss = model.load_state_dict(sd, strict=False)
        print(f"init from {a.init}: loaded {len(sd)} tensors, missing {len(miss.missing_keys)}", flush=True)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model {a.arch} params {n_par:.1f}M", flush=True)

    labels = ds_tr.labels
    if a.pos_weight > 0:
        pw = torch.full((10,), a.pos_weight)
    else:
        pos = labels.sum(0).clip(1)
        pw = torch.tensor((len(labels) - pos) / pos, dtype=torch.float32).clamp(1.0, a.pw_cap)
    print("pos_weight", [round(x, 2) for x in pw.tolist()], flush=True)
    if a.loss == "asl":
        crit = AsymmetricLoss(gamma_neg=a.asl_gn, gamma_pos=0.0, clip=a.asl_clip)
        print(f"loss ASL gamma_neg={a.asl_gn} clip={a.asl_clip}", flush=True)
    else:
        crit = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(dl_tr) * a.epochs
    warm = len(dl_tr) * a.warmup

    def lr_mult(s):
        if s < warm:
            return (s + 1) / max(1, warm)
        p = (s - warm) / max(1, steps - warm)
        return 0.5 * (1 + np.cos(np.pi * p)) * 0.999 + 0.001
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ema = EMA(model, a.ema) if a.ema > 0 else None

    best = -1
    log = []
    step = 0
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for img, y in dl_tr:
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if a.mixup > 0 and random.random() < 0.5:
                lam = float(np.random.beta(a.mixup, a.mixup))
                lam = max(lam, 1 - lam)
                perm = torch.randperm(img.size(0), device=img.device)
                img = lam * img + (1 - lam) * img[perm]
                y = lam * y + (1 - lam) * y[perm]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(img)
                loss = crit(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            tot += loss.item() * img.size(0)
            n += img.size(0)
            step += 1
            if step % 100 == 0:
                print(f"  ep{ep} step{step} loss {tot/n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{time.time()-t0:.0f}s", flush=True)
        dt = time.time() - t0
        rec = dict(epoch=ep, loss=tot / n, sec=dt, lr=sched.get_last_lr()[0])

        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            eval_model = build_model(a.arch, builder=builder).to(device)
            if ema:
                ema.copy_to(eval_model)
            else:
                eval_model.load_state_dict(model.state_dict())
            m1, _ = run_eval(eval_model, dl_va, meta.iloc[va_fold].reset_index(drop=True), device)
            m2, _ = run_eval(eval_model, dl_valid, meta.iloc[va_idx].reset_index(drop=True), device)
            rec.update(fold_auc=m1["study_auc"], fold_img_auc=m1["img_auc"],
                       valid_auc=m2["study_auc"], valid_img_auc=m2["img_auc"])
            print(f"ep{ep} loss {tot/n:.4f} | foldAUC {m1['study_auc']:.4f} (img {m1['img_auc']:.4f}) "
                  f"| validAUC {m2['study_auc']:.4f} | {dt:.0f}s", flush=True)
            score = m1["study_auc"]
            if score > best:
                best = score
                torch.save(dict(model=ema.shadow if ema else model.state_dict(),
                                arch=a.arch, size=a.size, epoch=ep, score=best, builder=builder,
                                args=vars(a)), os.path.join(a.out, "best.pt"))
                print(f"  * saved best {best:.4f}", flush=True)
            del eval_model
            torch.cuda.empty_cache()
        else:
            print(f"ep{ep} loss {tot/n:.4f} {dt:.0f}s", flush=True)
        log.append(rec)
        pd.DataFrame(log).to_csv(os.path.join(a.out, "log.csv"), index=False)
        torch.save(dict(model=(ema.shadow if ema else model.state_dict()),
                        arch=a.arch, size=a.size, epoch=ep, args=vars(a)),
                   os.path.join(a.out, "last.pt"))
    print("BEST", best, flush=True)


if __name__ == "__main__":
    main()