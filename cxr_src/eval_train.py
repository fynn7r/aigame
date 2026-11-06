# -*- coding: utf-8 -*-
"""诊断：在模型自己的训练数据子集上算 AUC，判断欠拟合/过拟合。

  python eval_train.py --ckpt .../best.pt --n 4000
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (load_meta, ImageCache, CXRDataset, build_model, macro_auc,
                    default_eval_aug, study_aggregate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    size = a.size or ck["size"]
    fold = a.fold if a.fold >= 0 else ck.get("args", {}).get("fold", 0)
    model = build_model(ck["arch"], builder=ck.get("builder", "timm")).cuda()
    model.load_state_dict(ck["model"])
    model.eval()

    meta = load_meta()
    if "fold" in meta.columns and fold >= 0:
        pool = meta.index[meta.fold != fold].values
    else:
        pool = meta.index[meta.split == "train"].values
    rng = np.random.RandomState(0)
    idx = rng.choice(pool, min(a.n, len(pool)), replace=False)
    sub = meta.iloc[idx]

    from common import CXRDataset
    ds = CXRDataset(sub, ImageCache(), idx, train=False, img_size=size, aug=default_eval_aug(size))
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)
    probs = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    probs = np.concatenate(probs)
    y = sub[[f"c{i}" for i in range(1, 11)]].values.astype(np.float32)
    m, per = macro_auc(y, probs)
    print(f"TRAIN-subset IMAGE macroAUC {m:.4f}  n={len(sub)}")
    print("per class:", " ".join(f"{v:.3f}" for v in per))
    df = sub.copy()
    for i in range(10):
        df[f"p{i+1}"] = probs[:, i]
    g = study_aggregate(df)
    yg = g[[f"c{i}" for i in range(1, 11)]].values.astype(np.float32)
    m2, _ = macro_auc(yg, g[[f"p{i}" for i in range(1, 11)]].values)
    print(f"TRAIN-subset STUDY macroAUC {m2:.4f}  n={len(g)}")


if __name__ == "__main__":
    main()