# -*- coding: utf-8 -*-
"""推理 / 评估：加载若干 checkpoint，对指定 split 预测并按 Study_id 聚合。

  python infer.py --ckpt run1/best.pt run2/best.pt --split valid --out preds_valid.csv
  python infer.py --ckpt ... --split test --out /data/cxr_first/submission.csv
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_meta, ImageCache, CXRDataset, default_eval_aug, build_model, macro_auc


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--size", type=int, default=0)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tta", type=int, default=1, help="1=无, 2=+水平翻转")
    p.add_argument("--scales", default="", help="多尺度 TTA, 如 384,448,512")
    p.add_argument("--out", default="")
    p.add_argument("--save_probs", default="", help="保存影像级概率 npz，供集成/阈值搜索")
    return p.parse_args()


def predict(model, ds, size, bs, workers, tta):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    out = []
    model.eval()
    with torch.no_grad():
        for img, _ in dl:
            img = img.cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                p = torch.sigmoid(model(img).float())
                if tta >= 2:
                    p = p + torch.sigmoid(model(torch.flip(img, dims=[3])).float())
                    p = p / 2
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def main():
    a = parse()
    meta = load_meta()
    cache = ImageCache()
    idx = np.where(meta["split"] == a.split)[0]
    ds = CXRDataset(meta, cache, idx, train=False, img_size=a.size or 384)
    df = meta.iloc[idx].reset_index(drop=True)
    acc = None
    for ck in a.ckpt:
        st = torch.load(ck, map_location="cpu")
        size = a.size or st.get("size", 384)
        ds.img_size = size
        from common import default_eval_aug
        clahe = bool(st.get("args", {}).get("clahe", False))
        ds.aug = default_eval_aug(size, clahe)
        model = build_model(st["arch"], builder=st.get("builder","timm")).cuda()
        sd = {k: v for k, v in st["model"].items()}
        model.load_state_dict({k: v.float() if v.dtype.is_floating_point else v
                               for k, v in sd.items()})
        print(f"loaded {ck} arch={st['arch']} size={size} epoch={st.get('epoch')}", flush=True)
        scales = [int(v) for v in a.scales.split(",")] if a.scales else [size]
        p = None
        for sc in scales:
            ds.img_size = sc
            ds.aug = default_eval_aug(sc, clahe)
            ps = predict(model, ds, sc, a.bs, a.workers, a.tta)
            p = ps if p is None else p + ps
        if len(scales) > 1:
            p = p / len(scales)
        ds.img_size = size
        acc = p if acc is None else acc + p
        del model
        torch.cuda.empty_cache()
    probs = acc / len(a.ckpt)

    agg = pd.DataFrame(probs, columns=[f"p{i}" for i in range(1, 11)])
    agg["Subject_id"] = df["Subject_id"].values
    agg["Study_id"] = df["Study_id"].values
    g = agg.groupby(["Subject_id", "Study_id"], sort=False)[[f"p{i}" for i in range(1, 11)]].mean().reset_index()

    if a.split != "test":
        y = df.groupby(["Subject_id", "Study_id"], sort=False)[[f"c{i}" for i in range(1, 11)]].first().reset_index()
        y = y.set_index(["Subject_id", "Study_id"]).loc[
            list(zip(g.Subject_id, g.Study_id))].values.astype(np.float32)
        m, per = macro_auc(y, g[[f"p{i}" for i in range(1, 11)]].values)
        print(f"STUDY macro AUC = {m:.4f}")
        for i, v in enumerate(per):
            print(f"  class {i+1:2d}: {v:.4f}")
        img_m, img_per = macro_auc(np.stack(df["labels"].values), probs)
        print(f"IMAGE macro AUC = {img_m:.4f}")

    if a.save_probs:
        np.savez_compressed(a.save_probs, probs=probs.astype(np.float16),
                            dicom=df["Dicom_id"].values,
                            study=df["Study_id"].values,
                            subject=df["Subject_id"].values,
                            split=a.split)
        print("probs saved", a.save_probs)

    if a.out:
        rows = []
        for _, r in g.iterrows():
            for c in range(1, 11):
                rows.append((r.Subject_id, r.Study_id, c, float(r[f"p{c}"])))
        sub = pd.DataFrame(rows, columns=["Subject_id", "Study_id", "Predict_class", "Probability"])
        sub.to_csv(a.out, index=False, encoding="utf-8")
        print("written", a.out, sub.shape)


if __name__ == "__main__":
    main()