import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/root/_dsh")
from common import macro_auc  # noqa: E402

R = "/data/cxr_first/runs"
meta = pd.read_csv("/data/cxr_first/proc/meta.csv", low_memory=False)
sy = meta.groupby("Study_id")[[f"c{i}" for i in range(1, 11)]].first()

rows = []
for d in sorted(glob.glob(os.path.join(R, "*"))):
    ck = os.path.join(d, "best.pt")
    if not os.path.exists(ck):
        continue
    a = torch.load(ck, map_location="cpu", weights_only=False)
    ar = a.get("args", {}) or {}
    tag = os.path.basename(d)
    va = ""
    pv = os.path.join(d, "probs_valid.npz")
    if os.path.exists(pv):
        z = np.load(pv, allow_pickle=True)
        df = pd.DataFrame(z["probs"].astype(np.float32), index=z["study"].astype(str))
        df = df.groupby(level=0).mean()
        y = sy.loc[df.index].values.astype(np.float32)
        va, _ = macro_auc(y, df.values)
        va = round(float(va), 4)
    wt = "ImageNet-1K"
    w = os.path.basename(str(ar.get("weights", "")))
    if "jft" in w:
        wt = "JFT-1k"
    elif "in22k" in w:
        wt = "IN-22k→1k"
    elif "ra_in1k" in w:
        wt = "IN-1k(RA)"
    elif not w:
        wt = "scratch"
    rows.append(dict(tag=tag, arch=ar.get("arch", a.get("arch")), size=ar.get("size", a.get("size")),
                     fold=ar.get("fold"), epochs=ar.get("epochs"), best_ep=a.get("epoch"),
                     bs=ar.get("bs"), lr=ar.get("lr"), wd=ar.get("wd"),
                     drop_path=ar.get("drop_path"), pos_weight=ar.get("pos_weight"),
                     mixup=ar.get("mixup"), clahe=1 if ar.get("clahe") else 0,
                     pretrain=wt, fold_auc=round(float(a.get("score", 0)), 4), valid_auc=va))

df = pd.DataFrame(rows).sort_values(["arch", "size", "tag"])
df["valid_auc"] = pd.to_numeric(df["valid_auc"], errors="coerce")
df.to_csv("/data/cxr_first/models_summary.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
print(df.to_string(index=False))

print("\n### 按架构-分辨率聚合的 foldAUC 均值")
g = df.groupby(["arch", "size"]).agg(n=("tag", "count"), fold_auc=("fold_auc", "mean"),
                                     valid_auc=("valid_auc", "mean")).round(4)
print(g.sort_values("fold_auc", ascending=False).to_string())

print("\n### 集成权重")
for tag in ("v4", "v5"):
    p = "/root/_dsh/%s.log" % tag
    if os.path.exists(p):
        for line in open(p):
            if "WEIGHTS" in line or "ENSEMBLE valid" in line:
                print(tag, line.strip())