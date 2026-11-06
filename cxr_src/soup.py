"""模型汤：把多个同架构种子的权重直接平均（不是预测平均），再评估。"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/_dsh")
from common import CXRDataset, ImageCache, build_model, load_meta, macro_auc  # noqa: E402

R = "/data/cxr_first/runs"
TAGS = ["full_cnxb_384_s42", "full_cnxb_384_s123", "full_cnxb_384_s777", "full_cnxb_384_s2024"]
OUT = os.path.join(R, "soup_cnxb_384")

sds = []
for t in TAGS:
    p = os.path.join(R, t, "last.pt")
    if not os.path.exists(p):
        p = os.path.join(R, t, "best.pt")
    if not os.path.exists(p):
        print("skip", t, flush=True)
        continue
    ck = torch.load(p, map_location="cpu")
    sds.append((t, ck["model"]))
    print("载入 %-24s 训练时分数 %s" % (t, ck.get("score")), flush=True)
print("参与平均 %d 个" % len(sds), flush=True)

base = sds[0][1]
soup = {}
for k, v in base.items():
    if v.dtype.is_floating_point:
        acc = torch.zeros_like(v, dtype=torch.float32)
        for _, sd in sds:
            acc += sd[k].float()
        soup[k] = (acc / len(sds)).to(v.dtype)
    else:
        soup[k] = v.clone()

os.makedirs(OUT, exist_ok=True)
torch.save({"model": soup, "arch": "convnext_base", "size": 384, "builder": "timm", "score": None},
           os.path.join(OUT, "soup.pt"))
print("写出", os.path.join(OUT, "soup.pt"), flush=True)

meta = load_meta()
meta["labels"] = list(np.stack(meta[[f"c{i}" for i in range(1, 11)]].values))
idx = np.where(meta["split"].values == "valid")[0]
ds = CXRDataset(meta, ImageCache(), idx, train=False, img_size=384)
df = meta.iloc[idx].reset_index(drop=True)

m = build_model("convnext_base", num_classes=10, builder="timm").cuda()
m.load_state_dict(soup)
m.eval()
dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=5)
out = []
with torch.no_grad():
    for img, _ in dl:
        img = img.cuda()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(m(img).float()) + torch.sigmoid(m(torch.flip(img, dims=[3])).float())
        out.append((p / 2).cpu().numpy())
P = np.concatenate(out)

agg = pd.DataFrame(P, columns=[f"p{i}" for i in range(1, 11)])
agg["Study_id"] = df["Study_id"].values
g = agg.groupby("Study_id", sort=False).mean()
yt = df.groupby("Study_id", sort=False)[[f"c{i}" for i in range(1, 11)]].first()
assert list(g.index) == list(yt.index), "study 顺序不一致"
mm, per = macro_auc(yt.values.astype(np.float32), g.values)
print("\nSOUP 单模型 study-AUC = %.4f" % mm, flush=True)
print("对照：单种子 best 0.8102 / 四种子预测平均 0.8150", flush=True)