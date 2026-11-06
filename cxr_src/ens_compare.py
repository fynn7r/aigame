"""CPU 上对比不同集成策略在固定验证集上的表现（不占 GPU）。"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/_dsh")
from common import macro_auc  # noqa: E402

R = "/data/cxr_first/runs"
meta = pd.read_csv("/data/cxr_first/proc/meta.csv", low_memory=False)
sy = meta.groupby("Study_id")[[f"c{i}" for i in range(1, 11)]].first()

FAM = {
    "b0": "EfficientNet-B0", "dn201": "DenseNet-201", "dn121": "DenseNet-121",
    "cnxs": "ConvNeXt-Small", "r50": "ResNet-50", "r34": "ResNet-34",
}


def family_of(tag):
    t = tag
    for k, v in FAM.items():
        if k.replace("xs", "xs") in t.replace("cnx-s", "cnxs"):
            pass
    if "cnxs" in t or "cnx_s" in t:
        return "ConvNeXt-Small"
    if "dn201" in t:
        return "DenseNet-201"
    if "dn121" in t:
        return "DenseNet-121"
    if "r50" in t:
        return "ResNet-50"
    if "r34" in t:
        return "ResNet-34"
    if "b0" in t:
        return "EfficientNet-B0"
    return "other"


probs, scores = {}, {}
for d in sorted(glob.glob(os.path.join(R, "*", "probs_valid.npz"))):
    tag = os.path.basename(os.path.dirname(d))
    z = np.load(d, allow_pickle=True)
    df = pd.DataFrame(z["probs"].astype(np.float32), index=z["study"].astype(str))
    df = df.groupby(level=0).mean()
    y = sy.loc[df.index].values.astype(np.float32)
    a, _ = macro_auc(y, df.values)
    probs[tag] = df
    scores[tag] = float(a)

tags = [t for t in probs if scores[t] > 0.77]
print("模型数 %d" % len(tags))
for t in sorted(tags, key=lambda x: -scores[x]):
    print("  %-22s %.4f  %s" % (t, scores[t], family_of(t)))

idx = probs[tags[0]].index
for t in tags[1:]:
    idx = idx.intersection(probs[t].index)
Y = sy.loc[idx].values.astype(np.float32)


def score(mat):
    a, _ = macro_auc(Y, np.asarray(mat, dtype=np.float32))
    return float(a)


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


print("\n验证集 study 数 %d" % len(idx))

# 1. 全部等权（概率域）
print("全部等权           %.4f" % score(np.mean([probs[t].loc[idx].values for t in tags], 0)))
# 2. 全部等权（logit 域）
print("全部等权(logit)    %.4f" % score(1 / (1 + np.exp(-np.mean(
    [logit(probs[t].loc[idx].values) for t in tags], 0)))))

# 3. Top-k
order = sorted(tags, key=lambda t: -scores[t])
for k in (3, 5, 8, 12):
    if k <= len(order):
        print("Top-%d 等权         %.4f" % (k, score(np.mean([probs[t].loc[idx].values for t in order[:k]], 0))))

# 4. 每族最优 + 等权
fam_best = {}
for t in order:
    fam_best.setdefault(family_of(t), t)
print("每族最优 %d 个等权   %.4f" % (len(fam_best),
      score(np.mean([probs[t].loc[idx].values for t in fam_best.values()], 0))))
print("    ", {k: (v, round(scores[v], 4)) for k, v in fam_best.items()})

# 5. 每族取前 n 个求族均值，再跨族平均
for n in (1, 2, 3):
    fams = {}
    for t in order:
        fams.setdefault(family_of(t), []).append(t)
    fam_mean = []
    for f, lst in fams.items():
        fam_mean.append(np.mean([probs[t].loc[idx].values for t in lst[:n]], 0))
    print("每族前%d取均值再跨族平均 %.4f" % (n, score(np.mean(fam_mean, 0))))

# 6. 贪心（带放回）
cur, sel = None, []
best = 0
for step in range(12):
    cand, cbest = None, -1
    for t in tags:
        m = probs[t].loc[idx].values
        s = score(m if cur is None else (cur * len(sel) + m) / (len(sel) + 1))
        if s > cbest:
            cbest, cand = s, t
    if cur is None:
        cur, sel, best = probs[cand].loc[idx].values.copy(), [cand], cbest
    else:
        if cbest <= best + 0.00005:
            break
        cur = (cur * len(sel) + probs[cand].loc[idx].values) / (len(sel) + 1)
        sel.append(cand)
        best = cbest
    print("贪心 step%d +%-20s %.4f" % (step, cand, cbest))
print("贪心最终           %.4f  %s" % (best, sel))