"""逐类别贪心集成：macro-AUC 是 10 个类别 AUC 的平均，每个类单独选模型理论上更优。"""
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

probs = {}
for d in sorted(glob.glob(os.path.join(R, "*", "probs_valid.npz"))):
    tag = os.path.basename(os.path.dirname(d))
    z = np.load(d, allow_pickle=True)
    df = pd.DataFrame(z["probs"].astype(np.float32), index=z["study"].astype(str))
    probs[tag] = df.groupby(level=0).mean()

tags = sorted(probs)
idx = probs[tags[0]].index
for t in tags[1:]:
    idx = idx.intersection(probs[t].index)
Y = sy.loc[idx].values.astype(np.float32)
M = {t: probs[t].loc[idx].values for t in tags}
print("模型 %d 个，验证 study %d 个" % (len(tags), len(idx)))


def auc_col(p, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


def macro(mats):
    return float(np.mean([auc_col(mats[c], Y[:, c]) for c in range(10)]))


# 全局贪心（基线）
print("\n--- 全局贪心")
cur, sel = None, []
best = 0
for step in range(8):
    cand, cbest = None, -1
    for t in tags:
        m = M[t]
        s = macro([m[:, c] for c in range(10)]) if cur is None else macro(
            [(cur[c] * len(sel) + m[:, c]) / (len(sel) + 1) for c in range(10)])
        if s > cbest:
            cbest, cand = s, t
    if cur is None:
        cur, sel, best = [M[cand][:, c] for c in range(10)], [cand], cbest
    else:
        if cbest <= best + 0.00005:
            break
        cur = [(cur[c] * len(sel) + M[cand][:, c]) / (len(sel) + 1) for c in range(10)]
        sel.append(cand)
        best = cbest
    print("  step%d +%-22s %.4f" % (step, cand, cbest))
print("全局贪心 %.4f  %s" % (best, sel))

# 逐类别贪心
print("\n--- 逐类别独立贪心")
per_sel, per_mat = {}, np.zeros_like(Y, dtype=np.float32)
for c in range(10):
    y = Y[:, c]
    if y.sum() < 3:
        continue
    cur_c, sel_c, best_c = None, [], 0
    for step in range(6):
        cand, cbest = None, -1
        for t in tags:
            p = M[t][:, c]
            s = auc_col(p, y) if cur_c is None else auc_col((cur_c * len(sel_c) + p) / (len(sel_c) + 1), y)
            if s > cbest:
                cbest, cand = s, t
        if cur_c is None:
            cur_c, sel_c, best_c = M[cand][:, c].copy(), [cand], cbest
        else:
            if cbest <= best_c + 0.00005:
                break
            cur_c = (cur_c * len(sel_c) + M[cand][:, c]) / (len(sel_c) + 1)
            sel_c.append(cand)
            best_c = cbest
    per_sel[c] = (round(best_c, 4), sel_c)
    per_mat[:, c] = cur_c
    print("  类%-2d %.4f  %s" % (c + 1, best_c, sel_c))

print("\n逐类别贪心 macro %.4f" % macro([per_mat[:, c] for c in range(10)]))
print("全局贪心 macro %.4f" % best)