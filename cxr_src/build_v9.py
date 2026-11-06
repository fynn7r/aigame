"""v9：逐类别贪心集成（75%）+ 全局贪心集成（25%）。
半分裂验证显示逐类别选择样本外 +0.004，最优混合权重 0.75。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/root/_dsh")

R = "/data/cxr_first/runs"
W_PERCLASS = 0.75

meta = pd.read_csv("/data/cxr_first/proc/meta.csv", low_memory=False)
sy = meta.groupby("Study_id")[[f"c{i}" for i in range(1, 11)]].first()
sub_info = meta.groupby("Study_id")["Subject_id"].first()

pv, pt = {}, {}
for d in sorted(glob.glob(os.path.join(R, "*", "probs_valid.npz"))):
    tag = os.path.basename(os.path.dirname(d))
    q = os.path.join(R, tag, "probs_test.npz")
    if not os.path.exists(q):
        continue
    a = np.load(d, allow_pickle=True)
    b = np.load(q, allow_pickle=True)
    pv[tag] = pd.DataFrame(a["probs"].astype(np.float32), index=a["study"].astype(str)).groupby(level=0).mean()
    pt[tag] = pd.DataFrame(b["probs"].astype(np.float32), index=b["study"].astype(str)).groupby(level=0).mean()

tags = sorted(pv)
idx = pv[tags[0]].index
for t in tags[1:]:
    idx = idx.intersection(pv[t].index)
Y = sy.loc[idx].values.astype(np.float32)
M = {t: pv[t].loc[idx].values for t in tags}
tidx = pt[tags[0]].index
for t in tags[1:]:
    tidx = tidx.intersection(pt[t].index)
T = {t: pt[t].loc[tidx].values for t in tags}
print("模型 %d 个 | 验证 study %d | 测试 study %d" % (len(tags), len(idx), len(tidx)))


def macro(mats, rows=None, y=None):
    yy = Y if y is None else y
    rr = np.arange(len(yy)) if rows is None else rows
    a = [roc_auc_score(yy[rr, c], mats[c][rr]) for c in range(10)
         if 3 < yy[rr, c].sum() < len(rr) - 3]
    return float(np.mean(a))


# ---------- 全局贪心 ----------
cur, sel, best = None, [], 0.0
while len(sel) < 6:
    cand, cbest = None, -1
    for t in tags:
        m = M[t]
        mats = [m[:, c] for c in range(10)] if cur is None else \
               [(cur[c] * len(sel) + m[:, c]) / (len(sel) + 1) for c in range(10)]
        s = macro(mats)
        if s > cbest:
            cbest, cand = s, t
    if cur is None:
        cur, sel, best = [M[cand][:, c] for c in range(10)], [cand], cbest
    else:
        if cbest <= best + 0.0001:
            break
        cur = [(cur[c] * len(sel) + M[cand][:, c]) / (len(sel) + 1) for c in range(10)]
        sel.append(cand)
        best = cbest
print("全局贪心 %.4f  %s" % (best, sel))
glob_v = {c: cur[c] for c in range(10)}
cnt = pd.Series(sel).value_counts()
glob_t = {c: sum(cnt[t] * T[t][:, c] for t in cnt.index) / cnt.sum() for c in range(10)}

# ---------- 逐类别贪心 ----------
per_v, per_t, detail = {}, {}, {}
for c in range(10):
    y = Y[:, c]
    cur_c, sel_c, best_c = None, [], 0.0
    while len(sel_c) < 5:
        cand, cbest = None, -1
        for t in tags:
            p = M[t][:, c]
            s = roc_auc_score(y, p) if cur_c is None else \
                roc_auc_score(y, (cur_c * len(sel_c) + p) / (len(sel_c) + 1))
            if s > cbest:
                cbest, cand = s, t
        if cur_c is None:
            cur_c, sel_c, best_c = M[cand][:, c].copy(), [cand], cbest
        else:
            if cbest <= best_c + 0.0001:
                break
            cur_c = (cur_c * len(sel_c) + M[cand][:, c]) / (len(sel_c) + 1)
            sel_c.append(cand)
            best_c = cbest
    cc = pd.Series(sel_c).value_counts()
    per_v[c] = cur_c
    per_t[c] = sum(cc[t] * T[t][:, c] for t in cc.index) / cc.sum()
    detail[c] = (round(float(best_c), 4), sel_c)
    print("  类%-2d %.4f  %s" % (c + 1, best_c, sel_c))

per_macro = macro([per_v[c] for c in range(10)])
glob_macro = macro([glob_v[c] for c in range(10)])
mix_v = [W_PERCLASS * per_v[c] + (1 - W_PERCLASS) * glob_v[c] for c in range(10)]
mix_t = [W_PERCLASS * per_t[c] + (1 - W_PERCLASS) * glob_t[c] for c in range(10)]
print("\n抽样内 逐类 %.4f | 全局 %.4f | 混合 %.4f" % (per_macro, glob_macro, macro(mix_v)))

# ---------- 写提交 ----------
rows = []
P = np.stack(mix_t, 1)
for i, s in enumerate(tidx):
    sid = sub_info.get(s, "")
    for k in range(10):
        rows.append((sid, s, k + 1, float(np.clip(P[i, k], 0.0, 1.0))))
out = pd.DataFrame(rows, columns=["Subject_id", "Study_id", "Predict_class", "Probability"])
out.to_csv("/data/cxr_first/v9.csv", index=False)
print("\n写出 /data/cxr_first/v9.csv", out.shape)
print("study 覆盖 %d / 类别数 %d" % (out["Study_id"].nunique(), out["Predict_class"].nunique()))
print("概率范围 [%.4f, %.4f] mean %.4f" % (out["Probability"].min(), out["Probability"].max(),
                                          out["Probability"].mean()))