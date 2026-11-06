"""在「一半选/一半评」框架下，找逐类别集成与全局集成的最佳混合权重。"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/root/_dsh")

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
N = len(idx)

def macro_on(mats, rows):
    a = [roc_auc_score(Y[rows, c], mats[c][rows]) for c in range(10)
         if 3 < Y[rows, c].sum() < len(rows) - 3]
    return float(np.mean(a))

def global_greedy(rows, iters=6):
    cur, sel, best = None, [], 0.0
    for _ in range(iters):
        cand, cbest = None, -1
        for t in tags:
            m = M[t]
            mats = [m[:, c] for c in range(10)] if cur is None else \
                   [(cur[c] * len(sel) + m[:, c]) / (len(sel) + 1) for c in range(10)]
            s = macro_on(mats, rows)
            if s > cbest: cbest, cand = s, t
        if cur is None: cur, sel, best = [M[cand][:, c] for c in range(10)], [cand], cbest
        else:
            if cbest <= best + 0.0001: break
            cur = [(cur[c] * len(sel) + M[cand][:, c]) / (len(sel) + 1) for c in range(10)]
            sel.append(cand); best = cbest
    return cur

def perclass_greedy(rows, iters=4):
    out = np.zeros((N, 10), dtype=np.float32)
    for c in range(10):
        y = Y[rows, c]
        if y.sum() < 3:
            out[:, c] = 0.5; continue
        cur, sel, best = None, [], 0.0
        for _ in range(iters):
            cand, cbest = None, -1
            for t in tags:
                p = M[t][:, c]
                s = roc_auc_score(y, p[rows]) if cur is None else \
                    roc_auc_score(y, ((cur * len(sel) + p) / (len(sel) + 1))[rows])
                if s > cbest: cbest, cand = s, t
            if cur is None: cur, sel, best = M[cand][:, c].copy(), [cand], cbest
            else:
                if cbest <= best + 0.0001: break
                cur = (cur * len(sel) + M[cand][:, c]) / (len(sel) + 1)
                sel.append(cand); best = cbest
        out[:, c] = cur
    return [out[:, c] for c in range(10)]

WS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
acc = {w: [] for w in WS}
for seed in [1, 2, 3, 4, 5, 6]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    A, B = perm[: N // 2], perm[N // 2:]
    g = global_greedy(A)
    p = perclass_greedy(A)
    for w in WS:
        mix = [(w * p[c] + (1 - w) * g[c]) for c in range(10)]
        acc[w].append(macro_on(mix, B))
    print("split%d 完成" % seed, flush=True)

print("\n混合权重  样本外 macro（6 次半分裂均值）")
for w in WS:
    print("  逐类 %.2f / 全局 %.2f  →  %.4f" % (w, 1 - w, np.mean(acc[w])))
best = max(WS, key=lambda w: np.mean(acc[w]))
print("\n最佳权重 逐类 %.2f（样本外 %.4f）" % (best, np.mean(acc[best])))