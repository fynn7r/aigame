"""用「一半选模型 / 另一半评模型」检验逐类别贪心是真增益还是过拟合。"""
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
    aucs = []
    for c in range(10):
        y = Y[rows, c]
        if y.sum() < 3 or y.sum() > len(rows) - 3:
            continue
        aucs.append(roc_auc_score(y, mats[c][rows]))
    return float(np.mean(aucs))


def global_greedy(rows, iters=6):
    cur, sel, best = None, [], 0.0
    for _ in range(iters):
        cand, cbest = None, -1
        for t in tags:
            m = M[t]
            mats = [m[:, c] for c in range(10)] if cur is None else \
                   [(cur[c] * len(sel) + m[:, c]) / (len(sel) + 1) for c in range(10)]
            s = macro_on(mats, rows)
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
    return cur, sel, best


def perclass_greedy(rows, iters=5):
    out = np.zeros((N, 10), dtype=np.float32)
    sels = {}
    for c in range(10):
        y = Y[rows, c]
        if y.sum() < 3:
            out[:, c] = 0.5
            continue
        cur_c, sel_c, best_c = None, [], 0.0
        for _ in range(iters):
            cand, cbest = None, -1
            for t in tags:
                p = M[t][:, c]
                s = roc_auc_score(y, p[rows]) if cur_c is None else \
                    roc_auc_score(y, ((cur_c * len(sel_c) + p) / (len(sel_c) + 1))[rows])
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
        out[:, c] = cur_c
        sels[c] = sel_c
    return [out[:, c] for c in range(10)], sels, macro_on([out[:, c] for c in range(10)], rows)


print("模型 %d 个 / 验证 study %d 个" % (len(tags), N))
print("\n%-6s %-10s %-10s %-10s %-10s" % ("split", "全局-选", "全局-评", "逐类-选", "逐类-评"))
res = []
for k, seed in enumerate([1, 2, 3, 4, 5]):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    A, B = perm[: N // 2], perm[N // 2:]
    mg, sg, ing = global_greedy(A)
    og = macro_on(mg, B)
    mp, sp, inp = perclass_greedy(A)
    op = macro_on(mp, B)
    res.append((ing, og, inp, op))
    print("%-6d %-10.4f %-10.4f %-10.4f %-10.4f" % (k + 1, ing, og, inp, op))

r = np.array(res)
print("\n均值   %-10.4f %-10.4f %-10.4f %-10.4f" % tuple(r.mean(0)))
print("\n全局：样本内 %.4f → 样本外 %.4f（虚高 %+.4f）" % (r[:, 0].mean(), r[:, 1].mean(),
                                                       r[:, 0].mean() - r[:, 1].mean()))
print("逐类：样本内 %.4f → 样本外 %.4f（虚高 %+.4f）" % (r[:, 2].mean(), r[:, 3].mean(),
                                                       r[:, 2].mean() - r[:, 3].mean()))
print("\n样本外谁更强：全局 %.4f vs 逐类 %.4f，差 %+.4f" % (r[:, 1].mean(), r[:, 3].mean(),
                                                          r[:, 3].mean() - r[:, 1].mean()))