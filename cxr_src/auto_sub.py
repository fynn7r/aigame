# -*- coding: utf-8 -*-
"""一步到位：读若干 run 的 valid/test 概率 -> 贪心加权 -> 写 submission。

  python auto_sub.py --tags "p_b0a_f0 p_b0a_f1" --name v2
"""
import os, sys, argparse
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_meta, macro_auc
from ensemble import agg_study, logit, sigmoid, P, C, KEY

R = "/data/cxr_first/runs"


def load(path):
    z = np.load(path, allow_pickle=True)
    d = pd.DataFrame(z["probs"].astype(np.float32), columns=P)
    d["Subject_id"] = z["subject"]
    d["Study_id"] = z["study"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--name", default="v2")
    ap.add_argument("--mode", default="logit_mean")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--min_gain", type=float, default=0.0002)
    a = ap.parse_args()
    tags = a.tags.split()

    dv, dt, used = {}, {}, []
    for t in tags:
        pv = os.path.join(R, t, "probs_valid.npz")
        pt = os.path.join(R, t, "probs_test.npz")
        if not (os.path.exists(pv) and os.path.exists(pt)):
            print("skip (missing probs)", t)
            continue
        dv[t] = agg_study(load(pv), a.mode)
        dt[t] = agg_study(load(pt), a.mode)
        used.append(t)
    assert used, "没有可用的概率文件"
    print(f"{len(used)} models: {used}")

    keys = dv[used[0]].index
    meta = load_meta()
    lab = meta.groupby(KEY, sort=False)[C].first().loc[keys]
    y = lab.values.astype(np.float32)
    V = {t: dv[t].loc[keys][P].values for t in used}

    print("单模型 valid AUC:")
    singles = {}
    for t in used:
        m, _ = macro_auc(y, V[t])
        singles[t] = m
        print(f"  {t:18s} {m:.4f}")

    sel, cur, best_sofar = [], None, -1.0
    for step in range(a.steps):
        best_t, best_s, best_cand = None, -1, None
        for t in used:
            cand = V[t] if cur is None else (cur * len(sel) + V[t]) / (len(sel) + 1)
            s, _ = macro_auc(y, cand)
            if s > best_s:
                best_t, best_s, best_cand = t, s, cand
        if best_s < best_sofar + a.min_gain and step >= 2:
            print(f"stop at step {step} (gain {best_s - best_sofar:+.4f})")
            break
        sel.append(best_t)
        cur, best_sofar = best_cand, best_s
        print(f"  step {step}: +{best_t:18s} -> {best_s:.4f}")

    w = Counter(sel)
    vscore, per = macro_auc(y, cur)
    print("WEIGHTS:", dict(w))
    print(f"ENSEMBLE valid macroAUC {vscore:.4f}")
    print("per class:", " ".join(f"{v:.3f}" for v in per))

    # 单模型最好成绩对照
    bt = max(singles, key=singles.get)
    print(f"best single {bt} {singles[bt]:.4f} -> ensemble {vscore:.4f} "
          f"({vscore - singles[bt]:+.4f})")

    tkeys = dt[used[0]].index
    T = {t: dt[t].loc[tkeys][P].values for t in used}
    acc = np.zeros_like(T[used[0]])
    tot = 0.0
    for t, c in w.items():
        acc += c * T[t]
        tot += c
    pred = acc / tot
    rows = []
    for i, (sub, stu) in enumerate(tkeys):
        for c in range(1, 11):
            rows.append((sub, stu, c, float(pred[i, c - 1])));
    out = pd.DataFrame(rows, columns=["Subject_id", "Study_id", "Predict_class", "Probability"])
    path = f"/data/cxr_first/{a.name}.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    print("written", path, out.shape)


if __name__ == "__main__":
    main()