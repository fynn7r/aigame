# -*- coding: utf-8 -*-
"""集成 + 聚合策略搜索 + 贪心加权(带放回) + 生成 submission。

  python ensemble.py --probs "runs/*/probs_valid.npz" --split valid --greedy
  python ensemble.py --probs "runs/*/probs_test.npz"  --split test --weights "a:2,b:1" --out sub.csv
"""
import os, sys, glob, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_meta, macro_auc

P = [f"p{i}" for i in range(1, 11)]
C = [f"c{i}" for i in range(1, 11)]
KEY = ["Subject_id", "Study_id"]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def agg_study(df, mode):
    g = df.groupby(KEY, sort=False)
    if mode == "mean":
        out = g[P].mean()
    elif mode == "max":
        out = g[P].max()
    elif mode == "logit_mean":
        t = df.copy()
        t[P] = logit(t[P].values)
        out = t.groupby(KEY, sort=False)[P].mean()
        out = pd.DataFrame(sigmoid(out.values), index=out.index, columns=P)
    elif mode == "mean_max":
        out = 0.5 * g[P].mean() + 0.5 * g[P].max()
    elif mode == "max_mean90":
        out = 0.9 * g[P].max() + 0.1 * g[P].mean()
    else:
        raise ValueError(mode)
    return out


def load_files(pats, split):
    files = []
    for p in pats:
        if p.endswith(":") or ":" in p:
            continue
        files.extend(sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p])
    out = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if str(z["split"]) != split:
            continue
        d = pd.DataFrame(z["probs"].astype(np.float32), columns=P)
        d["Subject_id"] = z["subject"]
        d["Study_id"] = z["study"]
        out.append((f, d))
    return out


def short(f):
    return os.path.basename(os.path.dirname(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", default="")
    ap.add_argument("--mode", default="", help="固定聚合方式，默认自动挑")
    ap.add_argument("--greedy", action="store_true", help="贪心带放回挑模型(权重)")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--weights", default="", help="test 阶段用: name:w,name:w")
    a = ap.parse_args()

    files = load_files(a.probs, a.split)
    print(f"{len(files)} prob files ({a.split})")
    assert files, "没有匹配的 npz"
    keys = files[0][1].groupby(KEY, sort=False).size().index

    meta = load_meta()
    lab = meta.groupby(KEY, sort=False)[C].first()
    y = lab.loc[keys].values.astype(np.float32) if a.split != "test" else None

    modes = [a.mode] if a.mode else ["mean", "max", "logit_mean", "mean_max", "max_mean90"]
    best_mode, best_auc = None, -1
    for mode in modes:
        acc = 0
        for _, d in files:
            acc = acc + agg_study(d, mode).loc[keys][P].values
        pred = acc / len(files)
        if y is not None:
            m, _ = macro_auc(y, pred)
            print(f"  agg={mode:11s} 等权集成 macroAUC {m:.4f}")
            if m > best_auc:
                best_mode, best_auc = mode, m
        else:
            best_mode = a.mode or "logit_mean"
    print("agg mode:", best_mode)

    M = [agg_study(d, best_mode).loc[keys][P].values for _, d in files]
    names = [short(f) for f, _ in files]

    if y is not None:
        print("\n单模型:")
        singles = []
        for n, v in zip(names, M):
            m, _ = macro_auc(y, v)
            singles.append(m)
            print(f"  {n:22s} {m:.4f}")
        order = np.argsort(singles)[::-1]
        print("ranked:", [(names[i], round(singles[i], 4)) for i in order])

    if a.split == "test":
        if not a.weights:
            print("test 阶段必须给 --weights")
            return
        wmap = {}
        for part in a.weights.split(","):
            if not part.strip():
                continue
            nm, w = part.split(":")
            wmap[nm.strip()] = float(w)
        acc = np.zeros_like(M[0])
        tot = 0.0
        used = []
        for n, v in zip(names, M):
            if n in wmap:
                acc += wmap[n] * v
                tot += wmap[n]
                used.append((n, wmap[n]))
        assert tot > 0, f"weights 没匹配任何模型: {names}"
        pred = acc / tot
        print("used:", used)
        rows = []
        for i, (sub, stu) in enumerate(keys):
            for c in range(1, 11):
                rows.append((sub, stu, c, float(pred[i, c - 1])))
        out = pd.DataFrame(rows, columns=["Subject_id", "Study_id", "Predict_class", "Probability"])
        out.to_csv(a.out, index=False, encoding="utf-8")
        print("written", a.out, out.shape)
        return

    # ---- valid: 贪心带放回 ----
    if a.greedy:
        sel = []
        cur = np.zeros_like(M[0])
        best_overall = -1
        for step in range(a.steps):
            best_i, best_s = None, -1
            for i, v in enumerate(M):
                cand = (cur * len(sel) + v) / (len(sel) + 1)
                s, _ = macro_auc(y, cand)
                if s > best_s:
                    best_i, best_s = i, s
            if best_s <= best_overall and step >= 2:
                print(f"  stop at step {step}, no gain")
                break
            sel.append(best_i)
            cur = (cur * (len(sel) - 1) + M[best_i]) / len(sel)
            best_overall = best_s
            print(f"  step {step}: +{names[best_i]:22s} -> {best_s:.4f}")
        from collections import Counter
        w = Counter(names[i] for i in sel)
        print("GREEDY WEIGHTS:", ",".join(f"{k}:{v}" for k, v in w.items()))
        m, per = macro_auc(y, cur)
        print(f"GREEDY ENSEMBLE valid macroAUC {m:.4f}")
        print("per class:", " ".join(f"{v:.3f}" for v in per))
    else:
        pred = np.mean(M, axis=0)
        m, per = macro_auc(y, pred)
        print(f"EQUAL ENSEMBLE valid macroAUC {m:.4f}")


if __name__ == "__main__":
    main()