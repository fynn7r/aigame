"""在 CPU 上对比「视图等权」与「正位0.8/侧位0.2」两种 study 聚合方式。

只读已保存的 probs_valid.npz，不动 GPU，可以在训练同时安全运行。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/root/_dsh")
from common import macro_auc  # noqa: E402

PROC = "/data/cxr_first/proc"
RUNS = "/data/cxr_first/runs"

meta = pd.read_csv(os.path.join(PROC, "meta.csv"), low_memory=False)
view = dict(zip(meta["Dicom_id"].values, meta["ViewPosition"].values))
front = {"AP", "PA", "PA LLD"}
lat = {"LATERAL", "LL"}


def agg_weighted(df, probs, w_front=0.8, w_lat=0.2):
    sids = df["study"].values
    vs = np.array([view.get(d, "AP") for d in df["dicom"].values])
    w = np.where([v in front for v in vs], w_front, np.where([v in lat for v in vs], w_lat, 0.5))
    order = pd.unique(sids)
    out = np.zeros((len(order), probs.shape[1]), dtype=np.float32)
    tot = np.zeros(len(order), dtype=np.float32)
    pos = {s: i for i, s in enumerate(order)}
    for k, s in enumerate(sids):
        i = pos[s]
        out[i] += probs[k] * w[k]
        tot[i] += w[k]
    out /= tot[:, None]
    bad = tot <= 1e-6
    if bad.any():
        _, m = agg_mean(df, probs)
        out[bad] = m[bad]
    return order, out


def agg_mean(df, probs):
    sids = df["study"].values
    order = pd.unique(sids)
    pos = {s: i for i, s in enumerate(order)}
    acc = np.zeros((len(order), probs.shape[1]), dtype=np.float32)
    cnt = np.zeros(len(order), dtype=np.float32)
    for k, s in enumerate(sids):
        i = pos[s]
        acc[i] += probs[k]
        cnt[i] += 1
    return order, acc / cnt[:, None]


def main():
    tags = sys.argv[1:] or [
        "p_b0_640_f2", "p_b0_640_f0", "p_b0_640_f1", "p_b0_640_f3", "p_b0_640_f4",
        "p_b0_448", "p_b0jft_f0", "x_b0_640_nopw",
    ]
    rows = []
    for tag in tags:
        p = os.path.join(RUNS, tag, "probs_valid.npz")
        if not os.path.exists(p):
            print("skip", tag)
            continue
        z = np.load(p, allow_pickle=True)
        probs = z["probs"].astype(np.float32)
        df = pd.DataFrame({"dicom": z["dicom"].astype(str), "study": z["study"].astype(str)})
        y = meta.set_index("Dicom_id").loc[df["dicom"].values, [f"c{i}" for i in range(1, 11)]]
        y = y.values.astype(np.float32)
        lab = y

        o1, a1 = agg_mean(df, probs)
        sy = pd.DataFrame(lab, index=df["study"].values).groupby(level=0).first().loc[o1].values
        auc_mean, _ = macro_auc(sy, a1)

        best = (0.5, auc_mean)
        for wf in (0.6, 0.7, 0.8, 0.9, 1.0):
            o2, a2 = agg_weighted(df, probs, wf, 1 - wf)
            sy2 = pd.DataFrame(lab, index=df["study"].values).groupby(level=0).first().loc[o2].values
            a, _ = macro_auc(sy2, a2)
            if a > best[1]:
                best = (wf, a)
        rows.append((tag, auc_mean, best[0], best[1]))
        print("%-16s 等权=%.4f  正位%.1f/侧位%.1f=%.4f  (增益 %+.4f)"
              % (tag, auc_mean, best[0], 1 - best[0], best[1], best[1] - auc_mean), flush=True)

    if rows:
        print("\n平均: 等权 %.4f -> 视图加权 %.4f" % (
            np.mean([r[1] for r in rows]), np.mean([r[3] for r in rows])))


if __name__ == "__main__":
    main()