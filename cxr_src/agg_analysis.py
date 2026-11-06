# -*- coding: utf-8 -*-
"""分析：体位(frontal/lateral) 对 AUC 的影响 + 各种 study 聚合方式。

  python agg_analysis.py /data/cxr_first/runs/r34_384_f0/probs_valid.npz
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_meta, macro_auc

P = [f"p{i}" for i in range(1, 11)]
C = [f"c{i}" for i in range(1, 11)]
KEY = ["Subject_id", "Study_id"]


def main(path):
    z = np.load(path, allow_pickle=True)
    probs = z["probs"].astype(np.float32)
    d = pd.DataFrame(probs, columns=P)
    d["Dicom_id"] = z["dicom"]
    d["Study_id"] = z["study"]
    d["Subject_id"] = z["subject"]

    meta = load_meta()
    m = meta[["Dicom_id", "ViewPosition"]].drop_duplicates("Dicom_id")
    d = d.merge(m, on="Dicom_id", how="left")
    d["ViewPosition"] = d["ViewPosition"].fillna("")

    lab = meta.groupby(KEY, sort=False)[C].first()
    lab_img = meta.set_index("Dicom_id")[C]

    front = d.ViewPosition.isin(["AP", "PA"])
    print(f"images {len(d)}  frontal {front.sum()} ({front.mean():.1%})  lateral {(~front).sum()}")

    for name, sub in [("ALL", d), ("FRONTAL only", d[front]), ("LATERAL only", d[~front])]:
        if len(sub) < 50:
            continue
        y = lab_img.loc[sub.Dicom_id].values.astype(np.float32)
        mm, per = macro_auc(y, sub[P].values)
        print(f"  IMAGE {name:14s} n={len(sub):5d} macroAUC {mm:.4f}")

    # study 级：只看有正位片的 study
    st_has_front = d.groupby(KEY).apply(lambda x: x.ViewPosition.isin(["AP", "PA"]).any())
    st_has_front.index.names = KEY
    for name, sel in [("all studies", d), ("frontal-bearing studies", d)]:
        g = sel.groupby(KEY, sort=False)
        keys = g.size().index
        y = lab.loc[keys].values.astype(np.float32)
        for mode, agg in [("mean", g[P].mean()),
                          ("max", g[P].max()),
                          ("mean+max", 0.5 * g[P].mean() + 0.5 * g[P].max())]:
            mm, _ = macro_auc(y, agg.loc[keys].values)
            print(f"  STUDY {name:22s} {mode:9s} n={len(keys):4d} macroAUC {mm:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])