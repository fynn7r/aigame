# -*- coding: utf-8 -*-
"""检查 train/valid/test 三划分的图像统计是否一致（是否有域偏移）。"""
import os, glob, random
import numpy as np
import pandas as pd
from PIL import Image

ROOT = r"E:\aigame\cxr_first\cxr_first"
random.seed(0)


def stats(img_dir, n=150):
    files = glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)
    sample = random.sample(files, min(n, len(files)))
    rows = []
    for f in sample:
        im = Image.open(f)
        a = np.asarray(im.resize((128, 128), Image.BILINEAR), dtype=np.float32)
        rows.append(dict(w=im.size[0], h=im.size[1],
                         mean=a.mean(), std=a.std(),
                         p1=np.percentile(a, 1), p50=np.percentile(a, 50), p99=np.percentile(a, 99),
                         dark=(a < 20).mean()))
    return pd.DataFrame(rows), len(files)


for split in ["train", "valid", "test"]:
    df, n_all = stats(os.path.join(ROOT, split))
    print(f"{split:6s} files={n_all:6d}  mean={df['mean'].mean():6.1f} std={df['std'].mean():6.1f} "
          f"p1={df['p1'].mean():5.1f} p50={df['p50'].mean():6.1f} p99={df['p99'].mean():6.1f} "
          f"dark={df['dark'].mean():.3f}  WxH={df['w'].mean():.0f}x{df['h'].mean():.0f}")

# 患者/检查密度
for split in ["train", "valid", "test"]:
    csv = os.path.join(ROOT, f"{split}.csv")
    d = pd.read_csv(csv)
    print(f"{split:6s} rows={len(d)} subjects={d.Subject_id.nunique()} studies={d.Study_id.nunique()} "
          f"studies/subject={d.Study_id.nunique()/d.Subject_id.nunique():.1f}")