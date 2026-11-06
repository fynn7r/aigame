# -*- coding: utf-8 -*-
"""检查原始 CXR 的黑边比例 / 内容 bbox / 长宽比，判断 512 缓存是否浪费了解析度。"""
import os, sys, glob, random
import numpy as np
from PIL import Image

ROOT = r"E:\aigame\cxr_first\cxr_first"
random.seed(0)
files = glob.glob(os.path.join(ROOT, "train", "**", "*.jpg"), recursive=True)
print("total", len(files))
sample = random.sample(files, 120)

ratios, bbox_frac, hw = [], [], []
for f in sample:
    im = Image.open(f)
    w, h = im.size
    hw.append((w, h))
    a = np.array(im.resize((256, 256), Image.BILINEAR), dtype=np.float32)
    # 用 3% 分位数做黑边阈值
    thr = max(8.0, np.percentile(a, 2) + 6)
    mask = a > thr
    if mask.sum() < 100:
        continue
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bbox_frac.append(((y1 - y0 + 1) * (x1 - x0 + 1)) / (256 * 256))
    ratios.append(mask.mean())

print("content pixels in bbox: mean %.3f  p10 %.3f  p50 %.3f" % (
    np.mean(bbox_frac), np.percentile(bbox_frac, 10), np.percentile(bbox_frac, 50)))
print("non-black fraction: mean %.3f" % np.mean(ratios))
w = np.array([x[0] for x in hw]); h = np.array([x[1] for x in hw])
print("width  mean %d  h mean %d" % (w.mean(), h.mean()))
print("portrait(w<h): %.2f  square: %.2f" % ((w < h).mean(), (abs(w - h) < 20).mean()))
print("sample sizes:", sorted(set(map(tuple, hw[:40])))[:8])