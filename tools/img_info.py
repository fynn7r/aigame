import os, glob
import numpy as np
import pandas as pd
from PIL import Image

ROOT = "/data/cxr_first"
d = pd.read_csv(os.path.join(ROOT, "train.csv"), nrows=40)
sizes = []
for s, st, dc in zip(d.Subject_id, d.Study_id, d.Dicom_id):
    p = os.path.join(ROOT, "train", str(s), str(st), str(dc) + ".jpg")
    if os.path.exists(p):
        im = Image.open(p)
        sizes.append((im.size, os.path.getsize(p) // 1024))
print("n", len(sizes))
for sz, kb in sizes[:12]:
    print(sz, kb, "KB")
ws = [s[0][0] for s in sizes]
hs = [s[0][1] for s in sizes]
print("width  min/med/max", min(ws), int(np.median(ws)), max(ws))
print("height min/med/max", min(hs), int(np.median(hs)), max(hs))