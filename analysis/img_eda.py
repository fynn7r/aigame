# -*- coding: utf-8 -*-
"""图片文件级探查：真实分辨率/模式/大小/解码耗时/与CSV尺寸一致性/每检查片数"""
import os, random, time
import pandas as pd
from PIL import Image

BASE = r"E:\aigame\cxr_first\cxr_first"
d = pd.read_csv(os.path.join(BASE, "train.csv"))

def path_of(r):
    dd = os.path.join(BASE, "train", r.Subject_id, r.Study_id)
    p = os.path.join(dd, r.Dicom_id + ".jpg")
    if os.path.exists(p):
        return p
    return os.path.join(dd, sorted(os.listdir(dd))[0])

rows = []
for i in random.Random(0).sample(range(len(d)), 400):
    r = d.iloc[i]
    p = path_of(r)
    try:
        im = Image.open(p)
        rows.append((im.size[0], im.size[1], im.mode, os.path.getsize(p) / 1e6, r.Rows, r.Columns))
    except Exception as e:
        print("ERR", p, e)
s = pd.DataFrame(rows, columns=["w", "h", "mode", "MB", "csvR", "csvC"])
print("mode:", s["mode"].value_counts().to_dict())
print(s[["w", "h", "MB"]].describe().loc[["min", "25%", "50%", "75%", "max"]].round(2).to_string())
print("w==csvC 比例: %.3f | h==csvR 比例: %.3f" % ((s.w == s.csvC).mean(), (s.h == s.csvR).mean()))
print("jpg MB mean %.2f, p95 %.2f, max %.2f" % (s.MB.mean(), s.MB.quantile(.95), s.MB.max()))
print("估算 train 总大小 %.1f GB" % (s.MB.mean() * len(d) / 1024))

# 解码耗时
t = time.time(); n = 0
for i in random.Random(1).sample(range(len(d)), 40):
    r = d.iloc[i]
    im = Image.open(path_of(r)); im = im.convert("L"); n += 1
print("PIL 解码+转L 平均 %.0f ms/张（原图）" % ((time.time() - t) / n * 1000))

# 每 study 片数分布
ips = d.groupby("Study_id").size()
print("每检查片数分布:", ips.value_counts().sort_index().to_dict())
print("每患者检查数:", d.groupby("Subject_id").Study_id.nunique().value_counts().sort_index().head(6).to_dict())

# 各 split 目录大小
for split in ["train", "valid", "test"]:
    tot = 0; cnt = 0
    for root, _, fs in os.walk(os.path.join(BASE, split)):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(root, f)); cnt += 1
            except OSError:
                pass
        if split == "test" and cnt > 5000:
            break
    print(f"{split}: {cnt} 文件, {tot/1024**3:.2f} GB")