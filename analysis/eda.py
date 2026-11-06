# -*- coding: utf-8 -*-
"""完整数据集探查：标签分布 / 共现 / 视图 / 尺寸 / 泄漏 / 图片文件校验"""
import os, sys, json
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

BASE = r"E:\aigame\cxr_first\cxr_first"
OUT = r"E:\aigame\cxr_first\analysis"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)

def load(n):
    df = pd.read_csv(os.path.join(BASE, f"{n}.csv"))
    df.columns = [c.strip() for c in df.columns]
    return df

train, valid, test = load("train"), load("valid"), load("test")
rep = []
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); rep.append(s)

P("="*78); P("【0】表结构"); P("="*78)
for n, d in [("train", train), ("valid", valid), ("test", test)]:
    P(f"{n:6s} rows={len(d):6d} cols={list(d.columns)}")

P("\n" + "="*78); P("【1】规模"); P("="*78)
for n, d in [("train", train), ("valid", valid), ("test", test)]:
    P(f"{n:6s} 影像={len(d):6d} 检查(study)={d.Study_id.nunique():6d} 患者={d.Subject_id.nunique():6d} "
      f"平均每检查={len(d)/d.Study_id.nunique():.2f}张")

P("\n" + "="*78); P("【2】标签分布（样本=影像级，保持原表口径）"); P("="*78)
lab = train["Predict_class"].astype(str)
n_lab = lab.apply(lambda s: len(s.split(";")))
P(f"单标签 {int((n_lab==1).sum())} ({(n_lab==1).mean()*100:.2f}%) | 多标签 {int((n_lab>1).sum())} ({(n_lab>1).mean()*100:.2f}%) | 平均标签数 {n_lab.mean():.2f} 最多 {n_lab.max()}")
P(f"标签组合数 {lab.nunique()}")
P("\nTop 20 组合:")
for k, v in Counter(lab).most_common(20):
    P(f"   {k:>16} {v:6d} {v/len(train)*100:6.2f}%")

cnt = Counter(); 
for s in lab:
    for x in s.split(";"):
        cnt[int(x)] += 1
NAMES = {1:"纵隔心影增大",2:"气胸",3:"肺部实变",4:"肺炎",5:"肺水肿",6:"心影扩大",7:"肺不张",8:"肺部阴影",9:"胸腔积液",10:"无异常"}
P("\n单类频次（影像级）:")
for k in range(1, 11):
    v = cnt.get(k, 0)
    P(f"  类{k:2d} {NAMES[k]:<7s} {v:6d} {v/len(train)*100:6.2f}%  {'#'*int(v/len(train)*60)}")

# 检查级（study）标签一致性
P("\n" + "="*78); P("【3】同一 study 内多张片标签是否一致 / 检查级流行率"); P("="*78)
g = train.groupby("Study_id")["Predict_class"].nunique()
P(f"study 数 {len(g)}，标签不一致的 study: {int((g>1).sum())} ({(g>1).mean()*100:.2f}%)")
study_lab = train.groupby("Study_id")["Predict_class"].first()
scnt = Counter()
for s in study_lab:
    for x in str(s).split(";"):
        scnt[int(x)] += 1
P("\n检查级流行率（分母=18513 study）:")
for k in range(1, 11):
    v = scnt.get(k, 0)
    P(f"  类{k:2d} {NAMES[k]:<7s} {v:6d} {v/len(study_lab)*100:6.2f}%")

P("\n" + "="*78); P("【4】标签共现（检查级，Jaccard/条件概率）"); P("="*78)
M = np.zeros((11, 11))
for s in study_lab:
    ks = [int(x) for x in str(s).split(";")]
    for i in ks:
        for j in ks:
            M[i][j] += 1
P("条件概率 P(列|行) %：")
hdr = "行\\列 " + "".join(f"{c:>6d}" for c in range(1, 11))
P(hdr)
for i in range(1, 11):
    row = f"  {i:2d}  "
    for j in range(1, 11):
        row += f"{M[i][j]/M[i][i]*100:6.1f}" if M[i][i] > 0 and i != j else "     -"
    P(row)
# 10号与其他类共现
P("\n无异常(10) 与其它类共现次数: " + ", ".join(f"{j}:{int(M[10][j])}" for j in range(1, 10)))

P("\n" + "="*78); P("【5】视图 / 体位"); P("="*78)
for n, d in [("train", train), ("valid", valid), ("test", test)]:
    c = d["ViewPosition"].fillna("<空>").value_counts()
    P(f"{n:6s} " + "  ".join(f"{k}:{v}({v/len(d)*100:.1f}%)" for k, v in c.items()))
P("\nPatientOrientation (train): " + str(dict(train["PatientOrientation"].fillna("<空>").value_counts())))

P("\n" + "="*78); P("【6】图像尺寸（CSV 元数据）"); P("="*78)
sz = train.groupby(["Rows", "Columns"]).size().sort_values(ascending=False)
P("Top10 Rows x Cols:")
for (r, c), v in sz.head(10).items():
    P(f"   {r}x{c}: {v} ({v/len(train)*100:.2f}%)")
P(f"Rows {train.Rows.min()}~{train.Rows.max()}  Cols {train.Columns.min()}~{train.Columns.max()}")

P("\n" + "="*78); P("【7】数据泄漏检查"); P("="*78)
for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    da, db = {"train": train, "valid": valid, "test": test}[a], {"train": train, "valid": valid, "test": test}[b]
    P(f"{a}∩{b}: subject={len(set(da.Subject_id)&set(db.Subject_id))} study={len(set(da.Study_id)&set(db.Study_id))} "
      f"dicom={len(set(da.Dicom_id)&set(db.Dicom_id))}")

P("\n" + "="*78); P("【8】ImageID 连续性（数据是否被切片划分）"); P("="*78)
ids = train.Dicom_id.str.replace("pic_", "").astype(int)
P(f"train Dicom 序号 {ids.min()}~{ids.max()}  单调递增? {ids.is_monotonic_increasing}")
vid = valid.Dicom_id.str.replace("pic_", "").astype(int)
tid = test.Dicom_id.str.replace("pic_", "").astype(int)
P(f"valid 序号 {vid.min()}~{vid.max()}  test 序号 {tid.min()}~{tid.max()}")
P(f"train/valid/test 序号区间是否重叠: train-valid={len(set(ids)&set(vid))} train-test={len(set(ids)&set(tid))}")
print("\n".join(rep[:0]))
with open(os.path.join(OUT, "eda_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(rep))