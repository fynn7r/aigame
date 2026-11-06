# -*- coding: utf-8 -*-
import csv, os
from collections import Counter, defaultdict

BASE = r"E:\aigame\cxr_first\cxr_first"

def load(name):
    with open(os.path.join(BASE, name), "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

train = load("train.csv")
valid = load("valid.csv")
test  = load("test.csv")

def cnt(rows, key):
    c = Counter(r[key] for r in rows)
    return c

def nunique(rows, key):
    return len(set(r[key] for r in rows))

print("="*60)
print("【1. 总体规模】")
print(f"  train : {len(train):>6} 片 | subjects {nunique(train,'Subject_id'):>5} | studies {nunique(train,'Study_id'):>5}")
print(f"  valid : {len(valid):>6} 片 | subjects {nunique(valid,'Subject_id'):>5} | studies {nunique(valid,'Study_id'):>5}")
print(f"  test  : {len(test):>6} 片 | subjects {nunique(test,'Subject_id'):>5} | studies {nunique(test,'Study_id'):>5}")

print("\n" + "="*60)
print("【2. 训练集类别分布（多标签，按 ';' 拆开）】")
tot = len(train)
# 整体标签（保留原始组合）
combo = Counter(r["Predict_class"] for r in train)
multi = sum(1 for k in combo if ";" in k)
print(f"  单标签样本: {tot - sum(combo[k] for k in combo if ';' in k)} ({(tot - sum(combo[k] for k in combo if ';' in k))/tot*100:.2f}%)")
print(f"  多标签样本: {sum(combo[k] for k in combo if ';' in k)} ({sum(combo[k] for k in combo if ';' in k)/tot*100:.2f}%)")
print(f"  不同标签组合数: {len(combo)}")
print("\n  Top 15 标签组合:")
for k, v in combo.most_common(15):
    print(f"    {k:>15}: {v:>5} ({v/tot*100:5.2f}%)")

# 单标签频次
cls = Counter()
n_labels_per_img = []
for r in train:
    labs = r["Predict_class"].split(";")
    n_labels_per_img.append(len(labs))
    for x in labs:
        cls[x.strip()] += 1
print(f"\n  每个样本平均标签数: {sum(n_labels_per_img)/len(n_labels_per_img):.2f} (最多 {max(n_labels_per_img)})")
print("  单标签频次（每片只要含该类就计数，总和 > 样本数）:")
for k in sorted(cls, key=lambda x: int(x)):
    v = cls[k]
    bar = "█" * int(v/tot*50)
    print(f"    类 {k:>2}: {v:>5} 片 ({v/tot*100:5.2f}%)  {bar}")

print("\n" + "="*60)
print("【3. 视图位置 ViewPosition】")
for label, rows in [("train", train), ("valid", valid), ("test ", test)]:
    c = cnt(rows, "ViewPosition")
    s = "  ".join(f"{k}:{v}({v/len(rows)*100:.1f}%)" for k,v in c.most_common())
    print(f"  {label}: {s}")

print("\n" + "="*60)
print("【4. 病人体位 PatientOrientation (train)】")
for k, v in cnt(train, "PatientOrientation").most_common():
    print(f"  {k:>12}: {v:>5} ({v/len(train)*100:.2f}%)")

print("\n" + "="*60)
print("【5. 图像尺寸分布 Top10 (Rows x Cols)】")
sizes = Counter(f"{r['Rows']}x{r['Columns']}" for r in train)
for k, v in sizes.most_common(10):
    print(f"  {k:>12}: {v:>5} ({v/len(train)*100:5.2f}%)")
rows_ = [int(r["Rows"]) for r in train]
cols_ = [int(r["Columns"]) for r in train]
print(f"  Rows 范围 {min(rows_)}~{max(rows_)}, 均值 {sum(rows_)/len(rows_):.0f}")
print(f"  Cols 范围 {min(cols_)}~{max(cols_)}, 均值 {sum(cols_)/len(cols_):.0f}")

print("\n" + "="*60)
print("【6. 多图情况】")
subj2studies = defaultdict(set)
study2imgs = Counter()
subj2imgs = Counter()
for r in train:
    subj2studies[r["Subject_id"]].add(r["Study_id"])
    study2imgs[r["Study_id"]] += 1
    subj2imgs[r["Subject_id"]] += 1
sps = [len(v) for v in subj2studies.values()]
ips = list(study2imgs.values())
ipi = list(subj2imgs.values())
lat_studies = set(r["Study_id"] for r in train if r["ViewPosition"]=="LATERAL")
print(f"  每患者 study 数: 均值 {sum(sps)/len(sps):.2f}, 最多 {max(sps)}")
print(f"  每 study 片数:    均值 {sum(ips)/len(ips):.2f}, 最多 {max(ips)}")
print(f"  每患者片数:       均值 {sum(ipi)/len(ipi):.2f}, 最多 {max(ipi)}")
print(f"  单 study 仅 1 片: {sum(1 for x in ips if x==1)} / {len(ips)} ({sum(1 for x in ips if x==1)/len(ips)*100:.1f}%)")
print(f"  含侧位(LATERAL)的 study: {len(lat_studies)}")

print("\n" + "="*60)
print("【7. 类别 × 视图位置 交叉 (train，按单标签频次)】")
ct = defaultdict(Counter)
for r in train:
    for lab in r["Predict_class"].split(";"):
        ct[lab.strip()][r["ViewPosition"]] += 1
vps = sorted({r["ViewPosition"] for r in train})
print("  class | " + " | ".join(f"{v:>8}" for v in vps) + " |   total")
for k in sorted(ct, key=lambda x: int(x)):
    row = ct[k]
    print(f"  {k:>5} | " + " | ".join(f"{row.get(v,0):>8}" for v in vps) + f" | {sum(row.values()):>6}")

print("\n" + "="*60)
print("【8. valid 单标签频次 vs train】")
vcls = Counter()
for r in valid:
    for x in r["Predict_class"].split(";"):
        vcls[x.strip()] += 1
for k in sorted(set(list(cls)+list(vcls)), key=lambda x: int(x)):
    tp = cls.get(k,0)/tot*100
    vp_ = vcls.get(k,0)/len(valid)*100
    print(f"  类 {k:>2}: train {tp:5.2f}%  |  valid {vp_:5.2f}%  |  差 {vp_-tp:+.2f}pp")

print("\n" + "="*60)
print("【9. 泄漏检查】")
tr_s = set(r["Subject_id"] for r in train); va_s = set(r["Subject_id"] for r in valid); te_s = set(r["Subject_id"] for r in test)
tr_st = set(r["Study_id"] for r in train); va_st = set(r["Study_id"] for r in valid); te_st = set(r["Study_id"] for r in test)
print(f"  train∩valid subject: {len(tr_s & va_s)}")
print(f"  train∩test  subject: {len(tr_s & te_s)}")
print(f"  valid∩test  subject: {len(va_s & te_s)}")
print(f"  train∩valid study:    {len(tr_st & va_st)}")
print(f"  train∩test  study:    {len(tr_st & te_st)}")
