# -*- coding: utf-8 -*-
"""校验 submission.csv 是否符合赛题格式，并给出基本统计。

用法: python check_submission.py /path/to/submission.csv
"""
import sys
import pandas as pd
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/data/cxr_first/submission.csv"
sub = pd.read_csv(path)
print("shape", sub.shape, "| columns", list(sub.columns))
need = ["Subject_id", "Study_id", "Predict_class", "Probability"]
missing = [c for c in need if c not in sub.columns]
assert not missing, f"缺少列 {missing}"

test = pd.read_csv("/data/cxr_first/test.csv")
want = set(zip(test.Subject_id, test.Study_id))
got = set(zip(sub.Subject_id, sub.Study_id))
print(f"study 覆盖: {len(got)} / {len(want)} | 缺失 {len(want-got)} | 多余 {len(got-want)}")

g = sub.groupby(["Subject_id", "Study_id"]).size()
print("每 study 行数分布:", g.value_counts().to_dict())
assert (g == 10).all(), "存在 study 不是 10 行"

cls = sub.groupby(["Subject_id", "Study_id"])["Predict_class"].apply(lambda s: sorted(s.tolist()))
bad = cls[cls.apply(lambda x: x != list(range(1, 11)))]
print(f"类别 id 不完整的 study: {len(bad)}")
assert len(bad) == 0, bad.head()

p = sub["Probability"].astype(float)
print(f"概率范围 [{p.min():.4f}, {p.max():.4f}] mean {p.mean():.4f}")
assert p.between(0, 1).all(), "概率越界"
print("NaN:", int(p.isna().sum()))

print("\n各类平均概率:")
for c in range(1, 11):
    d = sub[sub.Predict_class == c]["Probability"]
    print(f"  类{c:2d}: mean {d.mean():.3f}  p>0.5 {(d>0.5).mean()*100:5.1f}%  max {d.max():.3f}")

print("\n>0.5 的标签数分布:", sub[sub.Probability > 0.5].groupby(["Subject_id", "Study_id"]).size().value_counts().to_dict())
print("\nOK: 格式检查通过")