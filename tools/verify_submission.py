# -*- coding: utf-8 -*-
"""提交前自检：逐项核对 submission 文件是否符合赛方格式要求。

用法:
    python tools/verify_submission.py submissions/submission_77.59.csv

全部通过才显示「可以提交」。
"""
import hashlib
import sys

import pandas as pd

# 期望值（来自官方 test.csv）
EXPECT_STUDIES = 1414
EXPECT_CLASSES = set(range(1, 11))
EXPECT_ROWS = EXPECT_STUDIES * len(EXPECT_CLASSES)
EXPECT_COLS = ["Subject_id", "Study_id", "Predict_class", "Probability"]


def main(path="submissions/submission_77.59.csv", test_csv="cxr_first/test.csv"):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'[通过]' if cond else '[失败]'} {name}{('  ' + detail) if detail else ''}")
        if not cond:
            ok = False

    raw = open(path, "rb").read()
    print(f"文件: {path}")
    print(f"  md5 {hashlib.md5(raw).hexdigest()}   {len(raw):,} 字节\n")

    check("无 BOM 头", raw[:3] != b"\xef\xbb\xbf")
    try:
        text = raw.decode("utf-8")
        check("UTF-8 可解码", True)
    except UnicodeDecodeError as e:
        check("UTF-8 可解码", False, str(e))
        return 1

    lines = text.splitlines()
    check("行数 = 表头 1 + 数据 14140", len(lines) == EXPECT_ROWS + 1, f"实际 {len(lines)}")
    check("表头字段与顺序", lines[0].strip() == ",".join(EXPECT_COLS), repr(lines[0]))

    d = pd.read_csv(path)
    check("列名一致", list(d.columns) == EXPECT_COLS, str(list(d.columns)))
    check("数据行数 = 14140", len(d) == EXPECT_ROWS, f"实际 {len(d)}")
    check("study 数 = 1414", d["Study_id"].nunique() == EXPECT_STUDIES, f"实际 {d['Study_id'].nunique()}")
    check("每 study 恰好 10 行", set(d.groupby("Study_id").size().unique()) == {10})
    check("类别集合 = 1..10", set(d["Predict_class"].unique()) == EXPECT_CLASSES)
    check("无重复 (Study_id, Predict_class)", d.duplicated(["Study_id", "Predict_class"]).sum() == 0)
    check("无缺失值", d.isna().sum().sum() == 0)
    check("概率在 [0,1] 内", ((d["Probability"] >= 0) & (d["Probability"] <= 1)).all(),
          f"范围 [{d['Probability'].min():.6f}, {d['Probability'].max():.6f}]")

    # 与官方 test.csv 对齐
    try:
        t = pd.read_csv(test_csv)
        off = list(dict.fromkeys(t["Study_id"].astype(str)))
        mine = list(dict.fromkeys(d["Study_id"].astype(str)))
        check("study 集合与官方 test.csv 完全一致", set(off) == set(mine),
              f"缺失 {len(set(off) - set(mine))} / 多余 {len(set(mine) - set(off))}")
        check("study 顺序与官方一致", off == mine)
        m = dict(zip(t["Study_id"].astype(str), t["Subject_id"].astype(str)))
        first = d.groupby("Study_id", sort=False).first()
        bad = sum(1 for s, r in first.iterrows() if m.get(str(s)) != str(r["Subject_id"]))
        check("Subject_id 与官方映射一致", bad == 0, f"不一致 {bad} 个")
    except FileNotFoundError:
        print("  [跳过] 未找到官方 test.csv，跳过对齐检查")

    print("\n" + ("可以提交。" if ok else "存在问题，先别提交。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))