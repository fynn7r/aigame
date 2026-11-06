# -*- coding: utf-8 -*-
"""把模型清单做成带格式的 Excel（模型明细 + 架构汇总 + 训练配方）。"""
import os

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "models_summary.csv"), low_memory=False)
df = df[df["tag"] != "smoke"].copy()

NAME = {
    "efficientnet_b0": "EfficientNet-B0", "tf_efficientnet_b0.ns_jft_in1k": "EfficientNet-B0 (JFT)",
    "densenet201": "DenseNet-201", "densenet121": "DenseNet-121",
    "convnext_small": "ConvNeXt-Small", "resnet50": "ResNet-50", "resnet34": "ResNet-34",
}
df["模型"] = df["arch"].map(NAME).fillna(df["arch"])
df["是否集成"] = df["tag"].isin(
    ["ref_cnxs_384", "p_b0_640_f2", "x_b0_640_nopw", "p_b0jft_f0"]).map({True: "✔", False: ""})

out = df[["tag", "模型", "size", "fold", "epochs", "best_ep", "bs", "lr", "wd",
          "drop_path", "pos_weight", "mixup", "pretrain", "fold_auc", "valid_auc", "是否集成"]]
out.columns = ["实验代号", "模型", "分辨率", "折", "计划轮数", "最佳轮", "Batch",
               "学习率", "WeightDecay", "DropPath", "pos_weight", "mixup 强度",
               "预训练权重", "折内AUC", "固定验证集AUC", "v4集成"]
out = out.sort_values(["模型", "分辨率", "折"], ascending=[True, True, True])

agg = (df.groupby(["模型", "size"])
       .agg(折数=("tag", "count"), 折内AUC均值=("fold_auc", "mean"),
            固定验证集均值=("valid_auc", "mean"), 最佳轮=("best_ep", "median"))
       .reset_index().sort_values("折内AUC均值", ascending=False))
agg.columns = ["模型", "分辨率", "训练折数", "折内AUC均值", "固定验证集AUC均值", "典型最佳轮次"]

recipe = pd.DataFrame([
    ["数据切分", "按病人（Subject_id）分组 5 折；固定验证集 859 study 只用于选模型"],
    ["损失函数", "BCEWithLogitsLoss（早期版本带类别频率正样本权重，上限 20；后期 pos_weight=1 关闭）"],
    ["优化器", "AdamW"],
    ["学习率调度", "余弦退火，前 1 轮线性 warmup，衰减到 0"],
    ["混合精度", "AMP fp16 全程开启"],
    ["权重平滑", "EMA，衰减系数 0.999，用 EMA 权重做验证与保存"],
    ["数据增强", "随机裁剪 0.75–1.0、水平翻转 0.5、±10° 旋转、亮度 ±15%、伽马 0.7–1.4、随机擦除 0.25"],
    ["推理 TTA", "原图与水平翻转各 50% 等权平均（实测 +1.06 个千分点）"],
    ["Study 聚合", "同一 study 内多张视图概率等权平均（正侧位加权对照无增益）"],
    ["集成方式", "验证集上带放回贪心前向选择，选 4 个模型等权"],
    ["硬件", "海光 DCU K100_AI，12GB 显存 / 8 CPU / 32GB 内存，无外网"],
    ["吞吐实测", "b0@448 78 张/秒；b0@640 37 张/秒；dn201@448 26 张/秒；cnx-s@384 15 张/秒；dn121@1024 8 张/秒"],
], columns=["项目", "设置"])

path = os.path.join(BASE, "模型与参数清单.xlsx")
with pd.ExcelWriter(path, engine="openpyxl") as xw:
    out.to_excel(xw, sheet_name="模型明细", index=False)
    agg.to_excel(xw, sheet_name="架构汇总", index=False)
    recipe.to_excel(xw, sheet_name="训练配方", index=False)

    head_fill = PatternFill("solid", fgColor="1F3A5F")
    head_font = Font(color="FFFFFF", bold=True, size=10, name="微软雅黑")
    thin = Side(style="thin", color="C8D0DC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for ws, widths in [(xw.sheets["模型明细"],
                        [24, 22, 9, 6, 10, 8, 8, 10, 13, 11, 12, 11, 14, 11, 15, 9]),
                       (xw.sheets["架构汇总"], [22, 10, 11, 13, 18, 14]),
                       (xw.sheets["训练配方"], [16, 88])]:
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w
        for c in ws[1]:
            c.fill = head_fill
            c.font = head_font
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 32
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=len(widths)):
            for c in row:
                c.border = border
                c.font = Font(size=10, name="微软雅黑")
                if c.column > 3:
                    c.alignment = Alignment(horizontal="center")
                else:
                    c.alignment = Alignment(horizontal="left")
        ws.freeze_panes = "A2"

    ws = xw.sheets["模型明细"]
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, 3).value == 640 and "B0" in str(ws.cell(r, 2).value):
            for c in range(1, 17):
                ws.cell(r, c).fill = PatternFill("solid", fgColor="FFF4E0")

print("saved", path, os.path.getsize(path))
print(out.to_string(index=False, max_rows=60))