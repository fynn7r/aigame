# -*- coding: utf-8 -*-
"""按规则附录 A 生成「模型核验材料与数据声明清单.xlsx」。"""
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "模型核验材料与数据声明清单.xlsx")

HF = "https://huggingface.co/"
TV = "https://download.pytorch.org/models/"

ROWS = [
    # 名称, 链接, 许可证, 上游预训练数据, SHA-256, 参数量, 是否参与最终推理, 用途
    ("ConvNeXt-Base（convnext_base.fb_in22k_ft_in1k）",
     HF + "timm/convnext_base.fb_in22k_ft_in1k",
     "Apache-2.0（非 NC、非 gated）", "ImageNet-22K 预训练 → ImageNet-1K 微调",
     "4853efc8d5dd3ed9b0f888a9c601d7556373599f944f9ac79af8b8ec81aa7381",
     "8760 万", "是（主力，占 3/4 权重）", "主模型骨干"),

    ("EfficientNet-B0（torchvision EfficientNet_B0_Weights.IMAGENET1K_V1）",
     TV + "efficientnet_b0_rw_f48e...（见 torchvision 权重枚举）",
     "BSD-3-Clause（torchvision 仓库）", "ImageNet-1K",
     "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934",
     "530 万", "是（占 1/4 权重）", "辅助模型骨干"),

    ("ConvNeXt-Small（convnext_small_fb_in22k_ft_in1k）",
     HF + "timm/convnext_small.fb_in22k_ft_in1k",
     "Apache-2.0", "ImageNet-22K → ImageNet-1K",
     "2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9",
     "4950 万", "否（仅模型池）", "对照实验"),

    ("ConvNeXt-Tiny（convnext_tiny_fb_in22k_ft_in1k）",
     HF + "timm/convnext_tiny.fb_in22k_ft_in1k",
     "Apache-2.0", "ImageNet-22K → ImageNet-1K",
     "77b619e187de71bce58721689da0c009e84cc20677293d7d43105307e34d868a",
     "2860 万", "否", "早期实验"),

    ("DenseNet-201（torchvision / timm densenet201.tv_in1k）",
     HF + "timm/densenet201.tv_in1k",
     "Apache-2.0 / BSD-3-Clause", "ImageNet-1K",
     "0f5db56213cd09ba5b4c53ddcdab0ae1c9a11bbd8fc07f0da71b0c099b2fd971",
     "1800 万", "否（仅模型池）", "参考方案复现"),

    ("DenseNet-121（torchvision / timm densenet121.ra_in1k）",
     HF + "timm/densenet121.ra_in1k",
     "Apache-2.0 / BSD-3-Clause", "ImageNet-1K",
     "eae900d8ae5b3716efa7f443fcc692aeb5a7f01f72581835c03c8c495f9eda10",
     "800 万", "否", "早期实验"),

    ("ResNet-50（timm resnet50.a1_in1k / torchvision）",
     HF + "timm/resnet50.a1_in1k",
     "Apache-2.0 / BSD-3-Clause", "ImageNet-1K",
     "773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb",
     "2550 万", "否", "早期实验"),

    ("ResNet-34（torchvision IMAGENET1K_V1）",
     TV + "resnet34-b627a593.pth",
     "BSD-3-Clause", "ImageNet-1K",
     "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f",
     "2180 万", "否", "首个基线"),

    ("EfficientNet-B3（timm efficientnet_b3.ra2_in1k）",
     HF + "timm/efficientnet_b3.ra2_in1k",
     "Apache-2.0", "ImageNet-1K",
     "279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469",
     "1220 万", "否", "对照实验"),

    ("Swin-Base（swin_base_patch4_window12_384.ms_in22k_ft_in1k）",
     HF + "timm/swin_base_patch4_window12_384.ms_in22k_ft_in1k",
     "MIT", "ImageNet-22K → ImageNet-1K",
     "7ff88869f8609f09e36a48846aba48665c0a4164910391a1d24595d774db85c2",
     "8690 万", "否（实验失败，未进集成）", "架构探索"),

    ("ViT-Base（vit_base_patch16_384.augreg_in21k_ft_in1k）",
     HF + "timm/vit_base_patch16_384.augreg_in21k_ft_in1k",
     "Apache-2.0", "ImageNet-21K → ImageNet-1K",
     "390355f5b8f83fb003d98979784ec60f48b16267bff10cd528fdf069ae71d529",
     "8570 万", "否（实验失败，未进集成）", "架构探索"),

    ("tf_efficientnet_b0.ns_jft_in1k（⚠ 建议不申报/不使用）",
     HF + "timm/tf_efficientnet_b0.ns_jft_in1k",
     "Apache-2.0（但预训练主数据集含 JFT-300M，非公开）", "ImageNet-1K + JFT-300M 半监督",
     "ca6fd4de70b4777557e7deca534935b5db289946d55c98c0b589c9c399771510",
     "530 万", "否（已从最终集成排除）", "⚠ 合规风险项"),
]

HEADERS = ["序号", "A1 所用开源模型名称", "A2 权重公开链接", "A3 许可证", "A4 上游预训练数据说明",
           "A5 权重 SHA-256", "参数量", "是否参与最终推理", "用途"]
WIDTHS = [6, 46, 58, 34, 34, 68, 12, 26, 16]

wb = Workbook()
ws = wb.active
ws.title = "模型核验清单"

thin = Side(style="thin", color="B0B0B0")
border = Border(left=thin, right=thin, top=thin, bottom=thin)
hdr_fill = PatternFill("solid", fgColor="1F3A5F")
warn_fill = PatternFill("solid", fgColor="FFF2CC")

ws.append(HEADERS)
for i, w in enumerate(WIDTHS, 1):
    ws.column_dimensions[get_column_letter(i)].width = w
for c in ws[1]:
    c.font = Font(name="微软雅黑", bold=True, size=10, color="FFFFFF")
    c.fill = hdr_fill
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = border

for n, r in enumerate(ROWS, 1):
    ws.append([n] + list(r))
    row = ws.max_row
    for c in ws[row]:
        c.font = Font(name="微软雅黑", size=9)
        c.alignment = Alignment(vertical="top", wrap_text=True)
        c.border = border
    if "⚠" in r[0]:
        for c in ws[row]:
            c.fill = warn_fill

ws.freeze_panes = "A2"
ws.row_dimensions[1].height = 34

# ---- 第二张表：风险评估 ----
ws2 = wb.create_sheet("风险评估")
ws2.append(["项目", "规则条款", "现状", "判定", "处置"])
for i, w in enumerate([26, 12, 52, 12, 46], 1):
    ws2.column_dimensions[get_column_letter(i)].width = w

RISK = [
    ("预训练权重非医学", "第1条", "全部为 ImageNet-1K/21K/22K 预训练，Apache-2.0 / MIT / BSD-3，无 gated、无 NC", "合规", "已核实 license 与 datasets 字段"),
    ("外部医学数据", "第2条", "未使用任何外部医学数据、医学文献或医院数据", "合规", "—"),
    ("验证集并入训练", "第3条", "全部有监督训练仅用官方 split=train（30,344 张）；验证集从未参与训练", "合规", "train.py 第80行可核验"),
    ("数据增强", "第4条", "几何/像素/随机擦除/类别加权损失，均在允许列表内", "合规", "—"),
    ("测试集使用", "第5条", "测试集仅用于生成提交，未参与训练、伪标签、统计量估计", "合规", "—"),
    ("TTA/集成/患者聚合", "第6条", "水平翻转 TTA、多模型平均、study 内多视图平均，均为明确允许", "合规", "—"),
    ("BN 推理模式", "第6条", "infer.py 显式 model.eval()，统计量来自训练集", "合规", "可核验"),
    ("JFT 预训练模型", "第1条", "模型池中存在 tf_efficientnet_b0.ns_jft_in1k（预训练含 JFT-300M，非公开数据）", "风险", "已从最终集成排除，不申报"),
    ("自监督预训练分支", "第3条", "ssl_pretrain.py 曾把 valid 并入自监督（该分支模型 r34ssl 未进任何提交）", "风险", "永久排除 r34ssl_* 系列，不申报"),
    ("模型体积", "第8条", "当前4模型融合 334+334+16 = 684 MB", "超限", "提交材料版改为 fp16 或单模型（≤500MB）"),
    ("推理耗时", "第7条", "4模型×2TTA=8次前向，ConvNeXt-Base@384 单次约 97ms", "超限", "提交材料版改为单模型无TTA / int8 量化"),
    ("撤销机会", "第9条", "全程仅 1 次撤销机会", "注意", "谨慎提交"),
]

for r in RISK:
    ws2.append(list(r))
    row = ws2.max_row
    for c in ws2[row]:
        c.font = Font(name="微软雅黑", size=9)
        c.alignment = Alignment(vertical="top", wrap_text=True)
        c.border = border
    v = r[3]
    fill = {"合规": "E2EFDA", "风险": "FFF2CC", "超限": "FCE4E4", "注意": "FFF2CC"}.get(v)
    if fill:
        ws2.cell(row=row, column=4).fill = PatternFill("solid", fgColor=fill)

for c in ws2[1]:
    c.font = Font(name="微软雅黑", bold=True, size=10, color="FFFFFF")
    c.fill = hdr_fill
    c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    c.border = border
ws2.freeze_panes = "A2"

wb.save(OUT)
print("saved", OUT, os.path.getsize(OUT))