# -*- coding: utf-8 -*-
"""按「训练方案」体例生成现有最好结果的文档（纯文字 + 一张结果表）。"""
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "训练方案_现有最好结果.docx")

doc = Document()
st = doc.styles["Normal"]
st.font.name = "微软雅黑"
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")


def setfont(run, size=10.5, bold=False, color=None):
    run.font.name = "微软雅黑"
    run.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = color


p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
setfont(p.add_run("胸片多标签分类模型训练方案"), 19, True, RGBColor(0x1F, 0x3A, 0x5F))
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
setfont(p.add_run("现有最好结果 · 线上 76.70（榜首 82.20）"), 10, False, RGBColor(0x60, 0x6C, 0x80))


def h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    setfont(p.add_run(text), 13, True, RGBColor(0x1F, 0x3A, 0x5F))


def item(text, indent=0.0):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(indent)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.25
    setfont(p.add_run(text))
    return p


h1("一、模型架构与分辨率")
item("1. Model 1：EfficientNet-B0，输入尺寸 640×640 像素（主模型，5 折交叉验证）。")
item("2. Model 2：EfficientNet-B0，输入尺寸 448×448 像素（辅模型，5 折交叉验证）。")
item("3. Model 3：EfficientNet-B0（JFT 预训练版），输入尺寸 448×448 像素。")
item("4. 预训练权重：Model 1、2 采用官方 PyTorch 标准 ImageNet-1K 预训练权重"
     "（efficientnet_b0.pth）；Model 3 采用 timm 的 ImageNet-21K/JFT 预训练权重"
     "（tf_efficientnet_b0.ns_jft_in1k）。均属赛题规则允许的通用预训练范畴。")
item("5. 模型规模：EfficientNet-B0 参数量 530 万，单张 640 分辨率前向计算量约 6.7 GFLOPs。")

h1("二、训练超参数")
item("1. 数据集：使用官方 train.csv，按病人（Subject_id）分组做 5 折交叉验证，"
     "避免同一病人跨折泄露；单折训练 24,338 张、折内验证 6,006 张。")
item("2. 优化器：AdamW，Weight Decay 设为 5e-2。")
item("3. 学习率：统一使用 3e-4。")
item("4. 学习率调度：余弦退火（CosineAnnealingLR），前 1 轮线性 warmup，最终衰减至 0。")
item("5. Batch Size：640 分辨率下为 16；448 分辨率下为 40。")
item("6. 训练轮数：共 12 轮。实测在第 6 轮达到最优（见第四节），此后验证指标下滑。")
item("7. 损失函数：BCEWithLogitsLoss。默认按训练集类别频率计算正样本权重，"
     "并截断上限为 20；对照实验显示取消加权后仅变化 +0.0012，影响很小。")
item("8. 训练加速与稳定：全程启用混合精度 AMP（fp16），并维护指数滑动平均 EMA"
     "（衰减系数 0.999），以 EMA 权重作为最终保存结果。")
item("9. 吞吐实测：640 分辨率 37 张/秒，448 分辨率 78 张/秒；数据加载 1082 张/秒，不构成瓶颈。")

h1("三、数据增强与推理 TTA")
item("1. 训练增强：随机水平翻转（概率 0.5）、小角度旋转（正负 10 度）、"
     "随机裁剪缩放（比例 0.75–1.0）、亮度抖动（±15%）、伽马扰动（0.7–1.4，概率 0.2）"
     "以及随机擦除（概率 0.25）的组合增强策略。")
item("2. 测试推理 TTA：原图与水平翻转各占 50%，做等权平均预测，"
     "即 0.5×原图预测 + 0.5×翻转预测（实测较无 TTA 提升 1.06 个千分点）。")
item("3. 正侧位聚合：单个 Study 内多张图像聚合时采用等权平均（概率域）。"
     "另做过「正位 0.8 / 侧位 0.2」的加权对照，8 个模型上平均增益仅 +0.0001，"
     "故最终仍采用等权聚合。")
item("4. 模型融合比例：在验证集上用带放回的贪心前向选择自动确定权重，"
     "当前最优组合为 4 个模型等权——640 分辨率第 2 折、第 0 折、第 1 折 + JFT 预训练模型，"
     "融合后验证集 0.8075（最好单模型 0.8032）。")

h1("四、验证结果")

t = doc.add_table(rows=1, cols=4)
t.style = "Table Grid"
for i, txt in enumerate(["模型", "分辨率", "最佳轮次", "验证集 AUC"]):
    c = t.rows[0].cells[i]
    c.text = ""
    setfont(c.paragraphs[0].add_run(txt), 9.5, True)

rows = [
    ("EfficientNet-B0（5 折平均）", "640", "第 6 轮", "0.8016"),
    ("EfficientNet-B0（最好单折）", "640", "第 6 轮", "0.8032"),
    ("EfficientNet-B0 去加权 + mixup", "640", "第 6 轮", "0.8033"),
    ("EfficientNet-B0（JFT 预训练）", "448", "第 5 轮", "0.7989"),
    ("EfficientNet-B0", "448", "第 7 轮", "0.7997"),
    ("ResNet-50", "640", "—", "0.7959"),
    ("DenseNet-121", "1024", "第 13 轮", "0.7937"),
    ("ResNet-34（从零训练）", "384", "第 39 轮", "0.7745"),
    ("四模型融合（提交版本）", "640+448", "—", "0.8075"),
]
for r in rows:
    cells = t.add_row().cells
    for i, v in enumerate(r):
        cells[i].text = ""
        setfont(cells[i].paragraphs[0].add_run(v), 9.5, bold=(i == 3 and "0.8075" in v))

doc.add_paragraph()
item("说明：所有模型均在 12 轮内于第 6 轮附近达到峰值，之后训练 loss 继续下降而验证 AUC 单调下滑，"
     "属数据量不足导致的过拟合。继续增加轮数、分辨率或模型容量均未带来提升"
     "（DenseNet-121@1024 训练 24 轮仅 0.7937，且单轮耗时 47 分钟）。")

h1("五、线上验证")
item("1. 第 1 次提交：ResNet-34@384 从零训练，本地 0.7745 → 线上 73.13。")
item("2. 第 2 次提交：EfficientNet-B0@448 三模型集成，本地 0.8011 → 线上 76.62。")
item("3. 第 3 次提交：EfficientNet-B0@640 五折 + 448 系融合，本地 0.8075 → 线上 76.70（当前最好）。")
item("4. 本地提升 0.0064 仅换来线上 0.08，原因是测试集仅 1,414 例，罕见类别阳性样本约 30 例，"
     "单类 AUC 标准误约 0.07，宏平均后仍有 ±2 分抖动。故模型选择仅依据本地验证集指标。")

doc.save(OUT)
print("saved", OUT, os.path.getsize(OUT))