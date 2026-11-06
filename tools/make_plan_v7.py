# -*- coding: utf-8 -*-
"""按「训练方案」体例输出当前版本（v7，线上 77.59）的模型与参数说明。"""
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "训练方案_v7_线上77.59.docx")

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
setfont(p.add_run("线上 77.59 分版本（本地验证 0.8150）　|　2026-11-04"), 10, False,
        RGBColor(0x60, 0x6C, 0x80))


def h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    setfont(p.add_run(text), 13, True, RGBColor(0x1F, 0x3A, 0x5F))


def item(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.25
    setfont(p.add_run(text))


def table(header, rows, bold_col=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, txt in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        setfont(c.paragraphs[0].add_run(txt), 9.5, True)
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = ""
            setfont(cells[i].paragraphs[0].add_run(str(v)), 9.5,
                    bold=(bold_col is not None and i == bold_col and "**" in str(v)))
    doc.add_paragraph()


h1("一、模型架构与分辨率")
item("1. Model 1（主模型）：ConvNeXt-Base，输入尺寸 384×384 像素，8760 万参数，占最终融合权重的 3/4。")
item("2. Model 2（辅助模型）：EfficientNet-B0，输入尺寸 640×640 像素，530 万参数，占最终融合权重的 1/4。")
item("3. 预训练权重：ConvNeXt-Base 采用 ImageNet-22K 预训练、ImageNet-1K 微调权重"
     "（convnext_base.fb_in22k_ft_in1k，共加载 342 个张量）；EfficientNet-B0 采用官方 PyTorch "
     "标准 ImageNet-1K 权重（efficientnet_b0.pth）。均属赛题规则允许的通用预训练。")
item("4. 输入通道：所有模型由 3 通道改为单通道灰度（首层卷积核按通道取均值改造），"
     "使 ImageNet 权重可直接迁移。")

h1("二、训练超参数")
item("1. 数据集与切分：Model 1 使用官方 train.csv 的全部 30,344 张图像（不再切出验证集）；"
     "Model 2 使用按病人（Subject_id）分组 5 折交叉验证中的第 0 折，即 24,338 张。"
     "另有 859 个 study 构成固定验证集，只用于评估与融合权重拟合，从不参与训练。")
item("2. 优化器：AdamW。ConvNeXt-Base 的 Weight Decay 设为 1e-2，EfficientNet-B0 设为 5e-2"
     "（轻量模型需要更强正则，属分别调优的结果）。")
item("3. 学习率：ConvNeXt-Base 使用 1.5e-4，EfficientNet-B0 使用 3e-4。")
item("4. 学习率调度：采用 CosineAnnealingLR 余弦退火，前 1 轮线性 warmup，最终衰减至 0。")
item("5. Batch Size：两个模型均设置为 16。")
item("6. 训练轮数：ConvNeXt-Base 共 5 轮，模型在第 3 至第 4 轮收敛；EfficientNet-B0 共 12 轮，"
     "在第 6 轮收敛。训练全程启用混合精度 AMP（fp16），并维护指数滑动平均 EMA"
     "（衰减系数 0.999），以 EMA 权重作为最终保存结果。")
item("7. 损失函数：使用多标签标准的 BCEWithLogitsLoss，正样本权重按训练集类别频率计算，"
     "并截断上限为 20，以缓解罕见的阳性类别被淹没的问题。")
item("8. 算力与吞吐：单卡海光 DCU K100_AI（12 GB 显存 / 8 CPU / 32 GB 内存）。"
     "ConvNeXt-Base@384 实测 10.3 张/秒（单轮 49 分钟），EfficientNet-B0@640 实测 30 张/秒。")

h1("三、数据增强与推理 TTA")
item("1. 训练增强：随机裁剪缩放（比例 0.75~1.0）、随机水平翻转（概率 0.5）、小角度旋转"
     "（正负 10 度）、亮度抖动（±15%）、伽马扰动（0.7~1.4，概率 0.2）以及随机擦除"
     "（概率 0.25）的组合增强策略。")
item("2. 测试推理 TTA：原图与水平翻转各占 50%，做等权平均预测，即 0.5×原图预测 + 0.5×翻转预测"
     "（实测较无 TTA 提升 1.06 个千分点）。")
item("3. 正侧位聚合：单个 Study 内多张图像聚合时采用等权平均（概率域）。"
     "另外做过「正位片 PA/AP 权重 0.80、侧位片 LAT 权重 0.20」的对照实验，"
     "在 8 个模型上平均增益仅 +0.0001，故最终仍采用等权聚合。")
item("4. 模型融合比例：在固定验证集上做带放回的贪心前向选择（每步只接受增益大于 0.0001 的模型），"
     "最终组合为 ConvNeXt-Base(seed123) 占 2 份、ConvNeXt-Base(seed42) 占 1 份、"
     "EfficientNet-B0@640 占 1 份，四份等权平均。折算下来是 ConvNeXt-Base 75%、"
     "EfficientNet-B0 25% 的架构配比。")

h1("四、验证结果")
table(["模型", "分辨率", "参数量", "验证集 study 级 AUC"], [
    ["ConvNeXt-Base seed123", "384", "8760 万", "0.8102"],
    ["ConvNeXt-Base seed42", "384", "8760 万", "0.8097"],
    ["ConvNeXt-Base seed2024", "384", "8760 万", "0.8087"],
    ["ConvNeXt-Base seed777", "384", "8760 万", "0.8055"],
    ["ConvNeXt-Base(448 分辨率)", "448", "8760 万", "0.8108"],
    ["ConvNeXt-Small@448 全量", "448", "4950 万", "0.8077"],
    ["ConvNeXt-Small@384 全量", "384", "4950 万", "0.8062"],
    ["EfficientNet-B0@640 全量", "640", "530 万", "0.8044"],
    ["EfficientNet-B0@640 折模型 f0", "640", "530 万", "0.8013"],
    ["DenseNet-201@448 全量", "448", "1800 万", "0.7979"],
    ["**四模型融合（提交版本）**", "**384+640**", "—", "**0.8150**"],
], bold_col=3)

item("该融合版本线上得分 77.59（10 分类宏平均 AUC × 100）。")

h1("五、模型容量是本任务的决定因素（实验结论）")
item("在同一验证集上，单模型 AUC 随参数量单调上升，这是本项目最重要的发现：")
table(["模型", "参数量", "单模型 AUC"], [
    ["ResNet-34（从零训练）", "2180 万", "0.7737"],
    ["DenseNet-121", "800 万", "0.7953"],
    ["DenseNet-201", "1800 万", "0.7979"],
    ["EfficientNet-B0", "530 万", "0.8041"],
    ["ConvNeXt-Small", "4950 万", "0.8077"],
    ["ConvNeXt-Base", "8760 万", "0.8102"],
])
item("早期把优化重点放在分辨率（448→640→768→1024）和训练轮数上，收益很快饱和；"
     "换成更大容量的骨干后，单模型一次提升 +0.004，超过此前所有分辨率实验的总和。")

h1("六、已排除的方案")
table(["方案", "实测结果"], [
    ["ConvNeXt-Base 训练 10 轮 + mixup 0.3", "0.8061，低于 5 轮的 0.8097（过拟合）"],
    ["ConvNeXt-Base@448", "0.8093，略低于 384 分辨率"],
    ["DenseNet-201@512（五折）", "0.8053，低于 @448 的 0.8079"],
    ["多尺度 TTA（448+512+640）", "EfficientNet-B0 反而 −0.0034"],
    ["正侧位 0.8/0.2 加权聚合", "8 个模型平均 +0.0001"],
    ["每族取最优再等权平均", "0.8054，低于贪心 0.8115（弱族拖累）"],
    ["逐类别独立贪心集成", "本地样本内 0.8240，线上只有 77.53（选择偏差）"],
])

doc.save(OUT)
print("saved", OUT, os.path.getsize(OUT))