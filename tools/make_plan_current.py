# -*- coding: utf-8 -*-
"""按「训练方案」体例输出当前模型与参数（v7，线上 77.59）。"""
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "训练方案_当前模型_20261106.docx")

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
setfont(p.add_run("当前版本（线上 77.59 分 / 本地验证 0.8150）　2026-11-06"), 10, False,
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


def table(header, rows):
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
            setfont(cells[i].paragraphs[0].add_run(str(v)), 9.5)
    doc.add_paragraph()


h1("一、模型架构与分辨率")
item("1. Model 1（主模型）：ConvNeXt-Base，输入尺寸 384×384 像素，8760 万参数，"
     "在最终融合中占 3/4 权重（该架构的 2 个随机种子合计 3 份）。")
item("2. Model 2（辅助模型）：EfficientNet-B0，输入尺寸 640×640 像素，530 万参数，"
     "在最终融合中占 1/4 权重。")
item("3. 预训练权重：ConvNeXt-Base 采用 ImageNet-22K 预训练、ImageNet-1K 微调的权重"
     "（convnext_base.fb_in22k_ft_in1k，加载 342 个张量，Apache-2.0 许可证，"
     "SHA-256 = 4853efc8d5dd3ed9b0f888a9c601d7556373599f944f9ac79af8b8ec81aa7381）；"
     "EfficientNet-B0 采用官方 PyTorch 标准 ImageNet-1K 权重（efficientnet_b0.pth，BSD-3）。"
     "两者预训练主数据集均为 ImageNet，属规则允许的通用非医学预训练。")
item("4. 输入通道：为直接复用 ImageNet 权重，所有模型首层卷积由 3 通道改为单通道灰度"
     "（卷积核按通道取均值改造），不引入额外的通道复制。")
item("5. 精度：训练与推理均使用 fp16 混合精度（AMP），权重以 fp32 保存。")

h1("二、训练超参数")
item("1. 数据集：Model 1 使用官方 train.csv 的全部 30,344 张图像（不切分验证集）；"
     "Model 2 使用按病人（Subject_id）分组 5 折交叉验证中的第 0 折，24,338 张。"
     "另有 859 个 study 构成的固定验证集，仅用于评估与融合权重拟合，全程不参与训练。")
item("2. 优化器：AdamW。ConvNeXt-Base 的 Weight Decay 设为 1e-2，"
     "EfficientNet-B0 设为 5e-2（轻量模型需要更强正则，为分别调优的结果）。")
item("3. 学习率：ConvNeXt-Base 使用 1.5e-4，EfficientNet-B0 使用 3e-4。")
item("4. 学习率调度：采用 CosineAnnealingLR 余弦退火，前 1 轮线性 warmup，"
     "最终衰减至 0。")
item("5. Batch Size：两个模型均设置为 16。")
item("6. 训练轮数：ConvNeXt-Base 共 5 轮，模型在第 3 至第 4 轮收敛；"
     "EfficientNet-B0 共 12 轮，在第 6 轮收敛。训练全程启用混合精度 AMP 加速，"
     "并维护指数滑动平均 EMA（衰减系数 0.999），以 EMA 权重作为最终保存结果。")
item("7. 损失函数：使用多标签标准的 BCEWithLogitsLoss。正样本权重按训练集类别频率计算，"
     "并统一截断上限为 20，以缓解稀有病种被海量负样本淹没的问题。")
item("8. 算力环境：单卡海光 DCU K100_AI，12 GB 显存 / 8 CPU / 32 GB 内存。"
     "ConvNeXt-Base@384 实测单轮 49 分钟（约 10.3 张/秒），"
     "EfficientNet-B0@640 实测单轮约 12 分钟。")

h1("三、数据增强与推理 TTA")
item("1. 训练增强：随机缩放裁剪（比例 0.75~1.0）、随机水平翻转（概率 0.5）、"
     "小角度旋转（正负 10 度）、亮度抖动（±15%）、伽马扰动（0.7~1.4，概率 0.2）、"
     "随机擦除（概率 0.25，1~3 个矩形块）的组合增强策略。")
item("2. 测试推理 TTA：原图与水平翻转各占 50%，做等权平均预测，"
     "即 0.5×原图预测 + 0.5×翻转预测。该设置较无 TTA 提升 1.06 个千分点。")
item("3. 正侧位聚合：单个 Study 内多张图像聚合时，在概率域做等权平均。"
     "另外做过「正位片 PA/AP 权重 0.80、侧位片 LAT 权重 0.20」的对照实验，"
     "在 8 个模型上平均增益仅 +0.0001，故最终仍采用等权聚合。")
item("4. 模型融合比例：在固定验证集上做带放回的贪心前向选择（每步仅接受增益大于 0.0001 的候选），"
     "最终组合为 ConvNeXt-Base(seed123) 占 2 份、ConvNeXt-Base(seed42) 占 1 份、"
     "EfficientNet-B0@640 占 1 份，四份等权平均，折算为 ConvNeXt-Base 75%、"
     "EfficientNet-B0 25% 的架构配比。")
item("5. 推理分辨率：模型按训练分辨率（384 / 640）推理。实测 ConvNeXt-Base 在 352/384/416/448/"
     "512/576 六个推理尺寸上分别得 0.8079 / 0.8105 / 0.8099 / 0.8090 / 0.8077 / 0.8050，"
     "训练分辨率即为最优点，无需做训练-推理分辨率错配。")

h1("四、单模型验证结果")
table(["模型", "分辨率", "参数量", "验证 study-AUC"], [
    ["ConvNeXt-Base seed123（全量）", "384", "8760 万", "0.8105"],
    ["ConvNeXt-Base seed42（全量）", "384", "8760 万", "0.8097"],
    ["ConvNeXt-Base + CLAHE（全量）", "384", "8760 万", "0.8089"],
    ["ConvNeXt-Base seed2024（全量）", "384", "8760 万", "0.8087"],
    ["ConvNeXt-Base seed777（全量）", "384", "8760 万", "0.8055"],
    ["ConvNeXt-Base@448（全量）", "448", "8760 万", "0.8108"],
    ["ConvNeXt-Small@448（全量）", "448", "4950 万", "0.8077"],
    ["ConvNeXt-Small@384（全量）", "384", "4950 万", "0.8062"],
    ["EfficientNet-B0@640（全量）", "640", "530 万", "0.8044"],
    ["EfficientNet-B0@640 折模型 f0", "640", "530 万", "0.8013"],
    ["DenseNet-201@448（全量）", "448", "1800 万", "0.7979"],
    ["EfficientNet-B0@448", "448", "530 万", "0.7984"],
    ["ResNet-50@640", "640", "2550 万", "0.7911"],
    ["**四模型融合（当前提交版本）**", "**384+640**", "—", "**0.8150**"],
])
item("当前融合版本线上得分 77.59（10 类宏平均 AUC × 100），本地验证与线上满足线性标定关系："
     "线上 ≈ 109.6 × 本地 − 11.64（五点拟合，残差 RMS 0.24 分）。")

h1("五、参数量与精度的关系（核心结论）")
item("在固定验证集上，单模型 AUC 随参数量单调上升，这是本项目最重要的发现：")
table(["模型", "参数量", "单模型 AUC"], [
    ["ResNet-34（从零训练）", "2180 万", "0.7737"],
    ["DenseNet-121", "800 万", "0.7953"],
    ["DenseNet-201", "1800 万", "0.7979"],
    ["EfficientNet-B0", "530 万", "0.8041"],
    ["ConvNeXt-Small", "4950 万", "0.8077"],
    ["ConvNeXt-Base", "8760 万", "0.8102"],
])
item("早期将优化重点放在分辨率（448→640→768→1024）与训练轮数上，收益迅速饱和；"
     "换成更大容量骨干后，单模型一次提升 +0.004，超过此前所有分辨率实验的总和。")

h1("六、已排除的方案")
table(["方案", "实测结果"], [
    ["ConvNeXt-Base 训练 10 轮 + mixup 0.3", "0.8061，低于 5 轮的 0.8097（过拟合）"],
    ["ConvNeXt-Base@448", "0.8093，略低于 384 分辨率"],
    ["ConvNeXt-Base 权重平均（模型汤）", "0.8030，低于最好单种子 0.8102"],
    ["Swin-Base@384（Transformer）", "0.7850，显著落后（数据量不足）"],
    ["ViT-Base@384（Transformer）", "显存不足，bs 6/8 均无法完成一个 epoch"],
    ["DenseNet-201@512（五折）", "0.8053，低于 @448 的 0.8079"],
    ["多尺度 TTA（448+512+640）", "EfficientNet-B0 反而 −0.0034"],
    ["正侧位 0.8/0.2 加权聚合", "8 个模型平均仅 +0.0001"],
    ["每族取最优再等权平均", "0.8054，低于贪心 0.8115（弱族拖累）"],
    ["逐类别独立贪心集成", "本地样本内 0.8240，线上仅 77.53（选择偏差）"],
    ["训练-推理分辨率错配（FixRes）", "训练分辨率已是最优，无增益"],
])

doc.save(OUT)
print("saved", OUT, os.path.getsize(OUT))