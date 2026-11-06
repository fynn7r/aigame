# -*- coding: utf-8 -*-
"""生成初赛技术进展报告（Word + PNG 图表），全部依赖 Pillow / python-docx。"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(OUT), "report_assets")
os.makedirs(ASSETS, exist_ok=True)

FONT = "C:/Windows/Fonts/msyh.ttc"
def f(sz):
    return ImageFont.truetype(FONT, sz)

INK = (32, 38, 48)
MUTED = (110, 120, 135)
GRID = (226, 232, 240)
ACC = (37, 99, 235)
ACC2 = (245, 158, 11)
ACC3 = (16, 185, 129)
BG = (255, 255, 255)


def canvas(w, h):
    im = Image.new("RGB", (w, h), BG)
    return im, ImageDraw.Draw(im)


def text_center(d, box, s, font, fill=INK):
    x0, y0, x1, y1 = box
    bb = d.textbbox((0, 0), s, font=font)
    d.text((x0 + (x1 - x0 - (bb[2] - bb[0])) / 2 - bb[0],
            y0 + (y1 - y0 - (bb[3] - bb[1])) / 2 - bb[1]), s, font=font, fill=fill)


# ---------------- 图1：模型横向对比 ----------------
def chart_models():
    data = [
        ("b0@640 f4 + TTA", 0.8008),
        ("b0@640 f2 + TTA", 0.8032),
        ("b0@640 f1 + TTA", 0.8025),
        ("b0@640 f0 + TTA", 0.8029),
        ("x b0@640 去加权+mixup", 0.8033),
        ("b0@448 18ep + TTA", 0.7997),
        ("b0@448 JFT 预训练", 0.7989),
        ("b0@448 12ep f0 + TTA", 0.7985),
        ("r50@640 f0", 0.7959),
        ("dn121@1024 24ep", 0.7937),
        ("r50@448 18ep", 0.7905),
        ("r34@384 从零 40ep", 0.7745),
    ]
    W, H = 1180, 620
    im, d = canvas(W, H)
    text_center(d, (0, 18, W, 58), "各单模型验证集 study 级 macro-AUC", f(30))
    d.text((60, 66), "含 hflip TTA；数值越高越好，虚线为 0.80 参考线", font=f(17), fill=MUTED)
    left, right, top, bot = 330, W - 90, 110, H - 60
    lo, hi = 0.76, 0.81
    x = lambda v: left + (v - lo) / (hi - lo) * (right - left)
    for t in [0.76, 0.77, 0.78, 0.79, 0.80, 0.81]:
        px = x(t)
        d.line([(px, top), (px, bot)], fill=GRID, width=1)
        text_center(d, (px - 40, bot + 8, px + 40, bot + 32), f"{t:.2f}", f(16), MUTED)
    if left <= x(0.80) <= right:
        d.line([(x(0.80), top), (x(0.80), bot)], fill=(200, 120, 120), width=2)
    bh = (bot - top) / len(data)
    for i, (name, v) in enumerate(data):
        y0 = top + i * bh + bh * 0.18
        y1 = y0 + bh * 0.64
        col = ACC2 if "b0@640" in name or "去加权" in name else (ACC if v >= 0.7997 else (150, 160, 175))
        if "dn121" in name or "r50" in name or "r34" in name:
            col = (170, 180, 195)
        d.rounded_rectangle([left, y0, x(v), y1], radius=5, fill=col)
        d.text((left - 14 - d.textlength(name, font=f(19)), y0 + 3), name, font=f(19), fill=INK)
        d.text((x(v) + 10, y0 + 3), f"{v:.4f}", font=f(19), fill=INK)
    p = os.path.join(ASSETS, "chart_models.png")
    im.save(p, quality=95)
    return p


# ---------------- 图2：收敛曲线 ----------------
def chart_curves():
    base = [0.7587, 0.7833, 0.7929, 0.7998, 0.8037, 0.8059, 0.8069, 0.8048, 0.8031, 0.7997, 0.7967]
    nopw = [0.7629, 0.7848, 0.7933, 0.7997, 0.8051, 0.8064, 0.8081, 0.8074, 0.8045, 0.8012, 0.7987, 0.7971]
    dn = [0.7523, 0.7661, 0.7721, 0.7770, 0.7800, 0.7843, 0.7902, 0.7912, 0.7939, 0.7943,
          0.7973, 0.7976, 0.7978, 0.7984, 0.7963, 0.7941, 0.7934, 0.7923, 0.7904, 0.7887]
    W, H = 1180, 620
    im, d = canvas(W, H)
    text_center(d, (0, 18, W, 58), "训练曲线：所有配置都在第 6 轮见顶后下滑", f(30))
    d.text((60, 66), "纵轴为 fold 验证集 AUC；横轴为 epoch。这是数据量不足导致过拟合的直接证据", font=f(17), fill=MUTED)
    left, right, top, bot = 90, W - 190, 110, H - 70
    lo, hi = 0.745, 0.815
    n = max(len(base), len(nopw), len(dn))
    X = lambda i: left + i / (n - 1) * (right - left)
    Y = lambda v: bot - (v - lo) / (hi - lo) * (bot - top)
    for t in [0.75, 0.77, 0.79, 0.81]:
        d.line([(left, Y(t)), (right, Y(t))], fill=GRID, width=1)
        text_center(d, (left - 70, Y(t) - 12, left - 8, Y(t) + 12), f"{t:.2f}", f(16), MUTED)
    for i in range(n):
        text_center(d, (X(i) - 16, bot + 6, X(i) + 16, bot + 30), str(i), f(15), MUTED)
    peak = 6
    d.line([(X(peak), top), (X(peak), bot)], fill=(240, 200, 200), width=2)
    d.text((X(peak) + 6, top + 4), "第 6 轮峰值", font=f(17), fill=(190, 90, 90))
    for series, col, lab in [(base, ACC, "b0@640 基线（pos_weight 加权）"),
                             (nopw, ACC2, "b0@640 去加权 + mixup"),
                             (dn, (150, 160, 175), "dn121@1024 24ep")]:
        pts = [(X(i), Y(v)) for i, v in enumerate(series)]
        d.line(pts, fill=col, width=4, joint="curve")
        for px, py in pts:
            d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=col)
    ly = top + 10
    for col, lab in [(ACC, "b0@640 基线"), (ACC2, "b0@640 去加权+mixup"), ((150, 160, 175), "dn121@1024")]:
        d.line([(right + 24, ly + 10), (right + 54, ly + 10)], fill=col, width=4)
        d.text((right + 60, ly), lab, font=f(17), fill=INK)
        ly += 30
    p = os.path.join(ASSETS, "chart_curves.png")
    im.save(p, quality=95)
    return p


# ---------------- 图3：提交历史 ----------------
def chart_subs():
    subs = [("第1次\nr34@384", 0.7745, 73.13), ("第2次\nv2 集成", 0.8011, 76.62), ("第3次\nv3 集成", 0.8075, 76.70)]
    W, H = 1080, 560
    im, d = canvas(W, H)
    text_center(d, (0, 18, W, 58), "提交历史：本地稳步提升，线上被噪声压平", f(30))
    d.text((60, 66), "左轴=本地 valid AUC（蓝），右轴=线上分（橙）；榜首 82.2", font=f(17), fill=MUTED)
    left, right, top, bot = 100, W - 120, 120, H - 90
    n = len(subs)
    X = lambda i: left + (i + 0.5) / n * (right - left)
    lo, hi = 0.76, 0.82
    Y = lambda v: bot - (v - lo) / (hi - lo) * (bot - top)
    olo, ohi = 70, 84
    Y2 = lambda v: bot - (v - olo) / (ohi - olo) * (bot - top)
    for t in [0.76, 0.78, 0.80, 0.82]:
        d.line([(left, Y(t)), (right, Y(t))], fill=GRID, width=1)
        text_center(d, (left - 78, Y(t) - 12, left - 10, Y(t) + 12), f"{t:.2f}", f(16), MUTED)
    for t in [70, 74, 78, 82]:
        text_center(d, (right + 12, Y2(t) - 12, right + 80, Y2(t) + 12), str(t), f(16), MUTED)
    d.line([(left, Y2(82.2)), (right, Y2(82.2))], fill=(200, 120, 120), width=2)
    d.text((left + 8, Y2(82.2) - 28), "榜首 82.2", font=f(17), fill=(190, 90, 90))
    pts1 = [(X(i), Y(v)) for i, (_, v, _) in enumerate(subs)]
    pts2 = [(X(i), Y2(v)) for i, (_, _, v) in enumerate(subs)]
    d.line(pts1, fill=ACC, width=4)
    d.line(pts2, fill=ACC2, width=4)
    for (px, py), (_, v, o) in zip(pts1, subs):
        d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=ACC)
        d.text((px - 30, py - 34), f"{v:.4f}", font=f(17), fill=ACC)
    for (px, py), (_, _, o) in zip(pts2, subs):
        d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=ACC2)
        d.text((px - 26, py + 14), f"{o:.2f}", font=f(17), fill=ACC2)
    for i, (lab, _, _) in enumerate(subs):
        text_center(d, (X(i) - 90, bot + 12, X(i) + 90, bot + 70), lab, f(17), MUTED)
    p = os.path.join(ASSETS, "chart_subs.png")
    im.save(p, quality=95)
    return p


# ---------------- 图4：每类别 AUC ----------------
def chart_perclass():
    names = ["心纵隔增大", "气胸", "实变", "肺炎", "肺水肿", "心脏增大",
             "肺不张", "肺不透明度", "胸腔积液", "无异常"]
    vals = [0.737, 0.896, 0.770, 0.731, 0.887, 0.768, 0.786, 0.718, 0.908, 0.875]
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    W, H = 1180, 600
    im, d = canvas(W, H)
    text_center(d, (0, 18, W, 58), "v3 集成：逐类别 AUC（弱点定位）", f(30))
    d.text((60, 66), "红色=短板类别（<0.75），绿=强项类别（>0.88）", font=f(17), fill=MUTED)
    left, right, top, bot = 220, W - 110, 110, H - 60
    lo, hi = 0.68, 0.93
    x = lambda v: left + (v - lo) / (hi - lo) * (right - left)
    for t in [0.70, 0.75, 0.80, 0.85, 0.90]:
        px = x(t)
        d.line([(px, top), (px, bot)], fill=GRID, width=1)
        text_center(d, (px - 40, bot + 8, px + 40, bot + 32), f"{t:.2f}", f(16), MUTED)
    bh = (bot - top) / len(names)
    for row, i in enumerate(order):
        v = vals[i]
        y0 = top + row * bh + bh * 0.18
        y1 = y0 + bh * 0.64
        col = (220, 90, 90) if v < 0.75 else (ACC3 if v > 0.88 else ACC)
        d.rounded_rectangle([left, y0, x(v), y1], radius=5, fill=col)
        d.text((left - 14 - d.textlength(names[i], font=f(20)), y0 + 2), names[i], font=f(20), fill=INK)
        d.text((x(v) + 10, y0 + 2), f"{v:.3f}", font=f(19), fill=INK)
    p = os.path.join(ASSETS, "chart_perclass.png")
    im.save(p, quality=95)
    return p


# ---------------- Word 报告 ----------------
def build_docx(paths):
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "微软雅黑"
    st.font.size = Pt(10.5)
    from docx.oxml.ns import qn
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    def h(text, level=1):
        p = doc.add_heading(text, level=level)
        for r in p.runs:
            r.font.name = "微软雅黑"
            r.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
            r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        return p

    def para(text, bold=False, size=10.5, color=None):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.bold = bold
        r.font.size = Pt(size)
        r.font.name = "微软雅黑"
        r.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        if color:
            r.font.color.rgb = color
        return p

    def table(header, rows, widths=None):
        t = doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htxt in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            r = c.paragraphs[0].add_run(htxt)
            r.bold = True
            r.font.size = Pt(9.5)
            r.font.name = "微软雅黑"
            r.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v))
                r.font.size = Pt(9.5)
                r.font.name = "微软雅黑"
                r.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
        doc.add_paragraph()
        return t

    # 标题
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("胸片多标签分类 · 初赛技术进展报告")
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("2026 人工智能挑战赛　|　截至 2026-10-31　|　线上 76.70 / 榜首 82.20")
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x60, 0x6C, 0x80)

    h("一、结论摘要", 1)
    table(["指标", "数值", "说明"], [
        ["线上最好成绩", "76.70", "第 3 次提交（v3 集成）"],
        ["当前排名 / 榜首", "第 3 名 / 82.20", "差距 5.5 分"],
        ["本地最好单模型", "0.8081", "b0@640，去加权 + mixup"],
        ["本地最好集成", "0.8075", "v3，4 模型贪心加权"],
        ["可复现训练数", "21 个模型", "全部留存权重与推理概率"],
        ["已用算力", "约 90 GPU·小时", "单卡 Hygon DCU K100_AI"],
    ])
    para("核心判断：模型在同一分辨率下的单模型上限约 0.808，继续调参、加轮数、加分辨率都已饱和；"
         "线上分数被测试集规模（1414 例）带来的噪声主导，单次 ±1 分内不可信。"
         "下一步的增量只能来自「换架构族 + 更强预训练权重」。", bold=True)

    h("二、任务与数据", 1)
    table(["项目", "训练集", "验证集", "测试集"], [
        ["图像数", "30,344", "1,422", "2,218"],
        ["检查（study）数", "18,513", "859", "1,414"],
        ["病人数", "11,055", "228", "136"],
        ["标注", "公开", "公开", "主办方保密"],
    ])
    para("评价指标：10 类宏平均 AUC（macro-AUC），提交每日 2 次。"
         "类别 1–10 依次为：心纵隔增大、气胸、实变、肺炎、肺水肿、心脏增大、肺不张、肺不透明度、胸腔积液、无异常。")
    para("原始图像中位尺寸 2544×3056，我们的训练分辨率 448–640 相当于下采样 4–6.7 倍；"
         "已构建 512 与 1024 两套预处理缓存（.npy，分别为 8.9 GB 与 34 GB），避免重复解码。")

    h("三、方法", 1)
    para("整体流程：图像缓存 → 多模型多分辨率训练 → 逐模型 hflip-TTA 推理 → study 级 logit 平均 → 贪心加权集成 → 提交。")
    table(["环节", "做法"], [
        ["骨干网络", "EfficientNet-B0（ImageNet-1k / JFT）、ResNet-50、DenseNet-121、ResNet-34"],
        ["输入分辨率", "448 / 512 / 640 / 768 / 1024（消融见第四节）"],
        ["损失函数", "BCEWithLogits；默认按类别频率加权（上限 20），实验中对比了不加权"],
        ["数据增强", "随机裁剪 0.75–1.0、水平翻转、±10° 旋转、亮度/伽马抖动、随机擦除"],
        ["训练策略", "AdamW、cosine 退火、warmup 1 轮、EMA、混合精度"],
        ["集成", "study 级 logit 平均 + 带放回的贪心前向选择（在 valid 上拟合权重）"],
        ["TTA", "水平翻转（实测 +1.06 分）"],
    ])
    para("合规性：仅使用赛事提供的训练数据；ImageNet / JFT 等通用预训练权重属规则允许范围；"
         "未使用任何外部医学数据；测试集图像未参与训练、伪标注或模型选择；"
         "集成权重与阈值全部只在训练集/验证集上拟合。")

    h("四、实验结果", 1)
    doc.add_picture(paths["models"], width=Inches(6.4))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    table(["模型", "分辨率", "轮数", "fold AUC", "valid AUC"], [
        ["b0 × 5 折（最佳单折）", "640", "12", "0.8132", "0.8032"],
        ["b0 去加权 + mixup", "640", "12", "0.8081", "0.8033"],
        ["b0 基线", "640", "12", "0.8069", "0.8029"],
        ["b0（JFT 预训练）", "448", "12", "0.8024", "0.7989"],
        ["b0 基线", "448", "18", "0.8031", "0.7997"],
        ["ResNet-50", "640", "12", "0.7973", "0.7959"],
        ["DenseNet-121", "1024", "24", "0.7984", "0.7937"],
        ["ResNet-34（从零训练）", "384", "40", "0.7832", "0.7745"],
    ])
    para("分辨率消融：448 → 640 带来约 +0.004（valid 0.7997 → 0.8029）；640 → 768 仅 +0.0014，"
         "且单折耗时从 11 分钟/轮涨到 15.7 分钟/轮，性价比消失。")

    doc.add_picture(paths["curves"], width=Inches(6.4))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para("关键现象：所有配置都在第 6/12 轮见顶，之后训练 loss 继续下降而验证 AUC 单调下滑，"
         "典型的数据量不足型过拟合。DenseNet-121@1024 训练 24 轮仍只有 0.7984，且单轮耗时 47 分钟，"
         "在本算力上不具可行性。", bold=True)

    h("五、线上标定与噪声分析", 1)
    doc.add_picture(paths["subs"], width=Inches(6.4))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    table(["提交", "本地 AUC", "线上分", "差值"], [
        ["第 1 次（ResNet-34 从零）", "0.7745", "73.13", "−4.32"],
        ["第 2 次（v2 集成）", "0.8011", "76.62", "−3.49"],
        ["第 3 次（v3 集成）", "0.8075", "76.70", "−4.05"],
    ])
    para("第 3 次本地提升 +0.0064，线上只涨 0.08。若按前两次的斜率（1 个点本地≈1.3 分线上）应涨 0.8 分。"
         "原因是测试集仅 1414 个 study，罕见类别阳性样本约 30 例，单个类别 AUC 标准误约 "
         "sqrt(A(1−A)/n) ≈ 0.07，10 类宏平均后仍有 ±0.02（±2 分）抖动。", bold=True)
    para("因此：模型选择只使用本地 valid AUC；线上仅用于验证预期 +2 分以上的大台阶改动。")

    h("六、类别强弱分析", 1)
    doc.add_picture(paths["perclass"], width=Inches(6.4))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para("强项：胸腔积液 0.908、气胸 0.896、肺水肿 0.887、无异常 0.875、肺不张 0.786。")
    para("短板：肺不透明度 0.718、肺炎 0.731、心纵隔增大 0.737。这三类在原始报告中本身就存在标注歧义"
         "（“可能”“疑似”占比高），且病灶细微、对分辨率敏感，是后续重点。")

    h("七、算力与工程", 1)
    table(["项目", "情况"], [
        ["硬件", "Hygon DCU K100_AI，12 GB 显存 / 8 CPU / 32 GB 内存"],
        ["吞吐（实测）", "b0@448 78 张/秒；b0@640 37 张/秒；r50@640 28 张/秒；dn121@1024 8 张/秒"],
        ["算力利用诊断", "b0 只跑到 0.37 TFLOPs/s（深度可分离卷积受限），r50 可达 2.95 TFLOPs/s"],
        ["数据加载", "1082 张/秒，不是瓶颈"],
        ["总计已训练", "21 个模型，约 90 GPU·小时"],
    ])

    h("八、下一步计划", 1)
    table(["优先级", "动作", "预期增益"], [
        ["P0", "引入 ConvNeXt / EfficientNet-B4 / ViT-B-CLIP 等更强预训练骨干（规则允许）", "+0.01 ~ +0.03"],
        ["P1", "CLAHE 对比度均衡预处理（放射科标准操作，实验已启动）", "+0.005 ~ +0.015"],
        ["P2", "扩大集成规模至 15–20 个模型，补足架构多样性", "+0.003 ~ +0.005"],
        ["P3", "多尺度 TTA（448 + 640 + 768 概率平均）", "+0.003 ~ +0.006"],
        ["P4", "study 级多视图注意力池化，替代简单 logit 平均", "+0.003 ~ +0.008"],
    ])
    para("现实预期：本地 0.83–0.85，对应线上 79–82，冲击榜首区间。"
         "受算力与数据量限制，85 分（高于当前榜首）在本轮不具可行性。", bold=True)

    doc.add_paragraph()
    para("附：全部训练日志、权重、推理概率保存在容器 /data/cxr_first/ 与 /root/_dsh/，"
         "本地代码位于 E:\\aigame\\cxr_first\\cxr_src\\。", size=9)

    doc.save(paths["docx"])
    return paths["docx"]


if __name__ == "__main__":
    m = chart_models()
    c = chart_curves()
    s = chart_subs()
    pc = chart_perclass()
    print("charts:", m, c, s, pc)
    d = build_docx({"models": m, "curves": c, "subs": s, "perclass": pc,
                    "docx": os.path.join(os.path.dirname(OUT), "初赛技术进展报告_20261031.docx")})
    print("docx:", d)