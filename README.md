# 胸片多标签分类（CXR Multi-label Classification）

某人工智能挑战赛初赛方案 —— 胸部 X 光片 10 类病变多标签分类，评价指标为 **study 级宏平均 AUC**。

**线上最好成绩：77.59**（10 类宏观 AUC × 100，榜首 82.2）

---

## 1. 任务与数据

| 项 | 内容 |
|---|---|
| 任务 | 一张胸片可同时存在多种病变，10 个类别独立二分类 |
| 指标 | 每个 Study（检查）聚合多张影像后，计算 10 类 AUC 的宏平均 |
| 训练集 | 30,344 张影像 / 18,513 study / 11,055 病人 |
| 验证集 | 1,422 张 / 859 study（带标注） |
| 测试集 | 2,218 张 / 1,414 study（无标注，需提交概率） |
| 类别 | 1 心纵隔增大、2 气胸、3 实变、4 肺炎、5 肺水肿、6 心脏增大、7 肺不张、8 肺不透明度、9 胸腔积液、10 无异常 |

> 原始数据不随本仓库分发（赛方数据不得二次传播），仅保留代码与提交结果。

## 2. 最终方案

**ConvNeXt-Base@384（3 份权重）+ EfficientNet-B0@640（1 份权重）的等权融合。**

| | 模型 | 输入 | 参数量 | 权重占比 | 验证 AUC |
|---|---|---|---:|---:|---:|
| Model 1 | ConvNeXt-Base × 2 种子 | 384×384 | 8760 万 | 75% | 0.8105 / 0.8097 |
| Model 2 | EfficientNet-B0 | 640×640 | 530 万 | 25% | 0.8013 |
| — | **融合** | — | — | — | **0.8150** |

预训练权重均为 ImageNet 通用预训练（Apache-2.0 / BSD-3，非医学、非 NC、非 gated）。

### 训练配置

| 项 | ConvNeXt-Base | EfficientNet-B0 |
|---|---|---|
| 数据 | 官方 train 全量 30,344 张 | 病人分组 5 折之第 0 折，24,338 张 |
| 优化器 | AdamW，wd = 1e-2 | AdamW，wd = 5e-2 |
| 学习率 | 1.5e-4 | 3e-4 |
| 调度 | Cosine + 1 轮 warmup | 同左 |
| Batch Size | 16 | 16 |
| 轮数 | 5（第 3~4 轮收敛） | 12（第 6 轮收敛） |
| 损失 | BCEWithLogits，pos_weight 按类别频率、上限 20 | 同左 |
| 精度 | AMP fp16 + EMA 0.999 | 同左 |

**增强**：随机缩放裁剪 0.75~1.0、水平翻转 0.5、±10° 旋转、亮度 ±15%、伽马 0.7~1.4、随机擦除 0.25。
**TTA**：原图 0.5 + 水平翻转 0.5（+1.06 千分点）。
**聚合**：study 内多视图等权平均（正位 0.8 / 侧位 0.2 加权实测仅 +0.0001，未采用）。
**融合**：验证集上带放回的贪心前向选择，每步只接受增益 > 0.0001。

## 3. 关键结论

**参数量是本任务的决定因素**，单模型 AUC 随容量单调上升：

| 模型 | 参数量 | 单模型 AUC |
|---|---:|---:|
| ResNet-34（从零） | 2180 万 | 0.7737 |
| DenseNet-121 | 800 万 | 0.7953 |
| DenseNet-201 | 1800 万 | 0.7979 |
| EfficientNet-B0 | 530 万 | 0.8041 |
| ConvNeXt-Small | 4950 万 | 0.8077 |
| **ConvNeXt-Base** | **8760 万** | **0.8102** |

线上分与本地验证分满足线性标定（5 点拟合，残差 RMS 0.24）：

```
线上 ≈ 109.6 × 本地AUC − 11.64        （本地 +0.01 ⇒ 线上 +1.10）
```

## 4. 已排除的方案（负面结果）

| 方案 | 结果 |
|---|---|
| ConvNeXt-Base 训 10 轮 + mixup 0.3 | 0.8061 < 5 轮的 0.8097（过拟合） |
| ConvNeXt-Base@448 | 0.8093 < 384 的 0.8097 |
| ConvNeXt-Base 权重平均（model soup） | 0.8030 < 单种子 0.8102 |
| Swin-Base@384 | 0.7850（数据量不足，5 轮远未收敛） |
| ViT-Base@384 | 12 GB 显存下 bs 6/8 均 OOM |
| ASL 非对称损失 | 种子平均 0.8076 < BCE 的 0.8100 |
| DenseNet-201@512 | 五折 0.8053 < @448 的 0.8079 |
| 多尺度 TTA | EfficientNet-B0 反而 −0.0034 |
| 正位/侧位加权聚合 | 8 模型平均仅 +0.0001 |
| 每族取最优再等权 | 0.8054 < 贪心 0.8115 |
| 逐类别独立贪心集成 | 样本内 0.8240，线上仅 77.53（选择偏差） |
| 训练-推理分辨率错配（FixRes） | 384 已是最优，无增益 |

## 5. 目录结构

```
cxr_src/          训练与推理主代码
  common.py         数据缓存 / 模型构建 / EMA / 指标
  train.py          训练主脚本
  infer.py          推理 + TTA + 保存概率
  auto_sub.py       贪心集成 → submission.csv
  perclass_ens.py   逐类别集成实验
  soup.py           权重平均实验
  chain*.sh         各阶段实验编排脚本
tools/            容器驱动、权重下载、报告生成
analysis/         数据集分析与环境探测
submissions/      最好成绩的提交文件（submission_77.59.csv）
report_assets/    报告用图表与校验结果
```

## 6. 复现

```bash
# 训练（全量数据，ConvNeXt-Base@384）
python cxr_src/train.py --arch convnext_base \
    --weights weights/convnext_base.fb_in22k_ft_in1k.safetensors \
    --size 384 --bs 16 --fold -1 --epochs 5 --lr 1.5e-4 --wd 1e-2 \
    --warmup 1 --seed 42 --out runs/full_cnxb_384_s42

# 推理（含水平翻转 TTA）
python cxr_src/infer.py --ckpt runs/full_cnxb_384_s42/last.pt \
    --split test --tta 2 --save_probs runs/full_cnxb_384_s42/probs_test.npz

# 贪心集成 → submission.csv
python cxr_src/auto_sub.py --tags "full_cnxb_384_s42 full_cnxb_384_s123 p_b0_640_f0" --name v7
```

环境：Python 3.10、PyTorch 2.4.1、timm 1.0.30，单卡海光 DCU K100_AI（12 GB 显存）。

## 7. 合规说明

- 预训练权重全部为 ImageNet 通用非医学预训练（Apache-2.0 / MIT / BSD-3，非 NC、非 gated），完整清单与 SHA-256 见 `模型核验材料与数据声明清单.xlsx`。
- 未使用任何外部医学数据；官方验证集从未参与训练。
- 训练集仅使用赛方提供的 train split。
- 测试集仅用于生成提交，未参与训练、伪标签或统计量估计。