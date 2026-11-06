# -*- coding: utf-8 -*-
"""公共组件：数据读取、增强、模型、指标。"""
import os, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from sklearn.metrics import roc_auc_score

ROOT = "/data/cxr_first"
PROC = os.path.join(ROOT, "proc")
SIZE_CACHE = int(os.environ.get("DSH_CACHE", 512))
CLASS_NAMES = ["EnlargedCardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
               "Edema", "Cardiomegaly", "Atelectasis", "LungOpacity", "PleuralEffusion",
               "NoFinding"]


def load_meta():
    df = pd.read_csv(os.path.join(PROC, "meta.csv"))
    df["labels"] = df[[f"c{i}" for i in range(1, 11)]].values.astype(np.float32).tolist()
    return df


class ImageCache:
    def __init__(self, path=None):
        self.path = path or os.path.join(PROC, f"images_{SIZE_CACHE}.npy")
        self.arr = np.load(self.path, mmap_mode="r")


class CXRDataset(Dataset):
    def __init__(self, meta, cache, indices, train=False, img_size=384, aug=None, clahe=False):
        self.meta = meta
        self.cache = cache
        self.indices = np.asarray(indices)
        self.train = train
        self.img_size = img_size
        self.aug = aug or (default_train_aug(img_size, clahe) if train
                           else default_eval_aug(img_size, clahe))
        self.labels = np.stack(meta.loc[self.indices, "labels"].values).astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.indices[i]
        img = np.asarray(self.cache.arr[row])
        img = self.aug(np.array(img, copy=True))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = (img / 255.0 - 0.5) / 0.25
        return img, torch.from_numpy(self.labels[i])


# ---------------- augmentation (numpy/cv2) ----------------
import cv2


def _rand(a, b):
    return random.uniform(a, b)


def default_train_aug(size, clahe=False):
    _c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None

    def aug(img):
        h, w = img.shape
        # 随机缩放裁剪
        scale = _rand(0.75, 1.0)
        sh, sw = int(h * scale), int(w * scale)
        y0 = random.randint(0, h - sh)
        x0 = random.randint(0, w - sw)
        img = img[y0:y0 + sh, x0:x0 + sw]
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        if _c is not None:
            img = _c.apply(img)
        if random.random() < 0.5:
            img = img[:, ::-1]
        if random.random() < 0.5:
            ang = _rand(-10, 10)
            M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if random.random() < 0.3:
            img = np.clip(img.astype(np.float32) * _rand(0.85, 1.15) + _rand(-15, 15), 0, 255).astype(np.uint8)
        if random.random() < 0.2:
            g = _rand(0.7, 1.4)
            lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)]).astype(np.uint8)
            img = cv2.LUT(img, lut)
        if random.random() < 0.25:
            # 随机遮挡
            for _ in range(random.randint(1, 3)):
                eh, ew = random.randint(size // 16, size // 6), random.randint(size // 16, size // 6)
                ey, ex = random.randint(0, size - eh), random.randint(0, size - ew)
                img[ey:ey + eh, ex:ex + ew] = 0
        return img
    return aug


def default_eval_aug(size, clahe=False):
    if clahe:
        _c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        def aug(img):
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            return _c.apply(img)
        return aug

    def aug(img):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        return img
    return aug


# ---------------- model ----------------
TV_BUILDERS = {}


def _tv():
    import torchvision.models as tvm
    global TV_BUILDERS
    if not TV_BUILDERS:
        TV_BUILDERS = {
            "resnet34": tvm.resnet34, "resnet50": tvm.resnet50,
            "resnet18": tvm.resnet18, "resnet101": tvm.resnet101,
            "densenet121": tvm.densenet121, "densenet169": tvm.densenet169,
            "efficientnet_b0": tvm.efficientnet_b0, "efficientnet_b2": tvm.efficientnet_b2,
            "efficientnet_b3": tvm.efficientnet_b3, "efficientnet_b4": tvm.efficientnet_b4,
            "convnext_tiny": tvm.convnext_tiny, "convnext_small": tvm.convnext_small,
            "regnet_y_8g": tvm.regnet_y_8gf,
        }
    return TV_BUILDERS


def _first_conv_name(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            return name
    raise RuntimeError("no conv")


@torch.no_grad()
def to_grayscale_first_conv(model):
    """把第一个 conv 从 3 通道改成 1 通道（RGB 权重取均值）。"""
    name = _first_conv_name(model)
    mod = model.get_submodule(name)
    w = mod.weight.data.mean(dim=1, keepdim=True)
    new = nn.Conv2d(1, mod.out_channels, mod.kernel_size, mod.stride, mod.padding,
                    mod.dilation, mod.groups, mod.bias is not None)
    new.weight.data.copy_(w)
    if mod.bias is not None:
        new.bias.data.copy_(mod.bias.data)
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    setattr(parent, name.split(".")[-1], new)
    return name


def _replace_head(model, arch, num_classes):
    lin = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            lin = (name, m)
    if lin is None:
        raise RuntimeError("no linear head")
    name, m = lin
    new = nn.Linear(m.in_features, num_classes)
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    setattr(parent, name.split(".")[-1], new)


def build_model(arch="convnext_tiny", drop=0.0, drop_path=0.1, weights=None, num_classes=10,
                builder="timm"):
    """weights 给定时用 torchvision 结构 + 预训练权重（自动转 1 通道）。"""
    if builder == "tv":
        builders = _tv()
        if arch not in builders:
            raise KeyError(f"{arch} 不在 torchvision 支持列表: {list(builders)}")
        model = builders[arch](weights=None, num_classes=num_classes)
        if weights:
            sd = torch.load(weights, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            sd = {k: v for k, v in sd.items()
                  if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"pretrained: loaded {len(sd)} tensors "
                  f"(missing {len(missing)}, unexpected {len(unexpected)})", flush=True)
        to_grayscale_first_conv(model)
        model.builder = "tv"
        return model
    import timm
    kw = dict(pretrained=False, num_classes=num_classes, in_chans=(3 if weights else 1))
    if drop:
        kw["drop_rate"] = drop
    if drop_path:
        kw["drop_path_rate"] = drop_path
    try:
        m = timm.create_model(arch, **kw)
    except TypeError:
        kw.pop("drop_path_rate", None)
        kw.pop("drop_rate", None)
        m = timm.create_model(arch, **kw)
    if weights:
        if weights.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(weights)
        else:
            sd = torch.load(weights, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
        sd = {k: v for k, v in sd.items()
              if k in m.state_dict() and m.state_dict()[k].shape == v.shape}
        miss = m.load_state_dict(sd, strict=False)
        print(f"pretrained(timm): loaded {len(sd)} tensors, missing {len(miss.missing_keys)}",
              flush=True)
        to_grayscale_first_conv(m)
    m.builder = "timm"
    return m


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.n = 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        return model


# ---------------- metrics ----------------
def macro_auc(y_true, y_pred):
    aucs = []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        if yt.sum() > 0 and yt.sum() < len(yt):
            aucs.append(roc_auc_score(yt, y_pred[:, c]))
        else:
            aucs.append(float("nan"))
    return float(np.nanmean(aucs)), aucs


def study_aggregate(meta_df, probs, study_ids, mode="mean"):
    """按 Study_id 聚合影像级概率。"""
    df = pd.DataFrame(probs, columns=[f"p{i}" for i in range(10)])
    df["Study_id"] = list(study_ids)
    g = df.groupby("Study_id")
    agg = g.mean() if mode == "mean" else g.max()
    return agg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GroupKFold:
    """按 Subject_id 分组 K 折（患者级不泄漏）"""
    def __init__(self, meta, n_splits=5, seed=42):
        subs = meta["Subject_id"].unique()
        rng = np.random.RandomState(seed)
        rng.shuffle(subs)
        self.fold_of = {s: i % n_splits for i, s in enumerate(subs)}
        self.n_splits = n_splits
        self.meta = meta

    def split(self, k):
        f = self.meta["Subject_id"].map(self.fold_of).values
        return np.where(f != k)[0], np.where(f == k)[0]