# -*- coding: utf-8 -*-
"""SimCLR 自监督预训练（只用比赛给的图，不引入任何外部标注数据）。

  python ssl_pretrain.py --epochs 100 --size 224 --bs 256 --out /data/cxr_first/ssl/ssl_r34.pt

产物: checkpoint 里的 "backbone" 是 timm resnet34 的 state_dict，可直接喂给 train.py --init。
"""
import os, sys, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_meta, ImageCache, seed_all


class TwoView(Dataset):
    def __init__(self, meta, cache, idx, size=224):
        self.meta, self.cache, self.idx, self.size = meta, cache, np.asarray(idx), size

    def __len__(self):
        return len(self.idx)

    def _aug(self, img):
        s = self.size
        h, w = img.shape
        scale = random.uniform(0.3, 1.0)
        sh, sw = max(8, int(h * scale)), max(8, int(w * scale))
        y0, x0 = random.randint(0, h - sh), random.randint(0, w - sw)
        img = img[y0:y0 + sh, x0:x0 + sw]
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        if random.random() < 0.5:
            img = img[:, ::-1]
        if random.random() < 0.3:
            M = cv2.getRotationMatrix2D((s / 2, s / 2), random.uniform(-15, 15), 1.0)
            img = cv2.warpAffine(img, M, (s, s), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if random.random() < 0.8:
            img = np.clip(img.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-20, 20), 0, 255)
        if random.random() < 0.2:  # 高斯模糊
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        if random.random() < 0.2:  # 局部遮挡
            eh, ew = random.randint(s // 8, s // 3), random.randint(s // 8, s // 3)
            ey, ex = random.randint(0, s - eh), random.randint(0, s - ew)
            img[ey:ey + eh, ex:ex + ew] = 0
        return np.ascontiguousarray(img.astype(np.float32))

    def __getitem__(self, i):
        img = np.array(self.cache.arr[self.idx[i]], copy=True)
        a = self._aug(img)
        b = self._aug(img)
        x = np.stack([a, b])[:, None]  # (2,1,H,W)
        x = (x / 255.0 - 0.5) / 0.25
        return torch.from_numpy(x).float()


class SimCLR(nn.Module):
    def __init__(self, arch="resnet34", dim=512, proj=128):
        super().__init__()
        import timm
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=1)
        self.proj = nn.Sequential(nn.Linear(self.backbone.num_features, dim), nn.ReLU(),
                                  nn.Linear(dim, proj))

    def forward(self, x):
        h = self.backbone(x)
        return self.proj(h)


def nt_xent(z, temp=0.1):
    z = F.normalize(z.float(), dim=1)
    n = z.shape[0]
    sim = z @ z.t() / temp
    mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e4)
    pos = torch.cat([torch.diag(sim, n // 2), torch.diag(sim, -n // 2)])
    loss = -torch.log(torch.exp(pos) / torch.exp(sim).sum(1)).mean()
    return loss


def selftest():
    """检查 nt_xent 的正确性：随机特征≈ln(2N-1)，完全对齐→接近 0。"""
    import math
    n = 16
    z = torch.randn(2 * n, 32)
    print("random  loss", float(nt_xent(z)), "expected", math.log(2 * n - 1))
    a = torch.randn(n, 32)
    z = torch.cat([a, a + 1e-3 * torch.randn(n, 32)], 0)
    print("aligned loss", float(nt_xent(z)), "(应远小于 chance)")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="resnet34")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--bs", type=int, default=224)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="只取前 N 张做冒烟测试")
    a = p.parse_args()
    seed_all(a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    meta = load_meta()
    cache = ImageCache()
    idx = np.where(meta["split"].isin(["train", "valid"]))[0]
    if a.limit:
        idx = idx[:a.limit]
    ds = TwoView(meta, cache, idx, a.size)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    print(f"ssl imgs {len(idx)} steps {len(dl)}", flush=True)

    model = SimCLR(a.arch).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(dl) * a.epochs
    warm = len(dl) * 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    best = 1e9
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x in dl:
            b = x.shape[0]
            x = x.view(-1, 1, a.size, a.size).cuda(non_blocking=True)
            v0, v1 = x[0::2], x[1::2]          # 正样本对必须按 cat 布局排列
            x = torch.cat([v0, v1], dim=0)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                z = model(x)
                loss = nt_xent(z.float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        print(f"ep{ep} loss {tot/n:.4f} {time.time()-t0:.0f}s lr {sched.get_last_lr()[0]:.2e}", flush=True)
        torch.save(dict(backbone=model.backbone.state_dict(), arch=a.arch, epoch=ep,
                        loss=tot / n), a.out)
        if ep % 10 == 9:
            torch.save(dict(backbone=model.backbone.state_dict(), arch=a.arch, epoch=ep,
                            loss=tot / n), a.out.replace(".pt", f"_ep{ep}.pt"))


if __name__ == "__main__":
    main()