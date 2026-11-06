# -*- coding: utf-8 -*-
"""测 dataloader 吞吐，判断训练到底是 GPU 瓶颈还是取数瓶颈。"""
import sys, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_meta, ImageCache, CXRDataset

meta = load_meta()
cache = ImageCache()
print("cache", cache.arr.shape, cache.arr.dtype, "path", cache.path, "GB", cache.arr.nbytes/1e9)

tr = np.where(meta["split"] == "train")[0]
size = int(sys.argv[1]) if len(sys.argv) > 1 else 448

for nw in [4, 6, 8]:
    ds = CXRDataset(meta, cache, tr, train=True, img_size=size)
    dl = DataLoader(ds, batch_size=28, shuffle=True, num_workers=nw,
                    pin_memory=True, drop_last=True, persistent_workers=False)
    it = iter(dl)
    for _ in range(10):
        next(it)
    n, t0 = 0, time.time()
    for _ in range(60):
        x, y = next(it)
        n += x.size(0)
    dt = time.time() - t0
    print(f"workers={nw}: {n/dt:.1f} img/s  ({dt/n*1000:.2f} ms/img)", flush=True)