# -*- coding: utf-8 -*-
"""测不同 batch size 下的训练吞吐，找 GPU 利用率最优点。"""
import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_model

torch.backends.cudnn.benchmark = True


def bench(arch, size, bs, iters=14, wf=None):
    model = build_model(arch, num_classes=10,
                        weights=wf).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    crit = torch.nn.BCEWithLogitsLoss()
    x = torch.randn(bs, 1, size, size, device="cuda")
    y = (torch.rand(bs, 10, device="cuda") > 0.7).float()
    for _ in range(3):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = time.time() - t0
    ips = bs * iters / dt
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"{arch:22s} {size}px bs={bs:3d}: {ips:6.1f} img/s  {dt/iters*1000:6.0f} ms/step  "
          f"peak {mem:.2f} GB", flush=True)
    return (bs, ips, mem)
    del model, opt, x, y
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    arch = sys.argv[1] if len(sys.argv) > 1 else "efficientnet_b0"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 448
    res = []
    for bs in [int(v) for v in sys.argv[3].split(",")]:
        try:
            got = bench(arch, size, bs)
            if got:
                res.append(got)
        except Exception as e:
            print(f"{arch} {size} bs={bs}: FAILED {type(e).__name__} {str(e)[:80]}", flush=True)
            torch.cuda.empty_cache()
    if res:
        res.sort(key=lambda r: -r[1])
        print(f"BESTBS {arch} {size} bs={res[0][0]} ips={res[0][1]:.1f}", flush=True)
        for bs, ips in res:
            print(f"  bs={bs} -> {ips:.1f} img/s  ({ips/res[0][1]*100:.0f}% of best)")