# -*- coding: utf-8 -*-
"""GPU 吞吐基准：不同骨干/分辨率/批量的 train step 速度与显存占用"""
import time, torch, sys
import timm

def bench(arch, size, bs, amp=True, iters=12):
    try:
        m = timm.create_model(arch, pretrained=False, num_classes=10, in_chans=1).cuda()
    except Exception as e:
        print(f"{arch} build fail {e}")
        return
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    crit = torch.nn.BCEWithLogitsLoss()
    x = torch.randn(bs, 1, size, size, device="cuda")
    y = torch.rand(bs, 10, device="cuda")
    sc = torch.amp.GradScaler("cuda", enabled=amp)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = m(x); loss = crit(out.float(), y)
        sc.scale(loss).backward(); sc.step(opt); sc.update()
    for _ in range(4):
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t) / iters
    mem = torch.cuda.max_memory_allocated() / 1e9
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{arch:22s} size={size} bs={bs:3d} {dt*1000:7.0f} ms/step  {bs/dt:6.1f} img/s  "
          f"mem {mem:5.1f}GB  par {n_par:5.1f}M", flush=True)
    del m, x, y
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    jobs = [
        ("convnext_tiny", 384, 32),
        ("convnext_tiny", 448, 24),
        ("convnext_small", 384, 24),
        ("efficientnet_b0", 384, 32),
        ("resnet50", 384, 32),
        ("densenet121", 384, 24),
        ("swin_tiny_patch4_window12_384", 384, 24),
    ]
    for arch, size, bs in jobs:
        bench(arch, size, bs)