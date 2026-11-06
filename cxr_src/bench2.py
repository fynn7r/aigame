# -*- coding: utf-8 -*-
import time, torch, timm

def bench(arch, size, bs, iters=10):
    try:
        m = timm.create_model(arch, pretrained=False, num_classes=10, in_chans=1).cuda()
    except Exception as e:
        print(f"{arch} build fail {str(e)[:60]}", flush=True); return
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    crit = torch.nn.BCEWithLogitsLoss()
    x = torch.randn(bs, 1, size, size, device="cuda")
    y = torch.rand(bs, 10, device="cuda")
    sc = torch.amp.GradScaler("cuda", enabled=True)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = m(x); loss = crit(out.float(), y)
        sc.scale(loss).backward(); sc.step(opt); sc.update()
    for _ in range(3): step()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(iters): step()
    torch.cuda.synchronize(); dt = (time.time()-t)/iters
    mem = torch.cuda.max_memory_allocated()/1e9
    n = sum(p.numel() for p in m.parameters())/1e6
    print(f"{arch:18s} s={size} bs={bs:3d} {dt*1000:6.0f}ms {bs/dt:6.1f}img/s mem{mem:4.1f}G par{n:5.1f}M", flush=True)
    del m, x, y
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

for job in [("efficientnet_b0", 448, 32), ("efficientnet_b0", 512, 24),
            ("efficientnet_b1", 384, 32), ("efficientnet_b2", 384, 28),
            ("resnet34", 384, 32), ("resnet50", 448, 28),
            ("densenet121", 448, 24), ("efficientnet_b3", 320, 28)]:
    bench(*job)