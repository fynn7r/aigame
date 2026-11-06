# -*- coding: utf-8 -*-
"""带断点续传/重试的权重下载器，优先用 hf-mirror 镜像。"""
import os, sys, time, urllib.request

OUT = r"E:\aigame\cxr_first\outbox"
MODELS = ["resnet50.a1_in1k", "convnext_tiny.fb_in22k_ft_in1k", "efficientnet_b3.ra2_in1k",
          "tf_efficientnet_b0.ns_jft_in1k", "densenet121.ra_in1k",
          "convnext_small.fb_in22k_ft_in1k"]
BASES = ["https://hf-mirror.com", "https://huggingface.co"]


def fetch(url, path):
    tmp = path + ".part"
    for attempt in range(60):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(tmp, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            size = os.path.getsize(tmp)
            if total and size < total:
                print(f"  partial {size}/{total}, retry", flush=True)
                continue
            os.replace(tmp, path)
            return True
        except Exception as e:
            print(f"  err {repr(e)[:80]} (have {have}), retry", flush=True)
            time.sleep(3)
    return False


if __name__ == "__main__":
    for m in MODELS:
        fn = m.replace(".", "_") + ".safetensors"
        p = os.path.join(OUT, fn)
        if os.path.exists(p) and os.path.getsize(p) > 1e6:
            print(m, "exists", os.path.getsize(p) // 1024 // 1024, "MB", flush=True)
            continue
        ok = False
        for b in BASES:
            url = f"{b}/timm/{m}/resolve/main/model.safetensors"
            print("get", url, flush=True)
            if fetch(url, p):
                print(m, "OK", os.path.getsize(p) // 1024 // 1024, "MB", flush=True)
                ok = True
                break
        if not ok:
            print(m, "FAILED", flush=True)