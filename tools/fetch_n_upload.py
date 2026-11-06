# -*- coding: utf-8 -*-
"""下载 timm 权重到 weights/，再断点续传上到容器。可后台长跑。"""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upresume  # noqa: E402

BASE = r"E:\aigame\cxr_first"
OUT = os.path.join(BASE, "weights")
MIRRORS = ["https://hf-mirror.com", "https://huggingface.co"]
MODELS = sys.argv[1:] or [
    "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "vit_base_patch16_384.augreg_in21k_ft_in1k",
]


def fetch(url, path):
    tmp = path + ".part"
    for _ in range(80):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if have:
            req.add_header("Range", "bytes=%d-" % have)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with open(tmp, mode) as f:
                    while True:
                        c = r.read(1 << 20)
                        if not c:
                            break
                        f.write(c)
            size = os.path.getsize(tmp)
            if total and size < total:
                print("  partial %d/%d retry" % (size, total), flush=True)
                continue
            os.replace(tmp, path)
            return True
        except Exception as e:
            print("  err %s (have %d) retry" % (repr(e)[:70], have), flush=True)
            import time
            time.sleep(3)
    return False


for m in MODELS:
    fn = m.replace(".", "_") + ".safetensors"
    local = os.path.join(OUT, fn)
    if not (os.path.exists(local) and os.path.getsize(local) > 1e6):
        ok = False
        for b in MIRRORS:
            url = "%s/timm/%s/resolve/main/model.safetensors" % (b, m)
            print("下载", url, flush=True)
            if fetch(url, local):
                print("OK", m, os.path.getsize(local) // 1048576, "MB", flush=True)
                ok = True
                break
        if not ok:
            print("FAILED download", m, flush=True)
            continue
    else:
        print("已存在", m, os.path.getsize(local) // 1048576, "MB", flush=True)

    remote = "/root/_dsh/weights/%s" % fn
    print("开始上传", m, flush=True)
    sys.argv = ["upresume", local, remote]
    try:
        upresume.main()
    except SystemExit:
        pass
    print("上传流程结束", m, flush=True)