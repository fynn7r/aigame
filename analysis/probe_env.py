# -*- coding: utf-8 -*-
"""容器环境探测"""
import os, sys, subprocess, importlib

def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"ERR {e}"

print("== cpu/mem ==")
print("nproc:", sh("nproc"), "| cgroup cpu:", sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null"))
print("mem:", sh("free -g | sed -n 2p"))
print("ds:", sh("df -h /data / /root | tail -3"))

print("\n== python ==")
print("exe:", sys.executable, sys.version.split()[0])
for m in ["torch", "torchvision", "timm", "albumentations", "cv2", "PIL", "pandas",
          "numpy", "sklearn", "monai", "transformers", "numba", "turbojpeg", "jpeg4py", "safetensors"]:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  {m:14s} {v}")
    except Exception:
        print(f"  {m:14s} MISSING")

print("\n== torch ==")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("dev", p.name, "mem %.1fGB" % (p.total_memory / 1e9), "cc", p.major, p.minor,
          "mp", p.multi_processor_count)
print("cudnn", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)

print("\n== matmul bench ==")
import time
x = torch.randn(2048, 2048, device="cuda", dtype=torch.half)
for _ in range(3):
    y = x @ x
torch.cuda.synchronize()
t = time.time()
for _ in range(20):
    y = x @ x
torch.cuda.synchronize()
dt = (time.time() - t) / 20
print("fp16 2048^3 matmul: %.1f ms -> %.1f TFLOPS" % (dt * 1000, 2 * 2048**3 / dt / 1e12))

print("\n== net ==")
print(sh("timeout 8 curl -sI https://pypi.org/simple/ | head -1"))
print(sh("timeout 8 curl -sI https://huggingface.co | head -1"))
print(sh("pip config list 2>/dev/null"))
print(sh("cat /etc/pip.conf 2>/dev/null | head -5"))