# -*- coding: utf-8 -*-
"""验证各路预训练权重能否正确加载 + 前向。python verify_weights.py"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_model

W = "/root/_dsh/weights"
CASES = [
    ("tv", "resnet50", f"{W}/resnet50.pth"),
    ("tv", "densenet121", f"{W}/densenet121.pth"),
    ("tv", "efficientnet_b0", f"{W}/efficientnet_b0.pth"),
    ("tv", "efficientnet_b3", f"{W}/efficientnet_b3.pth"),
    ("tv", "convnext_tiny", f"{W}/convnext_tiny.pth"),
    ("timm", "resnet50.a1_in1k", f"{W}/resnet50_a1_in1k.safetensors"),
    ("timm", "convnext_tiny.fb_in22k_ft_in1k", f"{W}/convnext_tiny_fb_in22k_ft_in1k.safetensors"),
    ("timm", "efficientnet_b3.ra2_in1k", f"{W}/efficientnet_b3_ra2_in1k.safetensors"),
    ("timm", "tf_efficientnet_b0.ns_jft_in1k", f"{W}/tf_efficientnet_b0_ns_jft_in1k.safetensors"),
    ("timm", "densenet121.ra_in1k", f"{W}/densenet121_ra_in1k.safetensors"),
    ("timm", "convnext_small.fb_in22k_ft_in1k", f"{W}/convnext_small_fb_in22k_ft_in1k.safetensors"),
]

for builder, arch, path in CASES:
    try:
        m = build_model(arch, weights=path, builder=builder).eval()
        n = sum(p.numel() for p in m.parameters()) / 1e6
        with torch.no_grad():
            y = m(torch.randn(1, 1, 320, 320))
        print(f"OK   {arch:36s} {n:6.1f}M  out {tuple(y.shape)}", flush=True)
    except Exception as e:
        print(f"FAIL {arch:36s} {repr(e)[:160]}", flush=True)