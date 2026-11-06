cd /root/_dsh || exit 1
echo "===== 当前状态"
date
grep -E '^BEST' /root/_dsh/full_cnxb_384_s31337.log 2>/dev/null | tail -1
ps -eo etime,cmd | grep train.py | grep -v grep | head -1 | cut -c1-40
echo "===== Transformer 构建测试"
OMP_NUM_THREADS=2 python - <<'EOF'
import sys
sys.path.insert(0, '/root/_dsh')
import torch, timm
from common import build_model
W = '/root/_dsh/weights/'
cases = [
    ('swin_base_patch4_window12_384', W + 'swin_base_patch4_window12_384_ms_in22k_ft_in1k.safetensors'),
    ('vit_base_patch16_384', W + 'vit_base_patch16_384_augreg_in21k_ft_in1k.safetensors'),
]
for arch, wf in cases:
    try:
        m = build_model(arch, weights=wf)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        m.eval()
        with torch.no_grad():
            y = m(torch.randn(1, 1, 384, 384))
        print('OK %-32s %.1fM  out %s' % (arch, n, tuple(y.shape)), flush=True)
    except Exception:
        import traceback
        print('FAIL', arch, flush=True)
        traceback.print_exc()
EOF