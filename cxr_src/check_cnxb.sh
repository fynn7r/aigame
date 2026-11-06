cd /root/_dsh || exit 1
echo "===== 文件"
ls -la /root/_dsh/weights/convnext_base.fb_in22k_ft_in1k.safetensors
echo "===== 构建测试"
OMP_NUM_THREADS=2 python - <<'EOF'
import sys
sys.path.insert(0, '/root/_dsh')
import torch
from common import build_model
try:
    m = build_model('convnext_base', weights='/root/_dsh/weights/convnext_base.fb_in22k_ft_in1k.safetensors')
    print('OK params %.1fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(1, 1, 384, 384))
    print('forward ok', tuple(y.shape))
except Exception:
    import traceback
    traceback.print_exc()
EOF
echo "===== s123 进度"
grep -E '^ep|BEST' /root/_dsh/full_cnxs_448_s123.log 2>/dev/null | tail -3