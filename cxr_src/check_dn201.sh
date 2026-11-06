echo "===== file"
ls -la /root/_dsh/weights/densenet201.tv_in1k.safetensors
md5sum /root/_dsh/weights/densenet201.tv_in1k.safetensors
echo "===== build test"
cd /root/_dsh
OMP_NUM_THREADS=2 python - <<'EOF'
import sys
sys.path.insert(0, '/root/_dsh')
import torch
from common import build_model
try:
    m = build_model('densenet201', weights='/root/_dsh/weights/densenet201.tv_in1k.safetensors')
    print('OK params %.1fM' % (sum(p.numel() for p in m.parameters()) / 1e6))
    x = torch.randn(1, 1, 384, 384)
    m.eval()
    with torch.no_grad():
        y = m(x)
    print('forward ok', tuple(y.shape))
except Exception as e:
    import traceback
    traceback.print_exc()
EOF
echo "===== training progress"
grep -E '^ep[0-9]|BEST' /root/_dsh/ref_cnxs_384.log | tail -3