#!/bin/bash
# 第四阶段：
#   C: b0@640 + CLAHE（X 配方）—— 隔离「对比度增强」变量
#   D: b3@640（X 配方）—— 测更大容量
#   E: convnext_tiny@448（X 配方）—— 测现代架构
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export DSH_CACHE=1024

score_of() { python - <<EOF
import torch
try:
    c = torch.load('/data/cxr_first/runs/$1/best.pt', map_location='cpu', weights_only=False)
    print('%.4f' % c['score'])
except Exception:
    print('0.0')
EOF
}

run_exp() {  # tag fold arch weights size bs ep lr [extra...]
  local tag=$1 fold=$2 arch=$3 wf=$4 sz=$5 bs=$6 ep=$7 lr=$8; shift 8
  local out=/data/cxr_first/runs/$tag
  grep -q BEST /root/_dsh/$tag.log 2>/dev/null && { echo "skip $tag"; return 0; }
  for att in 1 2 3; do
    python wait_vram.py 10.0 >> /root/_dsh/$tag.log 2>&1
    echo "--- $tag attempt $att $(date)" >> /root/_dsh/$tag.log
    python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" --fold "$fold" \
        --epochs "$ep" --lr "$lr" --warmup 1 --workers 5 "$@" --out "$out" \
        >> /root/_dsh/$tag.log 2>&1
    grep -q BEST /root/_dsh/$tag.log && { echo "SCORE $tag $(score_of $tag)"; return 0; }
    echo "!!! $tag attempt $att failed"
    rm -rf "$out"
    sleep 90
  done
  return 1
}

pkill -9 -f chain13.sh
pkill -9 -f 'runs/x_dn121'
sleep 5
rm -rf /data/cxr_first/runs/x_dn121_1024 /data/cxr_first/runs/x_dn121_768
while pgrep -f 'train\.py' > /dev/null; do sleep 20; done
echo "### 第四阶段开始 $(date)"

echo "=== C: b0@640 + CLAHE $(date)"
run_exp x_b0_640_clahe 0 efficientnet_b0 "$W/efficientnet_b0.pth" 640 16 12 3e-4 \
    --pos_weight 1.0 --mixup 0.2 --clahe
echo "C = $(score_of x_b0_640_clahe)  (X 对照 = 0.8081)"

echo "=== D: b3@640 $(date)"
run_exp x_b3_640 0 efficientnet_b3 "$W/efficientnet_b3_ra2_in1k.safetensors" 640 12 12 3e-4 \
    --pos_weight 1.0 --mixup 0.2
echo "D = $(score_of x_b3_640)"

echo "=== E: convnext_tiny@448 $(date)"
run_exp x_cnx_448 0 convnext_tiny "$W/convnext_tiny_fb_in22k_ft_in1k.safetensors" 448 16 12 3e-4 \
    --pos_weight 1.0 --mixup 0.2
echo "E = $(score_of x_cnx_448)"
echo "CHAIN14 DONE $(date)"