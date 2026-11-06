#!/bin/bash
# 等 p_b0a 5 折跑完 -> 两个正则化/预处理小实验 -> 再跑 b0jft / b0@512 / b3 / dn121 各 5 折
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights

# 等 b0a 五折全部出现 BEST
while [ "$(grep -l BEST /root/_dsh/p_b0a_f*.log 2>/dev/null | wc -l)" -lt 5 ]; do sleep 60; done
echo "b0a folds done $(date)"

run1() {  # tag fold arch weights size bs ep lr extra...
  local tag=$1; shift
  local fold=$1; shift
  local arch=$1; shift
  local wf=$1; shift
  local sz=$1; shift
  local bs=$1; shift
  local ep=$1; shift
  local lr=$1; shift
  python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" --fold "$fold" \
      --epochs "$ep" --lr "$lr" --warmup 1 --workers 6 "$@" \
      --out /data/cxr_first/runs/$tag > /root/_dsh/$tag.log 2>&1
  python -c "
import torch
try:
    c = torch.load('/data/cxr_first/runs/$tag/best.pt', map_location='cpu')
    print('SCORE $tag %.4f' % c['score'])
except Exception:
    print('SCORE $tag 0.0')
"
}

echo "=== E1 mixup 0.2 @448 14ep $(date) ==="
run1 x_b0_mixup 0 efficientnet_b0 "$W/efficientnet_b0.pth" 448 28 14 4e-4 --mixup 0.2
echo "=== E2 clahe @448 12ep $(date) ==="
run1 x_b0_clahe 0 efficientnet_b0 "$W/efficientnet_b0.pth" 448 28 12 4e-4 --clahe

folds() {
  local PRE=$1 ARCH=$2 WF=$3 SZ=$4 BS=$5 EP=$6 LR=$7; shift 7
  echo "=== $PRE ($ARCH ${SZ}px ${EP}ep) start $(date) ==="
  for k in 0 1 2 3 4; do
    local OUT=/data/cxr_first/runs/${PRE}_f$k
    [ -f "$OUT/best.pt" ] && { echo "skip $OUT"; continue; }
    python train.py --arch "$ARCH" --weights "$WF" --size "$SZ" --bs "$BS" --fold $k \
        --epochs "$EP" --lr "$LR" --warmup 1 --workers 6 "$@" --out "$OUT" \
        > /root/_dsh/${PRE}_f$k.log 2>&1
    python -c "
import torch
try:
    c = torch.load('/data/cxr_first/runs/${PRE}_f$k/best.pt', map_location='cpu')
    print('SCORE ${PRE}_f$k %.4f' % c['score'])
except Exception:
    print('SCORE ${PRE}_f$k 0.0')
"
  done
  echo "=== $PRE done $(date) ==="
}

folds p_b0jft tf_efficientnet_b0.ns_jft_in1k "$W/tf_efficientnet_b0_ns_jft_in1k.safetensors" 448 28 12 4e-4
folds p_b0_512 efficientnet_b0 "$W/efficientnet_b0.pth" 512 22 12 4e-4
folds p_b3 efficientnet_b3 "$W/efficientnet_b3.pth" 448 18 15 3e-4
folds p_dn121 densenet121 "$W/densenet121.pth" 448 22 15 3e-4
echo "CHAIN5 DONE $(date)"