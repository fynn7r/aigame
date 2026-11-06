#!/bin/bash
# 多样性优先：先测 mixup 正则，再跑 4 个骨干 × 5 折
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights

score_of() { python -c "
import torch
try:
    c = torch.load('/data/cxr_first/runs/$1/best.pt', map_location='cpu')
    print('%.4f' % c['score'])
except Exception:
    print('0.0')
"; }

one() {  # tag fold arch weights size bs ep lr extra...
  local tag=$1; shift; local fold=$1; shift; local arch=$1; shift
  local wf=$1; shift; local sz=$1; shift; local bs=$1; shift
  local ep=$1; shift; local lr=$1; shift
  python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" --fold "$fold" \
      --epochs "$ep" --lr "$lr" --warmup 1 --workers 5 "$@" --out /data/cxr_first/runs/$tag \
      > /root/_dsh/$tag.log 2>&1
  echo "SCORE $tag $(score_of $tag)"
}

folds() {  # prefix arch weights size bs ep lr extra...
  local PRE=$1; shift; local ARCH=$1; shift; local WF=$1; shift
  local SZ=$1; shift; local BS=$1; shift; local EP=$1; shift; local LR=$1; shift
  echo "=== $PRE ($ARCH ${SZ}px ${EP}ep) start $(date) ==="
  for k in 0 1 2 3 4; do
    local OUT=/data/cxr_first/runs/${PRE}_f$k
    [ -f "$OUT/best.pt" ] && { echo "skip $OUT"; continue; }
    python train.py --arch "$ARCH" --weights "$WF" --size "$SZ" --bs "$BS" --fold $k \
        --epochs "$EP" --lr "$LR" --warmup 1 --workers 5 "$@" --out "$OUT" \
        > /root/_dsh/${PRE}_f$k.log 2>&1
    echo "SCORE ${PRE}_f$k $(score_of ${PRE}_f$k)"
  done
  echo "=== $PRE done $(date) ==="
}

if [ ! -f /data/cxr_first/runs/x_b0_mixup/best.pt ]; then
  echo "=== E1 mixup0.2 14ep @448 $(date) ==="
  one x_b0_mixup 0 efficientnet_b0 "$W/efficientnet_b0.pth" 448 28 14 4e-4 --mixup 0.2
fi
MX=$(score_of x_b0_mixup)
echo "mixup score=$MX (baseline b0a_f0=0.8021, b0_448=0.8031)"
if python -c "import sys; sys.exit(0 if $MX > 0.8050 else 1)"; then
  echo ">>> mixup 有效，b0 家族全部改用 mixup"
  MXARG="--mixup 0.2"
else
  echo ">>> mixup 无效，后续不用"
  MXARG=""
fi

folds p_b0jft tf_efficientnet_b0.ns_jft_in1k "$W/tf_efficientnet_b0_ns_jft_in1k.safetensors" 448 28 12 4e-4 $MXARG
folds p_b0_512 efficientnet_b0 "$W/efficientnet_b0.pth" 512 22 12 4e-4 $MXARG
folds p_b3 efficientnet_b3 "$W/efficientnet_b3.pth" 448 18 15 3e-4
folds p_dn121 densenet121 "$W/densenet121.pth" 448 22 15 3e-4
echo "CHAIN6 DONE $(date)"