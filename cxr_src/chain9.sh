#!/bin/bash
# 接管：停掉 chain7（不再补 jft f1/f3/f4，收益低），等当前折收尾后立刻做 640 探针
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

score_of() { python -c "
import torch
try:
    c = torch.load('/data/cxr_first/runs/$1/best.pt', map_location='cpu')
    print('%.4f' % c['score'])
except Exception:
    print('0.0')
"; }

one() {
  local tag=$1; shift; local fold=$1; shift; local arch=$1; shift
  local wf=$1; shift; local sz=$1; shift; local bs=$1; shift
  local ep=$1; shift; local lr=$1; shift
  python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" --fold "$fold" \
      --epochs "$ep" --lr "$lr" --warmup 1 --workers 5 "$@" --out /data/cxr_first/runs/$tag \
      > /root/_dsh/$tag.log 2>&1
  echo "SCORE $tag $(score_of $tag)"
}

folds() {
  local PRE=$1; shift; local ARCH=$1; shift; local WF=$1; shift
  local SZ=$1; shift; local BS=$1; shift; local EP=$1; shift; local LR=$1; shift
  echo "=== $PRE ($ARCH ${SZ}px bs$BS ${EP}ep) start $(date) ==="
  for k in 0 1 2 3 4; do
    local OUT=/data/cxr_first/runs/${PRE}_f$k
    [ -f "$OUT/best.pt" ] && grep -q BEST /root/_dsh/${PRE}_f$k.log 2>/dev/null && { echo "skip $OUT"; continue; }
    python train.py --arch "$ARCH" --weights "$WF" --size "$SZ" --bs "$BS" --fold $k \
        --epochs "$EP" --lr "$LR" --warmup 1 --workers 5 "$@" --out "$OUT" \
        > /root/_dsh/${PRE}_f$k.log 2>&1
    echo "SCORE ${PRE}_f$k $(score_of ${PRE}_f$k)"
  done
  echo "=== $PRE done $(date) ==="
}

pkill -9 -f chain7.sh
pkill -9 -f chain8.sh
echo "### 停掉旧链路，等当前折收尾 $(date)"
for i in $(seq 1 240); do
  pgrep -f 'train\.py' > /dev/null || break
  sleep 60
done
echo "### GPU 空闲 $(date)"
pgrep -f 'train\.py' && { echo "!!! 还有训练在跑，退出"; exit 1; }

echo "### 640 探针 $(date)"
if [ ! -f /data/cxr_first/runs/p_b0_640/best.pt ]; then
  DSH_CACHE=1024 one p_b0_640 0 efficientnet_b0 "$W/efficientnet_b0.pth" 640 16 12 3e-4
fi
S640=$(score_of p_b0_640)
echo "p_b0_640 = $S640  (448 baseline: b0a_f0 0.8021 / b0_448 0.8031)"

if python -c "import sys; sys.exit(0 if $S640 > 0.8045 else 1)"; then
  echo ">>> 640 胜出"
  export DSH_CACHE=1024
  folds p_b0_640 efficientnet_b0 "$W/efficientnet_b0.pth" 640 16 12 3e-4
  unset DSH_CACHE
  folds p_b3 efficientnet_b3 "$W/efficientnet_b3.pth" 448 16 15 3e-4
else
  echo ">>> 640 未胜出，跑 512"
  folds p_b0_512 efficientnet_b0 "$W/efficientnet_b0.pth" 512 20 12 4e-4
  folds p_b3 efficientnet_b3 "$W/efficientnet_b3.pth" 448 16 15 3e-4
fi
folds p_dn121 densenet121 "$W/densenet121.pth" 448 20 15 3e-4
echo "CHAIN9 DONE $(date)"