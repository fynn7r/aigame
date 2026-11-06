#!/bin/bash
# 接管 640 探针：把它当作 f0，补 f1-f4；再做 768 探针；然后补不同骨干
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
    grep -q BEST /root/_dsh/${PRE}_f$k.log 2>/dev/null && { echo "skip $OUT"; continue; }
    python train.py --arch "$ARCH" --weights "$WF" --size "$SZ" --bs "$BS" --fold $k \
        --epochs "$EP" --lr "$LR" --warmup 1 --workers 5 "$@" --out "$OUT" \
        > /root/_dsh/${PRE}_f$k.log 2>&1
    echo "SCORE ${PRE}_f$k $(score_of ${PRE}_f$k)"
  done
  echo "=== $PRE done $(date) ==="
}

# 1) 等探针跑完
for i in $(seq 1 120); do
  grep -q BEST /root/_dsh/p_b0_640.log 2>/dev/null && break
  sleep 60
done
echo "### 640 探针完成 $(date)"
pkill -9 -f chain9.sh
sleep 5

# 2) 把探针当作 f0
if [ ! -d /data/cxr_first/runs/p_b0_640_f0 ]; then
  cp -r /data/cxr_first/runs/p_b0_640 /data/cxr_first/runs/p_b0_640_f0
  cp /root/_dsh/p_b0_640.log /root/_dsh/p_b0_640_f0.log
  echo "把 640 探针登记为 f0: $(score_of p_b0_640_f0)"
fi

# 3) 补 f1-f4
export DSH_CACHE=1024
folds p_b0_640 efficientnet_b0 "$W/efficientnet_b0.pth" 640 16 12 3e-4
echo "### 640 五折完成 $(date)"
for k in 0 1 2 3 4; do echo "640 f$k = $(score_of p_b0_640_f$k)"; done
B640=$(score_of p_b0_640_f0)

echo "### 出 v3 提交 10/29/2026 22:52:54"
bash /root/_dsh/night_sub.sh

# 4) 768 探针
echo "### 768 探针 $(date)"
if [ ! -f /data/cxr_first/runs/p_b0_768/best.pt ]; then
  one p_b0_768 0 efficientnet_b0 "$W/efficientnet_b0.pth" 768 12 12 3e-4
fi
S768=$(score_of p_b0_768)
echo "p_b0_768 = $S768 ; 640 f0 = $B640"

if python -c "import sys; sys.exit(0 if $S768 > $B640 else 1)"; then
  echo ">>> 768 胜出"
  folds p_b0_768 efficientnet_b0 "$W/efficientnet_b0.pth" 768 12 12 3e-4
  folds p_r50_640 resnet50 "$W/resnet50.pth" 640 14 12 3e-4
else
  echo ">>> 640 保持最优"
  folds p_r50_640 resnet50 "$W/resnet50.pth" 640 14 12 3e-4
  folds p_b3_640 efficientnet_b3 "$W/efficientnet_b3.pth" 640 12 12 3e-4
fi
unset DSH_CACHE
echo "CHAIN10 DONE $(date)"