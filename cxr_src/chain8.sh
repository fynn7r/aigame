#!/bin/bash
# JFT 5 折跑完后：
#   1) 640 分辨率探针（fold0，1024 缓存）
#   2) 按探针结果选 640 或 512 跑满 5 折
#   3) b3 / densenet 补充多样性
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
  echo "=== $PRE ($ARCH ${SZ}px bs$BS ${EP}ep) start $(date) ==="
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

while [ "$(grep -l BEST /root/_dsh/p_b0jft_f*.log 2>/dev/null | wc -l)" -lt 5 ]; do sleep 60; done
echo "### jft 5 folds done $(date)"
# 掐掉 chain7（否则它会自己接着跑 512，和本脚本抢显存）
pkill -9 -f chain7.sh
pkill -9 -f 'runs/p_b0_512'
sleep 8
for k in 0 1 2 3 4; do
  L=/root/_dsh/p_b0_512_f$k.log
  if [ -f "$L" ] && ! grep -q BEST "$L"; then
    echo "清理中断的 p_b0_512_f$k"
    rm -rf /data/cxr_first/runs/p_b0_512_f$k
  fi
done
pgrep -c -f 'train\.py' || echo "no training running"
for k in 0 1 2 3 4; do echo "jft f$k = $(score_of p_b0jft_f$k)"; done

echo "### 640 probe $(date)"
if [ ! -f /data/cxr_first/runs/p_b0_640/best.pt ]; then
  DSH_CACHE=1024 one p_b0_640 0 efficientnet_b0 "$W/efficientnet_b0.pth" 640 16 12 3e-4
fi
S640=$(score_of p_b0_640)
echo "p_b0_640 probe = $S640  (448 baseline 0.8031)"

if python -c "import sys; sys.exit(0 if $S640 > 0.8045 else 1)"; then
  echo ">>> 640 胜出 -> 5 折跑 640"
  export DSH_CACHE=1024
  folds p_b0_640 efficientnet_b0 "$W/efficientnet_b0.pth" 640 16 12 3e-4
  unset DSH_CACHE
else
  echo ">>> 640 未胜出 -> 跑 512"
  folds p_b0_512 efficientnet_b0 "$W/efficientnet_b0.pth" 512 20 12 4e-4
fi

folds p_b3 efficientnet_b3 "$W/efficientnet_b3.pth" 448 16 15 3e-4
folds p_dn121 densenet121 "$W/densenet121.pth" 448 20 15 3e-4
echo "CHAIN8 DONE $(date)"