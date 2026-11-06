#!/bin/bash
# 多骨干 5 折流水线（b0 收敛很快，~ep8 就见顶，所以轮数不用多）
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights

while pgrep -f sweep_pretrained.sh >/dev/null; do sleep 30; done
echo "sweep finished $(date)"

score() { python -c "
import torch
try:
    c = torch.load('/data/cxr_first/runs/$1/best.pt', map_location='cpu')
    print('SCORE $1 %.4f' % c['score'])
except Exception:
    print('SCORE $1 0.0')
"; }

folds() {  # prefix arch weights size bs epochs lr
  local PRE=$1 ARCH=$2 WF=$3 SZ=$4 BS=$5 EP=$6 LR=$7
  echo "=== $PRE ($ARCH ${SZ}px ${EP}ep) start $(date) ==="
  for k in 0 1 2 3 4; do
    local OUT=/data/cxr_first/runs/${PRE}_f$k
    [ -f "$OUT/best.pt" ] && { echo "skip $OUT"; continue; }
    python train.py --arch "$ARCH" --weights "$WF" --size "$SZ" --bs "$BS" --fold $k \
        --epochs "$EP" --lr "$LR" --warmup 1 --workers 6 --out "$OUT" \
        > /root/_dsh/${PRE}_f$k.log 2>&1
    score ${PRE}_f$k
  done
  echo "=== $PRE done $(date) ==="
}

folds p_b0a   efficientnet_b0                 "$W/efficientnet_b0.pth"                448 28 12 4e-4
folds p_b0jft tf_efficientnet_b0.ns_jft_in1k  "$W/tf_efficientnet_b0_ns_jft_in1k.safetensors" 448 28 12 4e-4
folds p_b0_512 efficientnet_b0                "$W/efficientnet_b0.pth"                512 22 12 4e-4
folds p_b3    efficientnet_b3                  "$W/efficientnet_b3.pth"                448 18 15 3e-4
folds p_dn121 densenet121                      "$W/densenet121.pth"                    448 22 15 3e-4
echo "CHAIN4 DONE $(date)"