#!/bin/bash
# E1 跑完 -> 测 batch size -> 用最优 bs 跑 4 个骨干 × 5 折
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

while ! grep -q BEST /root/_dsh/x_b0_mixup.log 2>/dev/null; do sleep 30; done
MX=$(score_of x_b0_mixup)
echo "### mixup fold0 score=$MX  (baseline b0a_f0=0.8021, b0_448_18ep=0.8031) $(date)"

if python -c "import sys; sys.exit(0 if $MX > 0.8050 else 1)"; then
  MXARG="--mixup 0.2"; echo ">>> mixup 有效 -> b0 家族启用"
else
  MXARG=""; echo ">>> mixup 无效 -> 不启用"
fi

echo "### bench $(date)"
python bench_train.py efficientnet_b0 448 28,40 > /root/_dsh/bench448.log 2>&1
python bench_train.py efficientnet_b0 512 20,28 > /root/_dsh/bench512.log 2>&1
grep -E 'img/s|BESTBS' /root/_dsh/bench448.log
grep -E 'img/s|BESTBS' /root/_dsh/bench512.log
BS448=$(grep BESTBS /root/_dsh/bench448.log | sed 's/.*bs=\([0-9]*\).*/\1/')
BS512=$(grep BESTBS /root/_dsh/bench512.log | sed 's/.*bs=\([0-9]*\).*/\1/')
[ -z "$BS448" ] && BS448=28
[ -z "$BS512" ] && BS512=20
echo "picked bs448=$BS448 bs512=$BS512"

folds() {
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

folds p_b0jft tf_efficientnet_b0.ns_jft_in1k "$W/tf_efficientnet_b0_ns_jft_in1k.safetensors" 448 "$BS448" 12 4e-4 $MXARG
folds p_b0_512 efficientnet_b0 "$W/efficientnet_b0.pth" 512 "$BS512" 12 4e-4 $MXARG
folds p_b3 efficientnet_b3 "$W/efficientnet_b3.pth" 448 20 15 3e-4
folds p_dn121 densenet121 "$W/densenet121.pth" 448 24 15 3e-4
echo "CHAIN7 DONE $(date)"