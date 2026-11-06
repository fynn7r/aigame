#!/bin/bash
# 串行跑 K 折训练。用法:
#   bash run_folds.sh <tag> <arch> <size> <epochs> <bs> <lr> [fold_start] [fold_end] [extra args...]
set -u
TAG=$1; ARCH=$2; SIZE=$3; EPOCHS=$4; BS=$5; LR=$6
F0=${7:-0}; F1=${8:-4}
shift 8 2>/dev/null || shift $#
EXTRA="$*"
cd /root/_dsh || exit 1
for k in $(seq "$F0" "$F1"); do
  OUT=/data/cxr_first/runs/${TAG}_f${k}
  if [ -f "$OUT/best.pt" ]; then echo "skip $OUT (done)"; continue; fi
  echo "=== fold $k -> $OUT $(date) ==="
  python train.py --arch "$ARCH" --size "$SIZE" --fold "$k" --epochs "$EPOCHS" \
      --bs "$BS" --lr "$LR" --workers 6 --out "$OUT" $EXTRA > /root/_dsh/${TAG}_f${k}.log 2>&1
  echo "=== fold $k finished $(date) ==="
  sleep 5
done
echo "ALL DONE $(date)"