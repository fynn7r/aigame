#!/bin/bash
# 从 448 的 5 折模型出发，在 512 上再微调几轮（渐进式提升分辨率）
# 用法: bash finetune512.sh <前缀tag> [epochs] [lr]
set -u
cd /root/_dsh || exit 1
PRE=${1:?need tag prefix}
EP=${2:-6}
LR=${3:-1e-4}
W=/root/_dsh/weights/resnet50.pth

for k in 0 1 2 3 4; do
  SRC=/data/cxr_first/runs/${PRE}_f${k}/best.pt
  OUT=/data/cxr_first/runs/${PRE}_ft512_f${k}
  if [ -f "$OUT/best.pt" ]; then echo "skip $OUT"; continue; fi
  if [ ! -f "$SRC" ]; then echo "missing $SRC"; continue; fi
  echo "=== $PRE fold $k 512-ft $(date) ==="
  python train.py --arch resnet50 --weights "$W" --init "$SRC" --size 512 --bs 20 \
      --fold $k --epochs $EP --lr $LR --warmup 0 --workers 6 \
      --out "$OUT" > /root/_dsh/${PRE}_ft512_f${k}.log 2>&1
  grep BEST /root/_dsh/${PRE}_ft512_f${k}.log | tail -1
done
echo "FT512 DONE $(date)"