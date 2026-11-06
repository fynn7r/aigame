#!/bin/bash
# 预训练骨干横扫：每个 arch 用同一套配方跑 fold0，比较单模型验证 AUC
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
run() {
  local arch=$1 size=$2 ep=$3 bs=$4 lr=$5 tag=$6
  echo "=== $tag start $(date) ==="
  python train.py --arch "$arch" --weights "$W/$arch.pth" --size "$size" --fold 0 \
      --epochs "$ep" --bs "$bs" --lr "$lr" --warmup 1 --workers 6 \
      --out /data/cxr_first/runs/$tag > /root/_dsh/$tag.log 2>&1
  echo "=== $tag done $(date): $(grep BEST /root/_dsh/$tag.log | tail -1) ==="
}
run resnet50 448 18 24 4e-4 p_r50_448
run efficientnet_b0 448 18 28 4e-4 p_b0_448
echo "SWEEP DONE $(date)"