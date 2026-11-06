#!/bin/bash
# 免训练实验：train-test resolution discrepancy（训练 384，推理换分辨率）
set -u
cd /root/_dsh || exit 1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
CK=/data/cxr_first/runs/full_cnxb_384_s123/last.pt
for s in 352 384 416 448 512 576; do
  a=$(DSH_CACHE=1024 python infer.py --ckpt $CK --split valid --size $s --tta 2 2>&1 | grep 'STUDY macro' | head -1)
  echo "Base@train384  推理 $s  ->  $a"
done
CK2=/data/cxr_first/runs/p_b0_640_f0/last.pt
for s in 576 640 704 768; do
  a=$(DSH_CACHE=1024 python infer.py --ckpt $CK2 --split valid --size $s --tta 2 2>&1 | grep 'STUDY macro' | head -1)
  echo "B0@train640    推理 $s  ->  $a"
done
echo "RES SWEEP DONE"