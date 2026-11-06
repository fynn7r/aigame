#!/bin/bash
# Base 只训 5 轮其实没收敛（loss 还在猛降就被截断了）
# 试 10 轮 + mixup 0.3 加强正则，看能不能把单模型推上 0.812+
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
OUT=/data/cxr_first
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

run_full() {
  local tag=$1 sz=$2 bs=$3 ep=$4 lr=$5 wd=$6 seed=$7 mx=$8
  local out=$OUT/runs/$tag lg=/root/_dsh/$tag.log
  grep -q '^BEST' $lg 2>/dev/null && { echo "skip $tag"; return 0; }
  for att in 1 2; do
    python wait_vram.py 10.0 >> $lg 2>&1
    echo "--- $tag attempt $att $(date)" >> $lg
    DSH_CACHE=1024 python train.py --arch convnext_base \
        --weights "$W/convnext_base.fb_in22k_ft_in1k.safetensors" \
        --size "$sz" --bs "$bs" --fold -1 --epochs "$ep" --lr "$lr" --wd "$wd" \
        --warmup 1 --workers 5 --seed "$seed" --mixup "$mx" --out "$out" >> $lg 2>&1
    grep -q '^BEST' $lg && return 0
    rm -rf "$out"; sleep 60
  done
}

while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### Base 长训开始 $(date)"

run_full full_cnxb_384_L10_s42 384 16 10 1.5e-4 1e-2 42 0.3
echo "SCORE L10_s42 $(grep -h '^BEST' /root/_dsh/full_cnxb_384_L10_s42.log 2>/dev/null | tail -1)"

run_full full_cnxb_384_L10_s123 384 16 10 1.5e-4 1e-2 123 0.3
echo "SCORE L10_s123 $(grep -h '^BEST' /root/_dsh/full_cnxb_384_L10_s123.log 2>/dev/null | tail -1)"
echo "CHAIN24 DONE $(date)"