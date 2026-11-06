#!/bin/bash
# Transformer 批次：Swin-Base@384 然后 ViT-Base@384
# bs 8（注意力比卷积吃显存，12GB 卡上 16 会 OOM）
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
OUT=/data/cxr_first
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

run_full() {
  local tag=$1 arch=$2 wf=$3 sz=$4 bs=$5 ep=$6 lr=$7 seed=$8
  local out=$OUT/runs/$tag lg=/root/_dsh/$tag.log
  grep -q '^BEST' $lg 2>/dev/null && { echo "skip $tag"; return 0; }
  for att in 1 2; do
    python wait_vram.py 10.2 >> $lg 2>&1
    echo "--- $tag attempt $att $(date)" >> $lg
    DSH_CACHE=1024 python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" \
        --fold -1 --epochs "$ep" --lr "$lr" --wd 1e-2 --warmup 1 --workers 4 --seed "$seed" \
        --out "$out" >> $lg 2>&1
    grep -q '^BEST' $lg && return 0
    echo "!! $tag 失败，降低 batch 重试" >> $lg
    rm -rf "$out"; sleep 60
    bs=6
  done
}

while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### Transformer 批次开始 $(date)"

run_full full_swinb_384_s42 swin_base_patch4_window12_384 \
    "$W/swin_base_patch4_window12_384_ms_in22k_ft_in1k.safetensors" 384 8 5 1.5e-4 42
echo "SCORE swin $(grep -h '^BEST' /root/_dsh/full_swinb_384_s42.log 2>/dev/null | tail -1)"

run_full full_vitb_384_s42 vit_base_patch16_384 \
    "$W/vit_base_patch16_384_augreg_in21k_ft_in1k.safetensors" 384 8 5 1.5e-4 42
echo "SCORE vit $(grep -h '^BEST' /root/_dsh/full_vitb_384_s42.log 2>/dev/null | tail -1)"
echo "CHAIN26 DONE $(date)"