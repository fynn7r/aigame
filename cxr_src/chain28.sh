#!/bin/bash
# 最后一发：ConvNeXt-Large（1.98 亿参数）@320，5 轮
# 判据：单模型超过 0.8108（现最好单模型，含 TTA）才算成功
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
OUT=/data/cxr_first
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

run_full() {
  local tag=$1 sz=$2 bs=$3
  local out=$OUT/runs/$tag lg=/root/_dsh/$tag.log
  grep -q '^BEST' $lg 2>/dev/null && { echo "skip $tag"; return 0; }
  for att in 1 2 3; do
    python wait_vram.py 10.2 >> $lg 2>&1
    echo "--- $tag attempt $att bs=$bs $(date)" >> $lg
    DSH_CACHE=1024 python train.py --arch convnext_large \
        --weights "$W/convnext_large_fb_in22k_ft_in1k.safetensors" \
        --size "$sz" --bs "$bs" --fold -1 --epochs 5 --lr 1.5e-4 --wd 1e-2 \
        --warmup 1 --workers 4 --seed 42 --out "$out" >> $lg 2>&1
    grep -q '^BEST' $lg && return 0
    echo "!! 失败，降 batch / 降分辨率重试" >> $lg
    rm -rf "$out"; sleep 60
    bs=$((bs / 2)); [ "$bs" -lt 4 ] && { bs=4; sz=256; }
  done
}

while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### ConvNeXt-Large 开始 $(date)"

run_full full_cnxL_320_s42 320 8
echo "SCORE large $(grep -h '^BEST' /root/_dsh/full_cnxL_320_s42.log 2>/dev/null | tail -1)"
echo "CHAIN28 DONE $(date)"