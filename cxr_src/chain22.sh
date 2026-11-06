#!/bin/bash
# v7 之后：ConvNeXt-Base@448 全量训练（更高分辨率，预期再 +0.002）
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
OUT=/data/cxr_first
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

run_full() {
  local tag=$1 arch=$2 wf=$3 sz=$4 bs=$5 ep=$6 lr=$7 wd=$8 seed=$9
  local out=$OUT/runs/$tag lg=/root/_dsh/$tag.log
  grep -q '^BEST' $lg 2>/dev/null && { echo "skip $tag"; return 0; }
  for att in 1 2; do
    python wait_vram.py 10.0 >> $lg 2>&1
    echo "--- $tag attempt $att $(date)" >> $lg
    DSH_CACHE=1024 python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" \
        --fold -1 --epochs "$ep" --lr "$lr" --wd "$wd" --warmup 1 --workers 5 --seed "$seed" \
        --out "$out" >> $lg 2>&1
    grep -q '^BEST' $lg && return 0
    rm -rf "$out"; sleep 60
  done
}

for i in $(seq 1 240); do
  grep -q 'V7 DONE' /root/_dsh/v7run.log 2>/dev/null && break
  sleep 60
done
while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### Base@448 开始 $(date)"

run_full full_cnxb_448_s42 convnext_base "$W/convnext_base.fb_in22k_ft_in1k.safetensors" 448 12 5 1.5e-4 1e-2 42
echo "SCORE full_cnxb_448_s42 $(grep -h '^BEST' /root/_dsh/full_cnxb_448_s42.log 2>/dev/null | tail -1)"

run_full full_cnxb_448_s123 convnext_base "$W/convnext_base.fb_in22k_ft_in1k.safetensors" 448 12 5 1.5e-4 1e-2 123
echo "SCORE full_cnxb_448_s123 $(grep -h '^BEST' /root/_dsh/full_cnxb_448_s123.log 2>/dev/null | tail -1)"
echo "CHAIN22 DONE $(date)"