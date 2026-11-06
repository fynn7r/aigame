#!/bin/bash
# 在最强族（ConvNeXt-Small@448 全量）上补种子，等 ConvNeXt-Base 权重到位后接着训 Base
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

while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### 开始 $(date)"

echo "=== cnx-s@448 全量 s123/s777 $(date)"
run_full full_cnxs_448_s123 convnext_small "$W/convnext_small_fb_in22k_ft_in1k.safetensors" 448 16 5 1.5e-4 1e-2 123
run_full full_cnxs_448_s777 convnext_small "$W/convnext_small_fb_in22k_ft_in1k.safetensors" 448 16 5 1.5e-4 1e-2 777

# 等 ConvNeXt-Base 权重上传完成（最多等 3 小时）
echo "=== 等待 cnx-base 权重 $(date)"
for i in $(seq 1 180); do
  [ -s "$W/convnext_base.fb_in22k_ft_in1k.safetensors" ] && break
  sleep 60
done

WF="$W/convnext_base.fb_in22k_ft_in1k.safetensors"
if [ -s "$WF" ]; then
  echo "=== cnx-base@384 全量 x2 $(date)"
  for s in 42 123; do
    run_full full_cnxb_384_s$s convnext_base "$WF" 384 16 5 1.5e-4 1e-2 $s
    echo "SCORE full_cnxb_384_s$s $(grep -h '^BEST' /root/_dsh/full_cnxb_384_s$s.log 2>/dev/null | tail -1)"
  done
else
  echo "!! cnx-base 权重未就绪"
fi
echo "CHAIN21 DONE $(date)"