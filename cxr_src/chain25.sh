#!/bin/bash
# 10 轮长训反而更差（0.8061 < 5 轮的 0.8097），确认短轮数最优。
# 改走「多样性」：CLAHE 输入 + 新种子，都是 5 轮 Base@384 的已验证配置。
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
OUT=/data/cxr_first
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

run_full() {
  local tag=$1 sz=$2 bs=$3 ep=$4 lr=$5 wd=$6 seed=$7 extra=${8:-}
  local out=$OUT/runs/$tag lg=/root/_dsh/$tag.log
  grep -q '^BEST' $lg 2>/dev/null && { echo "skip $tag"; return 0; }
  for att in 1 2; do
    python wait_vram.py 10.0 >> $lg 2>&1
    echo "--- $tag attempt $att $(date)" >> $lg
    DSH_CACHE=1024 python train.py --arch convnext_base \
        --weights "$W/convnext_base.fb_in22k_ft_in1k.safetensors" \
        --size "$sz" --bs "$bs" --fold -1 --epochs "$ep" --lr "$lr" --wd "$wd" \
        --warmup 1 --workers 5 --seed "$seed" $extra --out "$out" >> $lg 2>&1
    grep -q '^BEST' $lg && return 0
    rm -rf "$out"; sleep 60
  done
}

# 干掉还在跑的 10 轮 s123（已知结论：长训更差，没必要再花 8 小时）
pkill -9 -f chain24.sh
pkill -9 -f 'full_cnxb_384_L10_s123'
sleep 5
while pgrep -f 'train\.py' > /dev/null; do sleep 20; done
echo "### 多样性批次开始 $(date)"

echo "=== Base@384 + CLAHE $(date)"
run_full full_cnxb_384_clahe_s42 384 16 5 1.5e-4 1e-2 42 "--clahe"
echo "SCORE clahe $(grep -h '^BEST' /root/_dsh/full_cnxb_384_clahe_s42.log 2>/dev/null | tail -1)"

echo "=== Base@384 新种子 $(date)"
run_full full_cnxb_384_s31337 384 16 5 1.5e-4 1e-2 31337
echo "SCORE s31337 $(grep -h '^BEST' /root/_dsh/full_cnxb_384_s31337.log 2>/dev/null | tail -1)"
echo "CHAIN25 DONE $(date)"