#!/bin/bash
# Transformer 线已证伪（Swin 0.7850 / ViT OOM），转回已验证的 ConvNeXt-Base@384
# 换损失函数 : ASL 针对多标签长尾，重点拉稀有病种（我们最弱的类 1/4/8）
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
OUT=/data/cxr_first
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

run_full() {
  local tag=$1 seed=$2 extra=${3:-}
  local out=$OUT/runs/$tag lg=/root/_dsh/$tag.log
  grep -q '^BEST' $lg 2>/dev/null && { echo "skip $tag"; return 0; }
  for att in 1 2; do
    python wait_vram.py 10.0 >> $lg 2>&1
    echo "--- $tag attempt $att $(date)" >> $lg
    DSH_CACHE=1024 python train.py --arch convnext_base \
        --weights "$W/convnext_base.fb_in22k_ft_in1k.safetensors" \
        --size 384 --bs 16 --fold -1 --epochs 5 --lr 1.5e-4 --wd 1e-2 \
        --warmup 1 --workers 5 --seed "$seed" $extra --out "$out" >> $lg 2>&1
    grep -q '^BEST' $lg && return 0
    rm -rf "$out"; sleep 60
  done
}

while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### ASL 批次开始 $(date)"

run_full full_cnxb_384_asl_s42 42 "--loss asl --asl_gn 4.0 --asl_clip 0.05"
echo "SCORE asl_s42 $(grep -h '^BEST' /root/_dsh/full_cnxb_384_asl_s42.log 2>/dev/null | tail -1)"

run_full full_cnxb_384_asl_s123 123 "--loss asl --asl_gn 4.0 --asl_clip 0.05"
echo "SCORE asl_s123 $(grep -h '^BEST' /root/_dsh/full_cnxb_384_asl_s123.log 2>/dev/null | tail -1)"
echo "CHAIN27 DONE $(date)"