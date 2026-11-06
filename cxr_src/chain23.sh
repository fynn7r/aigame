#!/bin/bash
# Base@448 跑完后：先出 v8，再补 Base@384 种子（384 是 Base 的甜点，448 反而略差）
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

for i in $(seq 1 300); do
  grep -q 'CHAIN22 DONE' /root/_dsh/chain22.log 2>/dev/null && break
  sleep 60
done
while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### chain22 结束，开始 v8 $(date)"

infer_tag() {
  local t=$1 c=$2
  local D=$OUT/runs/$t
  [ -f "$D/last.pt" ] || [ -f "$D/best.pt" ] || return
  local ck="$D/last.pt"; [ -f "$ck" ] || ck="$D/best.pt"
  [ -f "$D/probs_valid.npz" ] || DSH_CACHE=$c python infer.py --ckpt "$ck" --split valid --tta 2 \
      --save_probs "$D/probs_valid.npz" > /root/_dsh/${t}_iv.log 2>&1
  [ -f "$D/probs_test.npz" ] || DSH_CACHE=$c python infer.py --ckpt "$ck" --split test --tta 2 \
      --save_probs "$D/probs_test.npz" > /root/_dsh/${t}_it.log 2>&1
}

infer_tag full_cnxb_448_s42 1024
infer_tag full_cnxb_448_s123 1024

TAGS=""
for d in $(ls -d $OUT/runs/*/ 2>/dev/null); do
  t=$(basename $d)
  [ -f "$d/probs_valid.npz" ] && [ -f "$d/probs_test.npz" ] && TAGS="$TAGS $t"
done
echo "TAGS=$TAGS"
python auto_sub.py --tags "$TAGS" --name v8 > /root/_dsh/v8.log 2>&1
python check_submission.py $OUT/v8.csv >> /root/_dsh/v8.log 2>&1
grep -E 'step|WEIGHTS|ENSEMBLE|best single' /root/_dsh/v8.log | tail -14
echo "V8 DONE $(date)"

echo "=== Base@384 追加种子 $(date)"
for s in 777 2024; do
  run_full full_cnxb_384_s$s convnext_base "$W/convnext_base.fb_in22k_ft_in1k.safetensors" 384 16 5 1.5e-4 1e-2 $s
  echo "SCORE full_cnxb_384_s$s $(grep -h '^BEST' /root/_dsh/full_cnxb_384_s$s.log 2>/dev/null | tail -1)"
done
echo "CHAIN23 DONE $(date)"