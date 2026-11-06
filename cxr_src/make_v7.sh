#!/bin/bash
# 给新模型补推理 → 生成 v7
set -u
cd /root/_dsh || exit 1
OUT=/data/cxr_first
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### v7 开始 $(date)"

infer_tag() {
  local t=$1 c=$2
  local D=$OUT/runs/$t
  [ -f "$D/last.pt" ] || [ -f "$D/best.pt" ] || { echo "!! $t 无权重"; return; }
  local ck="$D/last.pt"; [ -f "$ck" ] || ck="$D/best.pt"
  [ -f "$D/probs_valid.npz" ] || DSH_CACHE=$c python infer.py --ckpt "$ck" \
      --split valid --tta 2 --save_probs "$D/probs_valid.npz" > /root/_dsh/${t}_iv.log 2>&1
  [ -f "$D/probs_test.npz" ] || DSH_CACHE=$c python infer.py --ckpt "$ck" \
      --split test --tta 2 --save_probs "$D/probs_test.npz" > /root/_dsh/${t}_it.log 2>&1
  if [ -f "$D/probs_valid.npz" ] && [ -f "$D/probs_test.npz" ]; then
    local v=$(DSH_CACHE=$c python infer.py --ckpt "$ck" --split valid --tta 2 2>&1 | grep 'STUDY' | head -1)
    echo "OK $t  $v"
  else
    echo "!! $t 推理失败"
  fi
}

echo "=== ConvNeXt-Base 推理 $(date)"
infer_tag full_cnxb_384_s42 1024
infer_tag full_cnxb_384_s123 1024
infer_tag full_cnxs_448_s123 1024

TAGS=""
for d in $(ls -d $OUT/runs/*/ 2>/dev/null); do
  t=$(basename $d)
  [ -f "$d/probs_valid.npz" ] && [ -f "$d/probs_test.npz" ] && TAGS="$TAGS $t"
done
echo "TAGS=$TAGS"
python auto_sub.py --tags "$TAGS" --name v7 > /root/_dsh/v7.log 2>&1
python check_submission.py $OUT/v7.csv >> /root/_dsh/v7.log 2>&1
grep -E 'step|WEIGHTS|ENSEMBLE|best single' /root/_dsh/v7.log | tail -14
echo "V7 DONE $(date)"