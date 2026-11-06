#!/bin/bash
# v10：把 CLAHE + ASL 两个新变体并进池子，重新贪心
set -u
cd /root/_dsh || exit 1
OUT=/data/cxr_first
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

infer_tag() {
  local t=$1 c=$2
  local D=$OUT/runs/$t
  [ -f "$D/last.pt" ] || [ -f "$D/best.pt" ] || { echo "!! $t 无权重"; return; }
  local ck="$D/last.pt"; [ -f "$ck" ] || ck="$D/best.pt"
  [ -f "$D/probs_valid.npz" ] || DSH_CACHE=$c python infer.py --ckpt "$ck" --split valid \
      --tta 2 --save_probs "$D/probs_valid.npz" > /root/_dsh/${t}_iv.log 2>&1
  [ -f "$D/probs_test.npz" ] || DSH_CACHE=$c python infer.py --ckpt "$ck" --split test \
      --tta 2 --save_probs "$D/probs_test.npz" > /root/_dsh/${t}_it.log 2>&1
  [ -f "$D/probs_valid.npz" ] && [ -f "$D/probs_test.npz" ] && echo "OK $t" || echo "!! $t 推理失败"
}

while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### v10 开始 $(date)"

echo "=== 新模型推理 $(date)"
infer_tag full_cnxb_384_clahe_s42 1024
infer_tag full_cnxb_384_asl_s42 1024
infer_tag full_cnxb_384_asl_s123 1024
infer_tag full_cnxb_384_s31337 1024

TAGS=""
for d in $(ls -d $OUT/runs/*/ 2>/dev/null); do
  t=$(basename $d)
  case "$t" in r34ssl*) continue;; esac          # 合规：SSL 分支动过验证集，永久排除
  case "$t" in *jft*) continue;; esac            # 合规：JFT 预训练来源不可核验，永久排除
  [ -f "$d/probs_valid.npz" ] && [ -f "$d/probs_test.npz" ] && TAGS="$TAGS $t"
done
echo "TAGS=$TAGS"

python auto_sub.py --tags "$TAGS" --name v10 > /root/_dsh/v10.log 2>&1
python check_submission.py $OUT/v10.csv >> /root/_dsh/v10.log 2>&1
grep -E 'step|WEIGHTS|ENSEMBLE|best single' /root/_dsh/v10.log | tail -16
grep -E 'shape|study 覆盖|NaN' /root/_dsh/v10.log | head -4
echo "V10 DONE $(date)"