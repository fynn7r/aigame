#!/bin/bash
# 端到端出提交：对若干 run 目录做 valid/test 推理 -> 集成 -> submission.csv
# 用法: bash make_sub.sh out_name "tag1 tag2 tag3 ..."
set -u
cd /root/_dsh || exit 1
NAME=${1:?out name}
TAGS=${2:?tags}
R=/data/cxr_first/runs
SP=""
for t in $TAGS; do
  D=$R/$t
  [ -f "$D/best.pt" ] || { echo "skip $t (no ckpt)"; continue; }
  if [ ! -f "$D/probs_valid.npz" ]; then
    echo "--- infer valid $t $(date)"
    python infer.py --ckpt "$D/best.pt" --split valid --tta 2 --save_probs "$D/probs_valid.npz" 2>&1 | grep -E 'STUDY|IMAGE'
  fi
  if [ ! -f "$D/probs_test.npz" ]; then
    echo "--- infer test $t $(date)"
    python infer.py --ckpt "$D/best.pt" --split test --tta 2 --save_probs "$D/probs_test.npz" 2>&1 | grep -E 'written|probs saved'
  fi
  SP="$SP $D/probs_valid.npz"
done

echo "=== valid ensemble $(date)"
python ensemble.py --probs $SP --split valid 2>&1 | grep -E 'agg=|best agg|per class|^  [a-z0-9_]+ '

ST=""
for t in $TAGS; do
  D=$R/$t
  [ -f "$D/probs_test.npz" ] && ST="$ST $D/probs_test.npz"
done
echo "=== test ensemble $(date)"
python ensemble.py --probs $ST --split test --out /data/cxr_first/${NAME}.csv 2>&1 | tail -2
python check_submission.py /data/cxr_first/${NAME}.csv 2>&1 | tail -3
echo "SUBMISSION READY /data/cxr_first/${NAME}.csv $(date)"