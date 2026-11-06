#!/bin/bash
# 一次性把现有模型的 valid/test 概率算出来，并生成 v3 提交
# 注意：必须在没有训练任务占用显存时运行
set -u
cd /root/_dsh || exit 1
OUT=/data/cxr_first
TAGS=""
run_one() {  # tag cache
  local t=$1 c=$2
  local D=$OUT/runs/$t
  [ -f "$D/best.pt" ] || { echo "skip $t (no ckpt)"; return; }
  if [ ! -f "$D/probs_valid.npz" ]; then
    echo "--- valid $t (cache$c) $(date)"
    DSH_CACHE=$c python infer.py --ckpt "$D/best.pt" --split valid --tta 2 \
        --save_probs "$D/probs_valid.npz" > /root/_dsh/${t}_iv.log 2>&1
  fi
  if [ ! -f "$D/probs_test.npz" ]; then
    echo "--- test  $t (cache$c) $(date)"
    DSH_CACHE=$c python infer.py --ckpt "$D/best.pt" --split test --tta 2 \
        --save_probs "$D/probs_test.npz" > /root/_dsh/${t}_it.log 2>&1
  fi
  [ -f "$D/probs_valid.npz" ] && [ -f "$D/probs_test.npz" ] && TAGS="$TAGS $t"
}

for t in p_b0a_f0 p_b0a_f1 p_b0_448 p_b0jft_f0 p_b0jft_f2; do run_one $t 512; done
for t in p_b0_640_f0 p_b0_640_f1 p_b0_640_f2 p_b0_640_f3 p_b0_640_f4; do run_one $t 1024; done

echo "TAGS=$TAGS"
if [ -n "$TAGS" ]; then
  python auto_sub.py --tags "$TAGS" --name v3 > /root/_dsh/v3.log 2>&1
  python check_submission.py /data/cxr_first/v3.csv >> /root/_dsh/v3.log 2>&1
  md5sum /data/cxr_first/v3.csv >> /root/_dsh/v3.log
  tail -14 /root/_dsh/v3.log
  curl -s -m 300 -X POST --data-binary @/data/cxr_first/v3.csv \
       'http://10.0.79.98:8765/upload?name=v3.csv'
  curl -s -m 60 -X POST --data-binary @/root/_dsh/v3.log \
       'http://10.0.79.98:8765/upload?name=v3.log'
else
  echo "没有可用模型"
fi