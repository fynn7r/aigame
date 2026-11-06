#!/bin/bash
# 每折一跑完就立刻算 valid/test 概率，最后只剩集成一步
set -u
cd /root/_dsh || exit 1
PRE=p_b0a
for k in 0 1 2 3 4; do
  LOG=/root/_dsh/${PRE}_f$k.log
  D=/data/cxr_first/runs/${PRE}_f$k
  while ! grep -q BEST "$LOG" 2>/dev/null; do sleep 60; done
  [ -f "$D/probs_valid.npz" ] || python infer.py --ckpt "$D/best.pt" --split valid --tta 2 \
      --save_probs "$D/probs_valid.npz" > /root/_dsh/${PRE}_f${k}_iv.log 2>&1
  [ -f "$D/probs_test.npz" ] || python infer.py --ckpt "$D/best.pt" --split test --tta 2 \
      --save_probs "$D/probs_test.npz" > /root/_dsh/${PRE}_f${k}_it.log 2>&1
  echo "probs ready for fold $k $(date)" >> /root/_dsh/probs_ready.log
done
echo "ALL PROBS READY $(date)" >> /root/_dsh/probs_ready.log