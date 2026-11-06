#!/bin/bash
# 基线跑完后自动接：SimCLR 预训练 -> 用 SSL 权重微调 fold0
set -u
cd /root/_dsh || exit 1
echo "waiting for baseline to finish $(date)"
while pgrep -f "train.py --arch resnet34 --size 384 --fold 0" >/dev/null; do sleep 60; done
echo "baseline done, start SSL $(date)"
mkdir -p /data/cxr_first/ssl

python ssl_pretrain.py --arch resnet34 --size 224 --bs 200 --epochs 80 \
    --lr 1e-3 --workers 6 --out /data/cxr_first/ssl/ssl_r34.pt > /root/_dsh/ssl.log 2>&1
echo "ssl done $(date)"
tail -3 /root/_dsh/ssl.log

INIT=$(ls -1 /data/cxr_first/ssl/ssl_r34_ep79.pt 2>/dev/null || ls -1 /data/cxr_first/ssl/ssl_r34.pt)
echo "init from $INIT"
python train.py --arch resnet34 --init "$INIT" --size 384 --fold 0 --epochs 40 \
    --bs 32 --lr 3e-4 --workers 6 --out /data/cxr_first/runs/r34ssl_384_f0 > /root/_dsh/r34ssl_f0.log 2>&1
echo "finetune done $(date)"
grep -E '^ep[0-9]|BEST' /root/_dsh/r34ssl_f0.log | tail -5