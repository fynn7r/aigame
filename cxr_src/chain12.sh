#!/bin/bash
# 第二阶段：放弃 768 五折（只有 +0.0014，ROI 太差），改为补「不同架构」的多样性
#   r50@640 x5  →  densenet121@640 x3  →  b3@640 x3
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export DSH_CACHE=1024

score_of() { python - <<EOF
import torch
try:
    c = torch.load('/data/cxr_first/runs/$1/best.pt', map_location='cpu', weights_only=False)
    print('%.4f' % c['score'])
except Exception:
    print('0.0')
EOF
}

train_retry() {  # tag fold arch weights size bs ep lr
  local tag=$1 fold=$2 arch=$3 wf=$4 sz=$5 bs=$6 ep=$7 lr=$8
  local out=/data/cxr_first/runs/$tag
  grep -q BEST /root/_dsh/$tag.log 2>/dev/null && { echo "skip $tag (done)"; return 0; }
  for att in 1 2 3; do
    python wait_vram.py 10.0 >> /root/_dsh/$tag.log 2>&1
    echo "--- $tag attempt $att $(date)" >> /root/_dsh/$tag.log
    python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" --fold "$fold" \
        --epochs "$ep" --lr "$lr" --warmup 1 --workers 5 --out "$out" \
        >> /root/_dsh/$tag.log 2>&1
    if grep -q BEST /root/_dsh/$tag.log; then
      echo "SCORE $tag $(score_of $tag)"
      return 0
    fi
    echo "!!! $tag attempt $att failed, retry in 90s"
    rm -rf "$out"
    sleep 90
  done
  echo "!!! $tag 三次失败"
  return 1
}

pkill -9 -f chain11.sh
pkill -9 -f 'runs/p_b0_768_f'
sleep 5
rm -rf /data/cxr_first/runs/p_b0_768_f1 /data/cxr_first/runs/p_b0_768_f2 \
       /data/cxr_first/runs/p_b0_768_f3 /data/cxr_first/runs/p_b0_768_f4
while pgrep -f 'train\.py' > /dev/null; do sleep 20; done
echo "### 第二阶段开始 $(date)"

# r50@640：这张 DCU 上 r50 的算力利用率是 b0 的 8 倍，容量也大
for k in 0 1 2 3 4; do
  train_retry p_r50_640_f$k $k resnet50 "$W/resnet50_a1_in1k.safetensors" 640 14 12 3e-4
done
echo "### r50@640 完成 $(date)"

# densenet121@640：稠密连接，和 b0/r50 差异最大
for k in 0 1 2; do
  train_retry p_dn121_640_f$k $k densenet121 "$W/densenet121_ra_in1k.safetensors" 640 14 12 3e-4
done
echo "### dn121@640 完成 $(date)"

# b3@640：更大容量
for k in 0 1 2; do
  train_retry p_b3_640_f$k $k efficientnet_b3 "$W/efficientnet_b3_ra2_in1k.safetensors" 640 12 12 3e-4
done
echo "CHAIN12 DONE $(date)"