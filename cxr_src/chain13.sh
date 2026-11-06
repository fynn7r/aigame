#!/bin/bash
# 第三阶段：配方实验
#   X: b0@640 12ep，取消 pos_weight、加 mixup —— 隔离「损失函数」这一个变量
#   Y: dn121@1024（或 768）24ep，新配方 —— 测「容量 + 分辨率 + 长训练」
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

train_retry() {  # tag fold arch weights size bs ep lr [extra...]
  local tag=$1 fold=$2 arch=$3 wf=$4 sz=$5 bs=$6 ep=$7 lr=$8; shift 8
  local out=/data/cxr_first/runs/$tag
  grep -q BEST /root/_dsh/$tag.log 2>/dev/null && { echo "skip $tag (done)"; return 0; }
  for att in 1 2 3; do
    python wait_vram.py 10.0 >> /root/_dsh/$tag.log 2>&1
    echo "--- $tag attempt $att $(date)" >> /root/_dsh/$tag.log
    python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" --fold "$fold" \
        --epochs "$ep" --lr "$lr" --warmup 1 --workers 5 "$@" --out "$out" \
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

# 1) 等 r50@640 f0 跑完，然后停掉 chain12 后面的折
for i in $(seq 1 200); do
  grep -q BEST /root/_dsh/p_r50_640_f0.log 2>/dev/null && break
  sleep 60
done
echo "### r50@640 f0 完成 $(date) = $(score_of p_r50_640_f0)"
pkill -9 -f chain12.sh
pkill -9 -f 'runs/p_r50_640_f[1-4]'
sleep 5
for k in 1 2 3 4; do
  [ -f /root/_dsh/p_r50_640_f$k.log ] && ! grep -q BEST /root/_dsh/p_r50_640_f$k.log && rm -rf /data/cxr_first/runs/p_r50_640_f$k
done
while pgrep -f 'train\.py' > /dev/null; do sleep 20; done
echo "### GPU 空闲 $(date)"

# 2) 实验 X：b0@640，取消 pos_weight + mixup（其余和 640 f0 完全相同，f0 基线 = 0.8069）
echo "=== 实验X: b0@640 12ep pw=1 mixup0.2 $(date) ==="
train_retry x_b0_640_nopw 0 efficientnet_b0 "$W/efficientnet_b0.pth" 640 16 12 3e-4 \
    --pos_weight 1.0 --mixup 0.2
echo "X 结果 = $(score_of x_b0_640_nopw)  (基线 640 f0 = 0.8069)"

# 3) 实验 Y：dn121@1024 24ep 新配方
echo "=== 实验Y: dn121@1024 24ep $(date) ==="
train_retry x_dn121_1024 0 densenet121 "$W/densenet121_ra_in1k.safetensors" 1024 6 24 2e-4 \
    --pos_weight 1.0 --mixup 0.2 --wd 0.02 --drop 0.2
SY=$(score_of x_dn121_1024)
echo "Y(1024) 结果 = $SY"

# 1024 若失败/崩，退回 768
if python -c "import sys; sys.exit(0 if $SY > 0.5 else 1)"; then
  echo "dn121@1024 OK"
else
  echo "=== 退回 dn121@768 $(date) ==="
  train_retry x_dn121_768 0 densenet121 "$W/densenet121_ra_in1k.safetensors" 768 8 24 2e-4 \
      --pos_weight 1.0 --mixup 0.2 --wd 0.02 --drop 0.2
  echo "Y(768) 结果 = $(score_of x_dn121_768)"
fi
echo "CHAIN13 DONE $(date)"