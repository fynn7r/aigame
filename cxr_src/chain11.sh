#!/bin/bash
# 稳健版：修好 640 f0，补 f1-f4（每折最多重试 3 次、跑前等显存），再出 v3，再做 768 探针
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

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

# ---- 1) 修复 640 f0（之前 cp 早于保存完成，拿到的是 ep0 的旧权重）----
GOOD=$(score_of p_b0_640)
echo "### p_b0_640 真实成绩 = $GOOD $(date)"
rm -rf /data/cxr_first/runs/p_b0_640_f0
cp -r /data/cxr_first/runs/p_b0_640 /data/cxr_first/runs/p_b0_640_f0
{ echo "ep11 (copied from p_b0_640)"; echo "BEST $GOOD"; } > /root/_dsh/p_b0_640_f0.log
echo "登记 f0 = $(score_of p_b0_640_f0)"

# ---- 2) 补 640 f1-f4 ----
export DSH_CACHE=1024
for k in 1 2 3 4; do
  train_retry p_b0_640_f$k $k efficientnet_b0 "$W/efficientnet_b0.pth" 640 16 12 3e-4
done
echo "### 640 五折完成 $(date)"
for k in 0 1 2 3 4; do echo "640 f$k = $(score_of p_b0_640_f$k)"; done

# ---- 3) 出 v3 ----
echo "### 出 v3 $(date)"
bash /root/_dsh/night_sub.sh

# ---- 4) 768 探针 ----
echo "### 768 探针 $(date)"
train_retry p_b0_768 0 efficientnet_b0 "$W/efficientnet_b0.pth" 768 12 12 3e-4
S768=$(score_of p_b0_768)
B640=$(score_of p_b0_640_f0)
echo "p_b0_768 = $S768 ; 640 f0 = $B640"

if python -c "import sys; sys.exit(0 if $S768 > $B640 else 1)"; then
  echo ">>> 768 胜出"
  for k in 1 2 3 4; do
    train_retry p_b0_768_f$k $k efficientnet_b0 "$W/efficientnet_b0.pth" 768 12 12 3e-4
  done
else
  echo ">>> 640 保持最优，补 r50@640"
  for k in 0 1 2 3 4; do
    train_retry p_r50_640_f$k $k resnet50 "$W/resnet50.pth" 640 14 12 3e-4
  done
fi
unset DSH_CACHE
echo "CHAIN11 DONE $(date)"