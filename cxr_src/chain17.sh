#!/bin/bash
# 把「新架构」和「好分辨率」结合：dn201@448 + cnx-s@448
# 等 v4 推理/集成跑完再开始，避免抢显存
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

run_exp() {  # tag fold arch weights size bs ep lr wd
  local tag=$1 fold=$2 arch=$3 wf=$4 sz=$5 bs=$6 ep=$7 lr=$8 wd=$9
  local out=/data/cxr_first/runs/$tag
  grep -q BEST /root/_dsh/$tag.log 2>/dev/null && { echo "skip $tag"; return 0; }
  for att in 1 2 3; do
    python wait_vram.py 10.0 >> /root/_dsh/$tag.log 2>&1
    echo "--- $tag attempt $att $(date)" >> /root/_dsh/$tag.log
    python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" --fold "$fold" \
        --epochs "$ep" --lr "$lr" --wd "$wd" --warmup 1 --workers 5 --out "$out" \
        >> /root/_dsh/$tag.log 2>&1
    grep -q BEST /root/_dsh/$tag.log && { echo "SCORE $tag $(score_of $tag)"; return 0; }
    echo "!!! $tag attempt $att failed"
    rm -rf "$out"
    sleep 90
  done
  return 1
}

# 等 v4 结束
for i in $(seq 1 240); do
  [ -f /data/cxr_first/v4.csv ] && ! pgrep -f 'infer\.py' > /dev/null && ! pgrep -f auto_sub > /dev/null && break
  sleep 30
done
while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### v4 结束，开始 448 训练 $(date)"

export DSH_CACHE=1024   # 448 也从高分辨率缓存降采样，细节更好
echo "=== dn201@448 6ep x5 $(date)"
for k in 0 1 2 3 4; do
  run_exp ref_dn201_448_f$k $k densenet201 "$W/densenet201.tv_in1k.safetensors" 448 16 6 2e-4 1e-2
done
echo "### dn201@448 完成 $(date)"

echo "=== cnx-s@448 5ep x2 $(date)"
for k in 0 1; do
  run_exp ref_cnxs_448_f$k $k convnext_small "$W/convnext_small_fb_in22k_ft_in1k.safetensors" 448 16 5 1.5e-4 1e-2
done
echo "### cnx-s@448 完成 $(date)"
echo "CHAIN17 DONE $(date)"