#!/bin/bash
# v5 之后：1) 多尺度 TTA 验证  2) dn201@512 五折  3) 生成 v6
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

run_exp() {
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
    rm -rf "$out"; sleep 90
  done
}

for i in $(seq 1 900); do
  grep -q 'CHAIN18 DONE' /root/_dsh/chain18.log 2>/dev/null && break
  sleep 60
done
while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### chain18 结束 $(date)"

# ---- 1. 多尺度 TTA 对比（只跑 valid，快）----
echo "=== 多尺度 TTA 验证 $(date)"
cat > /root/_dsh/ms_tta.sh <<'EOS'
set -u
cd /root/_dsh
for t in p_b0_640_f2 ref_dn201_448_f2 ref_cnxs_448_f0; do
  ck=$(ls /data/cxr_first/runs/$t/best.pt 2>/dev/null) || continue
  echo "--- $t 缺省尺度"
  DSH_CACHE=1024 python infer.py --ckpt $ck --split valid --tta 2 2>&1 | grep -E 'STUDY|='
  echo "--- $t 多尺度 448,512,640"
  DSH_CACHE=1024 python infer.py --ckpt $ck --split valid --tta 2 --scales 448,512,640 2>&1 | grep -E 'STUDY|='
done
EOS
bash /root/_dsh/ms_tta.sh

# ---- 2. dn201@512 五折 ----
echo "=== dn201@512 6ep x5 $(date)"
for k in 0 1 2 3 4; do
  run_exp ref_dn201_512_f$k $k densenet201 "$W/densenet201.tv_in1k.safetensors" 512 16 6 2e-4 1e-2
done
echo "### dn201@512 完成 $(date)"
echo "CHAIN19 DONE $(date)"