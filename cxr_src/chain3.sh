#!/bin/bash
# b0 扫完后:
#   A) JFT 预训练的 b0 @448 18ep fold0  —— 比较预训练来源
#   B) b0 @512 18ep fold0               —— 比较分辨率
#   然后 5 折 × 胜者配置 20ep
set -u
cd /root/_dsh || exit 1
EPF=20

while pgrep -f sweep_pretrained.sh >/dev/null; do sleep 30; done
echo "sweep finished $(date)"

score() { python -c "
import torch
try:
    c = torch.load('/data/cxr_first/runs/$1/best.pt', map_location='cpu')
    print('SCORE $1 %.4f' % c['score'])
except Exception:
    print('SCORE $1 0.0')
"; }

echo "=== A: tf_effnet_b0 JFT @448 $(date) ==="
python train.py --arch tf_efficientnet_b0.ns_jft_in1k \
    --weights /root/_dsh/weights/tf_efficientnet_b0_ns_jft_in1k.safetensors \
    --size 448 --fold 0 --epochs 18 --bs 28 --lr 4e-4 --warmup 1 --workers 6 \
    --out /data/cxr_first/runs/p_b0jft_448 > /root/_dsh/p_b0jft_448.log 2>&1
score p_b0jft_448

echo "=== B: b0 @512 $(date) ==="
python train.py --arch efficientnet_b0 --weights /root/_dsh/weights/efficientnet_b0.pth \
    --size 512 --fold 0 --epochs 18 --bs 22 --lr 4e-4 --warmup 1 --workers 6 \
    --out /data/cxr_first/runs/p_b0_512 > /root/_dsh/p_b0_512.log 2>&1
score p_b0_512

python - <<'EOF' | tee /tmp/pick.txt
import torch
cands = {
    "b0in1k_448": ("efficientnet_b0", "/root/_dsh/weights/efficientnet_b0.pth", 448, 28),
    "b0jft_448":  ("tf_efficientnet_b0.ns_jft_in1k",
                   "/root/_dsh/weights/tf_efficientnet_b0_ns_jft_in1k.safetensors", 448, 28),
    "b0in1k_512": ("efficientnet_b0", "/root/_dsh/weights/efficientnet_b0.pth", 512, 22),
}
best = None
for tag, cfg in cands.items():
    path = {"b0in1k_448": "/data/cxr_first/runs/p_b0_448/best.pt"}.get(tag) \
           or f"/data/cxr_first/runs/p_{tag}/best.pt"
    try:
        c = torch.load(path, map_location="cpu")
        print(tag, round(c["score"], 4))
        if best is None or c["score"] > best[1]:
            best = (tag, c["score"], cfg)
    except Exception as e:
        print(tag, "missing")
print("WINNER", best[0], round(best[1], 4))
EOF
WIN=$(grep WINNER /tmp/pick.txt | awk '{print $2}')
case "$WIN" in
  b0in1k_448) ARCH=efficientnet_b0; WF=/root/_dsh/weights/efficientnet_b0.pth; SZ=448; BS=28; PRE=p_b0 ;;
  b0jft_448)  ARCH=tf_efficientnet_b0.ns_jft_in1k; WF=/root/_dsh/weights/tf_efficientnet_b0_ns_jft_in1k.safetensors; SZ=448; BS=28; PRE=p_b0jft ;;
  *)          ARCH=efficientnet_b0; WF=/root/_dsh/weights/efficientnet_b0.pth; SZ=512; BS=22; PRE=p_b0_512 ;;
esac
echo "folds 0-4 -> arch=$ARCH size=$SZ bs=$BS ep=$EPF tag=$PRE $(date)"

for k in 0 1 2 3 4; do
  OUT=/data/cxr_first/runs/${PRE}_f$k
  [ -f "$OUT/best.pt" ] && { echo "skip $OUT"; continue; }
  python train.py --arch "$ARCH" --weights "$WF" --size "$SZ" --bs "$BS" --fold $k \
      --epochs $EPF --lr 4e-4 --warmup 1 --workers 6 --out "$OUT" \
      > /root/_dsh/${PRE}_f$k.log 2>&1
  score ${PRE}_f$k
done
echo "ALL FOLDS DONE $(date)"