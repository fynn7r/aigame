#!/bin/bash
# 等 b0 扫完 -> 448 vs 512 同轮数对照(都是 fold0) -> 用胜者配置补齐 fold 1-4
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights/resnet50.pth
EP=30

while pgrep -f sweep_pretrained.sh >/dev/null; do sleep 30; done
echo "sweep finished $(date)"

run() {  # tag fold size bs epochs lr pwcap
  local tag=$1 fold=$2 size=$3 bs=$4 ep=$5 lr=$6 pw=$7
  python train.py --arch resnet50 --weights "$W" --size "$size" --fold "$fold" \
      --epochs "$ep" --bs "$bs" --lr "$lr" --warmup 1 --pw_cap "$pw" --workers 6 \
      --out /data/cxr_first/runs/$tag > /root/_dsh/$tag.log 2>&1
  python -c "
import torch
try:
    c = torch.load('/data/cxr_first/runs/$tag/best.pt', map_location='cpu')
    print('SCORE $tag %.4f' % c['score'])
except Exception as e:
    print('SCORE $tag 0.0')
"
}

echo "=== A: 512px ${EP}ep $(date) ==="
run p_r50_512 0 512 20 $EP 4e-4 20
echo "=== B: 448px ${EP}ep $(date) ==="
run p_r50_448b 0 448 24 $EP 4e-4 20

python - <<'EOF' > /tmp/win.txt
import torch
best = None
for tag in ["p_r50_512", "p_r50_448b"]:
    try:
        c = torch.load(f"/data/cxr_first/runs/{tag}/best.pt", map_location="cpu")
        print(tag, round(c["score"], 4), flush=True)
        if best is None or c["score"] > best[1]:
            best = (tag, c["score"])
    except Exception as e:
        print(tag, "missing", e)
print("WINNER", best[0] if best else "p_r50_448b")
EOF
cat /tmp/win.txt
WIN=$(grep WINNER /tmp/win.txt | awk '{print $2}')
echo "winner=$WIN $(date)"

if [ "$WIN" = "p_r50_512" ]; then SIZE=512; BS=20; else SIZE=448; BS=24; fi
PRE=p_r50_${SIZE}
echo "folds 1-4 with size=$SIZE bs=$BS ep=$EP $(date)"
for k in 1 2 3 4; do
  run ${PRE}_f$k $k $SIZE $BS $EP 4e-4 20
done
echo "ALL FOLDS DONE $(date)"