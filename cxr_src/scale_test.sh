cd /root/_dsh || exit 1
CK=${1:-/data/cxr_first/runs/p_b0_448/best.pt}
echo "---- cache512 ----"
for sc in 448 512 448,512; do
  echo "== c512 scales $sc"
  DSH_CACHE=512 python infer.py --ckpt "$CK" --split valid --tta 2 --scales "$sc" 2>&1 | grep -E 'STUDY|Error|error' | head -2
done
echo "---- cache1024 ----"
for sc in 448 512 640 448,512 448,640 448,512,640; do
  echo "== c1024 scales $sc"
  DSH_CACHE=1024 python infer.py --ckpt "$CK" --split valid --tta 2 --scales "$sc" 2>&1 | grep -E 'STUDY|Error|error' | head -2
done