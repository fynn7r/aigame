#!/bin/bash
# 全量数据训练（fold=-1，用全部 30344 张，比五折单模多 25% 数据）
# 用 last.pt 而不是 best.pt：fold=-1 时 best 是按 valid 选的，会有选择泄漏
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
OUT=/data/cxr_first
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

run_full() {  # tag arch weights size bs ep lr wd seed [cache]
  local tag=$1 arch=$2 wf=$3 sz=$4 bs=$5 ep=$6 lr=$7 wd=$8 seed=$9; local c=${10:-1024}
  local out=$OUT/runs/$tag
  local lg=/root/_dsh/$tag.log
  grep -q '^BEST' $lg 2>/dev/null && { echo "skip $tag"; return 0; }
  for att in 1 2; do
    python wait_vram.py 10.0 >> $lg 2>&1
    echo "--- $tag attempt $att $(date)" >> $lg
    DSH_CACHE=$c python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" \
        --fold -1 --epochs "$ep" --lr "$lr" --wd "$wd" --warmup 1 --workers 5 --seed "$seed" \
        --out "$out" >> $lg 2>&1
    grep -q '^BEST' $lg && return 0
    rm -rf "$out"; sleep 60
  done
}

infer_tag() {  # tag cache
  local t=$1 c=$2
  local D=$OUT/runs/$t
  [ -f "$D/last.pt" ] || { echo "!! $t 无 last.pt"; return; }
  [ -f "$D/probs_valid.npz" ] || DSH_CACHE=$c python infer.py --ckpt "$D/last.pt" \
      --split valid --tta 2 --save_probs "$D/probs_valid.npz" > /root/_dsh/${t}_iv.log 2>&1
  [ -f "$D/probs_test.npz" ] || DSH_CACHE=$c python infer.py --ckpt "$D/last.pt" \
      --split test --tta 2 --save_probs "$D/probs_test.npz" > /root/_dsh/${t}_it.log 2>&1
}

while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### 全量训练开始 $(date)"

echo "=== b0@640 全量 x3 seed $(date)"
for s in 42 123 777; do
  run_full full_b0_640_s$s efficientnet_b0 "$W/efficientnet_b0.pth" 640 16 6 3e-4 0.05 $s 1024
  echo "SCORE full_b0_640_s$s $(grep -h '^BEST' /root/_dsh/full_b0_640_s$s.log 2>/dev/null | tail -1)"
done

echo "=== dn201@448 全量 x2 seed $(date)"
for s in 42 123; do
  run_full full_dn201_448_s$s densenet201 "$W/densenet201.tv_in1k.safetensors" 448 16 6 2e-4 1e-2 $s 1024
  echo "SCORE full_dn201_448_s$s $(grep -h '^BEST' /root/_dsh/full_dn201_448_s$s.log 2>/dev/null | tail -1)"
done

echo "=== cnx-s@448 全量 x1 $(date)"
run_full full_cnxs_448_s42 convnext_small "$W/convnext_small_fb_in22k_ft_in1k.safetensors" 448 16 5 1.5e-4 1e-2 42 1024
echo "SCORE full_cnxs_448_s42 $(grep -h '^BEST' /root/_dsh/full_cnxs_448_s42.log 2>/dev/null | tail -1)"

echo "### 全量模型推理 $(date)"
TAGS=""
for t in full_b0_640_s42 full_b0_640_s123 full_b0_640_s777 full_dn201_448_s42 full_dn201_448_s123 full_cnxs_448_s42; do
  infer_tag $t 1024
  [ -f "$OUT/runs/$t/probs_valid.npz" ] && TAGS="$TAGS $t"
done

# 合并积分池
for t in ref_cnxs_384 ref_cnxs_448_f0 ref_cnxs_448_f1 ref_dn201_448_f0 ref_dn201_448_f1 \
         ref_dn201_448_f2 ref_dn201_448_f3 ref_dn201_448_f4 ref_dn201_384 ref_dn201_384_f1 \
         ref_dn201_384_f2 ref_dn201_384_f3 ref_dn201_384_f4 ref_dn121_384; do
  [ -f "$OUT/runs/$t/probs_valid.npz" ] && [ -f "$OUT/runs/$t/probs_test.npz" ] && TAGS="$TAGS $t"
done
for t in p_b0_640_f0 p_b0_640_f1 p_b0_640_f2 p_b0_640_f3 p_b0_640_f4 x_b0_640_nopw \
         p_r50_640_f0 p_r50_640_f1; do
  [ -f "$OUT/runs/$t/probs_valid.npz" ] && TAGS="$TAGS $t"
done
for t in p_b0_448 p_b0jft_f0 p_b0jft_f2 p_b0a_f0; do
  [ -f "$OUT/runs/$t/probs_valid.npz" ] && TAGS="$TAGS $t"
done

echo "TAGS=$TAGS"
python auto_sub.py --tags "$TAGS" --name v6 > /root/_dsh/v6.log 2>&1
python check_submission.py $OUT/v6.csv >> /root/_dsh/v6.log 2>&1
grep -E 'step|WEIGHTS|ENSEMBLE|best single' /root/_dsh/v6.log | tail -14
echo "CHAIN20 DONE $(date)"