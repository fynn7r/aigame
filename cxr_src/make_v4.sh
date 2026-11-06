#!/bin/bash
# 给新模型补算 valid/test 概率，然后用全部模型做贪心集成，生成 v4
set -u
cd /root/_dsh || exit 1
OUT=/data/cxr_first
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
TAGS=""

run_one() {  # tag cache
  local t=$1 c=$2
  local D=$OUT/runs/$t
  [ -f "$D/best.pt" ] || { echo "skip $t (no ckpt)"; return; }
  if [ ! -f "$D/probs_valid.npz" ]; then
    echo "--- valid $t $(date)"
    DSH_CACHE=$c python infer.py --ckpt "$D/best.pt" --split valid --tta 2 \
        --save_probs "$D/probs_valid.npz" > /root/_dsh/${t}_iv.log 2>&1
  fi
  if [ ! -f "$D/probs_test.npz" ]; then
    echo "--- test  $t $(date)"
    DSH_CACHE=$c python infer.py --ckpt "$D/best.pt" --split test --tta 2 \
        --save_probs "$D/probs_test.npz" > /root/_dsh/${t}_it.log 2>&1
  fi
  if [ -f "$D/probs_valid.npz" ] && [ -f "$D/probs_test.npz" ]; then
    TAGS="$TAGS $t"
  else
    echo "!! $t 推理失败"
  fi
}

# 新模型（384 分辨率走 512 缓存）
for t in ref_dn201_384 ref_dn201_384_f1 ref_dn201_384_f2 ref_dn201_384_f3 ref_dn201_384_f4 \
         ref_cnxs_384 ref_dn121_384; do
  run_one $t 512
done
# 已有的 640 / 448 模型
for t in p_b0_640_f0 p_b0_640_f1 p_b0_640_f2 p_b0_640_f3 p_b0_640_f4; do run_one $t 1024; done
for t in p_b0_448 p_b0jft_f0 p_b0jft_f2 p_b0a_f0 x_b0_640_nopw; do
  case $t in
    x_b0_640_nopw) run_one $t 1024 ;;
    *) run_one $t 512 ;;
  esac
done
for t in p_r50_640_f0 p_r50_640_f1; do run_one $t 1024; done

echo "TAGS=$TAGS"
echo "$TAGS" > /root/_dsh/last_tags.txt
python auto_sub.py --tags "$TAGS" --name v4 > /root/_dsh/v4.log 2>&1
python check_submission.py /data/cxr_first/v4.csv >> /root/_dsh/v4.log 2>&1
tail -20 /root/_dsh/v4.log