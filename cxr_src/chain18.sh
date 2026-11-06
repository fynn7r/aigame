#!/bin/bash
# chain17（448 训练）结束后：给新模型补推理，生成 v5
set -u
cd /root/_dsh || exit 1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

for i in $(seq 1 900); do
  grep -q 'CHAIN17 DONE' /root/_dsh/chain17.log 2>/dev/null && break
  sleep 60
done
while pgrep -f 'train\.py|infer\.py' > /dev/null; do sleep 30; done
echo "### chain17 结束，开始 v5 $(date)"
grep -E '^SCORE' /root/_dsh/chain17.log | tail -10

TAGS=""
add() {
  local t=$1 c=$2
  local D=/data/cxr_first/runs/$t
  [ -f "$D/best.pt" ] || { echo "skip $t"; return; }
  if [ ! -f "$D/probs_valid.npz" ]; then
    DSH_CACHE=$c python infer.py --ckpt "$D/best.pt" --split valid --tta 2 \
        --save_probs "$D/probs_valid.npz" > /root/_dsh/${t}_iv.log 2>&1
  fi
  if [ ! -f "$D/probs_test.npz" ]; then
    DSH_CACHE=$c python infer.py --ckpt "$D/best.pt" --split test --tta 2 \
        --save_probs "$D/probs_test.npz" > /root/_dsh/${t}_it.log 2>&1
  fi
  [ -f "$D/probs_valid.npz" ] && [ -f "$D/probs_test.npz" ] && TAGS="$TAGS $t"
}

for k in 0 1 2 3 4; do add ref_dn201_448_f$k 1024; done
for k in 0 1; do add ref_cnxs_448_f$k 1024; done

# 复用之前已有的全部模型
for t in ref_dn201_384 ref_dn201_384_f1 ref_dn201_384_f2 ref_dn201_384_f3 ref_dn201_384_f4 \
         ref_cnxs_384 ref_dn121_384; do add $t 512; done
for t in p_b0_640_f0 p_b0_640_f1 p_b0_640_f2 p_b0_640_f3 p_b0_640_f4 x_b0_640_nopw \
         p_r50_640_f0 p_r50_640_f1; do add $t 1024; done
for t in p_b0_448 p_b0jft_f0 p_b0jft_f2 p_b0a_f0; do add $t 512; done

echo "TAGS=$TAGS"
python auto_sub.py --tags "$TAGS" --name v5 > /root/_dsh/v5.log 2>&1
python check_submission.py /data/cxr_first/v5.csv >> /root/_dsh/v5.log 2>&1
head -40 /root/_dsh/v5.log | tail -32
echo "CHAIN18 DONE $(date)"