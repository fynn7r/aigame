#!/bin/bash
# 参考配方第二步：等 chain15 跑完 → 训练 DenseNet-201@384 → 融合评估 → 铺 dn201 五折
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
export PYTORCH_HIP_ALLOC_PREFIX=x
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export DSH_CACHE=512

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

# 等第一步结束
while pgrep -f chain15.sh > /dev/null; do sleep 60; done
while pgrep -f 'train\.py' > /dev/null; do sleep 30; done
echo "### chain15 结束，开始 dn201 $(date)"
echo "cnx-s@384 = $(score_of ref_cnxs_384)"
echo "dn121@384 = $(score_of ref_dn121_384)"

WF="$W/densenet201.tv_in1k.safetensors"
if [ ! -s "$WF" ]; then echo "!!! dn201 权重缺失，只评估已有模型"; else
  echo "=== dn201@384 f0 $(date)"
  run_exp ref_dn201_384 0 densenet201 "$WF" 384 16 5 2e-4 1e-2
  echo "dn201@384 f0 = $(score_of ref_dn201_384)"
fi

echo "=== 融合评估 $(date)"
python - <<'EOF'
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/root/_dsh")
from common import macro_auc

R = "/data/cxr_first/runs"
meta = pd.read_csv("/data/cxr_first/proc/meta.csv", low_memory=False)
sy = meta.groupby("Study_id")[[f"c{i}" for i in range(1, 11)]].first()


def probs_of(tag):
    p = os.path.join(R, tag, "probs_valid.npz")
    if not os.path.exists(p):
        return None, None
    z = np.load(p, allow_pickle=True)
    df = pd.DataFrame(z["probs"].astype(np.float32), index=z["study"].astype(str))
    return df.groupby(level=0).mean(), None


def score(mat, idx):
    y = sy.loc[idx].values.astype(np.float32)
    a, _ = macro_auc(y, mat)
    return a


tags = ["ref_cnxs_384", "ref_dn201_384", "ref_dn121_384", "p_b0_640_f2", "p_b0_640_f0",
        "p_b0_640_f1", "p_b0jft_f0"]
data = {}
for t in tags:
    d, _ = probs_of(t)
    if d is not None:
        data[t] = d
        print("%-16s %.4f" % (t, score(d.values, d.index)))

def blend(names, ws):
    idx = data[names[0]].index
    for n in names[1:]:
        idx = idx.intersection(data[n].index)
    m = sum(w * data[n].loc[idx].values for n, w in zip(names, ws))
    return score(m, idx), idx

if "ref_cnxs_384" in data and "ref_dn201_384" in data:
    for w in (0.55, 0.6, 0.5, 0.7):
        a, _ = blend(["ref_dn201_384", "ref_cnxs_384"], [w, 1 - w])
        print("dn201 %.2f + cnx-s %.2f = %.4f" % (w, 1 - w, a))

if "ref_cnxs_384" in data and "ref_dn201_384" in data and "p_b0_640_f2" in data:
    a, _ = blend(["ref_dn201_384", "ref_cnxs_384", "p_b0_640_f2", "p_b0_640_f0"],
                 [0.3, 0.25, 0.225, 0.225])
    print("三路融合(参考配方+我们640) = %.4f" % a)
EOF

# 铺 dn201 其余四折
if [ -s "$WF" ] && grep -q BEST /root/_dsh/ref_dn201_384.log 2>/dev/null; then
  for k in 1 2 3 4; do
    run_exp ref_dn201_384_f$k $k densenet201 "$WF" 384 16 5 2e-4 1e-2
  done
  echo "### dn201 五折完成 $(date)"
fi
echo "CHAIN16 DONE $(date)"