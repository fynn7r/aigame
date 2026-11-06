#!/bin/bash
# 复现参考配方：ConvNeXt-Small@384 + DenseNet@384，5 轮、wd 1e-2、bs16
# 参考方案线上 79.63（我们当前 76.70）。先跑 fold0 验证，再决定是否铺开五折。
set -u
cd /root/_dsh || exit 1
W=/root/_dsh/weights
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

run_exp() {  # tag fold arch weights size bs ep lr wd [extra...]
  local tag=$1 fold=$2 arch=$3 wf=$4 sz=$5 bs=$6 ep=$7 lr=$8 wd=$9; shift 9
  local out=/data/cxr_first/runs/$tag
  grep -q BEST /root/_dsh/$tag.log 2>/dev/null && { echo "skip $tag"; return 0; }
  for att in 1 2 3; do
    python wait_vram.py 10.0 >> /root/_dsh/$tag.log 2>&1
    echo "--- $tag attempt $att $(date)" >> /root/_dsh/$tag.log
    python train.py --arch "$arch" --weights "$wf" --size "$sz" --bs "$bs" --fold "$fold" \
        --epochs "$ep" --lr "$lr" --wd "$wd" --warmup 1 --workers 5 "$@" --out "$out" \
        >> /root/_dsh/$tag.log 2>&1
    grep -q BEST /root/_dsh/$tag.log && { echo "SCORE $tag $(score_of $tag)"; return 0; }
    echo "!!! $tag attempt $att failed"
    rm -rf "$out"
    sleep 90
  done
  return 1
}

pkill -9 -f chain14.sh
pkill -9 -f 'runs/x_b0_640_clahe'
sleep 5
rm -rf /data/cxr_first/runs/x_b0_640_clahe
while pgrep -f 'train\.py' > /dev/null; do sleep 20; done
echo "### 参考配方复现开始 $(date)"

# 参考方案 Model 2：ConvNeXt-Small @384（容器内已有 22k→1k 微调权重，比官方 1k 更强）
echo "=== cnx-s@384 5ep $(date)"
run_exp ref_cnxs_384 0 convnext_small "$W/convnext_small_fb_in22k_ft_in1k.safetensors" \
    384 16 5 1.5e-4 1e-2
echo "cnx-s = $(score_of ref_cnxs_384)"

# 参考方案 Model 1：DenseNet-201 权重不在容器里，先用 DenseNet-121 占位
echo "=== dn121@384 5ep $(date)"
run_exp ref_dn121_384 0 densenet121 "$W/densenet121_ra_in1k.safetensors" \
    384 16 5 2e-4 1e-2
echo "dn121 = $(score_of ref_dn121_384)"

# 55/45 融合与贪心融合对比
echo "=== 融合评估 $(date)"
python - <<'EOF'
import numpy as np, os
from sklearn.metrics import roc_auc_score
R = "/data/cxr_first/runs"
import sys
sys.path.insert(0, "/root/_dsh")
import pandas as pd
from common import macro_auc

meta = pd.read_csv("/data/cxr_first/proc/meta.csv", low_memory=False)
sids = meta["Study_id"].values

def study_auc(tag):
    z = np.load(os.path.join(R, tag, "probs_valid.npz"), allow_pickle=True)
    p = z["probs"].astype(np.float32)
    st = z["study"].astype(str)
    df = pd.DataFrame(p, index=st).groupby(level=0).mean()
    order = df.index.values
    y = meta.groupby("Study_id")[[f"c{i}" for i in range(1, 11)]].first().loc[order].values
    a, _ = macro_auc(y.astype(np.float32), df.values)
    return a, df, y

try:
    a1, d1, y = study_auc("ref_cnxs_384")
    a2, d2, _ = study_auc("ref_dn121_384")
    print("cnx-s  %.4f" % a1)
    print("dn121  %.4f" % a2)
    idx = d1.index.intersection(d2.index)
    y = y.loc[idx].values
    f = 0.55 * d1.loc[idx].values + 0.45 * d2.loc[idx].values
    a, _ = macro_auc(y.astype(np.float32), f)
    print("55/45 融合 %.4f" % a)
    b = 0.5 * d1.loc[idx].values + 0.5 * d2.loc[idx].values
    a, _ = macro_auc(y.astype(np.float32), b)
    print("50/50 融合 %.4f" % a)
    m = 0.7 * d1.loc[idx].values + 0.3 * d2.loc[idx].values
    a, _ = macro_auc(y.astype(np.float32), m)
    print("70/30 融合 %.4f" % a)
except Exception as e:
    print("fusion err", e)
EOF
echo "CHAIN15 DONE $(date)"