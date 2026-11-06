echo "===== date"; date
echo "===== alive (chain11/monitor)"
pgrep -f chain11.sh | head -2
pgrep -f monitor.sh | head -2
echo "===== training proc"
ps -eo etime,cmd | grep 'train\.py' | grep -v grep | head -1 | cut -c1-90
echo "===== chain11 progress"
grep -E '^###|^>>>|^SCORE|^===|登记|!!!' /root/_dsh/chain11.log 2>/dev/null | tail -14
echo "===== ckpt table"
python - <<'EOF'
import os, torch
R = "/data/cxr_first/runs"
for n in sorted(os.listdir(R)):
    p = os.path.join(R, n, "best.pt")
    if not os.path.exists(p):
        continue
    try:
        c = torch.load(p, map_location="cpu", weights_only=False)
        print("%-16s size=%3d ep=%2d foldAUC=%.4f" % (n, c["size"], c["epoch"], c["score"]))
    except Exception as e:
        print(n, "ERR", str(e)[:40])
EOF
echo "===== probs"
ls /data/cxr_first/runs/*/probs_valid.npz 2>/dev/null | wc -l
ls /data/cxr_first/runs/*/probs_test.npz 2>/dev/null | wc -l
echo "===== files"
ls -la /data/cxr_first/*.csv 2>/dev/null
ls -la /root/_dsh/v3.log 2>/dev/null