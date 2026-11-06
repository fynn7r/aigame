echo "--- date"; date
echo "--- chain11 alive (expect 1)"; pgrep -c -f chain11.sh
echo "--- monitor alive (expect 1)"; pgrep -c -f monitor.sh
echo "--- training proc"
ps -eo pid,etime,cmd | grep 'train\.py' | grep -v grep | head -1 | cut -c1-95
echo "--- 640 f1 last line"
tail -2 /root/_dsh/p_b0_640_f1.log
echo "--- ckpt table"
python - <<'EOF'
import os, torch
R = "/data/cxr_first/runs"
names = [d for d in sorted(os.listdir(R)) if os.path.exists(os.path.join(R, d, "best.pt"))]
for n in names:
    try:
        c = torch.load(os.path.join(R, n, "best.pt"), map_location="cpu", weights_only=False)
        print("%-16s size=%3d ep=%2d foldAUC=%.4f" % (n, c["size"], c["epoch"], c["score"]))
    except Exception as e:
        print(n, "ERR", str(e)[:40])
EOF
echo "--- probs ready"
ls /data/cxr_first/runs/*/probs_valid.npz 2>/dev/null | wc -l
ls /data/cxr_first/runs/*/probs_test.npz 2>/dev/null | wc -l
echo "--- submissions on container"
ls -la /data/cxr_first/*.csv 2>/dev/null