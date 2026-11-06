echo "--- date"; date
echo "--- all python procs"
ps -eo pid,ppid,etime,cmd | grep python | grep -v grep | cut -c1-110
echo "--- rocm-smi"
rocm-smi 2>/dev/null | grep -E '^0 '
echo "--- mem_get_info"
python - <<'EOF'
import torch
try:
    free, total = torch.cuda.mem_get_info()
    print("free %.2f GB / total %.2f GB" % (free/1e9, total/1e9))
except Exception as e:
    print("ERR", e)
EOF
echo "--- check f0 ckpt"
python - <<'EOF'
import torch
p = "/data/cxr_first/runs/p_b0_640_f0/best.pt"
try:
    c = torch.load(p, map_location="cpu", weights_only=False)
    print("keys", list(c.keys()))
    print("score", c.get("score"), "size", c.get("size"), "epoch", c.get("epoch"))
except Exception as e:
    print("LOAD ERR", type(e).__name__, e)
EOF
ls -la /data/cxr_first/runs/p_b0_640_f0/ /data/cxr_first/runs/p_b0_640/