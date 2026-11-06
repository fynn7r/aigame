echo "--- saved best count (640)"
grep -c 'saved best' /root/_dsh/p_b0_640.log
grep -E 'saved best' /root/_dsh/p_b0_640.log | head -6
echo "--- ckpt info"
python - <<'EOF'
import torch, os
for t in ["p_b0_640", "p_b0_640_f0", "p_b0_448", "p_b0a_f0"]:
    p = f"/data/cxr_first/runs/{t}/best.pt"
    try:
        c = torch.load(p, map_location="cpu", weights_only=False)
        print(f"{t:14s} epoch={c['epoch']:2d} score={c['score']:.4f} size={c['size']} "
              f"mtime={os.path.getmtime(p):.0f}")
    except Exception as e:
        print(t, "ERR", e)
EOF