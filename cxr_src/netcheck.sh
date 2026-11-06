echo "===== fetch_weights.sh"
head -25 /root/_dsh/fetch_weights.sh 2>/dev/null
echo "===== net test huggingface"
timeout 15 python - <<'EOF' 2>&1 | tail -3
import urllib.request
try:
    r = urllib.request.urlopen("https://huggingface.co", timeout=8)
    print("HF OK", r.status)
except Exception as e:
    print("HF FAIL", type(e).__name__, str(e)[:120])
EOF
echo "===== net test pypi"
timeout 15 python - <<'EOF' 2>&1 | tail -2
import urllib.request
try:
    r = urllib.request.urlopen("https://pypi.org/simple/", timeout=8)
    print("PYPI OK", r.status)
except Exception as e:
    print("PYPI FAIL", type(e).__name__, str(e)[:120])
EOF
echo "===== pip config"
pip config list 2>/dev/null; cat /etc/pip.conf 2>/dev/null; cat ~/.pip/pip.conf 2>/dev/null
echo "===== weights dir"
ls -la /root/_dsh/weights/ 2>/dev/null
echo "===== hf cache?"
ls ~/.cache/huggingface/hub 2>/dev/null | head -10
ls ~/.cache/torch/hub/checkpoints 2>/dev/null | head -20