#!/bin/bash
# 看守进程：每 30 秒踢掉抢显卡的别人的任务；每 3 分钟推一次状态
KILLED=0
i=0
while true; do
  if pgrep -f train_swa.py >/dev/null 2>&1 || pgrep -f '/data/work' >/dev/null 2>&1; then
    pkill -9 -f train_swa.py
    pkill -9 -f '/data/work'
    KILLED=$((KILLED+1))
    echo "$(date) killed swa/glaucoma (#$KILLED)" >> /root/_dsh/guard.log
  fi
  i=$((i+1))
  if [ $((i % 6)) -eq 0 ]; then
    {
      date
      echo "HCU: $(rocm-smi 2>/dev/null | grep -E '^0 ' | tr -s ' ')"
      python - <<'EOF'
import torch
try:
    free, total = torch.cuda.mem_get_info()
    print("VRAM free %.2f / %.2f GB" % (free/1e9, total/1e9))
except Exception as e:
    print("vram err", e)
EOF
      echo "--- procs:"
      ps -eo etime,pcpu,cmd | grep -E 'python train\.py|python infer\.py' | grep -v grep | head -2 | cut -c1-95
      echo "--- logs:"
      for f in /root/_dsh/p_*.log /root/_dsh/x_*.log; do
        [ -f "$f" ] || continue
        last=$(grep -E '^ep[0-9]' "$f" | tail -1 | cut -c1-88)
        best=$(grep BEST "$f" | tail -1)
        echo "$(basename "$f"): ${last:-?} | ${best:-run}"
      done
      echo "--- chain11:"
      grep -E '^###|^>>>|^SCORE|^===|登记' /root/_dsh/chain11.log 2>/dev/null | tail -4
      echo "--- guard:"
      tail -2 /root/_dsh/guard.log 2>/dev/null
    } > /tmp/status.txt 2>&1
    curl -s -m 15 -X POST --data-binary @/tmp/status.txt 'http://10.0.79.98:8765/upload?name=status.txt' > /dev/null
  fi
  sleep 30
done