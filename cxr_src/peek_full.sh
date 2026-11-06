cd /root/_dsh || exit 1
date
for t in full_b0_640_s42 full_b0_640_s123 full_b0_640_s777 full_dn201_448_s42 full_dn201_448_s123 full_cnxs_448_s42; do
  echo "--- $t"
  grep -hE '^ep[0-9]|^BEST' "$t.log" 2>/dev/null | tail -7
done
echo "=== 当前进程"
ps -eo etime,cmd | grep train.py | grep -v grep | head -1 | cut -c1-60