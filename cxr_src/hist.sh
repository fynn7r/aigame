echo "=== Y (dn121@1024 24ep) 每轮 foldAUC"
grep -E '^ep[0-9]+ loss' /root/_dsh/x_dn121_1024.log | sed -E 's/^ep([0-9]+).*foldAUC ([0-9.]+).*validAUC ([0-9.]+).*/\1 fold=\2 valid=\3/'
echo "=== Y BEST"
grep BEST /root/_dsh/x_dn121_1024.log
echo
echo "=== X (b0@640 nopw+mixup) 每轮"
grep -E '^ep[0-9]+ loss' /root/_dsh/x_b0_640_nopw.log | sed -E 's/^ep([0-9]+).*foldAUC ([0-9.]+).*validAUC ([0-9.]+).*/\1 fold=\2 valid=\3/'
echo "=== X BEST"
grep BEST /root/_dsh/x_b0_640_nopw.log
echo
echo "=== 基线 640 f0 每轮"
grep -E '^ep[0-9]+ loss' /root/_dsh/p_b0_640.log | sed -E 's/^ep([0-9]+).*foldAUC ([0-9.]+).*validAUC ([0-9.]+).*/\1 fold=\2 valid=\3/'