#!/bin/bash
cd /root/_dsh/weights || exit 1
HOST=http://10.0.79.98:8765/get
for n in resnet50.pth resnet34.pth densenet121.pth efficientnet_b0.pth efficientnet_b3.pth \
         convnext_tiny.pth resnet50_a1_in1k.safetensors convnext_tiny_fb_in22k_ft_in1k.safetensors \
         efficientnet_b3_ra2_in1k.safetensors tf_efficientnet_b0_ns_jft_in1k.safetensors \
         densenet121_ra_in1k.safetensors convnext_small_fb_in22k_ft_in1k.safetensors; do
  if [ -s "$n" ]; then echo "have $n"; continue; fi
  curl -s -m 600 -o "$n" "$HOST/$n"
  echo "$n $(stat -c %s "$n" 2>/dev/null)"
done
du -sh /root/_dsh/weights