import sys, time
import torch

need = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
waited = 0
while True:
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as e:
        print("mem_get_info err", e)
        time.sleep(10)
        continue
    if free / 1e9 >= need:
        print("vram ok: %.1f GB free (need %.1f)" % (free / 1e9, need))
        break
    if waited % 60 == 0:
        print("waiting vram: %.1f GB free, need %.1f" % (free / 1e9, need), flush=True)
    time.sleep(10)
    waited += 10
    if waited > 1800:
        print("timeout waiting vram")
        break