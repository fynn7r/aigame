# -*- coding: utf-8 -*-
"""预处理：把 33984 张原始大图统一成 512x512 uint8 内存映射缓存。

输出 /data/cxr_first/proc/
  images_512.npy   (N,512,512) uint8
  meta.csv         split,idx,Subject_id,Study_id,Dicom_id,ViewPosition,Rows,Columns,label_str,c1..c10
"""
import os, sys, time
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool

ROOT = "/data/cxr_first"
OUT = os.path.join(ROOT, "proc")
SIZE = int(os.environ.get("CACHE_SIZE", 512))
os.makedirs(OUT, exist_ok=True)

NAMES = ["train", "valid", "test"]


def build_meta():
    frames = []
    for sp in NAMES:
        d = pd.read_csv(os.path.join(ROOT, f"{sp}.csv"))
        d["split"] = sp
        d["path"] = [os.path.join(ROOT, sp, s, st, f"{dcm}.jpg")
                     for s, st, dcm in zip(d.Subject_id, d.Study_id, d.Dicom_id)]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["idx"] = np.arange(len(df))
    lab = np.zeros((len(df), 10), dtype=np.int8)
    if "Predict_class" in df:
        for i, s in enumerate(df["Predict_class"].fillna("").astype(str)):
            for x in s.split(";"):
                x = x.strip()
                if x.isdigit() and 1 <= int(x) <= 10:
                    lab[i, int(x) - 1] = 1
    for k in range(10):
        df[f"c{k+1}"] = lab[:, k]
    return df


def load_one(path):
    try:
        im = Image.open(path)
        im = im.convert("L")
        w, h = im.size
        if w != h:
            s = max(w, h)
            canvas = Image.new("L", (s, s), 0)
            canvas.paste(im, ((s - w) // 2, (s - h) // 2))
            im = canvas
        im = im.resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)
    except Exception as e:
        print("ERR", path, e, flush=True)
        return np.zeros((SIZE, SIZE), dtype=np.uint8)


def main():
    t0 = time.time()
    df = build_meta()
    n = len(df)
    print(f"total {n} images", flush=True)
    mm = np.lib.format.open_memmap(os.path.join(OUT, f"images_{SIZE}.npy"),
                                   mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE))
    paths = df["path"].tolist()
    bs = 512
    with Pool(int(os.environ.get("CACHE_PROCS", 8))) as pool:
        for s in range(0, n, bs):
            chunk = paths[s:s + bs]
            arrs = pool.map(load_one, chunk, chunksize=8)
            for j, a in enumerate(arrs):
                mm[s + j] = a
            if (s // bs) % 10 == 0:
                print(f"{s}/{n} {time.time()-t0:.0f}s", flush=True)
    mm.flush()
    del mm
    df.drop(columns=["path"]).to_csv(os.path.join(OUT, "meta.csv"), index=False)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()