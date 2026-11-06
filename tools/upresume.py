# -*- coding: utf-8 -*-
"""断点续传上传：ct.py -u 中断后，从远端 .b64 已有长度继续。

用法: python tools/upresume.py <local> <remote>
"""
import base64
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ct  # noqa: E402

CHUNK = 120_000


def main():
    local, remote = sys.argv[1], sys.argv[2]
    raw = open(local, "rb").read()
    md5_local = hashlib.md5(raw).hexdigest()
    b = base64.b64encode(raw).decode()
    print("local %d bytes, b64 %d chars, md5 %s" % (len(raw), len(b), md5_local), flush=True)

    t = ct.Term()
    try:
        t.run("mkdir -p $(dirname %s)" % remote, 60)
        out, code = t.run("stat -c %%s %s.b64 2>/dev/null || echo 0" % remote, 120)
        nums = [x for x in out.replace("\n", " ").split() if x.isdigit()]
        have = int(nums[-1]) if nums else 0
        print("remote .b64 已有 %d 字符" % have, flush=True)

        if have > len(b):
            print("远端比本地还长，重来")
            t.run("rm -f %s.b64 %s" % (remote, remote), 60)
            have = 0
        if have and have % CHUNK:
            aligned = (have // CHUNK) * CHUNK
            t.run("truncate -s %d %s.b64" % (aligned, remote), 60)
            have = aligned
            print("对齐到 %d" % have, flush=True)

        total = len(b)
        i = have
        n = 0
        while i < total:
            piece = b[i:i + CHUNK]
            for attempt in range(3):
                try:
                    out, code = t.run("printf '%%s' '%s' >> %s.b64" % (piece, remote), 180)
                    if code == 0:
                        break
                except Exception as e:
                    print("  chunk 异常 %s，重连" % type(e).__name__, flush=True)
                    time.sleep(3)
                    t = ct.Term()
            else:
                print("!! 发送失败于 offset %d" % i, flush=True)
                return 1
            i += CHUNK
            n += 1
            if n % 50 == 0:
                print("  %d/%d (%.0f%%)" % (min(i, total), total, 100.0 * min(i, total) / total), flush=True)

        print("拼装…", flush=True)
        out, code = t.run("base64 -d %s.b64 > %s && rm -f %s.b64 && md5sum %s" % (remote, remote, remote, remote), 900)
        print(out, flush=True)
        print("本地 md5 %s" % md5_local)
        return 0
    finally:
        t.close()


if __name__ == "__main__":
    import time  # noqa: F401  (用于异常分支)
    sys.exit(main())