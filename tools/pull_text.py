"""Pull a text file from the container over the ttyd websocket by line ranges.

The websocket/PTY path is lossy for very long single lines, but plain
line-oriented text survives intact, so we chunk by line count and verify the
result against the remote md5.
"""
import hashlib
import subprocess
import sys
import os

CT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ct.py")


def run(cmd, wait=180):
    p = subprocess.run([sys.executable, "-u", CT, "--wait", str(wait), cmd],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = p.stdout
    if out.endswith("]") and "[exit " in out:
        out = out[:out.rindex("\n[exit ")]
    return out


def main():
    remote, local = sys.argv[1], sys.argv[2]
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    info = run(f"wc -l < {remote}; md5sum {remote} | cut -d' ' -f1", 120).split()
    total, md5_remote = int(info[0]), info[1]
    print("remote lines", total, "md5", md5_remote, flush=True)
    lines = []
    pos = 1
    while pos <= total:
        end = min(pos + chunk - 1, total)
        out = run(f"printf '@B@\\n'; sed -n '{pos},{end}p' {remote}; printf '@E@\\n'", 240)
        if "@B@" not in out or "@E@" not in out:
            print(f"!! chunk {pos}-{end}: markers missing", flush=True)
            return 1
        body = out.split("@B@", 1)[1].rsplit("@E@", 1)[0]
        if body.startswith("\n"):
            body = body[1:]
        if body.endswith("\n"):
            body = body[:-1]
        got = body.split("\n") if body else []
        exp = end - pos + 1
        if len(got) != exp:
            print(f"!! chunk {pos}-{end}: got {len(got)} want {exp}", flush=True)
            return 1
        lines.extend(got)
        pos = end + 1
        print(f"  {end}/{total}", flush=True)
    data = "\n".join(lines) + "\n"
    raw = data.encode("utf-8")
    if os.path.exists(remote + ".csvinfo"):
        pass
    with open(local, "wb") as f:
        f.write(raw)
    md5_local = hashlib.md5(raw).hexdigest()
    print("local md5", md5_local, "expect", md5_remote,
          "OK" if md5_local == md5_remote else "MISMATCH")
    return 0


if __name__ == "__main__":
    sys.exit(main())