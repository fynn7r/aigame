# -*- coding: utf-8 -*-
"""把命令真正的后台化（双 fork + setsid），脱离 ttyd 会话。

用法: python launch.py <日志文件> <命令> [参数...]
"""
import os, sys


def main():
    log = sys.argv[1]
    cmd = sys.argv[2:]
    pid = os.fork()
    if pid > 0:
        print(f"launched pid={pid} log={log}")
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    dn = os.open("/dev/null", os.O_RDONLY)
    os.dup2(dn, 0)
    os.chdir("/root/_dsh")
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()