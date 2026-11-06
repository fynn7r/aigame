# -*- coding: utf-8 -*-
"""极简文件收发服务：容器 -> 本机(上传) / 本机 -> 容器(下载)。

  POST /upload?name=foo.bin   body=文件内容   -> 保存到 INBOX
  GET  /ping                                 -> pong
  GET  /get/<name>                           -> 返回 OUTBOX 下的文件
"""
import os, sys, http.server, socketserver, urllib.parse

INBOX = r"E:\aigame\cxr_first\inbox"
OUTBOX = r"E:\aigame\cxr_first\outbox"
os.makedirs(INBOX, exist_ok=True)
os.makedirs(OUTBOX, exist_ok=True)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body=b"", ctype="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/ping":
            return self._send(200, b"pong")
        if u.path.startswith("/get/"):
            name = os.path.basename(urllib.parse.unquote(u.path[5:]))
            p = os.path.join(OUTBOX, name)
            if not os.path.exists(p):
                return self._send(404, b"no such file")
            with open(p, "rb") as f:
                data = f.read()
            return self._send(200, data, "application/octet-stream")
        if u.path == "/list":
            return self._send(200, "\n".join(os.listdir(INBOX)).encode())
        return self._send(404, b"?")

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        name = os.path.basename(q.get("name", ["upload.bin"])[0])
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        with open(os.path.join(INBOX, name), "wb") as f:
            f.write(data)
        print(f"[{name}] {len(data)} bytes", flush=True)
        return self._send(200, f"ok {len(data)}".encode())


class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"serving on 0.0.0.0:{PORT} inbox={INBOX}", flush=True)
    S(("0.0.0.0", PORT), H).serve_forever()