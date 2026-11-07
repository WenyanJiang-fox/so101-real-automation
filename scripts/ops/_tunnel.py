# -*- coding: utf-8 -*-
"""本机 127.0.0.1:8000 → AutoDL 容器 127.0.0.1:8000 的 SSH 直连转发。

    AUTODL_PW=$(cat /c/Users/jiang/.autodl_pw) python -u _tunnel.py [本地端口] [远端端口]

paramiko 没有现成的 `-L`，这里用 `Transport.open_channel('direct-tcpip')` 手搭。
端口/主机从 `AUTODL_HOST` / `AUTODL_PORT` 读，缺省是 2026-10-23 那一台。
"""
from __future__ import annotations

import os
import select
import socket
import sys
import threading

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "42710"))
PW = os.environ["AUTODL_PW"]
LOCAL = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
REMOTE = ("127.0.0.1", int(sys.argv[2]) if len(sys.argv) > 2 else 8000)


def main() -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username="root", password=PW,
                   timeout=15, banner_timeout=15, auth_timeout=15)
    tr = client.get_transport()
    assert tr is not None

    def pump(conn: socket.socket) -> None:
        try:
            chan = tr.open_channel("direct-tcpip", REMOTE, conn.getpeername())
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [隧道] 开通道失败：{type(exc).__name__}: {exc}", flush=True)
            conn.close()
            return
        try:
            while True:
                r, _, _ = select.select([conn, chan], [], [])
                if conn in r:
                    data = conn.recv(65536)
                    if not data:
                        break
                    chan.sendall(data)
                if chan in r:
                    data = chan.recv(65536)
                    if not data:
                        break
                    conn.sendall(data)
        except Exception:                                         # noqa: BLE001
            pass
        finally:
            try:
                chan.close()
            except Exception:                                     # noqa: BLE001
                pass
            conn.close()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", LOCAL))
    srv.listen(16)
    print(f"[隧道] 127.0.0.1:{LOCAL} → {HOST}:{PORT} → 远端 {REMOTE[0]}:{REMOTE[1]}",
          flush=True)
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=pump, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
