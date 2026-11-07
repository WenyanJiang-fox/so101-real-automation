# -*- coding: utf-8 -*-
"""把一段 shell 命令原样发到 AutoDL 那台执行，回显 stdout / stderr。

    AUTODL_PW=$(cat /c/Users/jiang/.autodl_pw) python -u _sshrun.py "<命令>"

主机 / 端口从 `AUTODL_HOST` / `AUTODL_PORT` 读（缺省 2026-10-23 那一台）。
需要后台跑的命令自己在末尾写 `> log 2>&1 &`。
"""
from __future__ import annotations

import os
import sys

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "42710"))


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=os.environ["AUTODL_PW"],
              timeout=15, banner_timeout=15, auth_timeout=15)
    i, o, e = c.exec_command(cmd)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(out, end="")
    if err.strip():
        print("--- stderr ---")
        print(err, end="")
    return o.channel.recv_exit_status()


if __name__ == "__main__":
    raise SystemExit(main())
