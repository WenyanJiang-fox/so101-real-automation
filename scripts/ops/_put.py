# -*- coding: utf-8 -*-
"""把一个本地文件原样传到 AutoDL 上。省得在命令行里跟引号打架。

    AUTODL_PW=$(cat /c/Users/jiang/.autodl_pw) python -u _put.py 本地路径 远端路径
"""
from __future__ import annotations

import os
import sys

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "42710"))


def main() -> int:
    if len(sys.argv) < 3:
        print("用法：_put.py <本地路径> <远端路径>")
        return 2
    local, remote = sys.argv[1], sys.argv[2]
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=os.environ["AUTODL_PW"],
              timeout=15, banner_timeout=15, auth_timeout=15)
    try:
        sftp = c.open_sftp()
        sftp.put(local, remote)
        print(f"{local} → {remote}  {sftp.stat(remote).st_size} B")
        sftp.close()
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
