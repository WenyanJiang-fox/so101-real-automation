# -*- coding: utf-8 -*-
"""把本机的末端数据集整个传到 AutoDL 的 HF_LEROBOT_HOME 下。

    AUTODL_PW=$(cat /c/Users/jiang/.autodl_pw) python -u _upload_endseg.py

目标路径 `/root/autodl-tmp/hf_lerobot/<repo_id>`，和原来那三份同一个根目录。
已存在同名文件直接覆盖（这份数据集是本地生成的，远端没有需要保护的东西）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "42710"))
SRC = Path(os.environ.get(
    "ENDSEG_OUT",
    r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_2cam_clean2_end_4hz"))
REPO = os.environ.get("ENDSEG_REPO_ID",
                      "so101_real/pick_grasp_2cam_clean2_end_4hz")
DST = f"/root/autodl-tmp/hf_lerobot/{REPO}"

#: `1` = 远端已经有同名且同大小的文件就跳过。给「往已有数据集里补新增 episode」用：
#: 不跳过的话整份 868MB 要重传，链路 0.06MB/s 得跑好几个小时。
SKIP_SAME = os.environ.get("SKIP_SAME", "") not in ("", "0", "false", "False")


def main() -> int:
    if not (SRC / "meta" / "info.json").exists():
        print(f"{SRC} 里没有 meta/info.json —— 数据集还没建完")
        return 1
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=os.environ["AUTODL_PW"],
              timeout=15, banner_timeout=15, auth_timeout=15)
    sftp = c.open_sftp()
    sftp.get_channel().settimeout(120)

    def mk(p: str) -> None:
        try:
            sftp.stat(p)
        except IOError:
            sftp.mkdir(p)

    for part in DST.split("/")[1:]:
        pass
    acc = ""
    for part in DST.strip("/").split("/"):
        acc += "/" + part
        mk(acc)

    # lerobot 写盘时会留 `tmpXXXX` 临时文件，不是数据集内容。
    files = sorted(p for p in SRC.rglob("*")
                   if p.is_file() and not p.name.startswith("tmp"))
    total = sum(p.stat().st_size for p in files)
    print(f"上传 {len(files)} 个文件 / {total / 1e6:.0f} MB → {DST}", flush=True)
    t0 = time.time()
    done = 0
    for i, p in enumerate(files, 1):
        rel = p.relative_to(SRC).as_posix()
        remote = f"{DST}/{rel}"
        parent = remote.rsplit("/", 1)[0]
        acc = ""
        for part in parent.strip("/").split("/"):
            acc += "/" + part
            mk(acc)
        local_size = p.stat().st_size
        # `meta/` 一律重传：`info.json` 改的是 `total_episodes` 的数字，202→285 长度一样、
        # 大小不变，按大小判会把它跳过，远端就永远停在旧集数。
        if SKIP_SAME and not rel.startswith("meta/"):
            try:
                if sftp.stat(remote).st_size == local_size:
                    done += local_size
                    if i % 20 == 0 or i == len(files):
                        print(f"  {i}/{len(files)}  {done / 1e6:.0f}MB  "
                              f"{time.time() - t0:.0f}s（跳过同名同大小）", flush=True)
                    continue
            except IOError:
                pass
        sftp.put(str(p), remote)
        done += p.stat().st_size
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {done / 1e6:.0f}MB  "
                  f"{time.time() - t0:.0f}s", flush=True)
    sftp.close()
    print(f"完成：{DST}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
