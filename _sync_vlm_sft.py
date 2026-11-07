# -*- coding: utf-8 -*-
"""把本机 `datasets/vlm_judge_sft` 同步到 AutoDL，并写远程 jsonl（绝对路径）。

    AUTODL_PW=$(cat /c/Users/jiang/.autodl_pw) python -u _sync_vlm_sft.py

只传远端没有的图；`train_remote.jsonl` / `val_remote.jsonl` 每次重写。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "42710"))
LOCAL = Path(r"G:\sim2real\so101-real-harness\datasets\vlm_judge_sft")
REMOTE = "/root/autodl-tmp/vlm_judge_sft"


def main() -> int:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=os.environ["AUTODL_PW"],
              timeout=15, banner_timeout=15, auth_timeout=15)
    sftp = c.open_sftp()
    sftp.get_channel().settimeout(120)

    imgs = sorted((LOCAL / "images").rglob("*.jpg"))
    sent = 0
    for p in imgs:
        rel = p.relative_to(LOCAL).as_posix()
        dst = f"{REMOTE}/{rel}"
        parent = dst.rsplit("/", 1)[0]
        acc = ""
        for part in parent.strip("/").split("/"):
            acc += "/" + part
            try:
                sftp.stat(acc)
            except IOError:
                sftp.mkdir(acc)
        try:
            sftp.stat(dst)
            continue
        except IOError:
            pass
        sftp.put(str(p), dst)
        sent += 1
    print(f"上传 {sent} 张新图（共 {len(imgs)} 张）", flush=True)

    for split in ("train", "val"):
        out = []
        for line in (LOCAL / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            r["images"] = [f"{REMOTE}/{Path(p).relative_to(LOCAL).as_posix()}"
                           for p in r["images"]]
            out.append(json.dumps(r, ensure_ascii=False))
        remote_path = f"{REMOTE}/{split}_remote.jsonl"
        with sftp.open(remote_path, "w") as f:
            f.write("\n".join(out) + "\n")
        print(f"{remote_path}：{len(out)} 条", flush=True)
    sftp.close()
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
