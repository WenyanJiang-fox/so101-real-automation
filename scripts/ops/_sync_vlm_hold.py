# -*- coding: utf-8 -*-
"""把 `out/vlm_gallery/` 里合爪后那一张打包，传到 AutoDL 做零样本探针。

    AUTODL_PW=$(cat /c/Users/jiang/.autodl_pw) python -u _sync_vlm_hold.py

标签只看文件名：`after_held.jpg` → yes，其余（empty / partial / unknown）→ no。
只传远端没有的图；`probe_remote.jsonl` 每次重写。
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "42710"))
ROOT = Path(r"G:\sim2real\so101-real-harness")
GALLERY = ROOT / "out" / "vlm_gallery"
LOCAL = ROOT / "datasets" / "vlm_hold_probe"
REMOTE = "/root/autodl-tmp/vlm_hold_probe"


def main() -> int:
    imgs = sorted(GALLERY.glob("*/after_*.jpg"))
    if not imgs:
        print("gallery 里还没有 after_*.jpg —— 先跑一轮采集")
        return 1
    (LOCAL / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in imgs:
        verdict = p.stem.replace("after_", "", 1)
        dst = LOCAL / "images" / p.parent.name / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        rows.append({"name": f"{p.parent.name}/{p.name}",
                     "label": "yes" if verdict == "held" else "no",
                     "verdict": verdict,
                     "images": [dst.relative_to(LOCAL).as_posix()]})
    (LOCAL / "probe_local.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    yes = sum(1 for r in rows if r["label"] == "yes")
    print(f"本地打包 {len(rows)} 张（yes {yes} / no {len(rows) - yes}）")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=os.environ["AUTODL_PW"],
              timeout=15, banner_timeout=15, auth_timeout=15)
    sftp = c.open_sftp()
    sftp.get_channel().settimeout(120)
    acc = ""
    for part in REMOTE.strip("/").split("/"):
        acc += "/" + part
        try:
            sftp.stat(acc)
        except IOError:
            sftp.mkdir(acc)
    sent = 0
    for r in rows:
        src = LOCAL / r["images"][0]
        dst = f"{REMOTE}/{r['images'][0]}"
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
        except IOError:
            sftp.put(str(src), dst)
            sent += 1
    print(f"上传 {sent} 张新图（共 {len(rows)} 张）", flush=True)

    out = []
    for r in rows:
        r = dict(r)
        r["images"] = [f"{REMOTE}/{r['images'][0]}"]
        out.append(json.dumps(r, ensure_ascii=False))
    with sftp.open(f"{REMOTE}/probe_remote.jsonl", "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"{REMOTE}/probe_remote.jsonl：{len(out)} 条", flush=True)
    sftp.close()
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
