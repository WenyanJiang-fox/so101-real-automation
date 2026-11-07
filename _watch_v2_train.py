# -*- coding: utf-8 -*-
"""盯 `pi05_so101_lora_v2` 这一轮训练：活着就报进度，死了或跑完就退出。

    AUTODL_PW=$(cat /c/Users/jiang/.autodl_pw) python -u _watch_v2_train.py

**不做任何自动关机**（用户 2026-09-21 明令禁止无人看管的定时任务）。它自己退出时
本会话会收到通知，然后由人决定下一步。轮询间隔从 `WATCH_EVERY` 读，默认 600 秒。
"""
from __future__ import annotations

import os
import re
import sys
import time

import paramiko

HOST = os.environ.get("AUTODL_HOST", "connect.nmb2.seetacloud.com")
PORT = int(os.environ.get("AUTODL_PORT", "42710"))
LOG = "/root/autodl-tmp/train_v2.log"
PATTERN = "scripts/train.py pi05_so101_lora_v2"
EVERY = float(os.environ.get("WATCH_EVERY", "600"))
MAX_WAIT_H = float(os.environ.get("WATCH_MAX_H", "6"))
TOTAL = 20_000

#: 两种进度行都认：`Step 159700: grad_norm=...` 和 tqdm 的
#: `Progress on: 184it/20.0kit rate:...`。只认前者会一直读出 −1（2026-10-25 实测）。
STEP_RE = re.compile(r"(?:Step\s+|Progress on:\s*)(\d+)")


def ssh(cmd: str, timeout: float = 60.0) -> str:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username="root", password=os.environ["AUTODL_PW"],
              timeout=15, banner_timeout=15, auth_timeout=15)
    try:
        _, o, _ = c.exec_command(cmd, timeout=timeout)
        return o.read().decode("utf-8", "replace")
    finally:
        c.close()


def main() -> int:
    t0 = time.time()
    last = -1
    stall = 0
    while True:
        alive = ssh(f"pgrep -f '{PATTERN}' | head -1").strip()
        tail = ssh(f"tail -c 200000 {LOG} 2>/dev/null | grep -aE 'Step |Progress on:' | tail -1").strip()
        df = ssh("df -h /root/autodl-tmp | tail -1").split()
        free = df[3] if len(df) > 3 else "?"
        m = STEP_RE.search(tail)
        step = int(m.group(1)) if m else -1
        el = (time.time() - t0) / 3600.0
        print(f"[{time.strftime('%H:%M:%S')}] alive={bool(alive)} step={step} "
              f"free={free} 盯了 {el:.1f}h", flush=True)
        if tail:
            print(f"    {tail[:160]}", flush=True)

        if not alive:
            print("训练进程不在了 —— 监视退出", flush=True)
            return 0
        if step >= TOTAL:
            print("到步数了，监视退出", flush=True)
            return 0
        if step == last:
            stall += 1
            if stall >= 3:
                print(f"步数连续 {stall} 次没动（{step}）—— 监视退出，等人看", flush=True)
                return 0
        else:
            stall = 0
        last = step
        if el > MAX_WAIT_H:
            print(f"盯够 {MAX_WAIT_H}h 了，监视退出（训练还在跑）", flush=True)
            return 0
        time.sleep(EVERY)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
