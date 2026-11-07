# -*- coding: utf-8 -*-
"""盯采集：日志不动太久 / 数据集集数涨了就报，采集自己结束了就退。

    G:/anaconda/envs/sim2real/python.exe -u _watch_collect.py --log <日志> [--stall-min 30]

判据只有两条：
- **采集自己结束**：日志里出现「结束：」→ 退出码 0；
- **日志超过 `--stall-min` 分钟没新行** → 退出码 2（臂可能卡住、串口掉了，要人看）。

每 30 秒看一次；数据集 `total_episodes` 一变就打一行，方便回头对时间。
不碰臂、不碰进程 —— 停采集一律走 `out/STOP`（`collect_real_dataset.py` 的停止文件）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def episodes(info: Path) -> int:
    try:
        with open(info, encoding="utf-8") as f:
            return int(json.load(f).get("total_episodes") or 0)
    except Exception:                                              # noqa: BLE001
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--info", default=r"G:\sim2real\so101-real-harness"
                                     r"\datasets\so101_real_grasp_960_4hz"
                                     r"\meta\info.json")
    ap.add_argument("--stall-min", type=float, default=30.0)
    ap.add_argument("--poll-s", type=float, default=30.0)
    args = ap.parse_args()

    log = Path(args.log)
    info = Path(args.info)
    n0 = episodes(info)
    print(f"[监视] 开始：日志 {log}，数据集已有 {n0} 集，"
          f"日志 {args.stall_min:.0f} 分钟不动就报", flush=True)
    if not log.is_file():
        time.sleep(args.poll_s)
    last_n = n0
    while True:
        time.sleep(args.poll_s)
        txt = ""
        try:
            txt = log.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            pass
        if "结束：" in txt or "[完成]" in txt:
            print(f"[监视] 采集自己结束了（数据集 {episodes(info)} 集）", flush=True)
            return 0
        try:
            idle_min = (time.time() - log.stat().st_mtime) / 60.0
        except OSError:
            idle_min = 0.0
        n = episodes(info)
        if n != last_n:
            print(f"[监视] {time.strftime('%H:%M:%S')} 数据集 {n} 集"
                  f"（+{n - last_n if last_n >= 0 else 0}）", flush=True)
            last_n = n
        if idle_min >= args.stall_min:
            print(f"[监视] ✗ 日志 {idle_min:.1f} 分钟没有新行 —— 停下来看："
                  f"{log}", flush=True)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
