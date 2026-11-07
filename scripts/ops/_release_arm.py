# -*- coding: utf-8 -*-
"""松爪子 / 握回去：给手让路。

    G:/anaconda/envs/sim2real/python.exe -u _release_arm.py
        六个舵机全部关扭矩 —— 臂变软，可以用手直接掰到想放的位置。
    G:/anaconda/envs/sim2real/python.exe -u _release_arm.py hold
        先把「现在这个位置」写成每个舵机的目标，再开扭矩 —— 松手后停在当前位置。
        （不先写目标直接开扭矩的话，舵机会朝上一次指令的目标冲过去。）

**松手那一下臂会塌**：关扭矩后没有任何力托着它，重力会把它压到桌面/方块上。
关之前先用手托住。

关扭矩之后串口就关了；舵机保持软的状态，直到下一次有程序打开并写使能。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from panel.rig_common import DEFAULT_CONFIG, load_cfg       # noqa: E402
from real.arm import SERVO_IDS                             # noqa: E402
from servo_bridge import open_controller                   # noqa: E402

mode = (sys.argv[1].strip().lower() if len(sys.argv) > 1 else "release")
cfg = load_cfg(DEFAULT_CONFIG)
rig = cfg["real"]
port = str(rig.get("port", "COM10"))
speed = int(rig.get("speed", 0))

ctrl = open_controller(port)
try:
    print(f"[arm] 串口 {port} 已打开", flush=True)
    now = ctrl.sync_read_positions(list(SERVO_IDS))
    print(f"[arm] 当前位置（原始刻度）：{now}", flush=True)

    if mode == "hold":
        if not now:
            raise RuntimeError("读不到当前位置，不敢开扭矩")
        ok_w = ctrl.sync_set_positions({int(k): int(v) for k, v in now.items()},
                                       speed)
        time.sleep(0.3)
        ok_t = ctrl.sync_enable_torque(list(SERVO_IDS))
        print(f"[arm] 目标已按当前位置写入：{'ok' if ok_w else '失败'}；"
              f"扭矩使能：{'ok' if ok_t else '失败'}", flush=True)
        print("[arm] 现在松手，它会停在当前位置", flush=True)
    else:
        ok = ctrl.sync_disable_torque(list(SERVO_IDS))
        print(f"[arm] 六个舵机扭矩已关：{'ok' if ok else '失败'}", flush=True)
        print("[arm] 现在可以用手掰；掰好之后跑 `_release_arm.py hold` 让它停住，"
              f"或者直接跑下一次抓取（那会按新位置使能）", flush=True)
finally:
    try:
        ctrl.close()
    except Exception as exc:                                # noqa: BLE001
        print(f"[arm] 关串口：{exc}", flush=True)
