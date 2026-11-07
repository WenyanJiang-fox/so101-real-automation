# -*- coding: utf-8 -*-
"""只开 6 号：把夹爪写到张开端，1~5 号一律不动。

6 号在总线上读不到。这个脚本只对它**写**张开目标（`grip_hi`），不需要读得到它；
其余关节的刻度从总线真读，原样写回去，所以臂不会动。
写完之后如果 6 号还是读不到，就把 6 号扭矩关掉，让爪子可以用手掰开。

    G:/anaconda/envs/sim2real/python.exe -u _manual_open_gripper.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np                                               # noqa: E402

from core.registry import make                                    # noqa: E402
import real.arm as RA                                             # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg             # noqa: E402

FALLBACK6 = 1434          # 6 号读不到时的回退值（只给 read_ticks 兜底）
# **张开目标（度，qpos 口径）。** 2026-10-21 用户定：任何时候张开都不许超过 20°
# （`grip_hi` = 100.0° 不许可达）。命令行给一个数就覆盖，但写下去前会夹到 20°。
OPEN_DEG = 20.0
OPEN_CEIL_DEG = float(RA.GRIP_OPEN_DEG)
if len(sys.argv) > 1:
    OPEN_DEG = float(sys.argv[1])
OPEN_DEG = min(OPEN_DEG, OPEN_CEIL_DEG)

cfg = load_cfg(DEFAULT_CONFIG)
rig = cfg["real"]
_orig_grip_q = RA.RealArm.grip_q
RA.RealArm.grip_q = lambda self: 36.0
try:
    arm = make("arm", "real", port=str(rig.get("port", "COM10")),
               speed=int(rig.get("speed", 0)))
finally:
    RA.RealArm.grip_q = _orig_grip_q
arm.last_ticks = {6: FALLBACK6}

try:
    now = arm.ctrl.sync_read_positions(list(range(1, 7))) or {}
    print(f"[开爪] 写前读数：{now}", flush=True)

    q = arm._ticks_to_qpos(arm.read_ticks())
    q[5] = float(np.clip(np.radians(OPEN_DEG), arm.grip_lo, arm.grip_open))
    tgt = arm._qpos_to_ticks(q)
    print(f"[开爪] 6 号目标：{tgt[6]} 刻（张到 {np.degrees(q[5]):.1f}° / "
          f"上限 {np.degrees(arm.grip_hi):.1f}°）", flush=True)

    for i in range(15):
        arm.ctrl.sync_set_positions({6: int(tgt[6])}, 0)
        time.sleep(0.1)
    time.sleep(0.5)

    after = arm.ctrl.sync_read_positions(list(range(1, 7))) or {}
    print(f"[开爪] 写后读数：{after}", flush=True)
    if after.get(6) is None:
        ok = arm.ctrl.sync_disable_torque([6])
        print(f"[开爪] 6 号仍然读不到 —— 扭矩已关（{'ok' if ok else '失败'}），"
              f"现在可以用手把爪子掰开", flush=True)
    else:
        print(f"[开爪] 6 号读回 {after[6]} 刻", flush=True)
finally:
    try:
        arm.close()
    except Exception as exc:                                     # noqa: BLE001
        print(f"[开爪] 关串口：{exc}", flush=True)
