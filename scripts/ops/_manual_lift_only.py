# -*- coding: utf-8 -*-
"""6 号舵机不回应时，只抬 1~5 号：验证爪里到底夹没夹住。

6 号读不到，`RealArm.__init__` 和 `read_ticks` 都会卡在它上面。这里给它一个
**占位读数**（只用于回退，不写进任何判据），其余五个关节照常真读。
抬升用 `grip=None` —— 这一段不碰 6 号的位置指令。

    G:/anaconda/envs/sim2real/python.exe -u _manual_lift_only.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np                                               # noqa: E402

from core.params import CARRY, LIFT_Z, speed_vmax                 # noqa: E402
from core.registry import make                                    # noqa: E402
import real.arm as RA                                             # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg             # noqa: E402

# 6 号读不到时的回退值：松手前读到的最后一个是 2935（张到 168°），合爪后停在闭端
# 附近（约 36° ≈ 1434 刻）。这个数只喂给 read_ticks 的回退，不参与任何判据。
FALLBACK6 = 1434

cfg = load_cfg(DEFAULT_CONFIG)
rig = cfg["real"]
_orig_grip_q = RA.RealArm.grip_q
RA.RealArm.grip_q = lambda self: 36.0                      # 只在构造时用
try:
    arm = make("arm", "real", port=str(rig.get("port", "COM10")),
               speed=int(rig.get("speed", 0)))
finally:
    RA.RealArm.grip_q = _orig_grip_q
arm.last_ticks = {6: FALLBACK6}
arm.terminated = False

try:
    from skills.pick_real.motion import _goto

    tcp0 = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    print(f"[只抬] 起点 TCP {np.round(tcp0 * 1000, 1)}mm", flush=True)
    up = _goto(arm, np.array([tcp0[0], tcp0[1], LIFT_Z], dtype=np.float64),
               vmax=speed_vmax(CARRY), grip=None, tol=0.012)
    tcp1 = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    print(f"[只抬] 抬升：{up}  TCP {np.round(tcp0 * 1000, 1)} → "
          f"{np.round(tcp1 * 1000, 1)}mm（该到 z={LIFT_Z * 1000:+.0f}mm）",
          flush=True)
finally:
    try:
        arm.close()
    except Exception as exc:                                     # noqa: BLE001
        print(f"[只抬] 关串口：{exc}", flush=True)
