# -*- coding: utf-8 -*-
"""手摆好位置之后：合爪 → 判夹住没有 → 竖直抬起来。

给 `_release_arm.py` 松手、人手把爪子摆到目标周围之后的下一步用。
不改任何现有文件：合爪和判据都直接调技能里同一份函数。

    G:/anaconda/envs/sim2real/python.exe -u _manual_close_lift.py
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
import real.arm                                                   # noqa: E402,F401
from grasp_check import classify_ref, load_thresholds             # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg             # noqa: E402

# 上一趟开头实测的行程两端（run16：合到底 36.7°、张到最大 147.0°）。这一次不重量：
# 爪现在就摆在目标周围，先合到底去量会把目标夹一下。
EMPTY_DEG = 36.7
OPEN_DEG = 147.0

cfg = load_cfg(DEFAULT_CONFIG)
rig = cfg["real"]
arm = make("arm", "real", port=str(rig.get("port", "COM10")),
           speed=int(rig.get("speed", 0)))
arm.terminated = False
try:
    from skills.pick_real.grasp import _close_until_stall, _gripper_deg
    from skills.pick_real.motion import _goto

    tcp0 = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    d0, _s0 = _gripper_deg(arm, arm.ctrl)
    print(f"[手摆位] 合爪前 TCP {np.round(tcp0 * 1000, 1)}mm  "
          f"6 号 {d0 if d0 is None else f'{d0:.1f}°'}", flush=True)

    stop, info = _close_until_stall(arm, None, half_m=None)
    grip_deg, src = _gripper_deg(arm, arm.ctrl)
    verdict = classify_ref(grip_deg, EMPTY_DEG, load_thresholds(), OPEN_DEG)
    print(f"[手摆位] 合爪：{stop}（{info['why']}，{info['steps']} 步）"
          f"  开合角 {info['q0']:+.3f} → {info['q']:+.3f} rad"
          f"  6 号 {grip_deg if grip_deg is None else f'{grip_deg:.1f}°'}（{src}）"
          f"  → 判 {verdict}", flush=True)

    if verdict != "held":
        print(f"[手摆位] 没夹住（判 {verdict}）—— 不抬，原地松爪", flush=True)
        raise SystemExit(0)

    up = _goto(arm, np.array([tcp0[0], tcp0[1], LIFT_Z], dtype=np.float64),
               vmax=speed_vmax(CARRY), grip=None, tol=0.012)
    z_now = float(arm.tcp()[2])
    d1, src1 = _gripper_deg(arm, arm.ctrl)
    v1 = classify_ref(d1, EMPTY_DEG, load_thresholds(), OPEN_DEG)
    print(f"[手摆位] 抬升：{up}  TCP z={z_now * 1000:+.1f}mm"
          f"（该到 {LIFT_Z * 1000:+.0f}mm）  6 号 {d1 if d1 is None else f'{d1:.1f}°'}"
          f"（{src1}）  → 判 {v1}", flush=True)
    print(f"[手摆位] 结果：{'夹住并抬起' if v1 == 'held' else '抬起来但没夹住'}", flush=True)
finally:
    try:
        arm.close()
    except Exception as exc:                                     # noqa: BLE001
        print(f"[手摆位] 关串口：{exc}", flush=True)
