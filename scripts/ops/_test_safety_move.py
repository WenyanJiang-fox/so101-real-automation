# -*- coding: utf-8 -*-
"""在**安全闸照常开着**的前提下，把 `_test_safety_trip.py` 那个点真的走到。

限幅用的是 VLM 刚给的值（`knobs.JOINT1_STEP_DEG`，默认就是 30°）：`_goto` 每次最多
让 1 号关节转 30°，下一趟从新姿态重新逆解，所以 110.6° 会被拆成几步。安全闸每趟照旧先问
一遍——真出问题时它还是会拦下、不写。

每一步之后都回到同一套判据：到点（差 < 8mm）就停；超过 8 趟还没到就停手，不硬来。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"G:\sim2real\so101-real-harness")

import collect_real_dataset as C                                  # noqa: E402
import real.arm                                                   # noqa: E402,F401
import real.cameras                                               # noqa: E402,F401
import skills.home_real                                           # noqa: E402,F401
import skills.pick_real                                           # noqa: E402,F401
from core.params import speed_vmax                                # noqa: E402
from skills.pick_real import knobs as K                           # noqa: E402
from skills.pick_real import motion                               # noqa: E402

GOAL_MM = (110.0, 100.0, 80.0)
MAX_TRIPS = 8
TOL_M = 0.008


def main() -> int:
    cfg = C.load_cfg(C.DEFAULT_CONFIG)
    C._setup_geom(cfg)
    rig = cfg["real"]

    arm = C.make("arm", "real", port=str(rig.get("port", "COM10")),
                 speed=int(rig.get("speed", 0)), fast=bool(rig.get("fast", False)))
    arm.terminated = False
    wrist = C.make("camera", "real_wrist", source=arm,
                   index=int(rig.get("cam_index", 1)))
    goal = np.array(GOAL_MM, dtype=np.float64) / 1000.0
    try:
        C.make("skill", "home_real").run(arm, {"real_wrist": wrist})
        print(f"[试验] 1 号关节单步上限 = {K.JOINT1_STEP_DEG}°；目标 "
              f"{np.round(goal * 1000, 1)}mm", flush=True)
        for i in range(1, MAX_TRIPS + 1):
            out = motion._goto(arm, goal, vmax=speed_vmax(), grip=1.0, tries=1)
            err = float(np.linalg.norm(goal - arm.tcp()))
            print(f"[试验] 第 {i} 趟：{out}，离目标 {err * 1000:.1f}mm，"
                  f"关节 {np.round(arm.angles(), 1)}", flush=True)
            if out == "safety":
                print("[试验] 安全闸拦下了，停手", flush=True)
                break
            if err < TOL_M:
                print("[试验] 到点", flush=True)
                break
        print(f"[试验] 最终 TCP {np.round(arm.tcp() * 1000, 1)}mm", flush=True)
        return 0
    finally:
        try:
            C.make("skill", "home_real").run(arm, {"real_wrist": wrist})
        except Exception as exc:                                   # noqa: BLE001
            print("[试验] 回初始位置失败：", exc, flush=True)
        try:
            wrist.close()
        except Exception:                                          # noqa: BLE001
            pass
        arm.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
