# -*- coding: utf-8 -*-
"""把 6 号（夹爪）从「顶着方块硬压」松开到指定角度，看它能不能退出过载。

**背景**：6 号回包带了错误位 0x20（32）。驱动 `read_position` 见到非零错误位就返回
`None`，于是 `RealArm.__init__` 在第一行 `grip_q()` 就抛「舵机 6 读不到」。这一趟 VLA
给它下过 −10°（顶到底），实测它被方块挡在 −0.7°，整整 7.2° 的位置误差一直压着。

做法只有一步：按**同一份标定**把 6 号写到 `--deg`（qpos 口径的度），压的量从 7.2°
降到 `|--deg − 实测|`，负载跟着降。写完立刻回读，打印错误位。

    G:/anaconda/envs/sim2real/python.exe -u _relax_servo6.py --deg -4
    G:/anaconda/envs/sim2real/python.exe -u _relax_servo6.py --deg -4 --apply   # 真写

不给 `--apply` 只算刻数、只读当前值，**不写**。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

import numpy as np                                               # noqa: E402

import real.arm as A                                             # noqa: E402
from servo_bridge import open_controller                         # noqa: E402

GRIP = 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=float, default=-4.0, help="要写到的 6 号角度（qpos 度）")
    ap.add_argument("--apply", action="store_true", help="真写下去；不给只预演")
    ap.add_argument("--re-enable", action="store_true", help="写位置之前先重新使能力矩")
    ap.add_argument("--cycle-torque", action="store_true", help="先关力矩再开，看能不能清掉过载")
    ap.add_argument("--port", default="COM10")
    args = ap.parse_args()

    cal = A.load_calibration(A.CAL_JSON)
    model, _, _, _ = A.load_model(A.MJCF)
    lo, hi = A.joint_limits(model)["gripper"]
    q = np.zeros(6)
    q[5] = np.radians(args.deg)
    tick = int(A.sim_rad_to_motor_ticks(q, calibration=cal,
                                        gripper_limits_rad=(lo, hi))["gripper"])

    c = open_controller(args.port, debug=True)
    now = c.read_position(GRIP)
    print(f"6 号现在 tick={now}", flush=True)
    print(f"要写到 {args.deg:+.1f}° = tick {tick}"
          f"（驱动口径 {(tick - 1024) * 180 / 2048:.2f}°）", flush=True)
    if not args.apply:
        print("（预演，没写）", flush=True)
        return 0

    if args.cycle_torque:
        # **过载是「有负载」触发的。** 力矩关掉，负载当场归零，错误位就有机会自己清掉
        # —— 这跟重新置使能位不是一回事，那个是「让位置指令生效」，这个是把负载拿掉。
        import time                                              # noqa: PLC0415
        print(f"关力矩：{c.disable_torque(GRIP)}", flush=True)
        time.sleep(0.6)
        print(f"  关力矩后读：{c.read_position(GRIP)}", flush=True)
        print(f"重开力矩：{c.enable_torque(GRIP)}", flush=True)
        time.sleep(0.3)
        print(f"  重开后读：{c.read_position(GRIP)}", flush=True)

    if args.re_enable:
        # 过载保护常常顺手把力矩关掉，关掉之后写位置等于没写（实测：写了 tick 1508，
        # 读回来还是 1554）。重新置一次使能位，才轮得到位置指令生效。
        print(f"重新使能力矩：{c.enable_torque(GRIP)}", flush=True)
        import time                                              # noqa: PLC0415
        time.sleep(0.3)

    c.set_servo_angle(GRIP, (tick - 1024) * 180 / 2048, speed=400)
    import time                                                  # noqa: PLC0415
    time.sleep(0.8)
    for i in range(3):
        print(f"写后第 {i + 1} 次读：{c.read_position(GRIP)}", flush=True)
        time.sleep(0.2)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
