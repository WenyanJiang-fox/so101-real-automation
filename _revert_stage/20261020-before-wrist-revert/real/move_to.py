"""真机第一步：让从臂从**此刻的姿态**匀速走到中立位（`core.params.WAYPOINT`）。可中断。

走的是**关节空间**的时间最优轨迹，用 harness 自己的 `core/trajectory.py`（toppra）
算速度曲线 —— 起来是匀加速、巡航恒速、收尾匀减速，没有一顿一顿。

**为什么不走 `core/follow.py`（笛卡尔那条）**：
  1. `follow` 每一步都问「手臂在哪」，而真机读一次要 66ms（驱动里每个舵机先睡
     10ms）。`follow` 一步要问三次，实测 4.5 步/秒，走起来就是一下一下的。
  2. `follow` 走位时默认要求工具轴朝下（`drive` 的 `z_des=[0,0,-1]`）。真机这个
     初始点位工具轴是侧着的，转腕的过程把 TCP 拽偏了 —— 目标在 +z 抬起来，实际
     z 从 0.086 掉到 0.043。
  关节空间没有第二个矛盾：从当前角直接解到目标角，中间不夹带姿态改变。

**走的过程中不读舵机。** 轨迹是提前算好的，舵机自己按位置指令走；中途插一次读
就是 66ms 的空档，那才是「一顿一顿」的来源。收尾连发几步让舵机跟上，再读一次。

安全约定（这个文件要一直守住）：
  * **默认什么都不动。** 不加 `--go` 就只读、只算、只打印，一个字节都不写舵机。
  * Ctrl+C 随时停：停在当时的位置，**不松力矩**（松了这条臂会塌下来）。
  * 走完也不松力矩 —— 真机那个「初始点位」是松手后重力塌出来的，每次都不一样；
    保持力矩才有一个可重复的落点。

用法（PowerShell 一行）：
    G:/anaconda/envs/sim2real/python.exe G:/sim2real/so101-real-harness/real/move_to.py --port COM10
    G:/anaconda/envs/sim2real/python.exe G:/sim2real/so101-real-harness/real/move_to.py --port COM10 --go

端口别写死：跑之前先 `G:/anaconda/envs/sim2real/python.exe G:/sim2real/so101-real-harness/identify_arms.py`
问一句哪个口是从臂（2026-10-14 那次是 COM10）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import params as P                                    # noqa: E402
from core.trajectory import build                               # noqa: E402
from real import arm as _arm                                    # noqa: F401,E402
from real.arm import RealArm, SERVO_IDS                         # noqa: E402
from servo_bridge import open_controller                        # noqa: E402

NAMES = ("肩转", "肩抬", "肘", "腕仰", "腕转", "爪")


def main() -> int:
    ap = argparse.ArgumentParser(description="真机：从当前姿态匀速走到中立位")
    ap.add_argument("--port", default="COM10", help="从臂所在串口（先跑 identify_arms.py 确认）")
    ap.add_argument("--go", action="store_true", help="真的动。不给就只读只算")
    ap.add_argument("--speed", type=int, default=0,
                    help="舵机运行速度 0-32767。0=舵机自身最高速（默认）。"
                         "**高帧率下每步位移本来就小，让舵机自己最快跟上**，"
                         "限速反而会让它永远落后于指令")
    ap.add_argument("--hz", type=float, default=30.0,
                    help="发指令的频率。写一次要 21ms，所以上限约 45Hz")
    ap.add_argument("--jvel", type=float, default=0.20, help="关节速度上限（弧度/秒）")
    ap.add_argument("--jacc", type=float, default=0.80, help="关节加速度上限（弧度/秒²）")
    ap.add_argument("--settle", type=int, default=15, help="到点后再连发几步让舵机跟上")
    ap.add_argument("--target", default=None, help="目标点 x,y,z，逗号分隔。默认中立位")
    ap.add_argument("--torque", action="store_true",
                    help="走之前显式使能力矩（臂现在是软的才需要）")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    target = (np.array([float(v) for v in args.target.split(",")]) if args.target
              else np.asarray(P.WAYPOINT, dtype=np.float64))

    ctrl = open_controller(args.port)
    arm = RealArm(ctrl=ctrl, speed=args.speed)
    try:
        q_now = arm._ticks_to_qpos(arm.read_ticks())
        tcp = arm.tcp()
        # 目标关节角：**保持此刻的朝向**解位置。不夹带姿态改变，就没有第二个矛盾。
        q_goal = arm._ik(target, arm._fk(q_now)[1], q_now)
        q_goal[5] = q_now[5]        # 夹爪跟过去，这一步不动它

        print(f"串口 {args.port}   舵机速度 {args.speed}（0=最高）   发指令 {args.hz:.0f} Hz")
        print(f"此刻 关节(度) {np.round(np.degrees(q_now), 1)}")
        print(f"      TCP(m)  {np.round(tcp, 4)}")
        print(f"目标 {np.round(target, 3)}   直线距离 {np.linalg.norm(target - tcp) * 1000:.1f}mm"
              f"   逆解到位后差 {arm.reach_residual(target) * 1000:.1f}mm")
        print(f"目标 关节(度) {np.round(np.degrees(q_goal), 1)}")
        move = np.degrees(np.abs(q_goal - q_now))
        mv = "  ".join(f"{NAMES[i]} {move[i]:.1f}°" for i in range(6) if move[i] > 0.5)
        print("每个关节要动：" + (mv if mv else "都不动"))

        traj = build([q_now, q_goal], vmax=args.jvel, amax=args.jacc, density=0.01)
        ts, qs, _ = traj.sample(1.0 / args.hz)
        print(f"轨迹 {traj.duration:.1f} 秒，{len(ts)} 个点"
              f"（峰值关节速度 {args.jvel} 弧度/秒 = {np.degrees(args.jvel):.1f}°/秒）")

        if not args.go:
            print("\n没给 --go，一步不动。上面这份没问题再加 --go。")
            return 0

        print("\n=== 开始走 ===")
        if args.torque:
            ctrl.sync_enable_torque(list(SERVO_IDS))
            print("力矩：已使能")

        dt = 1.0 / args.hz
        t0 = time.time()
        for k, q in enumerate(qs):
            arm.write_ticks(arm._qpos_to_ticks(q))
            if k % max(int(args.hz), 1) == 0:
                print(f"  点 {k:5d}/{len(ts)}  指令(度) {np.round(np.degrees(q), 1)}"
                      f"  （{time.time() - t0:.1f}s）")
            # 按**绝对时刻**睡，不是每次睡固定时长 —— 写一次要 21ms，
            # 每次 `sleep(dt)` 会把这 21ms 累加进去，越走越慢。
            nap = (k + 1) * dt - (time.time() - t0)
            if nap > 0:
                time.sleep(nap)

        for _ in range(max(args.settle, 0)):      # 收尾：同一条指令连发，让舵机追平
            arm.write_ticks(arm._qpos_to_ticks(q_goal))
            time.sleep(dt)

        el = time.time() - t0
        j = np.degrees(arm._ticks_to_qpos(arm.read_ticks()))
        print(f"\n走完 {el:.1f} 秒。读回 关节(度) {np.round(j, 1)}")
        print(f"      TCP(m)  {np.round(arm.tcp(), 4)}  离目标 "
              f"{np.linalg.norm(arm.tcp() - target) * 1000:.1f}mm")
        print("力矩仍使能 —— 臂会自己撑住这个姿势。要放开请断电。")
    except KeyboardInterrupt:
        print("\n打断。**停在当前位置，不松力矩。**")
        return 130
    finally:
        # 只关串口，不写任何寄存器 —— 力矩保持，臂留在原地。
        ctrl.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
