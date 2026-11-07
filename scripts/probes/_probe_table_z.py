"""量桌面在基座系里到底多高：回初始位置，从那儿垂直往下压，压不动了就是桌面。

**为什么要量这个。** `configs/real.yaml` 的 `table_z_mm: -9` 是从手眼标定的
`validated.target_pos_in_base_m` 抄来的，那是**靶标面**的高度，不是桌面 —— 平板本身
有厚度，这一截从来没算进去。这个数一路被当成「桌面」用在两个地方：下探终点的高度
（爪尖要停在桌面之上半个方块高），和感知把画面里的方块反推成水平坐标时用的那个平面。

**2026-10-15 这一版为什么重写。** 上一版把两套 TCP 口径混着用了，解出来的目标是自相
矛盾的，逆解交给一个顶死限位的折叠构型，臂把自己别住了：

    quat_des = _vertical(arm._fk(q_now)[1])   ← `_fk` 是**官方 site 口径**，
                                                它的工具 z 是开合轴，不是手指方向
    arm._ik([x, y, z], quat_des, q_seed)      ← `_ik` 只要官方口径的位姿，
                                                这里喂的是 TCP 口径的点，没走 `_site_pose`

上一版的日志（`out/probe_table_z.log` 旧档）里，臂从第 1 步起 5 号就钉在 −157.1°
（它的下限 −157.2°）、2 号 −100°、3 号从 74.4° 爬到 96.4°（上限 95°），「压不动了」
判在**实际 z 往回升**的三步上 —— 那不是桌面，是臂别住了。

这一版照 `skills/pick_real.py` 的 `_descend` 写，三处不能省：

1. 姿态一律走 **TCP 口径**：`_vertical(arm.quat())` 取竖直目标，`_site_pose` 换算成
   官方口径再交 `_ik`。`pick_real.py:262` 那两行注释就是这件事。
2. **一段的点一次解好再走**（`CHUNK` 个点），不一步一解。相邻点拿上一个点的解当种子。
3. 每走一段之前**先看解到哪儿了**：位置残差、工具轴离竖直多少度、六个关节各离限位
   多少。任一条不过就不走 —— 上一版从头到尾只打印了位置残差（而且那个数还是拿
   `reach_residual` 算的，它用的是**此刻的**朝向，跟真正要走的姿态不是一回事）。

**起点用初始位置**（`real_home.RAW_HOME`）。那儿手指本来就朝下（实测离正下 0.5°），
所以整段只是垂直下压，没有横移、没有重新摆姿态。回初始位置那一段用一半的速度走，
因为起点可能是个别住的姿势。

**为什么不用人喊停。** 舵机是位置控制的：爪尖顶住桌面之后，指令角度继续往下走，
实际角度不再跟 —— 「读数不跟着降了」就是碰上了。判据取「连着 2 段实际降幅不到指令
降幅的四成」，一段 6mm。

**姿态必须先锁竖直，不然量出来的不是桌面。** 4 号(wrist_flex) 角度不同，爪子的朝向
就不同；爪子一歪，最先碰桌面的就不是爪尖，而是某个指的边角（用户 2026-10-15 提醒：
4 号 30° 和 60° 落地的读数完全不一样）。所以整段下降都锁着「工具 z 轴朝下」。

用法（在 G:\\sim2real\\so101-real-harness 下）：

    G:/anaconda/envs/sim2real/python.exe _probe_table_z.py          # 只看，不动
    G:/anaconda/envs/sim2real/python.exe _probe_table_z.py --go     # 回初始位置再往下压

**压点就是初始位置爪尖的正下方**，不给 `--x/--y` 就不横移。**底下不能有方块。**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

import real.arm                                                  # noqa: E402,F401
from real import joint_move as jm                                # noqa: E402
from real.arm import RealArm                                     # noqa: E402
from servo_bridge import open_controller                         # noqa: E402
from skills.pick_real import _vertical                           # noqa: E402

STEP_MM = 2.0        # 每一步降多少毫米
CHUNK = 3            # 一段解几个点（3 × 2mm = 6mm 走一段）
MISS_RATIO = 0.4     # 一段实际降幅不到指令降幅的这个比例 = 这一段没走
MISS_N = 2           # 连着这么多段没走 = 碰上了
TILT_MAX = 5.0       # 解出来的姿态离竖直超过这么多度就不走（度）
MARGIN_MIN = 3.0     # 任何关节离限位不到这么多度就不走（度）
RESID_MAX = 2.0      # 一段里最坏那点离目标多少毫米就不走（毫米）
FLOOR_MM = -60.0     # 压到这个高度还没碰上就收手
HOME_JVEL = 0.25     # 回初始位置的速度（弧度/秒）—— 起点可能别着，用一半
HOME_JACC = 0.8


def _tilt_deg(arm, q) -> float:
    """关节角 → 工具轴离正下多少度。**走 TCP 口径**（`arm.quat()` 那一套）。"""
    _, off = _vertical(arm._sim_pose(*arm._fk(q))[1])
    return off


def _margins_deg(arm, q) -> np.ndarray:
    """六个关节各离自己的限位还有多少度（取上下两侧里小的那个）。"""
    return np.degrees(np.minimum(q - arm.q_lo, arm.q_hi - q))


def _plan(arm, xy, z_start, n, step_m, quat0, grip_q):
    """从 `z_start` 往下解 `n` 个点，相邻点拿上一个点的解当种子。返回 (点表, 最坏残差)。

    种子链从这里断开就没有意义 —— `_descend` 的注释写着「每一点拿上一个点的解当种子
    （相邻两点只差 5mm，解不会跳）」，一步一解、每步都从**真身读回来的姿态**重新起头的
    做法正是它要避免的。
    """
    pts, worst = [], 0.0
    q_prev = arm._ticks_to_qpos(arm.read_ticks())
    for i in range(1, int(n) + 1):
        goal = np.array([xy[0], xy[1], z_start - i * step_m], dtype=np.float64)
        q = arm._ik(*arm._site_pose(goal, quat0), q_prev)
        q[5] = float(grip_q)                       # 夹爪这一趟不动它
        got = arm._sim_pose(*arm._fk(q))[0]
        worst = max(worst, float(np.linalg.norm(got - goal)))
        pts.append(q)
        q_prev = q
    return pts, worst


def main() -> int:
    ap = argparse.ArgumentParser(description="压到桌面，量桌面在基座系的高度")
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--go", action="store_true", help="真的动。不给就只读只算")
    ap.add_argument("--x", type=float, default=None, help="压点的 x（毫米，基座系）")
    ap.add_argument("--y", type=float, default=None, help="压点的 y（毫米，基座系）")
    ap.add_argument("--step", type=float, default=STEP_MM, help="每一步降多少毫米")
    ap.add_argument("--speed", type=int, default=0)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ctrl = open_controller(args.port)
    arm = RealArm(ctrl=ctrl, speed=args.speed)
    try:
        q_now = arm._ticks_to_qpos(arm.read_ticks())
        tip = np.asarray(arm.tcp(), dtype=np.float64)
        print(f"此刻 关节(度) {np.round(np.degrees(q_now), 1)}")
        print(f"     爪尖 {np.round(tip * 1000, 1)}mm"
              f"   工具轴离正下 {_tilt_deg(arm, q_now):.1f}°")

        if not args.go:
            print("\n没给 --go，一步不动。确认爪尖正下方是空的，再加 --go。")
            return 0

        # ---- 回初始位置。**不是可选的**：起点是个别住的姿势时，从它直接往下解，
        # 解出来的构型也是别着的（2026-10-15 那一版的实测）。
        print("\n回初始位置（一半速度）")
        jm.goto_q(arm, jm.home_q(arm), jvel=HOME_JVEL, jacc=HOME_JACC)
        q_home = arm._ticks_to_qpos(arm.read_ticks())
        tip = np.asarray(arm.tcp(), dtype=np.float64)
        quat0, off0 = _vertical(arm.quat())
        print(f"到了 关节(度) {np.round(np.degrees(q_home), 1)}")
        print(f"     爪尖 {np.round(tip * 1000, 1)}mm   工具轴离正下 {off0:.1f}°")
        if off0 > TILT_MAX:
            print(f"\n初始姿态离竖直 {off0:.1f}°（上限 {TILT_MAX:.0f}°）—— 不压了。"
                  f"爪子是歪的，压下去最先碰桌面的不是爪尖，量出来的不是桌面。")
            return 1

        x = float(args.x) / 1000.0 if args.x is not None else float(tip[0])
        y = float(args.y) / 1000.0 if args.y is not None else float(tip[1])
        step_m = float(args.step) / 1000.0
        grip_q = float(q_home[5])
        print(f"\n压点 ({x * 1000:.1f}, {y * 1000:.1f})mm"
              f"   从 z={tip[2] * 1000:.1f}mm 往下压   每步 {args.step:.0f}mm，"
              f"一段 {CHUNK} 步")

        print("\n=== 开始压 ===")
        z_cmd = float(tip[2])
        missed = 0
        contact_z = None
        n_chunk = 0
        while z_cmd * 1000.0 > FLOOR_MM:
            n_chunk += 1
            pts, worst = _plan(arm, (x, y), z_cmd, CHUNK, step_m, quat0, grip_q)
            q_end = pts[-1]
            marg = _margins_deg(arm, q_end)
            marg[5] = 99.0                       # 夹爪这一趟不动它，不参与判据
            tilt = _tilt_deg(arm, q_end)
            j = int(np.argmin(marg))
            if worst * 1000.0 > RESID_MAX:
                print(f"  第 {n_chunk:2d} 段：解不到 —— 最坏一点还差 "
                      f"{worst * 1000:.1f}mm（上限 {RESID_MAX:.0f}mm），收手")
                print(f"    那点解成 {np.round(np.degrees(q_end), 1)}")
                break
            if tilt > TILT_MAX:
                print(f"  第 {n_chunk:2d} 段：姿态歪了 {tilt:.1f}°"
                      f"（上限 {TILT_MAX:.0f}°），收手")
                print(f"    那点解成 {np.round(np.degrees(q_end), 1)}")
                break
            if float(marg[j]) < MARGIN_MIN:
                print(f"  第 {n_chunk:2d} 段：{j + 1} 号离限位只剩 "
                      f"{float(marg[j]):.1f}°（下限 {MARGIN_MIN:.0f}°），收手")
                print(f"    那点解成 {np.round(np.degrees(q_end), 1)}"
                      f"   六个关节离限位 {np.round(marg, 1)} 度")
                break

            jm.goto_path(arm, pts, settle=0.3, hold=6)
            z_now = float(np.asarray(arm.tcp(), dtype=np.float64)[2])
            drop = (z_cmd - z_now) * 1000.0
            want = CHUNK * args.step
            print(f"  第 {n_chunk:2d} 段  指令 z={z_cmd * 1000:7.1f}mm"
                  f" → 实际 z={z_now * 1000:7.1f}mm  降了 {drop:5.2f}mm"
                  f"（该降 {want:.1f}）  姿态歪 {tilt:.1f}°"
                  f"  关节 {np.round(arm.angles(), 1)}")
            if drop < want * MISS_RATIO:
                missed += 1
                if missed >= MISS_N:
                    contact_z = z_now
                    print(f"\n**降不动了** —— 连着 {missed} 段实际 z 几乎没变。")
                    break
            else:
                missed = 0
            z_cmd = z_now

        q_end = arm._ticks_to_qpos(arm.read_ticks())
        tip_end = np.asarray(arm.tcp(), dtype=np.float64)
        marg_end = _margins_deg(arm, q_end)
        print(f"\n收手时 关节(度) {np.round(np.degrees(q_end), 1)}")
        print(f"       爪尖 {np.round(tip_end * 1000, 1)}mm"
              f"   工具轴离正下 {_tilt_deg(arm, q_end):.1f}°   4 号 "
              f"{np.degrees(q_end[3]):.1f}°")
        print(f"       六个关节离限位 {np.round(marg_end, 1)} 度（太小就是又快顶上了）")
        if contact_z is None:
            print(f"\n没碰上东西（到 {FLOOR_MM:.0f}mm 就收手了）"
                  f"—— 桌面不在这个范围里，或者底下是空的。")
            return 1
        print(f"\n【要的数】爪尖停在基座系 z = {contact_z * 1000:+.1f}mm 就降不动了。")
        print(f"  配置里现在写的是 table_z_mm: -9.0"
              f"   两者差 {contact_z * 1000 + 9.0:+.1f}mm")
        print("  **爪尖那一刻的姿态和上面的 4 号角度一起看** —— 这两条同时成立，"
              "读数才是桌面。")
        print("力矩仍使能 —— 臂会自己撑住。要放开请断电，或跑 _ref_pose.py unlock。")
        return 0
    except KeyboardInterrupt:
        print("\n打断。**停在当前位置，不松力矩。**")
        return 130
    finally:
        ctrl.close()


if __name__ == "__main__":
    sys.exit(main())
