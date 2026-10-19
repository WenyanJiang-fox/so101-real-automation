"""单个关节开环跟随测试：给一个明确的角变化，读回它到底往哪走、走了多少。

**为什么要这个。** `_probe_joint_dir.py` 把 12 份日志里 202 趟走位的带符号指令还原
出来，发现 2 号（肩抬）17%、3 号（肘）10% 的趟里**实际方向跟指令相反**，而且
**全是同一个方向**：指令往下、实际往上，反向的 0 次例外。模型里 2 号 +1° 让 TCP 降
2.32mm、3 号 +1° 降 2.14mm —— 也就是「指令往下」就是模型在要求**把 TCP 往上抬**。
所以那些反向趟实际是「模型说抬起来，它掉下去」，跟 run 69 抬高度那一步同一件事。

日志只能看到结果，看不到原因。这个脚本把动作拆到最小：**一次只让一个关节走**，
其余五个关节写回原位，看它跟不跟、跟多少、跟错方向没有。

**它不回答「为什么」。** 它只把「跟得上 / 跟不上 / 反向」这三种情况分开：
    * 大幅指令（±2°）也反向  → 不是死区的问题，是更底层的东西（标定/极性/机械）
    * 只有小幅指令（±0.5°）反向、大幅正常 → 小指令被负载吃掉
    * 全都正常                  → 那 202 趟里的反常跟「单关节」这个拆法无关，
                                   得回到多关节同时动的场景里查

用法（在 G:\\sim2real\\so101-real-harness 下）：

    G:/anaconda/envs/sim2real/python.exe _probe_joint_track.py           # 只看，不动
    G:/anaconda/envs/sim2real/python.exe _probe_joint_track.py --go      # 真的动

不给 `--go` 就把计划打出来然后退出。**每一趟只动一个关节，幅度 ≤3°，其余关节写回
原位**；不合爪、不碰方块、不回初始位置（就从它现在待的地方开始）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from real import joint_move as jm                                 # noqa: E402
from real.arm import RealArm                                      # noqa: E402
from servo_bridge import open_controller                          # noqa: E402

NAMES = ["1号肩转", "2号肩抬", "3号肘", "4号腕俯仰", "5号腕转", "6号夹爪"]
MAX_DEG = 3.0


def one_move(arm, q_base, joint: int, delta_deg: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """让 `joint` 从**此刻读回来的位置**走 `delta_deg`，其余五个关节钉在 `q_base`。

    返回 `(走之前读回的, 走之后读回的)`。**必须两个都返回。** 第一版只返回走完的角，
    再拿它减 `q_base` 当「实际走了多少」—— 可 `q_base` 是**最开始**那个姿态，而
    「回原位」那一趟并不总能回得去（第一版日志里那条 ⚠）。一旦回不去，后面的
    `走完 − q_base` 里就混进了累积偏移，读出来的「跟到 545%」全是假的。
    真正该减的是**这一趟走之前**的读数。

    **起点不给 `q_from`**：`goto_q` 自己去读当前关节角。给死 `q_base` 的话，「回原位」
    那一趟（`delta_deg=0`）起点和终点就是同一个点，`core.trajectory.build` 会当场抛
    「waypoints 全是同一个点，没有路可走」—— 第一版就是这么死在第二次走位上的。
    """
    q_now = arm._ticks_to_qpos(arm.read_ticks())
    q_goal = np.array(q_base, dtype=np.float64, copy=True)
    q_goal[joint] = float(np.clip(q_base[joint] + np.radians(delta_deg),
                                  arm.q_lo[joint], arm.q_hi[joint]))
    if np.allclose(q_now, q_goal, atol=1e-6):
        return q_now, q_now                # 已经在终点上，不用走
    return q_now, jm.goto_q(arm, q_goal)


def main() -> int:
    ap = argparse.ArgumentParser(description="单关节开环跟随测试")
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--go", action="store_true", help="真的动。不给就只读只算")
    ap.add_argument("--joints", default="2,3", help="测哪几号关节（1~6，逗号分隔）")
    ap.add_argument("--deltas", default="-2,-0.5,2,0.5", help="每个关节测这几个角（度）")
    ap.add_argument("--reps", type=int, default=3, help="每个组合重复几次")
    ap.add_argument("--watch", type=float, default=0.0,
                    help="每种组合的第一次走完，再每 0.5 秒读一次、连读这么秒（0=不读）")
    ap.add_argument("--speed", type=int, default=0)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    joints = [int(x) - 1 for x in args.joints.split(",")]
    deltas = [float(x) for x in args.deltas.split(",")]
    for d in deltas:
        if abs(d) > MAX_DEG:
            print(f"幅度 {d}° 超过 {MAX_DEG}° 的上限，不干")
            return 2

    ctrl = open_controller(args.port)
    arm = RealArm(ctrl=ctrl, speed=args.speed)
    try:
        q_base = arm._ticks_to_qpos(arm.read_ticks())
        tcp0 = np.asarray(arm.tcp(), dtype=np.float64)
        print(f"起点 关节(度) {np.round(np.degrees(q_base), 1)}")
        print(f"     TCP {np.round(tcp0 * 1000, 1)}mm")
        print(f"\n计划：关节 {[NAMES[j] for j in joints]} × 幅度 {deltas}° × "
              f"每种 {args.reps} 次（每次走完回原位）")
        print(f"     合计 {len(joints) * len(deltas) * args.reps * 2} 趟走位，"
              f"约 {len(joints) * len(deltas) * args.reps * 2 * 1.2:.0f} 秒\n")

        if not args.go:
            print("没给 --go，一步不动。确认臂周围是空的，再加 --go。")
            return 0

        tally: dict[tuple[int, float], list[float]] = {}
        backs: list[float] = []
        for j in joints:
            for d in deltas:
                for k in range(args.reps):
                    q_a, q_b = one_move(arm, q_base, j, d)
                    got = float(np.degrees(q_b[j] - q_a[j]))   # **相对这一趟的起点**
                    tcp1 = np.asarray(arm.tcp(), dtype=np.float64)
                    same = (np.sign(got) == np.sign(d)) if abs(got) > 0.15 else None
                    mark = "—" if same is None else ("同" if same else "**反**")
                    print(f"  {NAMES[j]} 第{k + 1}次  指令 {d:+.2f}°  实际 {got:+.2f}°"
                          f"  {mark}   跟到 {100 * got / d:6.1f}%"
                          f"   离起点姿态 {np.degrees(q_b[j] - q_base[j]):+5.2f}°"
                          f"   TCP z {tcp1[2] * 1000:+6.1f}mm（起点 "
                          f"{tcp0[2] * 1000:+6.1f}）")
                    tally.setdefault((j, d), []).append(got)
                    if args.watch > 0 and k == 0:
                        print(f"      走完盯 {args.watch:.0f} 秒（每 0.5 秒读一次，"
                              f"看它停不停、往哪边漂）")
                        n_t = int(args.watch / 0.5)
                        for it in range(n_t):
                            time.sleep(0.5)
                            # **必须把缓存作废。** `read_ticks` 有缓存，写完之后读一次
                            # 就转成「不脏」，再读多少次都返回同一个数 —— 第一版盯着看
                            # 会看到一条完全水平的线，那是假静止。
                            arm._dirty = True
                            qw = arm._ticks_to_qpos(arm.read_ticks())
                            print(f"        +{(it + 1) * 0.5:4.1f}s  {NAMES[j]} "
                                  f"{np.degrees(qw[j] - q_b[j]):+6.2f}°"
                                  f"（相对刚走完）  TCP z "
                                  f"{np.asarray(arm.tcp())[2] * 1000:+6.1f}mm")
                    q_c, q_d = one_move(arm, q_base, j, 0.0)   # 回原位
                    back = float(np.degrees(q_d[j] - q_c[j]))
                    backs.append(back)
                    if abs(back) > 0.3:
                        print(f"     ⚠ 回原位这一趟：要动 {np.degrees(q_base[j] - q_c[j]):+.2f}°"
                              f"（回到起点姿态），实际走了 {back:+.2f}°")

        print("\n== 汇总（实际 = 「走完的读数 − 走之前的读数」，跟 q_base 无关）==")
        for (j, d), gots in tally.items():
            g = np.array(gots)
            n_same = int(np.sum(np.sign(g) == np.sign(d)))
            n_flat = int(np.sum(np.abs(g) <= 0.15))
            print(f"  {NAMES[j]} 指令 {d:+.2f}°：{len(g)} 次里 方向对 {n_same}、"
                  f"反向 {len(g) - n_same - n_flat}、几乎没动 {n_flat}；"
                  f"实际中位 {np.median(g):+.2f}°")
        b = np.array(backs)
        print(f"\n  「回原位」{len(b)} 趟：指令幅度中位 "
              f"{np.median(np.abs(b)):.2f}°，超过 0.3° 的有 "
              f"{int(np.sum(np.abs(b) > 0.3))} 趟、超过 1° 的有 "
              f"{int(np.sum(np.abs(b) > 1.0))} 趟")
        return 0
    finally:
        try:
            ctrl.close()
        except Exception:                                          # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
