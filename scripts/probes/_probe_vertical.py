"""离线探针：纠位那一步 9° 的垂直残差，是逆解权重欠的还是执行欠的。

不碰硬件：只建运动学模型，拿日志里读回来的六个关节当起点，朝同一个悬停点解逆解，
比较不同 `ow` / 迭代次数下「位置误差」和「工具轴离垂直」这两个数。

日志那一眼（2026-10-18 21:01）：
    走到 [256.5 76.7 67.8]mm  关节 [-17.4 4.4 -3.9 80.5 -107.2 -8.9]
    目标是悬停点 [256.6 74.8 68.0]mm，报「爪尖离垂直 9.0°」。
"""
from __future__ import annotations

import numpy as np

import real.arm as RA
from skills.pick_real.pose import _vertical


def build():
    a = RA.RealArm.__new__(RA.RealArm)
    a.model, a.data, a.site, a.adr = RA.load_model(RA.MJCF)
    lim = RA.joint_limits(a.model)
    a.q_lo = np.array([lim[n][0] for n in RA.JOINT_ORDER], dtype=np.float64)
    a.q_hi = np.array([lim[n][1] for n in RA.JOINT_ORDER], dtype=np.float64)
    a.ow = RA.OW
    return a


def pose(a, q):
    return a._sim_pose(*a._fk(q))


def main():
    a = build()
    q0 = np.radians([-17.4, 4.4, -3.9, 80.5, -107.2, -8.9])
    pos0, quat0 = pose(a, q0)
    off0 = _vertical(quat0)[1]
    print(f"起点（日志读回）：TCP {np.round(pos0 * 1000, 1)}mm  "
          f"离垂直 {off0:.1f}°")

    goal = np.array([0.2566, 0.0748, 0.0680])
    q_down, off_tgt = _vertical(quat0)
    print(f"目标：{np.round(goal * 1000, 1)}mm  目标朝向离垂直应在 0"
          f"（相对此刻定义，{off_tgt:.1f}° 的偏差会被吃掉）")

    for iters in (3, 6, 12):
        for ow in (0.05, 0.1, 0.2, 0.5, 1.0):
            q = a._ik(*a._site_pose(goal, q_down), q0,
                      iters=iters, ow=ow)
            p, qt = pose(a, q)
            err = float(np.linalg.norm(goal - p)) * 1000.0
            off = _vertical(qt)[1]
            print(f"  iters {iters:2d}  ow {ow:4.2f} → 位置差 {err:6.1f}mm  "
                  f"离垂直 {off:5.1f}°  4号 {np.degrees(q[3]):+6.1f}° "
                  f"（上限 {np.degrees(a.q_hi[3]):+.1f}°）")


if __name__ == "__main__":
    main()
