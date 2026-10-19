"""离线：同一个 TCP 目标，换不同种子解逆解，看还有没有「4 号不顶 95° 限位」的解。

数据全部抄 run16 日志，一个数没改：
    悬停点（纠位目标）TCP [256.8, 78.1, 74.0]mm
    纠位走完的姿态  关节 [-17.6, 19.9, -26.5, 94.8, -110.0]  TCP [262.0, 79.4, 68.1]mm
    下探到底        关节 [-21.2, 28.6, -22.0, 84.9, -138.4]   TCP z=+24.6mm
不碰臂、不开串口。

    G:/anaconda/envs/sim2real/python.exe -u _probe_pose_search.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from real.arm import RealArm                                       # noqa: E402


class FakeCtrl:
    def sync_read_positions(self, ids):
        return {int(i): 2048 for i in ids}

    def sync_set_positions(self, *a, **k):
        raise AssertionError("探针不写舵机")

    def sync_enable_torque(self, ids):
        return True


HOME_JOINTS_DEG = [-17.6, 19.9, -26.5, 94.8, -110.0, 99.3]
HOVER = np.array([0.2568, 0.0781, 0.0740])
GRASP_JOINTS_DEG = [-21.2, 28.6, -22.0, 84.9, -138.4, 99.3]


def evaluate(arm, goal, quat, q0, ow=0.05):
    q = arm._ik(*arm._site_pose(goal, quat), q0, ow=ow)
    p, qu = arm._fk(q)
    p_c, _ = arm._sim_pose(p, qu)
    err = float(np.linalg.norm(p_c - goal)) * 1000
    margin = np.degrees(np.minimum(q - arm.q_lo, arm.q_hi - q))
    return q, err, margin


def main() -> None:
    arm = RealArm(ctrl=FakeCtrl())
    q_ref = np.radians(np.asarray(HOME_JOINTS_DEG, dtype=np.float64))
    _, quat = arm._fk(q_ref)
    _, quat_tcp = arm._sim_pose(arm._fk(q_ref)[0], quat)
    print(f"目标悬停点 {np.round(HOVER * 1000, 1)}mm，姿态取 run16 纠位后的实测姿态；"
          f"wrist_flex 限位 ±{np.degrees(arm.q_hi[3]):.0f}°\n")

    seeds = []
    for j2 in (-40, -20, 0, 20, 40):
        for j3 in (-60, -30, 0, 30, 60):
            for j4 in (40, 70, 90):
                s = q_ref.copy()
                s[1] = np.radians(j2)
                s[2] = np.radians(j3)
                s[3] = np.radians(j4)
                seeds.append(s)

    rows = []
    for s in seeds:
        q, err, margin = evaluate(arm, HOVER, quat_tcp, s)
        rows.append((err, float(margin[3]), float(margin.min()), q, s))
    rows.sort(key=lambda r: (r[0] > 5.0, -r[1], r[0]))

    print("残差<5mm 的解里，按 4 号余量从大到小（只列前 8 条）：")
    shown = 0
    for err, m4, mall, q, s in rows:
        if err > 5.0:
            continue
        shown += 1
        print(f"  残差 {err:5.1f}mm  4号 {np.degrees(q[3]):6.1f}°（余量 {m4:5.1f}°）"
              f"  最小余量 {mall:5.1f}°  解 {np.round(np.degrees(q[:5]), 1)}")
        if shown >= 8:
            break
    if shown == 0:
        print("  一条都没有")
    best = rows[0]
    print(f"\n总共 {len(rows)} 个种子，残差<5mm 的 {sum(1 for r in rows if r[0] <= 5.0)} 条；"
          f"最好残差 {best[0]:.1f}mm")

    print("\n== 把 wrist_flex 上限放宽到标定范围（±105°）再解一次 ==")
    hi0 = arm.q_hi.copy()
    arm.q_hi[3] = np.radians(105.0)
    for tag, s in (("run16 纠位后姿态", q_ref),
                   ("下探到底姿态", np.radians(np.asarray(GRASP_JOINTS_DEG)))):
        q, err, margin = evaluate(arm, HOVER, quat_tcp, s)
        print(f"  {tag:16s} → 残差 {err:5.1f}mm  4号 {np.degrees(q[3]):6.1f}°"
              f"  解 {np.round(np.degrees(q[:5]), 1)}")
    arm.q_hi[:] = hi0


if __name__ == "__main__":
    main()
