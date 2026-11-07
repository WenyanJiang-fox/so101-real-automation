"""离线探针：把 run 69 抬高度那一步的关节角拿模型正解/逆解各算一遍。**不碰臂、不开串口。**

问的是：`G:\\sim2real\\so101-real-harness\\out\\run_real_69.log` 里那三行对不上 ——
关节走了指令的 80%~100%，TCP 读回却离目标更远了。

    G:/anaconda/envs/sim2real/python.exe -u _probe_ik_fk.py

三个数全从日志抄，一个都没动：
    A 抬高度起点 关节 [-4.4 -3.2 34.7 60.9 -75.0 99.3]  日志报 TCP [198.5 20.7 -0.8]mm
    B 第 1 趟走完 关节 [-4.4 -2.8 34.9 61.4 -75.0 99.3]  日志报 TCP [195.4 20.4 -2.0]mm
    C 第 2 趟走完 关节 [-4.4 -2.3 35.1 61.9 -75.0 99.3]  日志报 TCP [192.2 20.4 -3.2]mm
    目标 [198.5 20.7 +0.8]mm（抬高度那一步的 goal，日志原样）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from real.arm import RealArm                                       # noqa: E402


class FakeCtrl:
    """只为了把 `RealArm` 建起来（它要读一次标定起点），**一个字节都不发到总线**。"""

    def __init__(self) -> None:
        self.ticks: dict[int, int] = {}

    def sync_read_positions(self, ids):
        return {int(i): int(self.ticks.get(int(i), 2048)) for i in ids}

    def sync_set_positions(self, goal, speed=0):
        raise AssertionError("探针脚本不许写舵机")


def show(tag: str, arm, q_deg) -> np.ndarray:
    q = np.radians(np.asarray(q_deg, dtype=np.float64))
    pos_s, quat_s = arm._fk(q)
    pos_c, quat_c = arm._sim_pose(pos_s, quat_s)
    print(f"  {tag}  正解 → TCP {np.round(pos_c * 1000, 1)}mm")
    return pos_c, quat_c


def main() -> None:
    arm = RealArm(ctrl=FakeCtrl())
    print(f"模型 {RealArm.__module__}  夹爪行程 {np.degrees(arm.grip_lo):.1f}°"
          f"~{np.degrees(arm.grip_hi):.1f}°，逆解迭代 {arm._ik.__defaults__[0]} 次\n")

    A = [-4.4, -3.2, 34.7, 60.9, -75.0, 99.3]
    B = [-4.4, -2.8, 34.9, 61.4, -75.0, 99.3]
    C = [-4.4, -2.3, 35.1, 61.9, -75.0, 99.3]
    goal = np.array([0.1985, 0.0207, 0.0008])

    print("== 1. 把日志里那三组关节角正解一遍，看跟日志报的 TCP 对不对得上 ==")
    print("     （日志报：A [198.5 20.7 -0.8]  B [195.4 20.4 -2.0]  C [192.2 20.4 -3.2]）")
    posA, quatA = show("A 起点  ", arm, A)
    show("B 第1趟", arm, B)
    show("C 第2趟", arm, C)

    print(f"\n== 2. 从 A 那组角度、同一个目标 {np.round(goal * 1000, 1)}mm 重解一次逆解 ==")
    q0 = np.radians(np.asarray(A, dtype=np.float64))
    tgt_s = arm._site_pose(goal, quatA)
    for iters in (3, 50):
        q_goal = arm._ik(*tgt_s, q0, iters=iters)
        q_goal[5] = arm.grip_hi                      # 跟 `_goto` 一样：夹爪那一维直接给行程端点
        p_s, qu_s = arm._fk(q_goal)
        p_c, _ = arm._sim_pose(p_s, qu_s)
        print(f"  迭代 {iters:2d} 次 → 关节 {np.round(np.degrees(q_goal), 1)}")
        print(f"              要动 {np.round(np.degrees(np.abs(q_goal - q0)), 1)} 度"
              f"  （日志第 1 趟是 [0. 0.5 0.2 0.6 0. 0.7]）")
        print(f"              正解 TCP {np.round(p_c * 1000, 1)}mm"
              f"  离目标 {np.linalg.norm(p_c - goal) * 1000:.1f}mm")

    print("\n== 3. 指令方向 vs 实际方向（日志那句 `要动` 取的是绝对值，看不出正负）==")
    q_goal = arm._ik(*tgt_s, q0)
    q_goal[5] = arm.grip_hi
    a = np.radians(np.asarray(A, dtype=np.float64))
    b = np.radians(np.asarray(B, dtype=np.float64))
    cmd = np.degrees(q_goal - a)
    got = np.degrees(b - a)
    print(f"  指令（带符号，度）：{np.round(cmd, 2)}")
    print(f"  实际（带符号，度）：{np.round(got, 2)}")
    print(f"  日志那句「要动」  ：{np.round(np.abs(cmd), 2)}")
    same = np.sign(cmd) == np.sign(got)
    print(f"  方向一致？         ：{['1号', '2号', '3号', '4号', '5号', '6号']} "
          f"{['同' if s else '反' if abs(c) > 1e-6 else '—' for s, c in zip(same, cmd)]}")

    print("\n== 4. 模型认为「从 A 沿直线走向逆解终点」这一路上 z 怎么变 ==")
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = a + t * (q_goal - a)
        p_c, _ = arm._sim_pose(*arm._fk(q))
        print(f"  走到 {t * 100:3.0f}%  TCP {np.round(p_c * 1000, 1)}mm"
              f"  离目标 {np.linalg.norm(p_c - goal) * 1000:.1f}mm")
    print(f"  实际那一组 B  TCP {np.round(arm._sim_pose(*arm._fk(b))[0] * 1000, 1)}mm")
    print(f"  A 离目标 {np.linalg.norm(posA - goal) * 1000:.1f}mm"
          f"   B 离目标 {np.linalg.norm(arm._sim_pose(*arm._fk(b))[0] - goal) * 1000:.1f}mm")


if __name__ == "__main__":
    main()
