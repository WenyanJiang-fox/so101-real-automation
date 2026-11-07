"""离线复算 2026-10-18 01:54 那一次大步走位：**模型里这条指令到底到不到点**。

只读运动学模型（`RealArm` + `StubCtrl`，不碰串口、不动臂）。关节角都抄自那一趟面板
日志：

    第 2 趟前   [ -11.5   2.2  26.1  67.1 -141.2  44.6]
    第 2 趟指令 [ -11.5   5.7  21.5  70.6 -141.2  44.6]   （打印是绝对值，符号按响应比推）
    第 2 趟走完 [ -11.5   5.9  23.6  70.4 -141.2  44.6]

三件事：
  1. 三组关节角各自的正解 TCP、离目标 [215.3 40.2 9.4]mm 多远；
  2. 从「第 2 趟前」用 `_ik_free`（放开 1~5 号、40 次）解同一个目标，残差多少、解出
     来的角和日志里那条指令对不对得上；
  3. 姿态权重 `ow` 从硬到软各解一遍 —— 残差是不是被姿态那一项抢走了。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from real.arm import RealArm                                       # noqa: E402
from skills.pick_real import _ik_free                              # noqa: E402

GOAL = np.array([215.3, 40.2, 9.4]) / 1000.0

Q_BEFORE = np.array([-11.5, 2.2, 26.1, 67.1, -141.2, 44.6])
Q_CMD = np.array([-11.5, 5.7, 21.5, 70.6, -141.2, 44.6])
Q_AFTER = np.array([-11.5, 5.9, 23.6, 70.4, -141.2, 44.6])


class StubCtrl:
    def sync_read_positions(self, ids):
        return {int(i): 2048 for i in ids}

    def sync_set_positions(self, mapping, speed=0):
        pass


def tcp_of(arm, q_deg):
    q = np.radians(np.asarray(q_deg, dtype=np.float64))
    return np.asarray(arm._sim_pose(*arm._fk(q))[0], dtype=np.float64)


def main() -> int:
    arm = RealArm(ctrl=StubCtrl())
    q_b = np.radians(Q_BEFORE)

    # 让 arm.quat() 读到的就是「第 2 趟前」那个姿态（和 `_goto` 里同一个口径）
    arm.read_ticks = lambda: {}
    arm._ticks_to_qpos = lambda ticks: np.array(q_b, dtype=np.float64)

    for name, q in (("第2趟前", Q_BEFORE), ("第2趟指令", Q_CMD), ("第2趟走完", Q_AFTER)):
        p = tcp_of(arm, q)
        print(f"{name} xyz {np.round(p * 1000, 1)}mm  离目标 {np.linalg.norm(p - GOAL) * 1000:.1f}mm")

    print(f"\n关节行程 [lo, hi]（度）：")
    for j in range(5):
        print(f"  {j + 1} 号 [{np.degrees(arm.q_lo[j]):7.1f}, {np.degrees(arm.q_hi[j]):7.1f}]"
              f"  此刻 {Q_BEFORE[j]:7.1f}")

    tgt_pos, tgt_quat = arm._site_pose(GOAL, arm.quat())
    for ow in (float(arm.ow), 0.5 * float(arm.ow), 0.0):
        for iters in (40,):
            qg = _ik_free(arm, tgt_pos, tgt_quat, q_b, (0, 1, 2, 3, 4), ow=ow, iters=iters)
            p = np.asarray(arm._sim_pose(*arm._fk(qg))[0], dtype=np.float64)
            print(f"\now={ow:.3f} iters={iters}  残差 {np.linalg.norm(p - GOAL) * 1000:.2f}mm"
                  f"  xyz {np.round(p * 1000, 1)}")
            print(f"  解出来 {np.round(np.degrees(qg), 1)}")
            print(f"  相对此刻 {np.round(np.degrees(qg) - Q_BEFORE, 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
