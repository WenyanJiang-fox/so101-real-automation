"""离线：`FREE_GAP=(0, 4)` 那一段，目标距离 vs 逆解出来的关节角。

不动臂、不碰串口。回答两件事：
  1. 每个关节动 1°，TCP / site 各挪多少毫米（`_step_min_deg` 挑关节的依据）；
  2. 给一个 `tgt`，`_ik_free` 解出来的各关节增量是多少、`_step_min_deg` 会挑哪个。

姿态取面板日志那一刻的六个关节角 —— `arm.angles()` 就是 `degrees(qpos)`（`real/arm.py:471`），
所以日志里那六个数直接就是 qpos（度）。

**位移一律 `*1000` 换成毫米再打印。** 单位是米，`{x:.2f}` 直接打出来全是 `0.00`。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from real.arm import RealArm                                      # noqa: E402
from skills import pick_real as PR                                # noqa: E402

# 日志那一刻：`走到 [200.3 29.7 9.8]mm  差 2.9mm  关节 [ -8.3 -8.6 35.6 65.1 -133.9 99.5]`
POSE_DEG = [-8.3, -8.6, 35.6, 65.1, -133.9, 99.5]


class StubCtrl:
    def sync_read_positions(self, ids):
        return {int(i): 2048 for i in ids}

    def sync_set_positions(self, mapping, speed=0):
        pass


def main():
    arm = RealArm(ctrl=StubCtrl())
    q_now = np.radians(np.array(POSE_DEG, dtype=np.float64))
    arm.read_ticks = lambda: {}
    arm._ticks_to_qpos = lambda ticks: np.array(q_now, dtype=np.float64)

    def tcp_of(q):
        """仿真那套 TCP 口径 —— `_goto` 的 `goal`、`tgt` 都是这个口径。"""
        return np.array(arm._sim_pose(*arm._fk(np.array(q, dtype=np.float64)))[0],
                        dtype=np.float64)

    def site_of(q):
        return np.array(arm._fk(np.array(q, dtype=np.float64))[0], dtype=np.float64)

    tcp = tcp_of(q_now)
    site0 = site_of(q_now)
    quat = np.asarray(arm.quat(), dtype=np.float64)
    print(f"姿态(度) {POSE_DEG}")
    print(f"  TCP(仿真口径) {np.round(tcp * 1000, 1)}mm   "
          f"site {np.round(site0 * 1000, 1)}mm")

    print("\n== 每个关节动 1° 的位移")
    for j in range(5):
        q = np.array(q_now, dtype=np.float64, copy=True)
        q[j] += np.radians(1.0)
        print(f"   {j + 1} 号   TCP {np.linalg.norm(tcp_of(q) - tcp) * 1000:5.2f}mm/度"
              f"   site {np.linalg.norm(site_of(q) - site0) * 1000:5.2f}mm/度")

    for free, name in (((0, 4), "FREE_GAP 缝"), ((3,), "FREE_AXIS 轴偏")):
        print(f"\n== {name} free={free}")
        for L in (0.00028, 0.00218, 0.00284, 0.006, 0.010, 0.020):
            for dname, d in (("+x", np.array([1.0, 0.0])),
                             ("+y", np.array([0.0, 1.0]))):
                tgt = np.array([tcp[0] + d[0] * L, tcp[1] + d[1] * L, tcp[2]])
                qg = PR._ik_free(arm, *arm._site_pose(tgt, quat), q_now, free)
                dd = np.degrees(np.abs(np.asarray(qg) - q_now))
                deg_pick, j_pick = PR._step_min_deg(arm, tgt, free)
                print(f"   {L * 1000:6.2f}mm {dname}  关节 {np.round(dd, 2)}  "
                      f"→ 定步长取 {deg_pick:.2f}°@{j_pick + 1}号  "
                      f"TCP 挪 {np.linalg.norm(tcp_of(qg) - tcp) * 1000:5.2f}mm  "
                      f"离目标还差 {np.linalg.norm(tcp_of(qg) - tgt) * 1000:5.2f}mm")


if __name__ == "__main__":
    main()
