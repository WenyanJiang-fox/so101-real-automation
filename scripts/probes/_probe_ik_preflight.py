"""离线：全关节逆解到「橙色点」到底到不到得了 —— 不动臂、不碰串口。

回答两件事：
  1. 从盘上那份检查点的姿态出发，解到一个 30mm 外的水平目标，残差是多少；
  2. 姿态权重从硬到软试一遍，哪个关节会被夹在限位上。

姿态取 `out/pick_checkpoint.json` 的 `angles`（度），`RealArm.angles()` 那六个数就是
qpos（度），和 `_probe_free_gap.py` 同一写法。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from real.arm import RealArm                                      # noqa: E402


class StubCtrl:
    def sync_read_positions(self, ids):
        return {int(i): 2048 for i in ids}

    def sync_set_positions(self, mapping, speed=0):
        pass


def main():
    cp = json.loads(Path("out/pick_checkpoint.json").read_text(encoding="utf-8"))
    pose_deg = np.asarray(cp["angles"], dtype=np.float64)
    arm = RealArm(ctrl=StubCtrl())
    q_now = np.radians(pose_deg)
    arm.read_ticks = lambda: {}
    arm._ticks_to_qpos = lambda ticks: np.array(q_now, dtype=np.float64)

    def tcp_of(q):
        return np.asarray(arm._sim_pose(*arm._fk(np.asarray(q, dtype=np.float64)))[0],
                          dtype=np.float64)

    tcp = tcp_of(q_now)
    print(f"姿态 {np.round(pose_deg, 1)} 度")
    print(f"TCP {np.round(tcp * 1000, 1)}mm   ow 默认 {float(arm.ow):.3f}")
    print(f"q_lo {np.round(np.degrees(arm.q_lo[:5]), 1)}")
    print(f"q_hi {np.round(np.degrees(arm.q_hi[:5]), 1)}")

    for step_mm in ((12.5, 27.0), (28.8, -14.7), (40.0, 0.0)):
        goal = tcp.copy()
        goal[:2] += np.asarray(step_mm, dtype=np.float64) / 1000.0
        print(f"\n== 目标：水平走 {np.round(step_mm, 1)}mm  → "
              f"{np.round(goal * 1000, 1)}mm（z 不动）")
        pos_s, quat_s = arm._site_pose(goal, arm.quat())
        for ow in (float(arm.ow), 0.5 * float(arm.ow), 0.0):
            q = arm._ik(pos_s, quat_s, q_now, ow=ow)
            got = tcp_of(q)
            res = float(np.linalg.norm(goal - got)) * 1000.0
            qgd = np.degrees(q)
            pinned = [j + 1 for j in range(5)
                      if abs(q[j] - arm.q_lo[j]) < 1e-3 or abs(q[j] - arm.q_hi[j]) < 1e-3]
            moved = np.round(np.degrees(q - q_now), 1)
            print(f"   ow={ow:.3f}  残差 {res:5.2f}mm  顶住 {pinned or '无'}"
                  f"  Δq {moved[:5]}")


if __name__ == "__main__":
    main()
