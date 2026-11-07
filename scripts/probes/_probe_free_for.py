"""离线：`_free_for` 到底补不补人。**不动臂、不碰串口**（`StubCtrl` 那套照抄
`_probe_free_gap.py`）。

三个问题：
  1. 那个姿态上，每个关节让爪尖往哪走、每度多少毫米；
  2. `need` 取切向（1 号满力那维）→ 应该原样返回 `(0,)`；
  3. `need` 取径向（1 号那一维转 90°，也就是那个退化的姿态）→ 应该补一个进来。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from real.arm import RealArm                                       # noqa: E402
from skills import pick_real as PR                                 # noqa: E402

POSE_DEG = [-8.3, -8.6, 35.6, 65.1, -133.9, 99.5]     # 跟 `_probe_free_gap.py` 同一姿态


class StubCtrl:
    def sync_read_positions(self, ids):
        return {int(i): 2048 for i in ids}

    def sync_set_positions(self, mapping, speed=0):
        pass


def main() -> int:
    arm = RealArm(ctrl=StubCtrl())
    q_now = np.radians(np.array(POSE_DEG, dtype=np.float64))
    arm.read_ticks = lambda: {}
    arm._ticks_to_qpos = lambda ticks: np.array(q_now, dtype=np.float64)

    tcp = np.asarray(arm._sim_pose(*arm._fk(q_now))[0], dtype=np.float64)
    xy = tcp[:2]
    tan = np.array([-xy[1], xy[0]], dtype=np.float64)
    tan = tan / float(np.linalg.norm(tan))
    rad = xy / float(np.linalg.norm(xy))
    print(f"TCP {np.round(tcp * 1000, 1)}mm")
    print(f"  切向（1 号满力那维）({tan[0]:+.2f}, {tan[1]:+.2f})")
    print(f"  径向（1 号出不上力那维）({rad[0]:+.2f}, {rad[1]:+.2f})")
    print(f"  两者正交性  tangent·radial = {float(tan @ rad):+.3f}")

    print("\n== 每个关节动 1°，爪尖往哪走")
    for j in range(5):
        d, mm = PR._joint_dir_xy(arm, q_now, j)
        if d is None:
            print(f"   {j + 1} 号   动不了")
            continue
        print(f"   {j + 1} 号   {mm:5.2f}mm/度   方向 ({d[0]:+.2f}, {d[1]:+.2f})"
              f"   投到切向 {float(d @ tan):+.2f}   投到径向 {float(d @ rad):+.2f}")

    print("\n== `_free_for`")
    for need, name in ((tan, "切向"), (rad, "径向")):
        for base, bname in (((0,), "FREE_GAP=(0,)"), ((3,), "FREE_AXIS=(3,)")):
            got = PR._free_for(arm, need, base)
            print(f"   need={name}  base={bname:16s} → {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
