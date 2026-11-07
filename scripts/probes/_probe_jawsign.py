"""离线探针：在两爪的不同开口下，量「固定爪尖 → 动爪尖」这个方向，看它稳不稳。**不动臂。**

落点的 `side` 要的是「动爪在固定爪的哪一侧」。`pick.py` 那一支用
`sign(dot(fixed_tip - src, xh))` —— 那边 TCP 和固定爪尖是**两个点**，这个差就是固定爪
横偏的方向。真机上 `tcp()` 报的就是固定爪尖，两者重合，这个差退化成「悬停点离盒子中心
多少」，是残差噪声。所以得换成从爪尖几何直接读。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco                                                  # noqa: E402

from real.arm import RealArm                                   # noqa: E402
from real_home import RAW_HOME                                 # noqa: E402

TIP_FIX = np.array([-0.0089, -0.0002, -0.1044])
TIP_MOV = np.array([-0.0087, -0.0820, 0.0189])
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
          "wrist_roll", "gripper"]


class _NoBus:
    def sync_read_positions(self, ids):
        return {int(k): int(v) for k, v in RAW_HOME.items()}

    def sync_set_positions(self, ticks, speed=0):
        pass


arm = RealArm(ctrl=_NoBus())
BG = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
BM = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")

# 悬停姿态（纠位停稳那一点）和下探到底那一列，都取 run_23 第 2 趟实测的读数。
POSES = {
    "悬停（第 2 趟纠位停稳）": [-7.4, -2.0, 2.2, 86.5, -141.6, None],
    "下探到底（第 2 趟）": [3.6, 3.9, 26.4, 61.8, -131.9, None],
}
# 6 号关节从「合到底」到「张到最大」。真机口径：合到底 36.6° / 张到最大 146.6°。
GRIP = {"合到底(0°)": -10.0, "中间(90°)": 45.0, "下探到底(≈180°)": 100.0}


def look(q):
    for i, n in enumerate(JOINTS):
        jid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        arm.data.qpos[arm.model.jnt_qposadr[jid]] = float(q[i])
    mujoco.mj_kinematics(arm.model, arm.data)
    f = arm.data.xpos[BG] + arm.data.xmat[BG].reshape(3, 3) @ TIP_FIX
    m = arm.data.xpos[BM] + arm.data.xmat[BM].reshape(3, 3) @ TIP_MOV
    return f, m


for pose_name, q0 in POSES.items():
    print(pose_name)
    for grip_name, g in GRIP.items():
        q = list(q0)
        q[5] = np.radians(g)
        f, m = look(q)
        d = m - f
        dh = np.array([d[0], d[1], 0.0])
        n = float(np.linalg.norm(dh))
        dh = dh / n if n > 1e-9 else dh
        mid = (f + m) / 2.0
        print(f"  {grip_name:<18} 两尖相距 {np.linalg.norm(d)*1000:6.1f}mm   "
              f"固定爪→动爪 水平方向 ({dh[0]:+.2f},{dh[1]:+.2f})   "
              f"开口中点离固定爪尖 {np.linalg.norm(mid - f)*1000:5.1f}mm")
    print()
