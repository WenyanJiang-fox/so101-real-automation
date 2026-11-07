"""离线探针：把「固定爪尖」相对官方 `gripperframe` 的差值**算出来**，别手推。**不动臂。**

`_sim_pose` 的写法是 `pos_site + R(site_quat) @ TCP_DELTA_BODY`，所以要求的就是
`d = R_site^T @ (爪尖 - 官方site)`。两个姿态各算一次，看它是不是同一个常数。
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

TIP_FIX = np.array([-0.0089, -0.0002, -0.1044])   # gripper 体系（so101-fingertip-points 量准的）
TIP_MOV = np.array([-0.0087, -0.0820, 0.0189])    # moving_jaw 体系
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
          "wrist_roll", "gripper"]


class _NoBus:
    def sync_read_positions(self, ids):
        return {int(k): int(v) for k, v in RAW_HOME.items()}

    def sync_set_positions(self, ticks, speed=0):
        pass


arm = RealArm(ctrl=_NoBus())
SID = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
BG = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
BM = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")

POSES = {
    "初始位置": arm._ticks_to_qpos(RAW_HOME),
    "下探到底（21 第 3 趟）": np.radians([0.7, 6.6, 16.5, 69.1, -134.4, 99.3]),
    "纠位（21 第 3 趟）": np.radians([0.6, -0.6, -4.3, 94.8, -134.4, -8.9]),
}


def look(q):
    for i, n in enumerate(JOINTS):
        jid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        arm.data.qpos[arm.model.jnt_qposadr[jid]] = float(q[i])
    mujoco.mj_forward(arm.model, arm.data)
    out = {
        "site_pos": np.array(arm.data.site_xpos[SID]),
        "site_R": np.array(arm.data.site_xmat[SID]).reshape(3, 3),
        "tip_fix": arm.data.xpos[BG] + arm.data.xmat[BG].reshape(3, 3) @ TIP_FIX,
        "tip_mov": arm.data.xpos[BM] + arm.data.xmat[BM].reshape(3, 3) @ TIP_MOV,
    }
    return out


ds = []
for name, q in POSES.items():
    w = look(q)
    d = w["site_R"].T @ (w["tip_fix"] - w["site_pos"])
    ds.append(d)
    tcp_now = w["site_pos"] + w["site_R"] @ np.array([0.0199, 0.0, 0.0])
    print(f"{name}")
    print(f"  官方 gripperframe {np.round(w['site_pos'] * 1000, 1)}mm")
    print(f"  固定爪尖          {np.round(w['tip_fix'] * 1000, 1)}mm")
    print(f"  动爪尖            {np.round(w['tip_mov'] * 1000, 1)}mm")
    print(f"  两尖中点          {np.round((w['tip_fix'] + w['tip_mov']) / 2 * 1000, 1)}mm"
          f"   两尖相距 {np.linalg.norm(w['tip_fix'] - w['tip_mov']) * 1000:.1f}mm")
    print(f"  现在的 arm.tcp()  {np.round(tcp_now * 1000, 1)}mm"
          f"   （它离固定爪尖 {np.linalg.norm(tcp_now - w['tip_fix']) * 1000:.1f}mm）")
    print(f"  求出来的 d = R_site^T @ (爪尖-site) = {np.round(d, 6)}")
    print()

print("三个姿态求出来的 d 一致吗：",
      np.allclose(ds[0], ds[1], atol=1e-9) and np.allclose(ds[0], ds[2], atol=1e-9))
print("平均 d =", np.round(np.mean(ds, axis=0), 6),
      f"  长度 {np.linalg.norm(np.mean(ds, axis=0)) * 1000:.1f}mm")

# ---- 改完之后的核对：走 `arm.tcp()` / `arm._site_pose` 自己那条路，别只看几何 ----
print()
print("核对（新 TCP_DELTA_BODY 已经写进 real/arm.py）：")
worst_tcp = worst_inv = 0.0
for name, q in POSES.items():
    w = look(q)
    tcp_code = arm._sim_pose(*arm._fk(q))[0]
    back = arm._site_pose(tcp_code, arm._sim_pose(*arm._fk(q))[1])[0]
    e_tcp = float(np.linalg.norm(tcp_code - w["tip_fix"]))
    e_inv = float(np.linalg.norm(back - w["site_pos"]))
    worst_tcp = max(worst_tcp, e_tcp)
    worst_inv = max(worst_inv, e_inv)
    print(f"  {name}: tcp()={np.round(tcp_code * 1000, 1)}mm  "
          f"爪尖={np.round(w['tip_fix'] * 1000, 1)}mm  差 {e_tcp * 1000:.3f}mm | "
          f"_site_pose 回官方 site 差 {e_inv * 1000:.3f}mm")
print(f"最大误差：tcp 对爪尖 {worst_tcp * 1000:.3f}mm，"
      f"_site_pose 回官方 {worst_inv * 1000:.3f}mm")
