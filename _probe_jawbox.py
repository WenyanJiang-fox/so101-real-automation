"""离线探针：拿真机跑出来的**真实关节角**，把量到的那只盒子按量出来的位置摆进去，
逐个开合角问一句：**两爪的料有没有进到盒子里面。**

不动臂、不开串口、不写任何文件。用 `mujoco.MjSpec` 现场加一个盒子 geom 编一份
**独立**的模型（臂那个模型一个字节都不改），按关节名把六个角塞进去。

**为什么不用 `mj_geomDistance` 判。** 网格撞盒子那一路是按**凸包**算的，固定指是块
弯钩，凸包把开口整个包进去，于是报出「重叠 1.35mm」这种假数（`_probe_jawbox.py`
上一版就踩着这个）。真问「碰到没有」只能逐顶点换到盒子坐标系里判内外。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco                                                  # noqa: E402
from scipy.spatial.transform import Rotation                   # noqa: E402

from real.arm import MJCF, RealArm                             # noqa: E402
from real_home import RAW_HOME                                 # noqa: E402

RESULTS = _ROOT / "out" / "run_real_25.log"

# (标签, 盒子中心, yaw 度, 盒子尺寸, 下探到底读回来的六个关节角)
CASES = [
    ("run25 第1趟", (0.248, 0.032, -0.001), 38.7, (0.015, 0.015, 0.016),
     [-10.2, 16.6, 8.9, 66.5, -59.5, 99.3]),
    ("run24 第2趟", (0.238, 0.004, -0.001), -46.4, (0.018, 0.017, 0.016),
     [-1.8, 13.4, 13.3, 65.5, -136.3, 99.3]),
    # 2026-10-15 加：`out/run_real_48.log` 下探到底那一刻，方块中心不是进场那一眼量的，
    # 而是拿 `out/jaw_raw.jpg`（`_jaw_check` 存的原始帧）在 `_descent_pose.py` 里**重量
    # 的一次**：量出来 (227.5, 17.7, −14.0)mm / yaw 3.7° / 19×16×16mm，重合度 0.88。
    # **x 扣掉感知固定偏移 35.8mm** —— 那 35.8 是「画面反推的 x 恒偏」，`run_sim.py` 把
    # 它加在 tcp 上，所以摆在模型里的方块得先减掉，不然等于把偏差又算了一遍。
    ("run48 下探到底 3D 重量", (0.1917, 0.0177, -0.0140), 3.7, (0.019, 0.016, 0.016),
     [-1.2, -3.6, 34.6, 60.8, -89.6, 99.3]),
]
GRIPS_DEG = [-8.8, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 45.0, 99.3]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
          "wrist_roll", "gripper"]
FIX_BODY = "gripper"
MOV_BODY = "moving_jaw_so101_v1"


class _NoBus:
    def sync_read_positions(self, ids):
        return {int(k): int(v) for k, v in RAW_HOME.items()}

    def sync_set_positions(self, ticks, speed=0):
        pass


arm = RealArm(ctrl=_NoBus())
spec = mujoco.MjSpec.from_file(str(MJCF))


def vertex_world(gid, mt, dt):
    mid = int(mt.geom_dataid[gid])
    if mid < 0:
        return None
    a, n = int(mt.mesh_vertadr[mid]), int(mt.mesh_vertnum[mid])
    v = np.asarray(mt.mesh_vert[a:a + n], dtype=np.float64)
    R = np.asarray(dt.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    return v @ R.T + np.asarray(dt.geom_xpos[gid], dtype=np.float64)


for tag, center, yaw_deg, size, q_deg0 in CASES:
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    # **每个 case 重新读一次 MJCF。** 原来 `spec` 是脚本级只建一次，第二个 case 往同一个
    # spec 里再加一个 `probe_box` 就撞名，`spec.compile()` 抛 "repeated name"。
    spec = mujoco.MjSpec.from_file(str(MJCF))
    b = spec.worldbody.add_body(pos=center.tolist(),
                                euler=[0.0, 0.0, float(np.radians(yaw_deg))])
    b.name = "probe_box"
    g = b.add_geom()
    g.name = "probe_box_geom"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = (size / 2.0).tolist()
    g.contype = 1
    g.conaffinity = 1
    model = spec.compile()
    data = mujoco.MjData(model)
    adr = {}
    for nm in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
        adr[nm] = model.jnt_qposadr[jid]
    gids = {}
    for nm, body in (("固定", FIX_BODY), ("动", MOV_BODY)):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        gids[nm] = [i for i in range(model.ngeom)
                    if model.geom_bodyid[i] == bid and model.geom_contype[i] != 0]

    print(f"===== {tag}  盒子 {np.round(size * 1000, 1)}mm @ "
          f"{np.round(center * 1000, 1)}mm yaw {yaw_deg}°")
    for gg in GRIPS_DEG:
        q_deg = np.array([q_deg0[0], q_deg0[1], q_deg0[2], q_deg0[3],
                          q_deg0[4], gg], dtype=np.float64)
        q_rad = np.radians(q_deg)
        for nm in JOINTS:
            data.qpos[adr[nm]] = float(np.radians(q_deg[JOINTS.index(nm)]))
        mujoco.mj_forward(model, data)
        f, m = arm.tips(q_rad)
        tcp, quat = arm._sim_pose(*arm._fk(q_rad))
        xt = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()[:, 0]
        xt_h = np.array([xt[0], xt[1], 0.0])
        xt_h = xt_h / np.linalg.norm(xt_h)
        u = m - f
        u_h = np.array([u[0], u[1], 0.0])
        u_h = u_h / np.linalg.norm(u_h)
        yaw = np.radians(yaw_deg)
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        h = size / 2.0
        ext = float(np.abs(h[:2]) @ np.abs(R.T @ u_h[:2]))
        s_mid = float((center[:2] - f[:2]) @ u_h[:2])
        # 盒子坐标：u 沿「固定爪→动爪」，v 与它垂直，z 竖直
        hv = np.array([0.0, 0.0, 1.0])
        ru = u_h / np.linalg.norm(u_h)
        rv = np.cross(hv, ru)
        row = [f"6号 {gg:6.1f}°"]
        for nm in ("固定", "动"):
            tot_in = 0
            deepest = 0.0
            for gid in gids[nm]:
                w = vertex_world(gid, model, data)
                if w is None:
                    continue
                rel = w - center
                cu, cv, cz = rel @ ru, rel @ rv, rel[:, 2]
                ins = (np.abs(cu) <= h[0]) & (np.abs(cv) <= h[1]) & (np.abs(cz) <= h[2])
                tot_in += int(ins.sum())
                slab = (np.abs(cv) <= h[1]) & (np.abs(cz) <= h[2])
                d = float(cu[slab].max()) if slab.any() else -9.9
                deepest = max(deepest, d - (float((f - center) @ ru) if nm == "固定"
                                            else float((m - center) @ ru)))
            row.append(f"{nm}爪进盒子 {tot_in:5d} 个点，最近的面在爪尖外 "
                       f"{deepest * 1000:+6.2f}mm")
        tip_m = float((m[:2] - f[:2]) @ u_h[:2])
        print("  " + "  |  ".join(row)
              + f"  || 动爪尖在开合轴 {tip_m * 1000:6.1f}mm，"
                f"盒子占 [{s_mid - ext:.3f}, {s_mid + ext:.3f}]"
                f"（米）")
    print()
