# -*- coding: utf-8 -*-
"""离线判定：`TIP_FIXED` / `TIP_MOVING` 这两个指尖常数到底是不是落爪子的网格上。

**不开串口、不开相机、不动臂、不用图。** 只读 MJCF。

判据只有一条：指尖这两点（`real/arm.py:102-103`）离它所在 body 的网格顶点有多远。
- 落在网格上（几个毫米以里）→ 指尖常数没问题，正解报出来的指尖就是真指尖；
- 离网格几十毫米 → 常数本身偏了，跟相机、跟姿态、跟时间差都没关系。

    G:/anaconda/envs/sim2real/python.exe -u _probe_tip_on_mesh.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

import mujoco                                                     # noqa: E402
import numpy as np                                                # noqa: E402

import real.arm as A                                              # noqa: E402

# 关节角随便给（只要网格能装出来）；刚体内部的相对距离跟姿态无关。
DEG = [-8.3, 36.3, -44.0, 104.9, -90.9, -8.6]


def body_vertices(model, data, body: str) -> np.ndarray:
    """这个 body 下所有网格 geom 的世界顶点。"""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    out = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        mid = int(model.geom_dataid[g])
        if mid < 0:
            continue
        adr = int(model.mesh_vertadr[mid])
        num = int(model.mesh_vertnum[mid])
        v = np.asarray(model.mesh_vert[adr:adr + num], dtype=np.float64)
        R = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        t = np.asarray(data.geom_xpos[g], dtype=np.float64)
        out.append(v @ R.T + t)
    return np.vstack(out) if out else np.zeros((0, 3))


def main() -> int:
    model, data, _site, adr = A.load_model(A.MJCF)
    q = np.radians(np.asarray(DEG, dtype=np.float64))
    for i, name in enumerate(A.JOINT_ORDER):
        data.qpos[adr[name]] = float(q[i])
    mujoco.mj_kinematics(model, data)

    for body, local in ((A.TIP_FIXED_BODY, A.TIP_FIXED),
                        (A.TIP_MOVING_BODY, A.TIP_MOVING)):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        pos = np.asarray(data.xpos[b], dtype=np.float64)
        R = np.asarray(data.xmat[b], dtype=np.float64).reshape(3, 3)
        tip_w = pos + R @ np.asarray(local, dtype=np.float64)
        verts = body_vertices(model, data, body)
        if not len(verts):
            print(f"{body}: 这个 body 下没有网格 geom")
            continue
        d = np.linalg.norm(verts - tip_w, axis=1)
        i = int(np.argmin(d))
        print(f"{body}: 网格顶点 {len(verts)} 个")
        print(f"  指尖常数 {np.round(np.asarray(local) * 1000, 2)} mm（body 系）")
        print(f"  最近顶点离它 {float(d[i]) * 1000:.2f} mm"
              f"（那个顶点 {np.round((verts[i] - pos) @ R.T * 1000, 1)} mm，body 系）")
        # 也在 body 系的三个方向上看看它离网格包络有多远
        loc_v = (verts - pos) @ R.T
        for k, axis in enumerate("xyz"):
            print(f"    body 系 {axis}: 指纹尖 {local[k] * 1000:+.1f}mm，"
                  f"网格 [{loc_v[:, k].min() * 1000:+.1f}, "
                  f"{loc_v[:, k].max() * 1000:+.1f}]mm")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
