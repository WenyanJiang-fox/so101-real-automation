# -*- coding: utf-8 -*-
"""同一瞬间读关节 + 拍侧相机，把正解出来的爪投影画上去。

**不动臂**：只读舵机位置、只拍图，不发任何位置指令，也不碰力矩位。
（力矩是舵机里记着的，读一下不会把它开回来。）

判据：投出来的爪（蓝框 + 绿点）如果压在画面里那块白色爪上 —— 正解、指尖常数、
侧相机内外参这一路就是自洽的；压不上，偏差就是**侧相机外参**的，跟姿态和时间都无关。

    G:/anaconda/envs/sim2real/python.exe -u _probe_now.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2                                                        # noqa: E402
import mujoco                                                     # noqa: E402
import numpy as np                                                # noqa: E402

from core.registry import make                                    # noqa: E402
from core.vision import project                                   # noqa: E402
import real.arm as A                                              # noqa: E402
import real.cameras                                              # noqa: F401, E402
from real.cameras import (CV2_TO_MJ, SIDE_INTRINSICS_JSON,        # noqa: E402
                          SIDE_MOUNT_JSON)
from panel.rig_common import DEFAULT_CONFIG, load_cfg             # noqa: E402

N_SHOTS = 3


def body_vertices(model, data, body: str) -> np.ndarray:
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
    cfg = load_cfg(DEFAULT_CONFIG)
    rig = cfg.get("real") or {}
    arm = make("arm", "real", port=str(rig.get("port", "COM10")),
               speed=int(rig.get("speed", 0)))
    cam = None
    try:
        cam = make("camera", "real_side", source=arm)
        intr = json.loads(SIDE_INTRINSICS_JSON.read_text(encoding="utf-8"))
        mount = json.loads(SIDE_MOUNT_JSON.read_text(encoding="utf-8"))
        k3 = np.asarray(intr["k"], dtype=np.float64)
        k4 = np.array([k3[0, 0], k3[1, 1], k3[0, 2], k3[1, 2]], dtype=np.float64)
        r_b2c = np.asarray(mount["R_base2cam"], dtype=np.float64)
        t_b2c = np.asarray(mount["t_base2cam"], dtype=np.float64).reshape(3)
        cam_pos = -r_b2c.T @ t_b2c
        cam_mat = r_b2c.T @ CV2_TO_MJ

        model, data, site, adr = A.load_model(A.MJCF)
        for n in range(N_SHOTS):
            ticks = arm.read_ticks()
            q = arm._ticks_to_qpos(ticks)     # 和相机同一瞬的关节角
            frame = np.asarray(cam.shot())
            for i, name in enumerate(A.JOINT_ORDER):
                data.qpos[adr[name]] = float(q[i])
            mujoco.mj_kinematics(model, data)

            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                  A.TIP_FIXED_BODY)
            fixed = (np.asarray(data.xpos[b], dtype=np.float64)
                     + np.asarray(data.xmat[b]).reshape(3, 3) @ A.TIP_FIXED)
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                  A.TIP_MOVING_BODY)
            moving = (np.asarray(data.xpos[b], dtype=np.float64)
                      + np.asarray(data.xmat[b]).reshape(3, 3) @ A.TIP_MOVING)
            mid = (fixed + moving) / 2.0

            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            px = []
            for body in (A.TIP_FIXED_BODY, A.TIP_MOVING_BODY):
                for p3 in body_vertices(model, data, body):
                    p = project(cam_pos, cam_mat, k4, p3)
                    if p is not None:
                        px.append([int(round(p[0])), int(round(p[1]))])
            if px:
                arr = np.asarray(px, dtype=np.int32).reshape(-1, 1, 2)
                hull = cv2.convexHull(arr)
                cv2.polylines(img, [hull], True, (255, 0, 0), 2)
            for p3, col in ((mid, (255, 0, 255)), (fixed, (255, 255, 0)),
                            (moving, (255, 255, 0))):
                p = project(cam_pos, cam_mat, k4, p3)
                if p is not None:
                    cv2.drawMarker(img, (int(round(p[0])), int(round(p[1]))),
                                   col, cv2.MARKER_TILTED_CROSS, 26, 2)
            out = ROOT / "out" / f"now_{n}_{int(time.time())}.png"
            cv2.imwrite(str(out), img)
            print(f"[{n}] 关节(度) {np.round(np.degrees(q), 1)}")
            print(f"    指尖中点 {np.round(mid * 1000, 1)}mm → "
                  f"像素 {np.round(project(cam_pos, cam_mat, k4, mid), 1)}")
            print(f"    {out}")
        return 0
    finally:
        if cam is not None:
            try:
                cam.close()
            except Exception:                                     # noqa: BLE001
                pass
        arm.close()          # 只关串口，不动力矩


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
