# -*- coding: utf-8 -*-
"""离线对账：拿一组关节角做正解，把两指尖投到**已存的侧相机图**上。

**不开串口、不开相机。** 只需要 MJCF、侧相机标定那两份 json、和一张已经存下来的图。
用来看「正解算出来的指尖」在侧相机画面里到底落在哪 —— 落在白色指尖上，
就说明指尖模型和侧相机外参这一路是自洽的。

    G:/anaconda/envs/sim2real/python.exe -u _probe_side_tip.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

import cv2                                                        # noqa: E402
import mujoco                                                     # noqa: E402
import numpy as np                                                # noqa: E402

from core.vision import project                                   # noqa: E402
import real.arm as A                                              # noqa: E402
from real.cameras import (CV2_TO_MJ, SIDE_INTRINSICS_JSON,        # noqa: E402
                          SIDE_MOUNT_JSON)

# 上一次锁死那一刻的关节角（`_free_and_fix.py fix` 打出来的那行）
DEG = [-8.3, 36.3, -44.0, 104.9, -90.9, -8.6]
SIDE_IMG = ROOT / "out" / "freefix_real_side_1792425972.png"
# 那一趟侧相机的读数（世界系，毫米）；z = 桌面 -22 + 半高 10
CUBE_MM = (296.6, 71.5, -12.0)


def main() -> int:
    model, data, _site, adr = A.load_model(A.MJCF)
    q = np.radians(np.asarray(DEG, dtype=np.float64))
    for i, name in enumerate(A.JOINT_ORDER):
        data.qpos[adr[name]] = float(q[i])
    mujoco.mj_kinematics(model, data)

    def tip(body: str, local) -> np.ndarray:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        return (np.asarray(data.xpos[b], dtype=np.float64)
                + np.asarray(data.xmat[b]).reshape(3, 3) @ np.asarray(local))

    fixed = tip(A.TIP_FIXED_BODY, A.TIP_FIXED)
    moving = tip(A.TIP_MOVING_BODY, A.TIP_MOVING)
    mid = (fixed + moving) / 2.0

    intr = json.loads(SIDE_INTRINSICS_JSON.read_text(encoding="utf-8"))
    mount = json.loads(SIDE_MOUNT_JSON.read_text(encoding="utf-8"))
    k3 = np.asarray(intr["k"], dtype=np.float64)
    k4 = np.array([k3[0, 0], k3[1, 1], k3[0, 2], k3[1, 2]], dtype=np.float64)
    r_b2c = np.asarray(mount["R_base2cam"], dtype=np.float64)
    t_b2c = np.asarray(mount["t_base2cam"], dtype=np.float64).reshape(3)
    cam_pos = -r_b2c.T @ t_b2c
    cam_mat = r_b2c.T @ CV2_TO_MJ

    print(f"指尖(正解, 基座系 mm)：固定 {np.round(fixed * 1000, 1)}   "
          f"动 {np.round(moving * 1000, 1)}")
    print(f"  中点 {np.round(mid * 1000, 1)}   两尖相距 "
          f"{np.linalg.norm(moving - fixed) * 1000:.1f}mm")

    img = cv2.imread(str(SIDE_IMG))
    if img is None:
        print(f"读不到图：{SIDE_IMG}")
        return 1
    print(f"画幅 {img.shape[1]}x{img.shape[0]}   内参说 "
          f"{intr['image_size'][0]}x{intr['image_size'][1]}")

    pm = project(cam_pos, cam_mat, k4, mid)
    pf = project(cam_pos, cam_mat, k4, fixed)
    pv = project(cam_pos, cam_mat, k4, moving)
    pc = project(cam_pos, cam_mat, k4, np.asarray(CUBE_MM) / 1000.0)
    for label, p in (("中点", pm), ("固定尖", pf), ("动尖", pv), ("方块中心", pc)):
        print(f"  {label} → 像素 {None if p is None else np.round(p, 1)}")

    if pm is not None:
        cv2.drawMarker(img, (int(round(pm[0])), int(round(pm[1]))),
                       (255, 0, 255), cv2.MARKER_TILTED_CROSS, 28, 2)
    # **整条链的锚点也投一遍。** 只看指尖分不清是「指尖常数偏了」还是
    # 「FK 这条路整个偏了」—— 把各节 body 原点和 `gripperframe` 一起标出来就分得清。
    anchored = [("gripper", np.asarray(data.xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")])),
        ("jaw", np.asarray(data.xpos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")])),
        ("wrist", np.asarray(data.xpos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")])),
        ("gripperframe", np.asarray(data.site_xpos[_site]))]
    for label, p3 in anchored:
        p = project(cam_pos, cam_mat, k4, p3)
        if p is None:
            continue
        u, v = int(round(p[0])), int(round(p[1]))
        cv2.circle(img, (u, v), 7, (0, 128, 255), 2)
        cv2.putText(img, label, (u + 10, v + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 128, 255), 1, cv2.LINE_AA)
        print(f"  {label} 原点 → 像素 {np.round(p, 1)}")
    for p in (pf, pv):
        if p is not None:
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 5,
                       (255, 255, 0), -1)

    # **把两个爪的网格顶点也投出来。** 光看几个点分不清「白对白」哪里是哪里；
    # 网格顶点铺出来就是爪在画面里的实际轮廓，跟白色打印件对不对得一目了然。
    n_vert = 0
    px_all: list = []
    for body in (A.TIP_FIXED_BODY, A.TIP_MOVING_BODY):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != bid:
                continue
            mid_g = int(model.geom_dataid[g])
            if mid_g < 0:
                continue
            adr_m = int(model.mesh_vertadr[mid_g])
            num = int(model.mesh_vertnum[mid_g])
            v = np.asarray(model.mesh_vert[adr_m:adr_m + num], dtype=np.float64)
            R = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
            t = np.asarray(data.geom_xpos[g], dtype=np.float64)
            pw = v @ R.T + t
            for p3 in pw:
                p = project(cam_pos, cam_mat, k4, p3)
                if p is not None:
                    cv2.circle(img, (int(round(p[0])), int(round(p[1]))), 1,
                               (0, 255, 0), -1)
                    px_all.append([int(round(p[0])), int(round(p[1]))])
                    n_vert += 1
    print(f"  爪网格顶点投出来 {n_vert} 个（绿色点）")
    if px_all:
        hull = cv2.convexHull(np.asarray(px_all, dtype=np.int32).reshape(-1, 1, 2))
        cv2.polylines(img, [hull], True, (255, 0, 0), 2)
    if pc is not None:
        cv2.drawMarker(img, (int(round(pc[0])), int(round(pc[1]))),
                       (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.putText(img, "magenta X = FK tip midpoint   yellow + = cube reading",
                (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1,
                cv2.LINE_AA)
    out = ROOT / "out" / f"offline_side_tip_{int(time.time())}.png"
    cv2.imwrite(str(out), img)
    print(f"那一帧：{out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
