"""解侧相机（固定支架）在**臂基座系**里的位姿。

思路：板子平放在桌面上，腕相机和侧相机**同时**看得见它。
  1. 腕相机那边整条链已经标好了（内参 `real_wrist.json` + 手眼 `real_wrist_mount.json`
     + `_fk`），板子角点能一路算到基座系的 3D 坐标；
  2. 侧相机同一时刻给出同一批角点的 2D 像素；
  3. 两组配对喂 `solvePnP`，出来的就是侧相机在基座系里的位姿。

**为什么要多放几处。** 一块板只给一组角点，解出来没有交叉验证。把板子换 3~5 个
位置各采一次（`--append`），最后一起解，每个角点的 3D 误差才被平摊掉。

**板子必须平放在桌面。** 不是靠这个假设解（3D 是腕相机实测的），是靠它做体检：
所有角点解出来的 z 应该落在桌面那一档附近，偏太多说明腕相机那一侧出问题了。

跑法（在 G:\\sim2real\\so101-real-harness 下）：
    G:/anaconda/envs/sim2real/python.exe real/side_extrinsic.py            # 采当前位置
    G:/anaconda/envs/sim2real/python.exe real/side_extrinsic.py --append   # 挪一处再采
    G:/anaconda/envs/sim2real/python.exe real/side_extrinsic.py --solve    # 一起解
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from color_check import open_camera, rotate_frame              # noqa: E402

CFG = _ROOT / "configs" / "real.yaml"
CALIB = _ROOT / "assets" / "camera_calibration"
SIDE_INTR = CALIB / "real_side.json"
VIEWS_JSON = _ROOT / "out" / "side_extr_views.json"
OUT_JSON = CALIB / "real_side_mount.json"

SQUARE_M = 0.028          # 平板上一个格子的实际边长（2026-10-19 用户给的口径）
CAM_W, CAM_H = 1920, 1080
FOURCC = "MJPG"


def _T(rot, tvec) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = np.asarray(rot, dtype=np.float64)
    m[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return m


def wrist_T_base_cv(arm, hand_eye: dict) -> np.ndarray:
    """腕相机在基座系里的位姿，**cv2 坐标约定**（solvePnP 用的那一套）。

    `pose_from_fk` 返回的是 MuJoCo 约定；这里直接 `T_b_g @ T_g_cv`，少换一次坐标系，
    也就不会在符号上绕进去。
    """
    from scipy.spatial.transform import Rotation            # noqa: PLC0415

    q = arm._ticks_to_qpos(arm.read_ticks())
    pos_g, quat_g = arm._fk(q)
    T_b_g = _T(Rotation.from_quat([quat_g[1], quat_g[2], quat_g[3], quat_g[0]])
               .as_matrix(), pos_g)
    T_g_cv = _T(hand_eye["R_cam2gripper"], hand_eye["t_cam2gripper"])
    return T_b_g @ T_g_cv


def matched(board, ch_c, ch_i):
    """把检测到的角点按编号整理成 {id: (x, y)}。"""
    if ch_c is None or ch_i is None:
        return {}
    ids = np.asarray(ch_i).reshape(-1)
    pts = np.asarray(ch_c, dtype=np.float64).reshape(-1, 2)
    return {int(i): (float(p[0]), float(p[1])) for i, p in zip(ids, pts)}


def object_points_m(board, ids) -> np.ndarray:
    """按角点编号取板子上的 3D 坐标（米）。"""
    allc = board.getChessboardCorners()          # (N,3)，单位是 squareLength（这里=1）
    return np.asarray([allc[i] for i in ids], dtype=np.float64) * SQUARE_M


def capture(args) -> int:
    from real.arm import RealArm                            # noqa: PLC0415
    from real.cameras import RealWristCamera, load_mount_cal  # noqa: PLC0415
    from servo_bridge import open_controller                # noqa: PLC0415
    import make_charuco as mc                               # noqa: PLC0415

    real = (yaml.safe_load(CFG.read_text(encoding="utf-8")) or {}).get("real") or {}
    _, board = mc.build(8, 6)
    det = cv2.aruco.CharucoDetector(board)

    cal = load_mount_cal()
    if cal is None:
        print("腕相机标定不全（内参或手眼缺一份），解不了 3D。")
        return 1
    side_intr = json.loads(SIDE_INTR.read_text(encoding="utf-8"))
    K_side = np.asarray(side_intr["k"], dtype=np.float64)
    d_side = np.asarray(side_intr["dist"], dtype=np.float64)

    ctrl = open_controller(args.port)
    arm = RealArm(ctrl=ctrl, speed=0)
    cam_w = RealWristCamera(source=arm, index=int(real["cam_index"]))
    side = open_camera(int(real["side_cam_index"]), size=(CAM_W, CAM_H),
                       fourcc=FOURCC, focus=real.get("side_cam_focus"))
    try:
        rgb = cam_w.shot()
        wrist_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if tuple(cam_w.size()) != (CAM_W, CAM_H):
            print(f"腕相机出的是 {cam_w.size()}，不是 {CAM_W}x{CAM_H}，内参对不上。")
            return 1
        T_b_cv = wrist_T_base_cv(arm, json.loads(
            (CALIB / "real_wrist_mount.json").read_text(encoding="utf-8")))

        # 侧相机连读几帧，等曝光稳定（第一帧常常还停在上一档）
        for _ in range(20):
            ok_s, side_raw = side.read()
        if not ok_s or side_raw is None:
            print("侧相机读不出帧。")
            return 1
        side_fr = rotate_frame(side_raw, int(real.get("side_cam_rotate") or 0))

        mw = matched(board, *det.detectBoard(
            cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2GRAY))[:2])
        ms = matched(board, *det.detectBoard(
            cv2.cvtColor(side_fr, cv2.COLOR_BGR2GRAY))[:2])
        common = sorted(set(mw) & set(ms))
        print(f"腕相机 {len(mw)} 个角点，侧相机 {len(ms)} 个，共同 {len(common)} 个")
        if len(common) < 6:
            print("**共同角点太少（<6），换个摆法或者把板子挪近一点。**")
            return 1

        # ---- 腕相机：角点 → 基座系 3D ----
        obj = object_points_m(board, common)
        img_w = np.asarray([mw[i] for i in common], dtype=np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj, img_w, cal["k3"],
                                      np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            print("腕相机 solvePnP 失败。")
            return 1
        R_w, _ = cv2.Rodrigues(rvec)
        pts_wrist_cam = (R_w @ obj.T).T + tvec.reshape(1, 3)
        pts_base = (T_b_cv[:3, :3] @ pts_wrist_cam.T).T + T_b_cv[:3, 3]

        obj_pts_proj, _ = cv2.projectPoints(obj, rvec, tvec, cal["k3"], np.zeros(5))
        err_w = float(np.sqrt(np.mean(np.sum(
            (img_w.reshape(-1, 2) - obj_pts_proj.reshape(-1, 2)) ** 2, axis=1))))
        print(f"腕相机重投影 {err_w:.2f} px   "
              f"角点 z 范围 {pts_base[:, 2].min() * 1000:.1f} ~ "
              f"{pts_base[:, 2].max() * 1000:.1f} mm（基座系）")

        rec = {
            "n": len(common),
            "ids": [int(i) for i in common],
            "image_side": [ms[i] for i in common],
            "base_xyz": [[float(v) for v in p] for p in pts_base],
            "wrist_reproj_px": err_w,
        }
    finally:
        cam_w.close()
        side.release()
        ctrl.close()

    views = json.loads(VIEWS_JSON.read_text(encoding="utf-8")) if (
        args.append and VIEWS_JSON.exists()) else []
    views.append(rec)
    VIEWS_JSON.write_text(json.dumps(views, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"记下第 {len(views)} 处摆法 → {VIEWS_JSON}")
    if len(views) < 3:
        print("**建议再挪 2 处（--append），一起解更稳。**")
    return 0


def solve() -> int:
    views = json.loads(VIEWS_JSON.read_text(encoding="utf-8"))
    side_intr = json.loads(SIDE_INTR.read_text(encoding="utf-8"))
    K = np.asarray(side_intr["k"], dtype=np.float64)
    d = np.asarray(side_intr["dist"], dtype=np.float64)

    obj_all, img_all = [], []
    for r in views:
        obj_all.append(np.asarray(r["base_xyz"], dtype=np.float64))
        img_all.append(np.asarray(r["image_side"], dtype=np.float64))
    obj = np.concatenate(obj_all).reshape(-1, 1, 3).astype(np.float32)
    img = np.concatenate(img_all).reshape(-1, 1, 2).astype(np.float32)
    print(f"用上 {len(views)} 处摆法，{len(obj)} 对点")

    # **不能用 IPPE。** 单个摆法里角点是共面的，但把 3 处摆法拼起来就不是了；
    # IPPE 要求严格共面，直接返回 False。SQPNP 不挑共面，实测 38 点全内点。
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, K, d, flags=cv2.SOLVEPNP_SQPNP, reprojectionError=3.0,
        iterationsCount=5000, confidence=0.999)
    if not ok:
        print("solvePnP 失败。")
        return 1
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec.reshape(3, 1)).reshape(3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, d)
    err = float(np.sqrt(np.mean(np.sum(
        (img.reshape(-1, 2) - proj.reshape(-1, 2)) ** 2, axis=1))))

    print(f"内点 {0 if inl is None else len(inl)}/{len(obj)}   重投影 {err:.2f} px")
    print(f"侧相机位置（基座系）   x {cam_pos[0] * 1000:8.1f}  "
          f"y {cam_pos[1] * 1000:8.1f}  z {cam_pos[2] * 1000:8.1f} mm")

    real = (yaml.safe_load(CFG.read_text(encoding="utf-8")) or {}).get("real") or {}
    OUT_JSON.write_text(json.dumps({
        "camera": "side",
        "frame": "arm_base_cv2",
        "R_base2cam": [[float(v) for v in row] for row in R],
        "t_base2cam": [float(v) for v in tvec.reshape(-1)],
        "pos_base": [float(v) for v in cam_pos],
        "k": [[float(v) for v in row] for row in K],
        "dist": [float(v) for v in d],
        "image_size": side_intr["image_size"],
        # **用这份 K 之前必须先按这个角度顺时针转正**，内参是在转正后的图上标的。
        "rotate_deg": int(real.get("side_cam_rotate") or 0),
        "rms_px": err,
        "n_views": len(views),
        "n_points": int(0 if inl is None else len(inl)),
        "square_m": SQUARE_M,
        "_calibrated_at": "2026-10-19",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写到 {OUT_JSON}")
    return 0


def main() -> int:                                             # noqa: C901
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="解侧相机在基座系里的位姿")
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--append", action="store_true", help="接着已有记录往下采（不覆盖）")
    ap.add_argument("--solve", action="store_true", help="拿已采的记录解外参，不采新数据")
    args = ap.parse_args()
    if args.solve:
        return solve()
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
