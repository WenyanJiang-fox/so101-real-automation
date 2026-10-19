"""从 `out/calib_views.json` 解手眼：真腕相机相对 gripperframe 的固定安装位姿。

**为什么非要这一步。** 内参只说明「像素怎么落到相机坐标系」。要知道「东西在世界里
哪儿」，还得知道相机装在手臂上的哪个位置。模型里那个 `wrist_cam` 是**建模时挂上去
的**，跟真机上这颗差多少没人量过 —— 手眼就是拿同一批图把这个差值解出来。

**算法：`cv2.calibrateHandEye`。** 喂两串：
    · 每个姿态 gripperframe 在基座系里的位姿 —— 模型正解（`RealArm._fk`）
    · 同一姿态板子在相机系里的位姿 —— `cv2.solvePnP`，用上面那份内参
它吐出「相机 → gripperframe」那个固定变换。**板子在世界里的位置不需要知道**，
这正是手眼的用处：它只用「手臂动了多少」和「相机看到的板子转了多少」这一对关系。

**怎么验。** 用手眼结果反推「板子在基座系里的位置」。那块板子全程摊在桌上没动过，
所以反推出来的位置在每张图上应该**一模一样**；散开多少就是这次标定的残差。

**五种算法都解一遍。** 和 `calibrate_camera.py` 那张钉法表一个道理：方法之间漂多少
就是这条手眼结果的可信区间，看残差挑最小的那个写盘。

跑法（在 G:\\sim2real\\so101-real-harness 下）：
    G:/anaconda/envs/sim2real/python.exe calibrate_handeye.py
出参：G:\\sim2real\\so101-real-harness\\assets\\camera_calibration\\real_wrist_mount.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


class _NoPort:
    """假的串口对象。

    这里只用模型做正解（`_fk`）和读数换算（`_ticks_to_qpos`），**一个字节都不上总线**。
    但 `RealArm(ctrl=None)` 会自己去开 COM10，而且 `__init__` 里还会读一次爪角给
    `_grip_cmd` 打底 —— 打开串口虽然无害（力矩已经在使能，开端口不会动臂），可标定
    这一步没理由碰硬件。所以塞个假的：读数直接回 `RAW_HOME`，反正那一次读只喂给
    `_grip_cmd`，而这个脚本从头到尾不用它（正解用的是 json 里记的实测 raw_ticks）。
    """

    @staticmethod
    def sync_read_positions(ids):
        from real_home import RAW_HOME                           # noqa: PLC0415
        return {int(i): int(RAW_HOME[int(i)]) for i in ids}


def T_of(rot, tvec) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = np.asarray(rot, dtype=np.float64)
    m[:3, 3] = np.asarray(tvec, dtype=np.float64).ravel()
    return m


def spread(poses: list[np.ndarray]):
    """一串位姿的散度：位置的标准差（毫米）+ 相对平均姿态的最大夹角（度）。"""
    p = np.array([m[:3, 3] for m in poses])
    rots = Rotation.from_matrix(np.array([m[:3, :3] for m in poses]))
    mean = rots.mean()
    ang = np.degrees(np.linalg.norm((mean.inv() * rots).as_rotvec(), axis=1))
    return p.mean(axis=0), np.linalg.norm(p - p.mean(axis=0), axis=1), float(ang.max())


def main() -> int:                                             # noqa: C901
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="解真机腕相机的手眼（相机相对 gripperframe）")
    ap.add_argument("--views", type=Path, default=ROOT / "out" / "calib_views.json")
    ap.add_argument("--intr", type=Path,
                    default=ROOT / "assets" / "camera_calibration" / "real_wrist.json")
    ap.add_argument("--squares", default="8x6", help="板子的格子数，和 make_charuco 出图时一致")
    ap.add_argument("--square", type=float, default=0.028, help="一个格子的物理边长（米）")
    ap.add_argument("--skip", type=int, default=0, help="跳过前 N 张（配合 --limit 取一段）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只用前 N 张（在 --skip 之后算）。**两轮之间平板被挪过，必须分开解**"
                         " —— 手眼的前提是「靶标全程不动」，把不同摆放的两轮混在一起解，"
                         "残差会大得没边")
    ap.add_argument("--use", default=None, choices=sorted(METHODS),
                    help="写进 json 的那一种。不给就自动挑残差最小的")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "assets" / "camera_calibration" / "real_wrist_mount.json")
    args = ap.parse_args()

    import make_charuco as mc                                    # noqa: PLC0415
    from real.arm import JOINT_ORDER, RealArm                     # noqa: PLC0415
    from real_home import RAW_HOME                                # noqa: PLC0415

    sx, sy = (int(v) for v in args.squares.lower().split("x"))
    _, board = mc.build(sx, sy)
    det = cv2.aruco.CharucoDetector(board)

    intr = json.loads(args.intr.read_text(encoding="utf-8"))
    k = np.array(intr["k"], dtype=np.float64)
    dist = np.array(intr["dist"], dtype=np.float64)
    print(f"内参用 {args.intr.name}：fx {k[0, 0]:.2f} fy {k[1, 1]:.2f} "
          f"cx {k[0, 2]:.2f} cy {k[1, 2]:.2f}（{intr['n_views']} 张，均方根 {intr['rms_px']:.3f}px）")

    arm = RealArm(ctrl=_NoPort(), speed=0)
    import mujoco                                                # noqa: PLC0415
    site_name = mujoco.mj_id2name(arm.model, mujoco.mjtObj.mjOBJ_SITE, arm.site)
    print(f"模型里当「抓手系」的那个 site = `{site_name}`")

    # 模型自己那个 `wrist_cam` 到 gripperframe 的距离，给个参照。**真机那颗是另外
    # 拿支架装的**，跟这个数差多少没人量过 —— 手眼解出来的就是那个差。数字差得离谱
    # 先别急着信手眼，回去看是不是这一批视图有问题。
    cam_id = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    if cam_id >= 0:
        q_home = arm._ticks_to_qpos(RAW_HOME)
        for name in JOINT_ORDER:
            arm.data.qpos[arm.adr[name]] = float(q_home[JOINT_ORDER.index(name)])
        mujoco.mj_kinematics(arm.model, arm.data)
        T_w_g = T_of(arm.data.site_xmat[arm.site].reshape(3, 3),
                     arm.data.site_xpos[arm.site])
        T_w_c = T_of(arm.data.cam_xmat[cam_id].reshape(3, 3), arm.data.cam_xpos[cam_id])
        T_g_cam = np.linalg.inv(T_w_g) @ T_w_c
        print(f"模型自己的 `wrist_cam` 到 gripperframe 是 "
              f"{np.linalg.norm(T_g_cam[:3, 3]) * 1000:.1f} mm "
              f"（真机那颗是另装的，只是拿来看差多少）")
    print()

    recs = json.loads(args.views.read_text(encoding="utf-8"))
    if args.skip > 0:
        recs = recs[args.skip:]
    if args.limit > 0:
        recs = recs[:args.limit]
    R_g2b, t_g2b, R_t2c, t_t2c, used = [], [], [], [], []
    seen: list[dict] = []
    for r in recs:
        p = args.views.parent / r["file"]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  {r['file']}  读不出来，丢")
            continue
        ch_c, ch_i, _m_c, _m_i = det.detectBoard(img)
        if ch_c is None or len(ch_c) < 4:
            print(f"  {r['file']}  认不出板子，丢")
            continue
        obj, imp = board.matchImagePoints(ch_c, ch_i)
        obj = obj.astype(np.float32) * args.square
        imp = imp.astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, imp, k, dist)
        if not ok:
            print(f"  {r['file']}  位姿解不出，丢")
            continue
        rv, _jac = cv2.projectPoints(obj, rvec, tvec, k, dist)
        d = imp.reshape(-1, 2) - rv.reshape(-1, 2)
        e = float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))

        # 关节读数 -> 模型正解。**用实测的 raw_ticks，不用当时发的指令值** ——
        # 舵机有静差，手眼吃的是「它当时真在哪」。
        ticks = {int(sid): int(v) for sid, v in r["raw_ticks"].items()}
        q = arm._ticks_to_qpos(ticks)
        pos, quat = arm._fk(q)

        R_g2b.append(Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix())
        t_g2b.append(pos.reshape(3, 1))
        R_t2c.append(cv2.Rodrigues(rvec)[0])
        t_t2c.append(tvec.reshape(3, 1))
        used.append(r["file"])
        seen.append({"file": r["file"], "reproj_px": e,
                     "pan_deg": r["pan_deg"], "flex_deg": r["flex_deg"]})

    n = len(used)
    print(f"用上 {n}/{len(recs)} 张（每张的板子位姿都是单独 `solvePnP` 解的）")
    if n < 4:
        print("**少于 4 张解不出来。** 先回去多采几张。")
        return 1
    worst = max(seen, key=lambda s: s["reproj_px"])
    print(f"逐张 `solvePnP` 重投影最大 {worst['reproj_px']:.3f}px（{worst['file']}）\n")

    print(f"{'算法':>12}  {'反推板心的位置残差 均方根/最大':>30}  {'朝向散度':>9}")
    results = {}
    for name, flag in METHODS.items():
        R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
        T_g_c = T_of(R_c2g, t_c2g)
        base = [T_of(R_g2b[i], t_g2b[i]) @ T_g_c @ T_of(R_t2c[i], t_t2c[i])
                for i in range(n)]
        mean_p, dev, max_ang = spread(base)
        # 板子的法线在基座系里 = 板子位姿的第三列。**板子摊平的话它该是 (0,0,±1)**。
        # 这条能把锅分清：反推世界坐标要打「水平面」，板子本身要是斜的，那个假设就
        # 不成立，误差跟这个倾角直接相关。
        nrm = np.mean([m[:3, :3][:, 2] for m in base], axis=0)
        nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-12)
        results[name] = {"T_g_c": T_g_c, "dev_m": dev, "mean_p": mean_p,
                         "max_ang_deg": max_ang,
                         "normal": nrm,
                         "tilt_deg": float(np.degrees(np.arccos(
                             min(abs(float(nrm[2])), 1.0))))}
        # **`dev` 是米。** 这里印过一版「0.49 mm（最大 0.13）」，那是把每张图的偏差
        # 向量又求了一次范数、单位也没换 —— 真值是均方根 75mm、最大 126mm。
        print(f"{name:>12}  {np.sqrt(np.mean(dev ** 2)) * 1000:13.2f} /"
              f"{dev.max() * 1000:7.2f} mm  {max_ang:7.2f}°  板心 {np.round(mean_p, 4)}")

    def _rms(dev):
        return float(np.sqrt(np.mean(dev ** 2)))

    pick = args.use or min(results, key=lambda m: _rms(results[m]["dev_m"]))
    T_g_c = results[pick]["T_g_c"]
    R_c2g = T_g_c[:3, :3]
    t_c2g = T_g_c[:3, 3]
    print(f"\n挑 {pick} 写盘（板子位置标准差最小那个）")
    print(f"相机 -> gripperframe  平移 {np.round(t_c2g, 5)} m   "
          f"（{np.linalg.norm(t_c2g) * 1000:.1f} mm 远）")
    print(f"相机 -> gripperframe  转轴 "
          f"{np.round(np.degrees(Rotation.from_matrix(R_c2g).as_rotvec()), 2)} 度")
    nrm = results[pick]["normal"]
    print(f"板子的法线在基座系里 {np.round(nrm, 4)}"
          f"  —— 离竖直 {results[pick]['tilt_deg']:.2f}°"
          f"（摊平在该是 0°；不是 0 就是靶标本身斜着在桌上）")
    dev = results[pick]["dev_m"]
    print(f"验证：反推板心在基座系里的位置，各图之间最大偏 {dev.max() * 1000:.2f} mm"
          f"（均方根 {np.sqrt(np.mean(dev ** 2)) * 1000:.2f} mm）")

    args.out.write_text(json.dumps({
        "_note": "real/calib_views.py 采图 + calibrate_handeye.py 解。"
                 "R_cam2gripper/t_cam2gripper 是 cv2.calibrateHandEye 的输出，"
                 "gripper 系 = MJCF 的 gripperframe site。",
        "method": pick,
        "R_cam2gripper": R_c2g.tolist(),
        "t_cam2gripper": t_c2g.tolist(),
        "intrinsics_from": args.intr.name,
        "square_m": args.square,
        "n_views": n,
        "views": used,
        "validated": {
            "target_pos_in_base_m": results[pick]["mean_p"].tolist(),
            "resid_mm_max": float(dev.max() * 1000),
            "resid_mm_rms": float(np.sqrt(np.mean(dev ** 2)) * 1000),
            "resid_deg_max": results[pick]["max_ang_deg"],
            "all_methods_mm_max": {m: float(v["dev_m"].max()) for m, v in results.items()},
        },
        "_calibrated_at": "2026-10-14",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写到 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
