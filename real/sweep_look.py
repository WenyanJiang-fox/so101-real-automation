"""掰腕子找视野外的东西：让 4 号(wrist_flex) 把相机从「正下方」往外推，扫一圈。

**为什么这招管用。** 腕相机离桌面只有约 159mm、光轴正对下方，能看到的地面只有约
0.33m × 0.18m 的一块。东西一挪出这块就彻底看不见，臂也够不着（实测：想靠平移找，
横移范围只有 ±0.18m；想靠 IK 掰姿态找，解出来的构型是肩抬 −98.9°、抓取点丢 90mm，
直接撞限位）。**4 号是唯一还能把光轴推出这块地板的杠杆**：把腕子往下掰，光轴离开
垂直方向，桌面落点跟着往外推几十厘米。

**为什么不做逆解、直接给关节角。** 4 号往下掰只会把爪子**抬起来**：实测 −30° 时
TCP 前移 80mm、抬高 21mm，往下压是不可能的。所以只动 1 号(pan) 和 4 号这两维
天然安全，不需要 IK 把关，也绕开了 IK 那个一掰就撞限位的毛病。

**动之前先算。** 每个姿态都用模型 `wrist_cam` 算出相机位置和光轴，再和桌面求交，
把「这个姿态往桌面哪儿看」先打出来 —— 这样一眼能看出覆盖了桌面哪一片，
不用瞎扫。

跑法（在 G:\\sim2real\\so101-real-harness 下）：
    G:/anaconda/envs/sim2real/python.exe real/sweep_look.py
    G:/anaconda/envs/sim2real/python.exe real/sweep_look.py --go
出图：G:\\sim2real\\so101-real-harness\\out\\look_p{m}_{f}.jpg
读数：G:\\sim2real\\so101-real-harness\\out\\sweep_look.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from real.joint_move import goto_q                       # noqa: E402
from real_home import RAW_HOME                           # noqa: E402
from servo_bridge import open_controller                 # noqa: E402

OUT = _ROOT / "out"
CAM_W, CAM_H = 1920, 1080
FOURCC = "MJPG"

# 关节序号（0 起）。1 号 pan -> 0，4 号 wrist_flex -> 3。
J_PAN, J_FLEX = 0, 3

# 1 号左右扫的角度，相对初始点位。**不超过 ±60°** —— 再大扫到桌外和那堆
# 3D 打印件上去了，撞了看不出来。
PAN_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)

# 4 号往下掰的角度，相对初始点位（初始是 88.1°，模型上限 ±95°，所以负向宽裕）。
FLEX_DEG = (0.0, -20.0, -40.0, -60.0)

# 安全带：TCP 不许比初始点位更低。**注意 `arm._fk` 给的是 TCP（gripperframe），
# 不是抓取点** —— 初始姿态下 TCP 是 0.0593、抓取点是 0.1175，两个差 58.7mm。
# 拿抓取点的值当门槛会把初始姿态自己判成不安全。
SAFE_MARGIN = 0.005

TABLE_Z = 0.0


def cam_ray(arm, q, cam_id):
    """这个关节角下，相机在哪、光轴指向哪、光轴打在桌面上哪一点。"""
    arm._fk(q)
    import mujoco                                          # noqa: PLC0415
    mujoco.mj_forward(arm.model, arm.data)
    pos = arm.data.cam_xpos[cam_id].copy()
    axis = -arm.data.cam_xmat[cam_id].reshape(3, 3)[:, 2]
    if axis[2] < -1e-3:
        hit = pos + axis * ((TABLE_Z - pos[2]) / axis[2])
    else:
        hit = None
    return pos, axis, hit


def poses(arm, *, pan_deg=PAN_DEG, flex_deg=FLEX_DEG, cam_name="wrist_cam",
          verbose=True):
    """扫视姿态表：从初始点位出发，1 号 pan 左右摆 × 4 号 flex 往下掰。

    返回 `(keep, dropped)`，两边都是 dict 的列表，每个带 `pan` / `flex` / `q`（关节角）
    / `gp_z`（TCP 高度）/ `cam`（相机位置）/ `axis`（光轴）/ `hit`（光轴打在桌面哪一点）。
    `dropped` 是被安全带滤掉的那些（TCP 比初始点位低）。

    **动之前先算。** 每个姿态都用模型算出相机在哪、往桌面哪儿看 —— 一眼能看出覆盖了
    桌面哪一片，不用瞎扫。`arm` 上得有模型（`RealArm` 有）。模型里没有 `cam_name` 就
    返回 `(None, None)`：算不出相机朝哪，扫了也是白扫。
    """
    import mujoco                                            # noqa: PLC0415

    cam_id = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        return None, None
    q_home = arm._ticks_to_qpos(RAW_HOME)
    tcp_home_z = float(arm._fk(q_home)[0][2])
    keep, dropped = [], []
    for pan in pan_deg:
        for flex in flex_deg:
            q = q_home.copy()
            q[J_PAN] += np.radians(pan)
            q[J_FLEX] += np.radians(flex)
            gp = arm._fk(q)[0]
            pos, axis, hit = cam_ray(arm, q, cam_id)
            rec = {"pan": pan, "flex": flex, "q": q, "gp_z": float(gp[2]),
                   "cam": [float(v) for v in pos],
                   "axis": [float(v) for v in axis],
                   "hit": None if hit is None else [float(v) for v in hit]}
            if rec["gp_z"] < tcp_home_z - SAFE_MARGIN:
                dropped.append(rec)
            else:
                keep.append(rec)

    if not verbose:
        return keep, dropped
    print(f"相机离桌面 {keep[0]['cam'][2] if keep else float('nan'):.3f} m。"
          f"初始姿态 TCP z = {tcp_home_z:.4f}，任何姿态的 TCP 都不许低于它。")
    print(f"{'姿态':>12}  {'TCP z':>8}  {'相机位置 x,y,z':>24}  {'光轴离正下':>9}  "
          f"{'桌面落点 x,y':>16}")
    for p in dropped:
        print(f"  pan{p['pan']:+5.0f} flex{p['flex']:+5.0f}  {p['gp_z']:8.4f}  "
              f"**比初始低，丢**")
    for p in keep:
        off = np.degrees(np.arccos(np.clip(-p["axis"][2], -1, 1)))
        h = "光轴朝上，不打桌面" if p["hit"] is None else \
            f"({p['hit'][0]:+.3f},{p['hit'][1]:+.3f})"
        print(f"  pan{p['pan']:+5.0f} flex{p['flex']:+5.0f}  {p['gp_z']:8.3f}  "
              f"({p['cam'][0]:+.3f},{p['cam'][1]:+.3f},{p['cam'][2]:+.3f})  "
              f"{off:8.1f}°  {h:>16}")
    print(f"\n能走 {len(keep)}/{len(keep) + len(dropped)} 个姿态。")
    return keep, dropped


def open_cam(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"打不开相机 index={index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    for _ in range(10):
        ok, fr = cap.read()
    if not ok or fr is None or (fr.shape[1], fr.shape[0]) != (CAM_W, CAM_H):
        cap.release()
        raise RuntimeError(f"要 {CAM_W}x{CAM_H}，拿不到")
    return cap


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="掰腕子扫一圈找视野外的东西")
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--jvel", type=float, default=0.20)
    ap.add_argument("--jacc", type=float, default=0.8)
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--go", action="store_true", help="真的动。不给就只打印")
    args = ap.parse_args()

    import make_charuco as mc                                    # noqa: PLC0415
    _, board = mc.build(8, 6)
    det = cv2.aruco.CharucoDetector(board)

    from real.arm import RealArm                                 # noqa: PLC0415
    ctrl = open_controller(args.port)
    arm = RealArm(ctrl=ctrl, speed=0)
    keep, dropped = poses(arm)
    if keep is None:
        print("**模型里没有 wrist_cam**，算不出相机朝哪，停。")
        ctrl.close()
        return 1
    if not args.go:
        print("没给 --go，一步不动。")
        ctrl.close()
        return 0

    cap = open_cam(args.index)
    recs = []
    q_prev = arm._ticks_to_qpos(arm.read_ticks())
    try:
        for p in keep:
            goto_q(arm, p["q"], q_from=q_prev, hz=args.hz,
                   jvel=args.jvel, jacc=args.jacc, settle=args.settle)
            q_prev = p["q"]
            for _ in range(3):
                ok, fr = cap.read()
            if not ok or fr is None:
                continue
            name = f"look_p{p['pan']:+.0f}_f{p['flex']:+.0f}.jpg"
            cv2.imwrite(str(OUT / name), fr)
            ch_c, _ci, m_c, _mi = det.detectBoard(fr)
            n_c = 0 if ch_c is None else len(ch_c)
            n_m = 0 if m_c is None else len(m_c)
            q_now = arm._ticks_to_qpos(arm.read_ticks())
            recs.append({"file": name, "pan": p["pan"], "flex": p["flex"],
                         "corners": n_c, "markers": n_m,
                         "hit_pred": p["hit"],
                         "q_deg": [float(v) for v in np.degrees(q_now)]})
            flag = "  <<< 看见板子了" if n_c >= 8 else ""
            print(f"  pan{p['pan']:+5.0f} flex{p['flex']:+5.0f}  角点 {n_c:3d}  码 {n_m:3d}"
                  f"  {name}{flag}")
    except KeyboardInterrupt:
        print("\n中断。臂停在当时位置，**力矩不松**。")
    finally:
        cap.release()

    (OUT / "sweep_look.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    best = max(recs, key=lambda r: r["corners"]) if recs else None
    if best and best["corners"] >= 8:
        print(f"\n最好的一帧 {best['file']}：{best['corners']} 个角点"
              f"（pan{best['pan']:+.0f} flex{best['flex']:+.0f}）")
    else:
        print("\n**这一圈没拍到板子。** 把它往哪个方向挪，或者加大扫描范围。")
    ctrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
