"""下探到底那一刻，方块在 3D 里还量不量得准、两爪相对它偏在哪。**不动臂、不开串口。**

**问这个干嘛。** 微调现在量的是画面像素（固定爪轮廓 vs 红区剪影），方块贴近爪尖时这个
量会跳（`out/run_real_46.log`：同一条 8mm 指令，方块先动 −25 像素、再动 −66 像素）。
想换成 3D：下探到底重量一次方块位姿，拿它跟实测关节角算出来的爪子几何一比，误差是毫米。
**前提是这一刻还量得准** —— 固定爪压在方块上，画面里方块被挡掉一块。这一趟就是量这个前提。

输入只有两份离线东西：
  · `out/jaw_raw.jpg` —— `_jaw_check` 存的那张**没画过**的原始帧（画过轮廓的那张没法用，
    洋红轴线正压在方块边上）；
  · 日志里下探到底那一行的六个关节角，机位按线上同一个 `pose_from_fk` 复算。
感知跑的是线上同一条链 `BrownRectPerceptor.locate`，只把相机换成一个「返回固定帧 +
固定机位」的假相机 —— 不另写一份看着一样的。

跑：

    G:/anaconda/envs/sim2real/python.exe _descent_pose.py
    G:/anaconda/envs/sim2real/python.exe _descent_pose.py --log out/run_real_47.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import mujoco                                                   # noqa: E402

from core import geom                                           # noqa: E402
from core.vision import px_per_m_xy                             # noqa: E402
from real.arm import JOINT_ORDER, RealArm, load_model           # noqa: E402
from real.cameras import load_mount_cal, pose_from_fk           # noqa: E402
from real.perception import BrownRectPerceptor                  # noqa: E402

JOINT_RE = re.compile(r"关节 \[\s*([-0-9. ]+?)\s*\]")
MEAS_RE = re.compile(r"量出来：x=([-0-9.]+) y=([-0-9.]+) z=([-0-9.]+)mm\s+yaw ([-0-9.]+)°")
HIT_MARK = "下探到底"


class _NoBus:
    """`RealArm` 要一个控制器才建得起来 —— 这个假的不开串口、读回初始位置。"""

    def sync_read_positions(self, ids):
        from real_home import RAW_HOME

        return {int(k): int(v) for k, v in RAW_HOME.items()}

    def sync_set_positions(self, ticks, speed=0):
        pass


class _StillCam:
    """一个「永远给我这张帧、永远是这个机位」的假相机。`locate` 只用 `shot()`/`mount()`。"""

    def __init__(self, rgb: np.ndarray, mount: dict):
        self._rgb, self._mount = rgb, mount

    def shot(self):
        return self._rgb

    def mount(self):
        return self._mount


def _from_log(log: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """从日志里抠出：下探到底那一刻的六个关节角（度）、以及进场那一眼量出来的方块中心。"""
    q_deg, meas = None, None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = MEAS_RE.search(line)
        if m:
            meas = np.array([float(m.group(i)) for i in (1, 2, 3)], dtype=np.float64)
        if HIT_MARK in line:
            break
        m = JOINT_RE.search(line)
        if m:
            q_deg = np.array([float(v) for v in m.group(1).split()], dtype=np.float64)
    if q_deg is None:
        raise SystemExit(f"{log} 里没找到「{HIT_MARK}」之前的那行关节角")
    return q_deg, meas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(_ROOT / "out" / "run_real_48.log"))
    ap.add_argument("--img", default=str(_ROOT / "out" / "jaw_raw.jpg"))
    a = ap.parse_args()

    cfg = yaml.safe_load((_ROOT / "configs" / "real.yaml").read_text(encoding="utf-8"))
    rig = cfg["real"]
    objs = cfg["objects"]                       # **顶层**，不在 `real:` 段里
    geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
    geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]], dtype=np.float64) / 1000.0
    off = np.array([float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
                   dtype=np.float64) / 1000.0

    q_deg, meas = _from_log(Path(a.log))
    q = np.deg2rad(q_deg)
    print(f"日志 {Path(a.log).name}   下探到底关节（度）{np.round(q_deg, 1)}")
    print(f"  进场那一眼量出来 {None if meas is None else np.round(meas, 1)} mm   "
          f"感知固定偏移 {np.round(off * 1000, 1)} mm")

    # 机位：跟线上 `RealWristCamera.mount()` 同一条链（FK → 手眼标定）。
    model, data, site, adr = load_model()
    for name in JOINT_ORDER:
        data.qpos[adr[name]] = float(q[JOINT_ORDER.index(name)])
    mujoco.mj_kinematics(model, data)
    pos_s = np.array(data.site_xpos[site], dtype=np.float64)
    quat_s = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_s, data.site_xmat[site])
    mount = pose_from_fk(load_mount_cal(), pos_s, quat_s)
    print(f"  下探到底相机位置 {np.round(np.asarray(mount['pos']) * 1000, 1)} mm")

    img = cv2.imread(a.img)
    if img is None:
        print(f"读不到 {a.img}")
        return 1
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  原始帧 {Path(a.img).name}  {img.shape[1]}x{img.shape[0]}")

    percep = BrownRectPerceptor(cams={"real_wrist": _StillCam(rgb, mount)},
                                objects=objs,
                                cam_name="real_wrist")
    guess = None if meas is None else meas / 1000.0
    center, facts = percep.locate(str(objs[0]["color"]), guess=guess)
    print(f"  found={facts.get('found')}  矩形度 {facts.get('rect_fill')}  "
          f"重合度 iou={facts.get('iou')}  面积 {facts.get('area_px')}")
    print(f"  估出来的形状 {facts.get('size')}")
    if center is None:
        print(f"  ✗ 这一刻量不出来：{facts.get('no_world_why') or facts.get('why')}")
        return 1

    size = facts.get("est") or objs[0]
    print(f"  ✓ 量出来：{np.round(np.asarray(center) * 1000, 1)} mm   "
          f"yaw {facts.get('yaw_deg')}°")
    if meas is not None:
        d = (np.asarray(center) * 1000) - meas
        print(f"      跟进场那一眼比：x {d[0]:+.1f}  y {d[1]:+.1f}  z {d[2]:+.1f} mm")

    # ---- 爪子几何：固定爪尖、动爪尖，合爪轴 ----
    arm = RealArm(ctrl=_NoBus())
    f, m = arm.tips(q)
    tcp, quat = arm._sim_pose(*arm._fk(q))
    u = m - f
    u[2] = 0.0
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    n = np.array([-u[1], u[0], 0.0])
    print(f"  tcp() {np.round(tcp * 1000, 1)} mm  固定爪尖 {np.round(f * 1000, 1)} "
          f"动爪尖 {np.round(m * 1000, 1)}  开口（沿轴）{float(np.dot(m - f, u)) * 1000:.1f} mm")

    rel = np.asarray(center, dtype=np.float64) - f
    s = float(np.dot(rel, u))                      # 方块中心沿合爪轴，从固定爪尖算起
    p = float(np.dot(rel, n))                      # 离轴多远（垂直）
    print(f"  方块中心相对固定爪尖：沿轴(合爪方向) {s * 1000:+.1f} mm   "
          f"离轴 {p * 1000:+.1f} mm   竖着 {rel[2] * 1000:+.1f} mm")
    tcp_off = np.asarray(center, dtype=np.float64) - (tcp + np.array([off[0], off[1], 0.0]))
    print(f"  方块中心相对「tcp()+感知偏移」：{np.round(tcp_off * 1000, 1)} mm    "
          f"|水平| {np.linalg.norm(tcp_off[:2]) * 1000:.1f} mm")

    # ---- 像素到毫米：把画面里的轴偏/沿轴读数换算成毫米 ----
    J = px_per_m_xy(mount, np.asarray(mount["k"]), np.asarray(center, dtype=np.float64))
    if J is not None:
        J = np.asarray(J, dtype=np.float64)
        print(f"  这一点上画面里 1mm 走多少像素（J，行=u/v 像素，列=x/y 毫米）：")
        print(np.array2string(J, precision=2, prefix="      "))
        px_along = float(np.linalg.norm(J @ u[:2]))
        px_perp = float(np.linalg.norm(J @ n[:2]))
        print(f"  沿合爪轴 1mm -> {px_along:.2f} 像素；垂直 1mm -> {px_perp:.2f} 像素")
        print(f"  照这个，画面里的「沿轴 −29 像素」约合 {29 / max(px_along, 1e-9):.1f} mm；"
              f"「缝 ≥8 像素」约合 {8 / max(px_perp, 1e-9):.1f} mm 的离面距离")
    # ---- 把这个点投回画面画出来（写 out/descent_pose_mark.jpg）----
    from core import vision                                     # noqa: PLC0415

    uv = vision.project(np.asarray(mount["pos"]), np.asarray(mount["mat"]),
                        np.asarray(mount["k"]), np.asarray(center, dtype=np.float64))
    if uv is None:
        print("  投不出来（这一点在相机背后）—— 不画")
        return 0
    u_px, v_px = int(round(float(uv[0]))), int(round(float(uv[1])))
    mark = cv2.imread(str(a.img))
    if mark is None:
        print(f"  读不到 {a.img}，不画")
        return 0
    cv2.drawMarker(mark, (u_px, v_px), (0, 255, 0), cv2.MARKER_STAR, 56, 5)
    cv2.circle(mark, (u_px, v_px), 22, (0, 255, 0), 3, cv2.LINE_AA)
    txt = (f"THIS LOOK {np.round(np.asarray(center) * 1000, 1)} mm  px ({u_px}, {v_px})")
    for col, w in (((0, 0, 0), 5), ((0, 255, 0), 2)):
        cv2.putText(mark, txt, (u_px + 30, v_px - 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, col, w, cv2.LINE_AA)
    out_img = _ROOT / "out" / "descent_pose_mark.jpg"
    cv2.imwrite(str(out_img), mark)
    print(f"  画到 {out_img} —— 绿星就是这一眼量出来的方块中心，落在画面 ({u_px}, {v_px})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
