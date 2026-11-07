"""从整轮视频里抽一帧出来，把准星和红色区域画上，存成一张整幅的图。

准星用的就是 `skills/pick_real.py` 里那一套口径：固定像素（`real_wrist_tip_px.json`）
加 `pointPolygonTest`。掩膜也照抄线上那条链（`thresholds.json` 里 red 的 HSV 区间 +
最小面积 + 矩形度门槛），不另选一块。

跑：`G:/anaconda/envs/sim2real/python.exe _mark_frame.py 87 out/last_before_close.jpg`

准星和视频都可以从命令行给（仿真那一路的视频是 640x480、准星是相机自己投出来的，
跟真机那份 `real_wrist_tip_px.json` 不是同一个数）：

    _mark_frame.py <帧号> <输出名> [准星u] [准星v] [视频路径]

**再画「感知判的方块中心」和「判的那条法向」**，加 `--pose`（真机那一趟的 `round_*.json`
里存了每一帧的关节角，拿它复算这一帧的机位，跟线上 `mount()` 走的是同一个
`pose_from_fk`）：

    _mark_frame.py 151 f151.jpg --pose 212 47 -14 71.8 --json out/round_01.json

`--pose` 给的是基座系毫米 / 度（就是日志里「量出来」那一行的四个数）。**画之前先自检
两条**：固定爪尖按这套复算投出来的像素必须等于标定那份 `tip_px`；工具轴跟法向的夹角
必须等于日志里「离侧面还差 N 度」那个数。两条对不上就只打数字、不画。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(_ROOT))

from real.perception import brown_mask, candidates, load_thresholds   # noqa: E402

# 桌上那只方块的边长全长（毫米）。跟 `configs/real.yaml` 的 `objects:` 那一行对齐 ——
# 只用来定法向箭头画多长（画到那条侧面为止）和算 `support()`。
SIDE_MM = 18.0


def _draw_pose(img, idx, pose, json_path, tip_ref):
    """标出「感知判的中心」和「判的那条侧面法向」。**先自检，过了才画。**

    自检两条，都是为了证明「复算出来的机位跟线上那一刻是同一个」：
      · 固定爪尖按这套复算投回图像，必须等于标定那份 `tip_px`；
      · 工具轴跟法向的夹角必须等于日志里「离侧面还差 N 度」那个数。
    对不上就只打数字不画 —— 画一张错的比不画糟。
    """
    import mujoco
    from scipy.spatial.transform import Rotation

    from core import shapes as _shapes
    from core import vision as _vision
    from real.arm import (JOINT_ORDER, TIP_FIXED, TIP_FIXED_BODY, TIP_MOVING,
                          TIP_MOVING_BODY, RealArm, load_model)
    from real.cameras import load_mount_cal, pose_from_fk

    x_mm, y_mm, z_mm, yaw_deg = pose
    rec = json.loads(Path(json_path).read_text(encoding="utf-8"))
    q = np.radians(np.asarray(rec["arm"][idx], dtype=np.float64))

    model, data, site, adr = load_model()
    shim = type("FK", (), {})()
    shim.model, shim.data, shim.site, shim.adr = model, data, site, adr
    pos_s, quat_s = RealArm._fk(shim, q)             # 官方 gripperframe 的位姿

    cal = load_mount_cal()
    if cal is None:
        print("！标定不全，中心/法向不画")
        return
    mount = pose_from_fk(cal, pos_s, quat_s)

    # ---- 自检 1：固定爪尖投影 ----
    tips_uv = {}
    for name, local in ((TIP_FIXED_BODY, TIP_FIXED), (TIP_MOVING_BODY, TIP_MOVING)):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        w = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ local
        tips_uv[name] = _vision.project(mount["pos"], mount["mat"], mount["k"], w)
    tip_uv = tips_uv[TIP_FIXED_BODY]

    # ---- 自检 2：工具轴 vs 法向的夹角 ----
    pos_c, quat_c = RealArm._sim_pose(pos_s, quat_s)
    tool_x = Rotation.from_quat(
        [quat_c[1], quat_c[2], quat_c[3], quat_c[0]]).as_matrix()[:, 0]
    sh = _shapes.from_spec({"shape": "cube", "color": "red", "side_mm": SIDE_MM})
    nrm = np.asarray(sh.normal(np.radians(yaw_deg), tool_x), dtype=np.float64)
    nrm = nrm / np.linalg.norm(nrm)
    cos = float(np.dot(nrm, tool_x)) / max(float(np.linalg.norm(tool_x)), 1e-12)
    off_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    sup_mm = float(sh.support(np.radians(yaw_deg), tool_x)) * 1000.0

    d_tip = float(np.linalg.norm(tip_uv - np.asarray(tip_ref, dtype=np.float64)))
    print(f"自检 1 固定爪尖投影 {tip_uv.round(1)}（标定 "
          f"{np.asarray(tip_ref).round(1)}，差 {d_tip:.1f} 像素）")
    print(f"自检 2 工具轴离法向 {off_deg:.1f}°（日志 7.5°）  沿工具轴鼓出来 {sup_mm:.1f}mm"
          f"（日志 10.1mm）")
    if d_tip > 8.0:
        print(f"✗ 自检 1 没过（差 {d_tip:.0f} 像素 > 8）—— 这套复算的机位跟线上那一刻"
              f"不是同一个，中心/法向不画。只画准星和画面里那块红的轮廓中心。")
        return


    # ---- 画 ----
    GREEN = (0, 200, 0)
    c_w = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    c_uv = _vision.project(mount["pos"], mount["mat"], mount["k"], c_w)
    e_uv = _vision.project(mount["pos"], mount["mat"], mount["k"],
                           c_w + (SIDE_MM / 2000.0) * nrm)
    if c_uv is None or e_uv is None:
        print("！中心/法向投不到画面里（在相机背后），不画")
        return
    print(f"中心像素 {c_uv.round(1)}   法向那端点 {e_uv.round(1)}"
          f"   箭头长 {np.linalg.norm(e_uv - c_uv):.0f} 像素")
    cv2.arrowedLine(img, tuple(np.round(c_uv).astype(int)),
                    tuple(np.round(e_uv).astype(int)), GREEN, 4, cv2.LINE_AA, tipLength=0.18)
    cv2.drawMarker(img, tuple(np.round(c_uv).astype(int)), GREEN,
                   cv2.MARKER_TILTED_CROSS, 40, 4)
    cv2.circle(img, tuple(np.round(c_uv).astype(int)), 14, GREEN, 3)
    txt = (f"sense center ({x_mm:.0f},{y_mm:.0f},{z_mm:.0f})mm yaw {yaw_deg:.1f}"
           f"  -> px {c_uv[0]:.0f},{c_uv[1]:.0f}")
    cv2.putText(img, txt, (28, 204), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, txt, (28, 204), cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 2, cv2.LINE_AA)
    txt2 = f"face normal (green arrow) {off_deg:.1f} deg off tool axis"
    cv2.putText(img, txt2, (28, 252), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, txt2, (28, 252), cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 2, cv2.LINE_AA)


def main() -> int:
    # **先把 `--` 那两个开关摘出去，再按位置读。** 混在一起读的话 `--pose 212 47 -14 71.8`
    # 里的 `-14` 会被当成第 5 个位置参数（视频路径），然后报「取不到第 N 帧」。
    rest = list(sys.argv[1:])
    pose = None
    if "--pose" in rest:
        i = rest.index("--pose")
        pose = [float(v) for v in rest[i + 1:i + 5]]
        del rest[i:i + 5]
    json_path = _ROOT / "out" / "round_01.json"
    if "--json" in rest:
        i = rest.index("--json")
        json_path = rest[i + 1]
        del rest[i:i + 2]

    idx = int(rest[0])
    out_path = _ROOT / "out" / (rest[1] if len(rest) > 1 else "frame.jpg")
    video = Path(rest[4]) if len(rest) > 4 else _ROOT / "out" / "round_01_完整.mp4"

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        print(f"取不到第 {idx} 帧")
        return 1

    th = load_thresholds()
    ranges = np.array(th["color"]["hsv"]["red"], dtype=np.uint8)
    min_area = 2000
    if len(rest) > 3:
        tip = np.array([float(rest[2]), float(rest[3])], dtype=np.float64)
    else:
        tip = np.asarray(json.loads(
            (_ROOT / "assets" / "camera_calibration" / "real_wrist_tip_px.json")
            .read_text(encoding="utf-8"))["tip_px"], dtype=np.float64)

    mask = brown_mask(img, ranges)
    cands = candidates(mask, min_area)
    print(f"第 {idx} 帧：{len(cands)} 个候选，面积 "
          f"{[c['area_px'] for c in cands]}，矩形度 {[round(c['rect_fill'], 3) for c in cands]}")

    for c in cands:
        cv2.drawContours(img, [c["contour"]], -1, (255, 255, 0), 3)

    uv = (int(round(tip[0])), int(round(tip[1])))
    cv2.drawMarker(img, uv, (255, 0, 255), cv2.MARKER_CROSS, 60, 4)
    cv2.circle(img, uv, 24, (255, 0, 255), 2)

    if cands:
        main_c = max(cands, key=lambda c: c["area_px"])
        px = cv2.pointPolygonTest(main_c["contour"], tip, True)
        text = (f"crosshair ({uv[0]},{uv[1]})  -> red region {px:+.0f} px"
                f"   ('+' = inside red, '-' = outside)")
        cv2.putText(img, text, (28, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, text, (28, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 2,
                    cv2.LINE_AA)
        print(text)

        # 画面里这块红的轮廓中心 —— **纯像素，不经过模型任何一步**，所以它跟准星一样
        # 是能直接信的。模型那条链投出来的中心放在旁边做对照。
        m = cv2.moments(main_c["contour"])
        cxy = (int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"])))
        cv2.drawMarker(img, cxy, (0, 200, 0), cv2.MARKER_TILTED_CROSS, 40, 4)
        cv2.circle(img, cxy, 14, (0, 200, 0), 3)
        txt = f"contour centroid {cxy}  (pixels only, no model)"
        cv2.putText(img, txt, (28, 156), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, txt, (28, 156), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 0), 2,
                    cv2.LINE_AA)
        print(txt)

    if pose is not None:
        _draw_pose(img, idx, pose, json_path, tip)

    cv2.imwrite(str(out_path), img)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
