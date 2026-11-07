"""离线复算：把「爪上好几个候选点」都投到那一帧画面上，看哪个落在白爪尖上。

**不动臂、不开串口。** 输入只有两份东西：
  · `out/round_01.json` 里第 N 帧的六个关节角（`arm.angles()` 存的就是度数）；
  · `assets/camera_calibration/` 那两份标定。
正解和手眼都调**线上真跑的那两个函数**（`RealArm._sim_pose` / `pose_from_fk`），
不另写一份看着一样的。

跑：`G:/anaconda/envs/sim2real/python.exe _probe_cross.py 39 out/cross_aim_f039.jpg`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(_ROOT))

import mujoco                                                      # noqa: E402

from core import geom                                              # noqa: E402
from core.vision import project, px_per_m                          # noqa: E402
from real.arm import (JOINT_ORDER, RealArm, TIP_FIXED,             # noqa: E402
                      TIP_FIXED_BODY, TIP_MOVING, TIP_MOVING_BODY,
                      load_model)
from real.cameras import load_mount_cal, pose_from_fk              # noqa: E402

# 线上那一轮打在日志里的读数，用来验离线这条链复算得对不对。
LIVE_TCP = np.array([0.202, 0.036, 0.059])       # 纠位 stall
LIVE_CROSS = (1031.0, 483.0)                     # 纠位后（对准）准星


def main() -> int:
    frame = int(sys.argv[1]) if len(sys.argv) > 1 else 39
    img_path = Path(sys.argv[2]) if len(sys.argv) > 2 else _ROOT / "out" / "cross_aim_f039.jpg"

    rec = json.loads((_ROOT / "out" / "round_01.json").read_text(encoding="utf-8"))
    q = np.deg2rad(np.asarray(rec["arm"][frame], dtype=np.float64))

    model, data, site, adr = load_model()
    for name in JOINT_ORDER:
        data.qpos[adr[name]] = float(q[JOINT_ORDER.index(name)])
    mujoco.mj_kinematics(model, data)
    pos_s = np.array(data.site_xpos[site], dtype=np.float64)
    quat_s = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_s, data.site_xmat[site])

    pos_tcp, quat_tcp = RealArm._sim_pose(pos_s, quat_s)

    def body_point(body: str, local) -> np.ndarray:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        return (np.asarray(data.xpos[b], dtype=np.float64)
                + np.asarray(data.xmat[b]).reshape(3, 3) @ np.asarray(local))

    tip_fix = body_point(TIP_FIXED_BODY, TIP_FIXED)
    tip_mov = body_point(TIP_MOVING_BODY, TIP_MOVING)
    mid = (tip_fix + tip_mov) / 2.0

    # `geom.REAL_PERCEP_OFFSET` 是 `run_sim.py` 开工时从配置里灌进去的全局量，
    # 离线跑没灌过，所以直接读配置文件，别读那个全局变量。
    import yaml                                                     # noqa: PLC0415
    rig = yaml.safe_load((_ROOT / "configs" / "real.yaml")
                         .read_text(encoding="utf-8"))["real"]
    off = np.array([float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
                   dtype=np.float64) / 1000.0
    cal = load_mount_cal()
    mount = pose_from_fk(cal, pos_s, quat_s)
    k = mount["k"]

    print(f"第 {frame} 帧   相机→基座 T 前三行：")
    print(np.array2string(np.column_stack([mount["mat"], mount["pos"]]), precision=4))
    print(f"标定：{cal['source']}   k={k}   画面 {cal['image_size']}")
    print(f"感知偏移 percep_offset_mm = {off*1000}（`+off` 只在 tcp 那两行加）")
    print()
    print(f"正解出来的官方 site 位置 {pos_s*1000} mm")
    print(f"tcp()                  {pos_tcp*1000} mm   ← 日志里纠位 stall 是 {LIVE_TCP*1000} mm")
    print(f"                         差 {(pos_tcp-LIVE_TCP)*1000} mm")
    print(f"固定爪尖 tips()[0]      {tip_fix*1000} mm   离 tcp {(tip_fix-pos_tcp)*1000} mm")
    print(f"动爪尖   tips()[1]      {tip_mov*1000} mm")
    print(f"两尖中点                {mid*1000} mm   两尖相距 {np.linalg.norm(tip_mov-tip_fix)*1000:.1f} mm")
    print()

    cands: list[tuple[str, np.ndarray, tuple[int, int, int]]] = [
        ("加偏移 tcp()+off", pos_tcp + np.array([off[0], off[1], 0.0]), (255, 0, 255)),
        ("不加偏移 tcp()", pos_tcp, (0, 0, 0)),
        ("动爪尖", tip_mov, (0, 165, 255)),
        ("官方 site", pos_s, (255, 0, 0)),
    ]
    img = cv2.imread(str(img_path))
    print(f"画面上（原图 {img.shape[1]}x{img.shape[0]}）")
    for label, p, col in cands:
        uv = project(mount["pos"], mount["mat"], k, p)
        if uv is None:
            print(f"  {label:22s} 投不出来")
            continue
        scale = px_per_m(mount, k, p)
        d_live = np.hypot(uv[0] - LIVE_CROSS[0], uv[1] - LIVE_CROSS[1])
        print(f"  {label:22s} ({uv[0]:6.0f},{uv[1]:6.0f})   基准 {scale:7.0f} px/m"
              f"   离线上那个准星 {d_live:5.0f} px")
        u, v = int(round(uv[0])), int(round(uv[1]))
        cv2.drawMarker(img, (u, v), col, cv2.MARKER_CROSS, 30, 2)
        cv2.circle(img, (u, v), 10, col, 1)
        cv2.putText(img, label, (u + 14, v - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)

    # 锚点：日志里量出来的方块中心（毫米），看它投回画面是不是还在方块身上。
    # 相机位姿换成这一帧的，方块的世界坐标不动，所以它应该照样落在红块上。
    cube = np.array([209.0, 36.0, -14.0]) / 1000.0
    for label, p in (("方块+off", cube + np.array([off[0], off[1], 0.0])),
                     ("方块不加off", cube)):
        uv = project(mount["pos"], mount["mat"], k, p)
        print(f"  {label:22s} ({uv[0]:6.0f},{uv[1]:6.0f})" if uv is not None else f"  {label}: 投不出来")

    out = _ROOT / "out" / f"cross_cand_f{frame:03d}.jpg"
    cv2.imwrite(str(out), img)
    print(f"\n候选点全画在这张上：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
