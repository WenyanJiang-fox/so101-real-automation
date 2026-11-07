"""量「看一眼」要多久：色块掩膜、剪影套框（`estimate_pose`）各占多少毫秒。

**为什么量这个。** 用户问微调那几种方案的延迟。三种方案（画面像素 / 3D 位姿 / 学习）
都要**每步看一眼**，这一眼的成本是共同的底盘；不量出来，「谁快谁慢」全是拍脑袋。

输入只用两份离线东西，**不动臂、不开串口**：`out/round_01.json` 里某一帧的六个关节
角（复算机位，走线上同一个 `pose_from_fk`），加一张现成的整幅原图。

跑：

    G:/anaconda/envs/sim2real/python.exe _time_percep.py
    G:/anaconda/envs/sim2real/python.exe _time_percep.py --img out/别的帧.jpg --frame 39
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import mujoco                                                   # noqa: E402

from core.vision import color_mask, estimate_pose               # noqa: E402
from real.arm import JOINT_ORDER, load_model                    # noqa: E402
from real.cameras import load_mount_cal, pose_from_fk           # noqa: E402

REPS = 5


def _mount_at(q_deg: np.ndarray) -> dict:
    """按给定的关节角复算相机位姿 —— 跟线上 `RealWristCamera.mount()` 同一条链。"""
    q = np.deg2rad(np.asarray(q_deg, dtype=np.float64))
    model, data, site, adr = load_model()
    for name in JOINT_ORDER:
        data.qpos[adr[name]] = float(q[JOINT_ORDER.index(name)])
    mujoco.mj_kinematics(model, data)
    pos_s = np.array(data.site_xpos[site], dtype=np.float64)
    quat_s = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_s, data.site_xmat[site])
    return pose_from_fk(load_mount_cal(), pos_s, quat_s)


def _ms(fn, n: int = REPS) -> float:
    fn()                                    # 先跑一次，别把首次分配算进去
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=str(_ROOT / "out" / "real_37_preclose_f82.jpg"))
    ap.add_argument("--frame", type=int, default=0)
    a = ap.parse_args()

    img = cv2.imread(a.img)
    if img is None:
        print(f"读不到 {a.img}")
        return 1
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rec = json.loads((_ROOT / "out" / "round_01.json").read_text(encoding="utf-8"))
    mount = _mount_at(np.asarray(rec["arm"][a.frame], dtype=np.float64))
    print(f"图 {Path(a.img).name}  {img.shape[1]}x{img.shape[0]}   "
          f"机位取 round_01.json 第 {a.frame} 帧   相机位置 "
          f"{np.round(np.asarray(mount['pos']) * 1000, 1)} mm")

    ms_mask = _ms(lambda: color_mask(rgb, "red"))
    print(f"  色块掩膜           {ms_mask:7.1f} ms")

    guess = np.asarray(rec["meta"].get("target", [205.0, 18.0, -14.0])
                       if isinstance(rec["meta"].get("target"), (list, tuple))
                       else [205.0, 18.0, -14.0], dtype=np.float64) / 1000.0
    print(f"  初值（盒子中心）{np.round(guess * 1000, 1)} mm")
    ms_pose = _ms(lambda: estimate_pose(rgb, mount, "red", guess))
    print(f"  剪影套框 estimate_pose {ms_pose:7.1f} ms")
    center, yaw, info = estimate_pose(rgb, mount, "red", guess)
    print(f"  这一帧量出来 {None if center is None else np.round(np.asarray(center) * 1000, 1)} mm "
          f"/ {None if yaw is None else round(np.degrees(yaw), 1)}°   "
          f"重合度 {info.get('iou')}   {info.get('why') or ''}")
    print(f"  合计一眼           {ms_mask + ms_pose:7.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
