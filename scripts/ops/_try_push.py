# -*- coding: utf-8 -*-
"""单独试一次推方块（不动抓取流程）。

    # 只看：串口通不通、方块在哪、打算往哪推
    G:/anaconda/envs/sim2real/python.exe -u _try_push.py
    # 真推：回位 → 认方块 → 往正前方推 15mm → 再认一次 → 回位
    G:/anaconda/envs/sim2real/python.exe -u _try_push.py --go [--toward front|center|pick]

默认 `--toward front`：目标 = 机械臂**正前方**、可达环中段那个点
（`push.front_target()`，基座系约 [239, 0]mm），和采集脚本同一口径。
`--toward center|pick` 是旧口径（朝环心 / 朝标定点挪 `--step-mm`），留着对比用。
**只推这一次**：这个脚本自己不再重试。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

for _k in ("PICK_CONFIRM_STEP", "PICK_CONFIRM", "PICK_CONFIRM_DESCEND",
           "PICK_RESUME", "PICK_START_AT"):
    os.environ.pop(_k, None)

import numpy as np                                               # noqa: E402

from core import geom                                            # noqa: E402
from core.registry import make                                   # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="真推；不加只读数")
    ap.add_argument("--toward", choices=("front", "center", "pick"),
                    default="front",
                    help="front=机械臂正前方（采集时的口径）；center=环心；pick=标定点")
    ap.add_argument("--step-mm", type=float, default=40.0,
                    help="center/pick 用：目标点 = 方块中心朝那个方向挪这么多")
    args = ap.parse_args()

    cfg = load_cfg(DEFAULT_CONFIG)
    rig = cfg["real"]
    geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
    geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                                 dtype=np.float64) / 1000.0
    geom.REAL_PERCEP_OFFSET = np.array(
        [float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
        dtype=np.float64) / 1000.0
    geom.OBJECTS = list(cfg.get("objects") or [])

    import real.arm                                              # noqa: E402,F401
    import real.cameras                                          # noqa: E402,F401
    import real.perception                                       # noqa: E402,F401
    import skills.home_real                                      # noqa: E402,F401
    import skills.pick_real                                      # noqa: E402,F401
    from skills.pick_real.push import front_target, push_toward    # noqa: E402

    color = (cfg.get("objects") or [{"color": "green"}])[0].get("color", "green")
    task = f"pick up the {color} box"

    arm = make("arm", "real", port=str(rig.get("port", "COM10")),
               speed=int(rig.get("speed", 0)), fast=bool(rig.get("fast", False)))
    arm.terminated = False
    cams = {"real_wrist": make("camera", "real_wrist", source=arm,
                               index=int(rig.get("cam_index", 0)))}
    percep_kw = ({"cam_name": str(cfg["perception_cam"])}
                 if cfg.get("perception_cam") else {})
    percep = make("perception", "brown_rect", cams=cams, ref=None,
                  objects=geom.object_specs(), **percep_kw)

    try:
        tcp = np.asarray(arm.tcp(), dtype=np.float64) * 1000.0
        print(f"[臂] 串口通，当前 TCP {np.round(tcp, 1)}mm", flush=True)

        c, facts = percep.locate(task, guess=geom.real_pick_xy())
        if c is None:
            print(f"[方块] 没认到：{facts}", flush=True)
            return 2
        c_mm = np.asarray(c, dtype=np.float64) * 1000.0
        print(f"[方块] 中心 {np.round(c_mm, 1)}mm", flush=True)

        ring = np.asarray(geom.MARKER_CENTER, dtype=np.float64).reshape(-1)[:2]
        if args.toward == "front":
            # 和采集脚本同一条口径：目标 = 正前方、可达环中段那个点，
            # `push_toward` 自己按 15mm 一步推，不在这里改目标。
            tgt = front_target()
        else:
            base = ring if args.toward == "center" else np.asarray(geom.real_pick_xy())[:2]
            u = base - np.asarray(c)[:2]
            n = float(np.linalg.norm(u))
            if n < 1e-9:
                u = np.array([1.0, 0.0])
                n = 1.0
            tgt = np.asarray(c)[:2] + (u / n) * (args.step_mm / 1000.0)
        print(f"[目标] 朝 {args.toward}：方块从 {np.round(c_mm[:2], 1)}mm "
              f"推到 {np.round(tgt * 1000, 1)}mm", flush=True)
        print(f"[环] 工作台中心 {np.round(ring * 1000, 1)}mm", flush=True)

        if not args.go:
            print("[只看] 没动臂；要真推加 --go", flush=True)
            return 0

        home = make("skill", "home_real")
        h = home.run(arm, cams)
        print(f"[回初始位置] {h}", flush=True)
        if not h.get("ok"):
            print("[停] 回位没到位，不推", flush=True)
            return 3

        # 回位之后方块可能被挡住/位置没变，再认一次
        c, facts = percep.locate(task, guess=geom.real_pick_xy())
        if c is None:
            print(f"[停] 回位后没认到方块：{facts}", flush=True)
            return 2
        pr = push_toward(arm, cams, percep, np.asarray(c, dtype=np.float64), tgt,
                         text=task)
        print(f"[结果] {pr}", flush=True)

        h = home.run(arm, cams)
        print(f"[回初始位置] {h}", flush=True)
        return 0 if pr.get("ok") else 1
    finally:
        try:
            cams["real_wrist"].close()
        except Exception as exc:                                  # noqa: BLE001
            print("关相机：", exc)
        try:
            arm.close()
        except Exception as exc:                                  # noqa: BLE001
            print("关串口：", exc)


if __name__ == "__main__":
    raise SystemExit(main())
