# -*- coding: utf-8 -*-
"""只拍一张：连上臂和腕相机，读当前姿态、出一张带爪形/合爪轴的图。**不动臂。**

给「停在对正之后」那一趟看姿态用（用户 2026-10-19）。写
`out/after_align_20261019.jpg`（就是 `_jaw_check` 画的那张的副本）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np                                               # noqa: E402

from core import geom                                            # noqa: E402
from core.registry import make                                   # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg            # noqa: E402

cfg = load_cfg(DEFAULT_CONFIG)
rig = cfg["real"]
geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                             dtype=np.float64) / 1000.0
geom.REAL_PERCEP_OFFSET = np.array(
    [float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
    dtype=np.float64) / 1000.0
geom.OBJECTS = list(cfg.get("objects") or [])

import impls.judge_held                                          # noqa: E402,F401
import impls.policy_real                                          # noqa: E402,F401
import real.arm                                                   # noqa: E402,F401
import real.cameras                                               # noqa: E402,F401
import real.perception                                            # noqa: E402,F401

arm = make("arm", "real", port=str(rig.get("port", "COM10")),
           speed=int(rig.get("speed", 0)))
arm.terminated = False
cams = {"real_wrist": make("camera", "real_wrist", source=arm,
                           index=int(rig.get("cam_index", 0)))}
percep_kw = ({"cam_name": str(cfg["perception_cam"])}
             if cfg.get("perception_cam") else {})
percep = make("perception", "brown_rect", cams=cams, ref=None,
              objects=geom.object_specs(), **percep_kw)

try:
    print("当前 TCP：", np.round(np.asarray(arm.tcp()) * 1000.0, 1), "mm", flush=True)
    from skills.pick_real.rulers import _jaw_check                # noqa: PLC0415
    out = _jaw_check(cams["real_wrist"], percep, arm, "对正之后停住",
                     axis_only=True)
    import shutil                                                 # noqa: PLC0415
    dst = ROOT / "out" / "after_align_20261019.jpg"
    shutil.copyfile(ROOT / "out" / "jaw_check.jpg", dst)
    print("图 ->", dst, flush=True)
    print("读数：", {k: out.get(k) for k in ("gap", "axis", "along", "box_uv")},
          flush=True)
finally:
    try:
        cams["real_wrist"].close()
    except Exception as exc:                                      # noqa: BLE001
        print("关相机：", exc, flush=True)
    try:
        arm.close()
    except Exception as exc:                                      # noqa: BLE001
        print("关串口：", exc, flush=True)
