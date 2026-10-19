# -*- coding: utf-8 -*-
"""回到初始位置，拍一张（带目标轮廓标注的）画面。

不改任何判据、不碰夹爪：只走 `home_real` 回初始位置，然后按配置里那个颜色认一次，
把整幅原始帧和带标注的帧都写到 `out/`。
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

import cv2                                                       # noqa: E402
import impls.judge_held                                          # noqa: E402,F401
import impls.policy_real                                          # noqa: E402,F401
import real.arm                                                   # noqa: E402,F401
import real.cameras                                               # noqa: E402,F401
import real.perception                                            # noqa: E402,F401
import skills.home_real                                           # noqa: E402,F401
import skills.pick_real                                           # noqa: E402,F401
from real.perception import draw                                  # noqa: E402

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
    home = make("skill", "home_real")
    print("[回初始位置]", home.run(arm, cams), flush=True)

    cam = cams["real_wrist"]
    bgr = cv2.cvtColor(np.asarray(cam.shot()), cv2.COLOR_RGB2BGR)
    raw = ROOT / "out" / "look_init_raw.png"
    cv2.imwrite(str(raw), bgr)

    center, facts = percep.locate("pick up the green box", guess=geom.REAL_PICK_XY)
    ann = ROOT / "out" / "look_init.png"
    cv2.imwrite(str(ann), draw(bgr, facts or {}))
    print(f"[中心] {None if center is None else np.round(center * 1000, 1)} mm")
    print(f"[事实] found={facts.get('found')} area={facts.get('area_px')} "
          f"rect_fill={facts.get('rect_fill')} uv={facts.get('uv')} "
          f"yaw={facts.get('yaw_deg')} iou={facts.get('iou')}")
    print(f"[形状] {facts.get('shape')} {facts.get('size')}  候选前三：{facts.get('top')}")
    print(f"[文件] {raw}\n       {ann}", flush=True)
finally:
    try:
        cams["real_wrist"].close()
    except Exception as exc:                                      # noqa: BLE001
        print("关相机：", exc, flush=True)
    try:
        arm.close()
    except Exception as exc:                                      # noqa: BLE001
        print("关串口：", exc, flush=True)
