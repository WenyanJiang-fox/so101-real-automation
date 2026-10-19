"""一次性诊断：面板说「没认出盒子」。**不动臂**，只抓一帧、存整幅、把感知这一眼说了什么打出来。

跑：
    G:/anaconda/envs/sim2real/python.exe _split_tmp/diag_box.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from core import geom, vision                                   # noqa: E402
from core.registry import make                                  # noqa: E402
import impls.judge_held                                         # noqa: F401,E402
import real.arm                                                 # noqa: F401,E402
import real.cameras                                             # noqa: F401,E402
import real.perception                                          # noqa: F401,E402
from real.perception import find                                # noqa: E402

cfg = __import__("yaml").safe_load((_ROOT / "configs" / "real.yaml").read_text(encoding="utf-8"))
rig = cfg["real"]
geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]]) / 1000.0
geom.REAL_PERCEP_OFFSET = np.array([float(v) for v in rig["percep_offset_mm"]]) / 1000.0
geom.OBJECTS = list(cfg["objects"])

arm = make("arm", "real", port=str(rig["port"]), speed=int(rig.get("speed", 0)))
cam = make("camera", "real_wrist", source=arm, index=int(rig["cam_index"]))
percep = make("perception", "brown_rect", cams={"real_wrist": cam}, ref=None,
              objects=geom.object_specs())

rgb = np.asarray(cam.shot())
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
out = _ROOT / "out" / "diag_box_full.png"
cv2.imwrite(str(out), bgr)
print(f"整幅存到 {out}  {bgr.shape[1]}x{bgr.shape[0]}")

mount = cam.mount()
print(f"mount: {'None' if mount is None else 'ok'}   "
      f"TCP {np.round(np.asarray(arm.tcp()) * 1000, 1)}mm")
q = np.degrees(arm._ticks_to_qpos(arm.read_ticks()))
print(f"关节(度): {np.round(q, 1)}")

hit, facts = find(bgr, percep.ranges, percep.fill_min, percep.min_area, color=percep.color)
print(f"find: found={hit is not None}")
for k in ("why", "uv", "area_px", "rect_fill", "aspect"):
    print(f"  {k}: {facts.get(k)}")

center, facts2 = percep.locate(percep.color, guess=None)
print(f"locate: center={None if center is None else np.round(np.asarray(center) * 1000, 1)}mm")
for k in ("found", "where", "why", "no_world_why", "iou", "seen_c", "est_info"):
    print(f"  {k}: {facts2.get(k)}")

cam.close()
arm.close()
