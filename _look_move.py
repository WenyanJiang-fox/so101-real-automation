# -*- coding: utf-8 -*-
"""初始位置那一眼目标贴着画面边缘 → 把镜头挪过去、再拍。

只做这一件事：回初始位置 → 认一次 → 看轮廓有没有贴边 → 贴边就把 TCP 平移到
目标 xy 正上方（z 不动）再认一次，最多两趟。合爪一步不碰。
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
from core.params import SPEED, speed_vmax                        # noqa: E402
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
import impls.policy_real                                         # noqa: E402,F401
import real.arm                                                   # noqa: E402,F401
import real.cameras                                               # noqa: E402,F401
import real.perception                                            # noqa: E402,F401
import skills.home_real                                           # noqa: E402,F401
import skills.pick_real                                           # noqa: E402,F401
from real.perception import brown_mask, draw                      # noqa: E402
from skills.pick_real.motion import _goto                         # noqa: E402

EDGE_PX = 3          # 外接矩形离画面边界不超过这么多像素 = 贴边


def edge_touch(bgr, ranges, min_area):
    """返回 (贴没贴边, 外接矩形, 画面尺寸)。只认最大那块色块。"""
    mask = brown_mask(bgr, ranges)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < int(min_area):
            continue
        if best is None or a > best[0]:
            best = (a, tuple(int(v) for v in stats[i, :4]))
    if best is None:
        return False, None, mask.shape[::-1]
    _, (x, y, w, h) = best
    H, W = mask.shape[:2]
    touch = (x <= EDGE_PX or y <= EDGE_PX or x + w >= W - EDGE_PX
             or y + h >= H - EDGE_PX)
    return touch, (x, y, w, h), (W, H)


arm = make("arm", "real", port=str(rig.get("port", "COM10")),
           speed=int(rig.get("speed", 0)))
arm.terminated = False
cams = {"real_wrist": make("camera", "real_wrist", source=arm,
                           index=int(rig.get("cam_index", 0)))}
percep_kw = ({"cam_name": str(cfg["perception_cam"])}
             if cfg.get("perception_cam") else {})
percep = make("perception", "brown_rect", cams=cams, ref=None,
              objects=geom.object_specs(), **percep_kw)


def shoot(tag):
    bgr = cv2.cvtColor(np.asarray(cams["real_wrist"].shot()), cv2.COLOR_RGB2BGR)
    center, facts = percep.locate("pick up the green box", guess=guess_now())
    touch, bb, (W, H) = edge_touch(bgr, percep.ranges, percep.min_area)
    cv2.imwrite(str(ROOT / "out" / f"look_{tag}.png"), draw(bgr, facts or {}))
    print(f"[{tag}] 贴边={touch}  外接矩形={bb}  画面={W}x{H}  "
          f"中心={None if center is None else np.round(center * 1000, 1)}mm  "
          f"yaw={facts.get('yaw_deg')} iou={facts.get('iou')}", flush=True)
    return center, facts, touch


_guess = {"v": np.asarray(geom.REAL_PICK_XY, dtype=np.float64)}


def guess_now():
    return _guess["v"]


try:
    print("[回初始位置]", make("skill", "home_real").run(arm, cams), flush=True)
    center, facts, touch = shoot("init")

    for k in range(1, 3):
        if not touch or center is None:
            break
        z = float(np.asarray(arm.tcp(), dtype=np.float64)[2])
        target = np.array([float(center[0]), float(center[1]), z], dtype=np.float64)
        print(f"  ↻ 贴边 → 镜头挪到 ({target[0] * 1000:.0f}, {target[1] * 1000:.0f})mm"
              f"（z 不动 {z * 1000:.1f}mm）", flush=True)
        got = _goto(arm, target, vmax=speed_vmax(SPEED))
        tcp = np.asarray(arm.tcp(), dtype=np.float64)
        print(f"    走位：{got}  TCP ({tcp[0] * 1000:.1f}, {tcp[1] * 1000:.1f}, "
              f"{tcp[2] * 1000:.1f})mm", flush=True)
        _guess["v"] = tcp.copy()
        center, facts, touch = shoot(f"move{k}")

    print("[结果] 贴边=", touch, " 最后中心=",
          None if center is None else np.round(center * 1000, 1), flush=True)
finally:
    try:
        cams["real_wrist"].close()
    except Exception as exc:                                      # noqa: BLE001
        print("关相机：", exc, flush=True)
    try:
        arm.close()
    except Exception as exc:                                      # noqa: BLE001
        print("关串口：", exc, flush=True)
