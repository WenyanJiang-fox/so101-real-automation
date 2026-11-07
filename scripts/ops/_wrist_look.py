# -*- coding: utf-8 -*-
"""只用腕相机量：拍 N 眼，打世界坐标 + 存带标注的图。**不动臂。**

    G:/anaconda/envs/sim2real/python.exe -u _wrist_look.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

import cv2                                                        # noqa: E402
import numpy as np                                                # noqa: E402

from panel.rig import Rig                                         # noqa: E402
from real.perception import draw                                  # noqa: E402

N = 3


def main() -> int:
    rig = Rig(cfg_path=str(ROOT / "configs" / "real.yaml"))
    try:
        print(f"感知相机 = {rig.percep.cam_name}   颜色 = {rig.color}")
        print(f"当前 TCP {np.round(np.asarray(rig.arm.tcp()) * 1000, 1)}mm   "
              f"两爪中点 {np.round(np.asarray(rig.arm.jaw_gap_center()) * 1000, 1)}mm")
        for i in range(N):
            center, f = rig.percep.locate(rig.color)
            if center is None:
                print(f"[{i}] 没量出来：{f.get('no_world_why') or f.get('why')}")
                continue
            c = np.asarray(center, dtype=np.float64).reshape(-1)
            mid = np.asarray(rig.arm.jaw_gap_center(), dtype=np.float64)
            d = (c[:2] - mid[:2]) * 1000.0
            print(f"[{i}] 方块 ({c[0] * 1000:+.1f}, {c[1] * 1000:+.1f}, "
                  f"{c[2] * 1000:+.1f})mm  像素 {tuple(np.round(f['uv'], 1))}  "
                  f"yaw {f['yaw_deg']:+.1f}°  IoU {f['iou']}  {f.get('size')}")
            print(f"     离两爪中点 dx={d[0]:+.1f} dy={d[1]:+.1f}mm")
            if i == N - 1:
                bgr = cv2.cvtColor(np.asarray(rig.cams["real_wrist"].shot()),
                                   cv2.COLOR_RGB2BGR)
                out = ROOT / "out" / f"wrist_look_{time.strftime('%H%M%S')}.png"
                cv2.imwrite(str(out), draw(bgr.copy(), f))
                print(f"     图 -> {out}")
        return 0
    finally:
        rig.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
