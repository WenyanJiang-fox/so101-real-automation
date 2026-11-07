# -*- coding: utf-8 -*-
"""回初始位置，然后在**臂不在方块附近**的状态下把目标中心量几遍。

不动方块、不合爪。只回答一个问题：现在这个静止状态下，侧相机报的方块中心是多少、
几遍之间差多少。

    G:/anaconda/envs/sim2real/python.exe -u _home_and_remeasure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402

from panel.rig import Rig                                         # noqa: E402

N = 3


def main() -> int:
    rig = Rig(cfg_path=str(ROOT / "configs" / "real.yaml"))
    try:
        print("回初始位置…")
        rig._home()
        print(f"  现在 TCP {np.round(rig.arm.tcp() * 1000, 1)}mm")
        if rig.percep_side is None:
            print("✗ 侧相机没建起来")
            return 2
        for i in range(N):
            center, facts = rig.percep_side.locate(rig.color)
            if center is None:
                print(f"[{i}] 没认到：{facts.get('why') or facts.get('no_world_why')}")
                continue
            c = np.asarray(center, dtype=np.float64).reshape(-1)
            print(f"[{i}] 中心 ({c[0] * 1000:+.1f}, {c[1] * 1000:+.1f}, "
                  f"{c[2] * 1000:+.1f})mm   "
                  f"半径 {float(np.hypot(c[0], c[1])) * 1000:.1f}mm   "
                  f"矩形度 {facts.get('rect_deg')}   IoU {facts.get('iou')}   "
                  f"外接 {facts.get('rect_mm')}")
            print(f"      facts: "
                  f"{ {k: v for k, v in facts.items() if k in ('why', 'no_world_why', 'size_mm', 'shape_mm', 'label')} }")
        return 0
    finally:
        rig.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
