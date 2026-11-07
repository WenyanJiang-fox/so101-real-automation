# -*- coding: utf-8 -*-
"""把两爪中点挪到方块正上方（z=+60mm 那一档），然后停在那儿不回家。

**只做一件事**：侧相机读方块中心 → `_goto` 到它正上方、爪张着 → 再按侧相机量一次
两爪中点与方块中心的差，差超过 2mm 就用 `rig._nudge` 补一步。**不合爪、不下探。**

    G:/anaconda/envs/sim2real/python.exe -u _align_tip_over_cube.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402

from panel.rig import Rig                                         # noqa: E402
from skills.pick_real.motion import _goto                         # noqa: E402

GO_Z = 0.060          # 停在这一高度（米）——方块顶面在 −12mm 那一档，上面留足
TOL = 0.002           # 两爪中点离方块中心多近算对上了（米）


def main() -> int:
    rig = Rig(cfg_path=str(ROOT / "configs" / "real.yaml"))
    try:
        if rig.percep_side is None:
            print("✗ 侧相机没建起来")
            return 2
        print("爪张到底…")
        rig._grip("open", None)
        center, facts = rig.percep_side.locate(rig.color)
        if center is None:
            print(f"✗ 侧相机没认到方块：{facts.get('why') or facts.get('no_world_why')}")
            return 2
        c = np.asarray(center, dtype=np.float64).reshape(-1)
        print(f"方块中心 ({c[0] * 1000:+.1f}, {c[1] * 1000:+.1f}, {c[2] * 1000:+.1f})mm")

        goal = np.array([c[0], c[1], GO_Z], dtype=np.float64)
        print(f"走到 ({goal[0] * 1000:+.1f}, {goal[1] * 1000:+.1f}, "
              f"{goal[2] * 1000:+.1f})mm…")
        out = _goto(rig.arm, goal, vmax=0.2, grip=+1.0, tries=5)
        print(f"  -> {out}  现在 TCP {np.round(rig.arm.tcp() * 1000, 1)}mm")

        for i in range(4):
            mid = np.asarray(rig.arm.jaw_gap_center(), dtype=np.float64)
            tcp = np.asarray(rig.arm.tcp(), dtype=np.float64).reshape(-1)
            center, _f = rig.percep_side.locate(rig.color)
            if center is None:
                print("✗ 这一眼没认到方块")
                return 1
            c = np.asarray(center, dtype=np.float64).reshape(-1)
            d = (c[:2] - mid[:2]) * 1000.0
            print(f"[{i}] 两爪中点 ({mid[0] * 1000:+.1f}, {mid[1] * 1000:+.1f})mm  "
                  f"方块 ({c[0] * 1000:+.1f}, {c[1] * 1000:+.1f})mm  "
                  f"差 dx={d[0]:+.1f} dy={d[1]:+.1f}mm  z={mid[2] * 1000:+.1f}mm")
            if float(np.hypot(*d)) <= TOL * 1000:
                print("  ✓ 对上了")
                break
            # **用 `_goto` 而不是 `_nudge`**：`_nudge` 只走一趟、舵机带载欠的那一截
            # 没人补（实测挪 +9mm 实际只动了 0.1mm）；`_goto` 会按实测放大、重解几趟。
            # 目标按「TCP 比两爪中点偏多少」换算，z 回到 `GO_Z` 那一档。
            off = tcp[:2] - mid[:2]
            tgt = np.array([c[0] + off[0], c[1] + off[1], GO_Z], dtype=np.float64)
            _goto(rig.arm, tgt, vmax=0.2, grip=+1.0, tries=3)
        mid = np.asarray(rig.arm.jaw_gap_center(), dtype=np.float64)
        print(f"停在：两爪中点 ({mid[0] * 1000:+.1f}, {mid[1] * 1000:+.1f}, "
              f"{mid[2] * 1000:+.1f})mm  爪角 "
              f"{float(np.degrees(rig.arm.grip_q())):+.1f}°")
        print("（不回家、不合爪、不松力矩 —— 下一步用面板点）")
        return 0
    finally:
        rig.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
