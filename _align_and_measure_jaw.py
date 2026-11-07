# -*- coding: utf-8 -*-
"""把两爪中点摆到方块中心正上方（不下落），然后量「缝」和「轴偏」并把差距画出来。

**不下探、不合爪。** 走完就停在那儿，写两张图：
  · `out/jaw_check.jpg` —— 腕相机那一帧，画着爪轮廓、合爪轴、红区、缝和轴偏的连线
  · `out/jaw_align_<时分秒>.png` —— 同一帧的副本，带时间戳

    G:/anaconda/envs/sim2real/python.exe -u _align_and_measure_jaw.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402

from panel.rig import Rig                                         # noqa: E402
from skills.pick_real import knobs                                # noqa: E402
from skills.pick_real.motion import _goto                         # noqa: E402
from skills.pick_real.rulers import _jaw_check                    # noqa: E402

HOVER_Z = 0.014       # 方块中心 −12mm + 26mm 那一档（跟技能下探前的悬停高度同一个）


def main() -> int:
    rig = Rig(cfg_path=str(ROOT / "configs" / "real.yaml"))
    try:
        if rig.percep_side is None:
            print("✗ 侧相机没建起来")
            return 2
        center, _f = rig.percep_side.locate(rig.color)
        if center is None:
            print("✗ 侧相机没认到方块")
            return 2
        c = np.asarray(center, dtype=np.float64).reshape(-1)
        print(f"方块中心 ({c[0] * 1000:+.1f}, {c[1] * 1000:+.1f}, {c[2] * 1000:+.1f})mm")

        print("爪张到底…")
        rig._grip("open", None)
        for i in range(4):
            mid = np.asarray(rig.arm.jaw_gap_center(), dtype=np.float64)
            tcp = np.asarray(rig.arm.tcp(), dtype=np.float64).reshape(-1)
            off = tcp[:2] - mid[:2]
            d = (c[:2] - mid[:2]) * 1000.0
            print(f"[{i}] 两爪中点 ({mid[0] * 1000:+.1f}, {mid[1] * 1000:+.1f}, "
                  f"{mid[2] * 1000:+.1f})mm   离方块中心 dx={d[0]:+.1f} dy={d[1]:+.1f}mm")
            if float(np.hypot(*d)) <= 2.0 and abs(mid[2] - HOVER_Z) <= 0.005:
                print("  ✓ 到位置了")
                break
            tgt = np.array([c[0] + off[0], c[1] + off[1], HOVER_Z], dtype=np.float64)
            out = _goto(rig.arm, tgt, vmax=0.2, grip=+1.0, tries=4)
            print(f"   -> {out}")

        print("\n=== 量缝 / 轴偏（腕相机，不动臂）===")
        res = _jaw_check(rig.cams["real_wrist"], rig.percep, rig.arm, "手动对准")
        for k in ("ok", "why", "gap", "axis", "along", "which", "box_uv", "c_uv",
                  "gap_near", "gap_exit", "axis_ref_px", "axis_dir_px"):
            if k in res:
                v = res[k]
                if isinstance(v, np.ndarray):
                    v = np.round(v, 1).tolist()
                elif isinstance(v, float):
                    v = round(v, 2)
                print(f"  {k}: {v}")
        src = Path(knobs.JAW_CHECK_IMG)
        if src.exists():
            out = ROOT / "out" / f"jaw_align_{time.strftime('%H%M%S')}.png"
            shutil.copyfile(src, out)
            print(f"\n图：{src}\n副本：{out}")
        else:
            print("！图没写出来")
        mid = np.asarray(rig.arm.jaw_gap_center(), dtype=np.float64)
        print(f"停在：两爪中点 ({mid[0] * 1000:+.1f}, {mid[1] * 1000:+.1f}, "
              f"{mid[2] * 1000:+.1f})mm  爪角 "
              f"{float(np.degrees(rig.arm.grip_q())):+.1f}°")
        return 0
    finally:
        rig.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
