# -*- coding: utf-8 -*-
"""只用腕相机闭环对准：读方块中心 → 把两爪中点挪到它正上方 → 再读一遍。

**不下探、不合爪。** 每一轮都重测（腕相机的读数随臂姿在变，不能一次算完）。
最后打 `_jaw_check` 的缝 / 轴读数并存图。

    G:/anaconda/envs/sim2real/python.exe -u _wrist_align.py
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

HOVER_ABOVE_CUBE = 0.026      # 两爪中点停在方块中心之上这么多（米）
TOL = 0.004                   # 差多少以内算对上了（米）
MAX_ROUNDS = 6


def look(rig, tag: str):
    c, f = rig.percep.locate(rig.color)
    if c is None:
        print(f"  {tag} 没量出来：{f.get('no_world_why') or f.get('why')}")
        return None
    return np.asarray(c, dtype=np.float64).reshape(-1)


def main() -> int:
    rig = Rig(cfg_path=str(ROOT / "configs" / "real.yaml"))
    try:
        print(f"感知相机 = {rig.percep.cam_name}")
        print("爪张到底…")
        rig._grip("open", None)

        for r in range(MAX_ROUNDS):
            c = look(rig, f"[{r}]")
            if c is None:
                return 1
            mid = np.asarray(rig.arm.jaw_gap_center(), dtype=np.float64)
            d = (c[:2] - mid[:2]) * 1000.0
            print(f"[{r}] 方块 ({c[0] * 1000:+.1f}, {c[1] * 1000:+.1f}, "
                  f"{c[2] * 1000:+.1f})mm   两爪中点 ({mid[0] * 1000:+.1f}, "
                  f"{mid[1] * 1000:+.1f}, {mid[2] * 1000:+.1f})mm   "
                  f"差 dx={d[0]:+.1f} dy={d[1]:+.1f}  "
                  f"平面 {float(np.hypot(*d)):.1f}mm")
            if float(np.hypot(*d)) <= TOL * 1000:
                print("  ✓ 平面已对上")
                break
            tcp = np.asarray(rig.arm.tcp(), dtype=np.float64).reshape(-1)
            off = tcp[:2] - mid[:2]
            tgt = np.array([mid[0] + (c[0] - mid[0]), mid[1] + (c[1] - mid[1]),
                            c[2] + HOVER_ABOVE_CUBE], dtype=np.float64)
            tgt[:2] += off
            out = _goto(rig.arm, tgt, vmax=0.2, grip=+1.0, tries=3)
            print(f"   -> 走 ({tgt[0] * 1000:+.1f}, {tgt[1] * 1000:+.1f}, "
                  f"{tgt[2] * 1000:+.1f})mm  {out}")

        print("\n=== 量缝 / 轴偏（腕相机，不动臂）===")
        res = _jaw_check(rig.cams["real_wrist"], rig.percep, rig.arm, "腕相机对准")
        for k in ("ok", "why", "gap", "axis", "along", "which", "box_uv", "c_uv"):
            if k in res:
                v = res[k]
                if isinstance(v, np.ndarray):
                    v = np.round(v, 1).tolist()
                elif isinstance(v, float):
                    v = round(v, 2)
                print(f"  {k}: {v}")
        src = Path(knobs.JAW_CHECK_IMG)
        if src.exists():
            dst = ROOT / "out" / f"wrist_align_{time.strftime('%H%M%S')}.png"
            shutil.copyfile(src, dst)
            print(f"\n图：{dst}")
        mid = np.asarray(rig.arm.jaw_gap_center(), dtype=np.float64)
        print(f"停在：两爪中点 ({mid[0] * 1000:+.1f}, {mid[1] * 1000:+.1f}, "
              f"{mid[2] * 1000:+.1f})mm  爪角 {float(np.degrees(rig.arm.grip_q())):+.1f}°")
        return 0
    finally:
        rig.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
