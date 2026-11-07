# -*- coding: utf-8 -*-
"""只用腕相机把方块摆进两爪之间：先降到悬停高度，再按画面里的「轴偏 / 缝」闭环挪。

量的口径全是 `_jaw_check`（腕相机 + 描好的爪子轮廓），不碰侧相机、不碰世界坐标。
推进靠**现场探出来的响应矩阵**（TCP 走 1mm，画面里的轴偏/沿轴各动多少像素），
不用任何写死的像素/毫米换算。**不下探、不合爪。**

    G:/anaconda/envs/sim2real/python.exe -u _wrist_center.py
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

HOVER_ABOVE = 0.026       # 两爪中点停在方块中心之上多少（米）
PROBE_MM = 6.0            # 探响应：指令走多少毫米
GAIN = 0.6                # 每步只走算出位移的这么多
MAX_STEP_MM = 8.0
TOL_AXIS_PX = 10.0        # 算「对准了」的轴偏（判据那边是 35）
GAP_BAND = (20.0, 75.0)
MAX_ITERS = 10


def look(rig, tag: str) -> dict:
    return _jaw_check(rig.cams["real_wrist"], rig.percep, rig.arm, tag)


def move_tcp(rig, dp_m):
    t = np.asarray(rig.arm.tcp(), dtype=np.float64).reshape(-1) + np.asarray(dp_m)
    return _goto(rig.arm, t, vmax=0.2, grip=+1.0, tries=2)


def err(res: dict) -> tuple[float, float] | None:
    a = res.get("axis")
    if a is None:
        return None
    if not res.get("axis_in"):
        return float(a), 0.0          # 轴还没穿过方块：只纠垂直那一维
    g = res.get("gap")
    if g is None:
        return float(a), 0.0
    return float(a), float(g) - 45.0


def main() -> int:
    rig = Rig(cfg_path=str(ROOT / "configs" / "real.yaml"))
    try:
        print(f"感知相机 = {rig.percep.cam_name}")
        rig._grip("open", None)

        c, f = rig.percep.locate(rig.color)
        if c is None:
            print(f"✗ 腕相机没量到方块：{f.get('no_world_why') or f.get('why')}")
            return 1
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        z_target = float(c[2]) + HOVER_ABOVE
        tcp = np.asarray(rig.arm.tcp(), dtype=np.float64).reshape(-1)
        print(f"方块（腕相机）({c[0] * 1000:+.1f}, {c[1] * 1000:+.1f}, "
              f"{c[2] * 1000:+.1f})mm   现 TCP z {tcp[2] * 1000:+.1f}mm "
              f"→ 悬停 {z_target * 1000:+.1f}mm")
        if abs(tcp[2] - z_target) > 0.004:
            tgt = tcp.copy()
            tgt[2] = z_target
            print(f"   降高度 → {move_tcp(rig, tgt - tcp)}")

        J = None
        prev_dp = None
        prev_e = None
        for it in range(MAX_ITERS):
            res = look(rig, f"腕对准{it}")
            e = err(res)
            if e is None:
                print("✗ 这一眼量不出轴偏")
                return 1
            print(f"[{it}] 轴偏 {res.get('axis'):+.0f}px  沿轴 {res.get('along'):+.0f}  "
                  f"缝 {('—' if res.get('gap') is None else format(res['gap'], '+.0f'))}  "
                  f"轴穿过方块={res.get('axis_in')}  目标 "
                  f"({e[0]:+.0f}, {e[1]:+.0f})")
            if abs(e[0]) <= TOL_AXIS_PX and (not res.get("axis_in")
                                             or GAP_BAND[0] <= (res.get("gap") or 0)
                                             <= GAP_BAND[1]):
                print("  ✓ 画面这一眼对上了")
                break

            if J is None:
                # 探响应：+x 走 PROBE_MM、+y 走 PROBE_MM，各看一眼
                base = np.asarray(rig.arm.tcp(), dtype=np.float64).reshape(-1)
                cols = []
                for k in range(2):
                    d = np.zeros(3)
                    d[k] = PROBE_MM / 1000.0
                    move_tcp(rig, d)
                    r2 = look(rig, f"探{k}")
                    e2 = err(r2)
                    cur = np.asarray(rig.arm.tcp(), dtype=np.float64).reshape(-1)
                    if e2 is None:
                        raise RuntimeError("探响应那一眼量不出轴偏")
                    col = np.array([e2[0] - e[0], e2[1] - e[1]])
                    cols.append(col)
                    moved = (cur - base) * 1000.0
                    print(f"   探 {'xy'[k]} +{PROBE_MM:.0f}mm → 轴偏/缝 "
                          f"动了 ({col[0]:+.0f}, {col[1]:+.0f})px  "
                          f"TCP 实际 ({moved[0]:+.1f}, {moved[1]:+.1f})mm")
                    if float(np.linalg.norm(moved)) < 1.0:
                        print("✗ 指令走了但臂没动（stall），探不出响应，停下")
                        return 1
                    move_tcp(rig, base - cur)
                J = np.column_stack(cols) / (PROBE_MM / 1000.0)   # px per meter
                det = float(np.linalg.det(J)) * 1e6
                print(f"   响应矩阵（px/mm）{np.round(J * 1e-3, 1).tolist()} det {det:.1f}")
                if abs(det) < 1e-6:
                    print("✗ 两个方向在画面里几乎动成一回事，解不出来")
                    return 1

            dp = np.linalg.solve(J, -np.asarray(e)) * GAIN
            n = float(np.linalg.norm(dp))
            if n > MAX_STEP_MM / 1000.0:
                dp *= (MAX_STEP_MM / 1000.0) / n
            print(f"   走 ({dp[0] * 1000:+.1f}, {dp[1] * 1000:+.1f})mm → "
                  f"{move_tcp(rig, np.array([dp[0], dp[1], 0.0]))}")

        res = look(rig, "最后一眼")
        src = Path(knobs.JAW_CHECK_IMG)
        if src.exists():
            dst = ROOT / "out" / f"wrist_center_{time.strftime('%H%M%S')}.png"
            shutil.copyfile(src, dst)
            print(f"\n图：{dst}")
        mid = np.asarray(rig.arm.jaw_gap_center(), dtype=np.float64)
        print(f"停在：两爪中点 ({mid[0] * 1000:+.1f}, {mid[1] * 1000:+.1f}, "
              f"{mid[2] * 1000:+.1f})mm  爪角 {float(np.degrees(rig.arm.grip_q())):+.1f}°")
        print(f"最后读数：轴偏 {res.get('axis')}  沿轴 {res.get('along')}  "
              f"缝 {res.get('gap')}  轴穿过方块 {res.get('axis_in')}  ok {res.get('ok')}")
        return 0
    finally:
        rig.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
