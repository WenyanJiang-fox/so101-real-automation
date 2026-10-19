"""离线：下探余量改「中心 / 顶面两条取大」之后，各尺寸盒子算出来是多少。**不动臂、不碰串口。**

判据只有一条：**18mm 那个盒子（现在唯一跑通的那件）必须跟改之前一模一样 = 25mm。**

改之前：`焦点 = 中心 + GRASP_Z_MARGIN_REAL`（固定 25mm）。
改之后：`焦点 = 中心 + max(GRASP_Z_MARGIN_REAL, 半高 + GRASP_Z_OVER_TOP_REAL)`。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

from core.shapes import Box, Cylinder                              # noqa: E402
from skills.pick_real import (GRASP_Z_MARGIN_REAL,                 # noqa: E402
                              GRASP_Z_OVER_TOP_REAL)


def row(label: str, rest_z: float) -> None:
    """`rest_z` = 物体半高（米）。中心离桌面就是这个数。"""
    old = GRASP_Z_MARGIN_REAL
    new = max(GRASP_Z_MARGIN_REAL, rest_z + GRASP_Z_OVER_TOP_REAL)
    over_old = old - rest_z
    over_new = new - rest_z
    same = abs(new - old) < 1e-12
    print(f"   {label:22s} 半高 {rest_z * 1000:5.1f}mm"
          f"  旧余量 {old * 1000:5.1f}  新余量 {new * 1000:5.1f}"
          f"  新下探高度 {new * 1000:5.1f}mm"
          f"  离顶面 旧 {over_old * 1000:5.1f} → 新 {over_new * 1000:5.1f}mm"
          f"   {'← 与改之前相同 ✓' if same else ''}")


def main() -> int:
    print(f"GRASP_Z_MARGIN_REAL   = {GRASP_Z_MARGIN_REAL * 1000:.1f}mm（从中心量）")
    print(f"GRASP_Z_OVER_TOP_REAL = {GRASP_Z_OVER_TOP_REAL * 1000:.1f}mm（从顶面量）")
    print("\n== 正方体，边长一路翻倍")
    for side in (12.0, 18.0, 24.0, 30.0, 36.0, 50.0, 70.0):
        row(f"边长 {side:.0f}mm", Box(side / 2000.0).rest_z)
    print("\n== 长方体（高矮不一，看的是高那一维）")
    for a, b, h in ((40, 30, 20), (50, 20, 20), (100, 80, 20), (60, 40, 60)):
        row(f"{a}x{b}x{h}mm", Box(a / 2000.0, b / 2000.0, h / 2000.0).rest_z)
    print("\n== 圆柱")
    for d, h in ((40, 40), (45, 45), (60, 80)):
        row(f"Φ{d}×{h}mm", Cylinder(d / 2000.0, h / 2000.0).rest_z)
    print("\n== 边界：新余量从哪一档开始超过 25mm")
    for side in (18.0, 18.1, 19.0, 20.0):
        rz = Box(side / 2000.0).rest_z
        new = max(GRASP_Z_MARGIN_REAL, rz + GRASP_Z_OVER_TOP_REAL)
        print(f"   边长 {side:5.1f}mm  新余量 {new * 1000:6.2f}mm"
              f"   {'（还是 25，没顶起来）' if abs(new - GRASP_Z_MARGIN_REAL) < 1e-12 else '（顶起来了）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
