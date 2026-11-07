"""离线：长方体上 `Box.normal` 到底挑哪一对侧面。**不动臂、不碰串口。**

用户 2026-10-17 定的口径：**爪垫贴在面积最大的那一对面上，两爪跨最短的那一维**
（例子是「细长的棍子不能夹两个圆面，得抓棍子长边」）。

对 `a > b` 的长方体：两对侧面的跨度是 `a` 和 `b`。跨 `b` 那一对，爪垫贴的是
`a×h` 两个大面 —— 就是要的那一对。所以判据 = **取跨度最小的那一对**。

这个探针把当前 `Box.normal` 在「两对都塞得进 56mm」时挑了什么打出来，
改完再跑一遍对比。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Windows 控制台默认 GBK，平方号 `²` 打不出来。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.shapes import MAX_JAW_SPAN, Box                            # noqa: E402

# (a, b, h) 三个**全长**（毫米）
CASES = ((40.0, 30.0, 20.0),      # 两对都塞得进 56mm —— 要看它挑哪一对
         (50.0, 20.0, 20.0),      # 同上，更悬殊
         (200.0, 10.0, 10.0),     # 细长棍：只有跨直径那一对塞得进
         (100.0, 80.0, 20.0),     # 两对都塞不进 —— 观察它退回什么
         (30.0, 30.0, 30.0))      # 正方体：两对等价，改动前后应当一模一样


def which(bx: Box, yaw: float, tool_xy) -> tuple[str, float, float]:
    """返回 (挑中的正常量是 ±x 还是 ±y, 跨度, 爪垫贴的那两个面的面积)。"""
    n = bx.normal(yaw, tool_xy)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    dx = float(np.dot(n[:2], R[:, 0]))
    dy = float(np.dot(n[:2], R[:, 1]))
    if abs(dx) >= abs(dy):
        return "±x", 2.0 * bx.half_x, 2.0 * bx.half_y * 2.0 * bx.half_z
    return "±y", 2.0 * bx.half_y, 2.0 * bx.half_x * 2.0 * bx.half_z


def main() -> int:
    print(f"MAX_JAW_SPAN = {MAX_JAW_SPAN * 1000:.0f}mm")

    print("\n== 新旧逐组对比（旧 = 改之前那份 `normal`，抄在这里）")
    import numpy as np2
    from core.shapes import _h_axis, _yaw_matrix                  # noqa: PLC0415

    def old_normal(bx: Box, yaw: float, tool_xy) -> np.ndarray:
        want = _h_axis(tool_xy)
        R = _yaw_matrix(yaw)
        pairs = ((2.0 * bx.half_y, R[:, 1]), (2.0 * bx.half_x, R[:, 0]))
        fits = [v for span, v in pairs if span <= MAX_JAW_SPAN]
        cands = fits if fits else [v for _span, v in pairs]
        out = []
        for v in cands:
            out += [v, -v]
        return max(out, key=lambda v: float(np.dot(v[:2], want[:2])))

    def span_of(bx: Box, yaw: float, n) -> float:
        """这条法向跨的那一维有多宽。"""
        R = _yaw_matrix(yaw)
        if abs(float(np2.dot(n, R[:, 0]))) >= abs(float(np2.dot(n, R[:, 1]))):
            return 2.0 * bx.half_x
        return 2.0 * bx.half_y

    cases = ((40.0, 30.0, 20.0), (50.0, 20.0, 20.0), (200.0, 10.0, 10.0),
             (100.0, 80.0, 20.0), (30.0, 30.0, 30.0), (56.0, 56.0, 20.0),
             (56.0, 55.0, 20.0), (60.0, 40.0, 20.0), (25.0, 25.0, 60.0))
    tot = dif = 0
    n_wider = n_narrower = 0
    for a_mm, b_mm, h_mm in cases:
        bx = Box(a_mm / 2000.0, b_mm / 2000.0, h_mm / 2000.0)
        n_diff = n_up = n_dn = 0
        for yaw_deg in np2.arange(0.0, 180.0, 3.0):
            for tool_deg in np2.arange(0.0, 360.0, 3.0):
                tool = np2.array([np2.cos(np2.radians(tool_deg)),
                                  np2.sin(np2.radians(tool_deg))])
                yr = np2.radians(float(yaw_deg))
                o = old_normal(bx, yr, tool)
                n = bx.normal(yr, tool)
                tot += 1
                if float(np.dot(o, n)) < 0.9999:
                    dif += 1
                    n_diff += 1
                    so, sn = span_of(bx, yr, o), span_of(bx, yr, n)
                    if sn < so - 1e-9:
                        n_narrower += 1
                        n_dn += 1
                    elif sn > so + 1e-9:
                        n_wider += 1
                        n_up += 1
                elif span_of(bx, yr, n) > span_of(bx, yr, o) + 1e-9:
                    n_up += 1          # 方向没变，但同一对里换了 ± —— 不追
        print(f"   {a_mm:5.0f} x {b_mm:5.0f} x {h_mm:3.0f}mm  "
              f"{'正方截面' if abs(a_mm - b_mm) < 1e-9 else '长方截面'}"
              f"   变了 {n_diff:4d} 组"
              f"（改成更窄那一对 {n_dn}，改成更宽那一对 {n_up}）")
    print(f"   合计变了 {dif} / {tot} 组；其中改成更窄那一对 {n_narrower}，"
          f"改成更宽那一对 {n_wider}")
    print("   判据：正方截面那几行必须 0 组；变了的那几组必须**全部**是「改成更窄那一对」，"
          "「改成更宽」必须是 0。")

    for a_mm, b_mm, h_mm in CASES:
        bx = Box(a_mm / 2000.0, b_mm / 2000.0, h_mm / 2000.0)
        print(f"\n== {a_mm:.0f} x {b_mm:.0f} x {h_mm:.0f}mm"
              f"   跨度 {a_mm:.0f} / {b_mm:.0f}mm"
              f"   {'两对都塞得进' if max(a_mm, b_mm) <= MAX_JAW_SPAN * 1000 else '只有一对（或没有）塞得进'}")
        n_small = 0
        for yaw_deg in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0):
            for tool_deg in (0.0, 30.0, 60.0, 90.0, 120.0):
                yaw = np.radians(yaw_deg)
                tool = np.array([np.cos(np.radians(tool_deg)),
                                 np.sin(np.radians(tool_deg))])
                axis, span, face = which(bx, yaw, tool)
                small = span <= min(a_mm, b_mm) / 1000.0 + 1e-9
                n_small += int(small)
                if tool_deg == 0.0:
                    print(f"   yaw {yaw_deg:3.0f}°  tool {tool_deg:3.0f}°"
                          f"  → {axis}  跨 {span * 1000:5.1f}mm"
                          f"  爪垫贴的面 {face * 1e6:6.0f}mm²"
                          f"  {'← 小的那对 ✓' if small else '← 大的那对 ✗'}")
        print(f"   30 组合里挑中「小跨度那一对」的有 {n_small} 组")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
