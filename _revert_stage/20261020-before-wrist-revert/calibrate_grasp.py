"""标定模块 A 的阈值：跑两次采样，一次空抓、一次夹住，写进 thresholds.json。

会动真机（合爪）。跑之前确认：
    · 爪里没有东西的时候不会夹到别的东西（线、桌面）
    · 串口没被 mcp_server.py 占着

    python calibrate_grasp.py --port COM12 --seconds 4

建议连续跑 2~3 遍，看两次算出来的阈值差多少。差得多说明这个夹爪的重复性不够，
阈值要取更保守的一边（本来想留 2° 余量就留 8°）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from grasp_check import load_thresholds, save_thresholds  # noqa: E402


def sample(c, seconds: float, hz: float) -> list[float]:
    """在 seconds 秒内按 hz 采 6 号舵机角度，返回所有成功的读数。"""
    from servo_bridge import read_gripper_deg

    out = []
    n = max(1, int(seconds * hz))
    for _ in range(n):
        a = read_gripper_deg(c, samples=3, delay=0.0)
        if a is not None:
            out.append(a)
        time.sleep(1.0 / hz)
    return out


def describe(tag: str, vals: list[float]) -> None:
    if not vals:
        print(f"  {tag}：一个读数都没拿到（串口/供电有问题？）")
        return
    vals_sorted = sorted(vals)
    p05 = vals_sorted[int(0.05 * (len(vals_sorted) - 1))]
    p95 = vals_sorted[int(0.95 * (len(vals_sorted) - 1))]
    print(f"  {tag}：n={len(vals)}  中位数={statistics.median(vals):6.1f}°  "
          f"最小={vals_sorted[0]:6.1f}°  p05={p05:6.1f}°  "
          f"p95={p95:6.1f}°  最大={vals_sorted[-1]:6.1f}°")


def _main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="标定夹爪判据阈值")
    ap.add_argument("--port", default=None)
    ap.add_argument("--seconds", type=float, default=4.0, help="每组采多久")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=5.0,
                    help="空抓和夹住之间留的余量（度）。重复性差就调大")
    ap.add_argument("--no-write", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    from servo_bridge import open_controller

    cal = load_thresholds()
    port = args.port or (cal.get("gripper") or {}).get("_port") or "COM12"
    c = open_controller(port=port)
    print(f"串口 {port} 已开")

    try:
        print("\n【第 1 组：空抓】爪里不放任何东西，让爪子合到底。")
        print("  先手动把爪子合上（或者等脚本发合爪指令）。准备好按回车。")
        input("  按回车开始采样…")
        from servo_bridge import close_gripper
        close_gripper(c, speed=0)
        time.sleep(0.8)
        print("  采集中…")
        empty = sample(c, args.seconds, args.hz)
        describe("空抓", empty)

        print("\n【第 2 组：夹住】把方块夹起来，抬离桌面，全程保持夹住。")
        print("  合爪之后按回车开始采样，采样期间请保持举起别放下。")
        input("  按回车开始采样…")
        time.sleep(0.3)
        print("  采集中…（保持举着）")
        held = sample(c, args.seconds, args.hz)
        describe("夹住", held)
    finally:
        c.close()

    if not empty or not held:
        print("\n有一组没采到，不写文件。")
        return 1

    e = sorted(empty)
    h = sorted(held)
    empty_max = e[int(0.95 * (len(e) - 1))]        # 空抓的 p95，留上面那 5% 给抖动
    held_min = h[int(0.05 * (len(h) - 1))]         # 夹住的 p05
    empty_max += args.margin
    held_min -= args.margin

    print(f"\n空抓 p95 = {empty_max - args.margin:.1f}°  ->  空抓阈值 empty_max_deg = {empty_max:.1f}°")
    print(f"夹住 p05 = {held_min + args.margin:.1f}°  ->  夹住阈值 held_min_deg  = {held_min:.1f}°")

    if held_min <= empty_max:
        print("\n！！两组重叠了。夹爪的分辨力不够，或者采样时动了。")
        print("   处理办法：把方块换成更宽的，或者增大 --margin，或者改用负载判据。")
        return 1

    gap = held_min - empty_max
    print(f"两阈值间隔 {gap:.1f}°。" + ("间隔偏小，阈值取保守里可能误判。" if gap < 5 else ""))

    if args.no_write:
        print("--no-write，没写文件。")
        return 0

    data = load_thresholds()
    g = data.setdefault("gripper", {})
    g["empty_median_deg"] = round(statistics.median(empty), 2)
    g["empty_max_deg"] = round(empty_max, 2)
    g["held_median_deg"] = round(statistics.median(held), 2)
    g["held_min_deg"] = round(held_min, 2)
    g.setdefault("drop_delta_deg", 10.0)
    g["_calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    g["_port"] = port
    save_thresholds(data)
    print(f"\n写好了：{HERE / 'thresholds.json'}")
    print("下一步跑一遍 python grasp_check.py --check 验证。")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
