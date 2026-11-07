"""标定模块 B 的 HSV 区间：喂进某颜色的样本图，算出区间写进 thresholds.json。

输入两种：
    python calibrate_color.py --color red  --images "G:\\sim2real\\so101-real-harness\\samples\\red\\*.jpg"
    python calibrate_color.py --color red  --camera 0 --n 20

只统计「够彩」的像素（S>=SAMPLE_S），所以桌面、白墙、灰背景会被自动剔掉，
不用手动裁 ROI。要是背景里也有大片同色，就得裁 —— 用 thresholds.json 的
color.roi 把画面限制住，再重标。

色相跨 0 的处理：把 H 从 0-179 平移到 -90..89，红色就连续了，取完分位再平移回来。
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from color_check import COLORS, crop_roi, load_thresholds, open_camera, save_thresholds  # noqa: E402

SAMPLE_S = 50          # 采样阶段认定"够彩"的饱和度下限（比判定时的 60 松一点，多留样本）
SAMPLE_V = 40
P_LO, P_HI = 5, 95     # 色相取这两档分位，砍掉两头的离群像素


def _shift(h: np.ndarray) -> np.ndarray:
    """0..179 -> -90..89，让红色跨 0 的问题消失。"""
    return np.where(h > 90, h - 180, h)


def _unshift(x: np.ndarray) -> np.ndarray:
    return np.where(x < 0, x + 180, x)


def collect_pixels(frames: list[np.ndarray], roi) -> np.ndarray:
    """把所有帧里够彩的像素的 HSV 拼成 (N, 3)。"""
    chunks = []
    for f in frames:
        r = crop_roi(f, roi)
        hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, SAMPLE_S, SAMPLE_V], np.uint8),
                           np.array([179, 255, 255], np.uint8))
        if cv2.countNonZero(mask) == 0:
            continue
        chunks.append(hsv[mask > 0])
    if not chunks:
        raise RuntimeError("所有样本里都没有够彩的像素 —— 图太暗/太灰/拍的不是方块？")
    return np.concatenate(chunks, axis=0)


def hue_ranges(hsv_px: np.ndarray) -> list:
    """算出 [ [lo,hi], ... ] 的色相段（可能两段），每段再配上 S/V 的下限。"""
    h = _shift(hsv_px[:, 0].astype(np.int32))
    lo = int(np.percentile(h, P_LO))
    hi = int(np.percentile(h, P_HI))
    lo, hi = int(_unshift(np.array(lo))), int(_unshift(np.array(hi)))

    s_lo = int(max(50, np.percentile(hsv_px[:, 1], P_LO)))
    v_lo = int(max(40, np.percentile(hsv_px[:, 2], P_LO)))

    if lo > hi:                      # 跨 0 了，拆两段
        return [[lo, s_lo, v_lo], [179, 255, 255]], [[0, s_lo, v_lo], [hi, 255, 255]]
    return [[lo, s_lo, v_lo], [hi, 255, 255]],


def _main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="标定颜色 HSV 区间")
    ap.add_argument("--color", required=True, choices=list(COLORS))
    ap.add_argument("--images", help="glob，例如 G:\\...\\samples\\red\\*.jpg")
    ap.add_argument("--camera", type=int, help="或者直接用相机采")
    ap.add_argument("--n", type=int, default=20, help="--camera 时采几张")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    cal = load_thresholds()
    ccal = cal.get("color") or {}
    roi = ccal.get("roi")

    frames = []
    if args.images:
        paths = sorted(globmod.glob(args.images))
        if not paths:
            print(f"这个 glob 一个文件都没匹配到：{args.images}")
            return 1
        for p in paths:
            f = cv2.imread(p)
            if f is None:
                print(f"  读不出来，跳过：{p}")
                continue
            frames.append(f)
        print(f"读了 {len(frames)} 张图")
    elif args.camera is not None:
        cap = open_camera(args.camera)
        try:
            from color_check import grab_frame
            for i in range(args.n):
                frames.append(grab_frame(cap))
                print(f"  采了 {i + 1}/{args.n}", end="\r", flush=True)
                time.sleep(0.15)
            print()
        finally:
            cap.release()
    else:
        ap.error("要么 --images，要么 --camera")

    if not frames:
        print("没有可用样本。")
        return 1

    px = collect_pixels(frames, roi)
    print(f"够彩的像素共 {len(px)} 个")
    hs = _shift(px[:, 0].astype(np.int32))
    print(f"  色相（平移后）：p{P_LO}={np.percentile(hs, P_LO):.0f}  "
          f"中位={np.percentile(hs, 50):.0f}  p{P_HI}={np.percentile(hs, P_HI):.0f}")
    print(f"  饱和度：p{P_LO}={np.percentile(px[:, 1], P_LO):.0f}  "
          f"亮度：p{P_LO}={np.percentile(px[:, 2], P_LO):.0f}")
    if np.percentile(px[:, 1], 50) < 90:
        print("  ！饱和度中位数偏低，光太散或者方块脏了，判定会不稳。")

    ranges = list(hue_ranges(px))
    print(f"\n{args.color} 的区间：")
    for lo, hi in ranges:
        print(f"  H {lo[0]:3d}-{hi[0]:3d}  S>={lo[1]:3d}  V>={lo[2]:3d}")
    if len(ranges) > 1:
        print("  （两段 —— 色相跨了 0 点，正常）")

    if args.no_write:
        print("--no-write，没写文件。")
        return 0

    data = load_thresholds()
    cc = data.setdefault("color", {})
    cc.setdefault("hsv", {})[args.color] = ranges
    cc["_calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_thresholds(data)
    print(f"\n写好了：{HERE / 'thresholds.json'}")
    print(f"三个色都标完之后，跑 python color_check.py --image <一张测试图> 验证。")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
