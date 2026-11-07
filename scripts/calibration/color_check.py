"""模块 B：从一张（腕部相机）图里判夹住的是哪个颜色。

方法：HSV 阈值分割。纯几何，没有模型，中间结果都能打印出来自己核对。
    · 红色跨色相 0 点，所以有两段区间（0 附近和 179 附近）。
    · 输出三个颜色各自的像素占比，占比最大的那个算数。
    · 三个占比都低于 min_ratio -> "unknown"，不硬猜。

抗光照的备选：channel_order_guess()，不看色相，只看"有明显颜色的那些像素里
B / G / R 哪个通道最强"。换灯、调亮度都不用重标。用来和 HSV 结果交叉验证。

标定：跑 calibrate_color.py 得出每个颜色的 HSV 区间，写进 thresholds.json。

命令行：
    python color_check.py --image G:\\...\\shot.jpg
    python color_check.py --camera 0
    python color_check.py --camera 0 --n 20 --save G:\\...\\samples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[2]
TH_PATH = HERE / "thresholds.json"

COLORS = ("red", "green", "blue")


def load_thresholds(path: Path = TH_PATH) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_thresholds(data: dict, path: Path = TH_PATH) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def color_cal(cal: dict) -> dict:
    c = cal.get("color") or {}
    missing = [k for k in COLORS if not c.get("hsv", {}).get(k)]
    if missing:
        raise RuntimeError(
            f"thresholds.json 里的 color.hsv 还没标定（缺 {missing}）。"
            f"先跑：python {HERE / 'calibrate_color.py'}"
        )
    return c


def crop_roi(frame_bgr: np.ndarray, roi) -> np.ndarray:
    """roi = 归一化 [x0, y0, x1, y1]。None 或 [0,0,1,1] 就是整幅。"""
    if not roi:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = roi
    if (x0, y0, x1, y1) == (0.0, 0.0, 1.0, 1.0):
        return frame_bgr
    return frame_bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def foreground_mask(frame_bgr: np.ndarray, s_min: int = 60, v_min: int = 50) -> np.ndarray:
    """不管是什么色相，只要"够彩、够亮"就算前景。用来把桌面/背景剔掉。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([0, s_min, v_min], np.uint8),
                       np.array([179, 255, 255], np.uint8))


def color_ratios(frame_bgr: np.ndarray, cal: dict) -> dict:
    """三个颜色各自占"前景像素"的比例，加起来可能不到 1（有彩但不像三色的像素）。

    分母是前景像素数（不是全图），这样对手离得远近不敏感。
    """
    c = color_cal(cal)
    roi = crop_roi(frame_bgr, c.get("roi"))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    fg = foreground_mask(roi)
    n_fg = int(cv2.countNonZero(fg))
    if n_fg == 0:
        return {name: 0.0 for name in COLORS}

    out = {}
    for name in COLORS:
        mask = np.zeros(fg.shape, np.uint8)
        for lo, hi in c["hsv"][name]:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.bitwise_and(mask, fg)
        out[name] = int(cv2.countNonZero(mask)) / n_fg
    return out


def which_color(frame_bgr: np.ndarray, cal: dict):
    """返回 (颜色名 or 'unknown', 三色占比 dict)。占比并列或都太低就是 unknown。"""
    c = color_cal(cal)
    ratios = color_ratios(frame_bgr, cal)
    best = max(ratios, key=ratios.get)
    if ratios[best] < c.get("min_ratio", 0.03):
        return "unknown", ratios
    return best, ratios


def channel_order_guess(frame_bgr: np.ndarray, cal: dict):
    """不看色相，只看前景像素里 B / G / R 哪个通道均值最大。返回 (颜色名, 通道均值 dict)。

    这个判据对灯光色温不敏感，换灯不用重标。但红和蓝在某些白平衡下会互相串，
    所以当成交叉验证用，别单独当唯一判据。
    """
    c = cal.get("color") or {}
    roi = crop_roi(frame_bgr, c.get("roi"))
    fg = foreground_mask(roi)
    if int(cv2.countNonZero(fg)) == 0:
        return "unknown", {"blue": 0.0, "green": 0.0, "red": 0.0}
    b, g, r = (float(cv2.mean(ch, mask=fg)[0]) for ch in cv2.split(roi))
    means = {"blue": b, "green": g, "red": r}
    return max(means, key=means.get), means


# ---------------------------------------------------------------- 取帧

def open_camera(index: int, size: tuple[int, int] | None = None,
                fourcc: str | None = None):
    """开相机。`size` / `fourcc` 不给就是老行为（什么都不设，用相机自己的默认档）。

    **要高清就得走这一条而且两样都得给。** USB 摄像头默认出 640x480；`cap.set` 在
    DSHOW 上还经常返回 True 而实际没生效，所以分辨率和 MJPG 都得显式要一遍。标定
    是按 1920x1080 做的，拿 640x480 的画面套那份内参，反推出来的坐标全错。
    （同一套写法在 `real/calib_views.py::open_cam` 里已经跑通了。）
    """
    if size is None and fourcc is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(size[0]))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size[1]))
    if not cap.isOpened():
        raise RuntimeError(f"打不开相机 index={index}。换个数试试，或者用 --image 喂文件。")
    return cap


def grab_frame(cap, warmup: int = 5):
    """连读几帧丢缓存 —— 打开相机后的头几帧经常是空的或过曝的。"""
    frame = None
    for _ in range(max(1, warmup)):
        ok, frame = cap.read()
        if not ok:
            frame = None
    if frame is None:
        raise RuntimeError("相机读不出帧")
    return frame


def _main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="颜色判定（模块 B）")
    ap.add_argument("--image", type=Path, help="读一张图")
    ap.add_argument("--camera", type=int, help="读相机，如 --camera 0")
    ap.add_argument("--n", type=int, default=1, help="读几张/连续判几次")
    ap.add_argument("--every", type=float, default=0.2, help="连续判时的间隔秒数")
    ap.add_argument("--save", type=Path, help="配合 --camera，把读到的帧存到这个目录（标定采样用）")
    ap.add_argument("--test-hue", action="store_true",
                    help="不管阈值，直接打印前景像素的色相直方图，用来定区间")
    args = ap.parse_args()

    cal = load_thresholds()

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            print(f"读不出来：{args.image}")
            return 1
        _report(frame, cal, args.test_hue)
        return 0

    if args.camera is None:
        ap.error("要么 --image，要么 --camera")
    cap = open_camera(args.camera)
    try:
        if args.save:
            args.save.mkdir(parents=True, exist_ok=True)
        for i in range(max(1, args.n)):
            frame = grab_frame(cap)
            if args.save:
                p = args.save / f"{int(time.time() * 1000)}_{i:03d}.jpg"
                cv2.imwrite(str(p), frame)
                print(f"存了 {p}")
            _report(frame, cal, args.test_hue)
            if i + 1 < args.n:
                time.sleep(args.every)
    finally:
        cap.release()
    return 0


def _report(frame, cal: dict, test_hue: bool) -> None:
    if test_hue:
        c = cal.get("color") or {}
        roi = crop_roi(frame, c.get("roi"))
        fg = foreground_mask(roi)
        if int(cv2.countNonZero(fg)) == 0:
            print("前景像素为 0 —— 画面里没有够彩的东西（太暗？太灰？）")
            return
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], fg, [180], [0, 180]).ravel()
        top = np.argsort(hist)[::-1][:6]
        print("前景像素色相直方图前 6 名（H, 像素数）：")
        for h in top:
            if hist[h] > 0:
                print(f"  H={h:3d}  {int(hist[h])}")
        return

    if not _is_calibrated(cal):
        print("thresholds.json 的 color 还没标定，只打印原始色相。加 --test-hue 看直方图。")
        return

    name, ratios = which_color(frame, cal)
    guess, means = channel_order_guess(frame, cal)
    print(f"颜色 = {name}")
    print("  三色占比：" + "  ".join(f"{k}={v:.3f}" for k, v in ratios.items()))
    print("  通道均值交叉验证：" + "  ".join(f"{k}={v:6.1f}" for k, v in means.items())
          + f"  -> {guess}")
    if name != "unknown" and guess != "unknown" and name != guess:
        print("  ！两个判据不一致，这帧别信。")


def _is_calibrated(cal: dict) -> bool:
    c = cal.get("color") or {}
    return all(c.get("hsv", {}).get(k) for k in COLORS)


if __name__ == "__main__":
    sys.exit(_main())
