"""离线：`_box_center_px` 换成 V 通道之后，换颜色的方块还量不量得出顶面中心。**不动臂、不碰串口。**

造一张合成的剪影：轮廓是个上下两块的长方形（下面那块 = 正面、暗；上面那块 = 顶面、亮），
亮块的真中心在 (100, 75)。拿红/绿/蓝三种颜色各跑一遍 R 通道和 V 通道。

判据：V 通道三种颜色都该报回 (100, 75)；R 通道只该在红块上报对。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

import cv2                                                         # noqa: E402

from skills import pick_real as PR                                 # noqa: E402

# 每种颜色给一个本色 BGR（满亮时是什么样的）。实际亮度 = 本色 × 照度场。
COLORS = {
    "red":   (0, 0, 255),
    "green": (0, 255, 0),
    "blue":  (255, 0, 0),
}
TRUE_UV = np.array([99.5, 74.5])        # 顶面那一块（上半 100×50）的中心

# **照度场不能是两级台阶。** 第一版就是「顶面恒 1.0、正面恒 0.55」两个值 —— 二值分布的
# Otsu 阈值在两级之间处处并列，`cv2.threshold` 返回最低那个，等于两半全被算成「亮」，
# 报回来的是整块重心。真实画面本来也有光斑渐变，所以这里给两半各一条小梯度。
_FRONT = np.linspace(0.55, 0.75, 50)[:, None]         # 正面：上暗下亮
_TOP = np.linspace(0.92, 1.00, 50)[:, None]           # 顶面：受光，整体亮


def make(base) -> tuple[np.ndarray, np.ndarray]:
    f = np.full((200, 200), 0.40)                     # 背景
    f[50:100, 50:150] = _TOP
    f[100:150, 50:150] = _FRONT
    img = np.clip(f[:, :, None] * np.asarray(base, np.float64), 0, 255).astype(np.uint8)
    cnt = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], np.int32)
    return img, cnt


def run(chan: str) -> None:
    old = PR.knobs.BOX_CENTER_CHAN
    PR.knobs.BOX_CENTER_CHAN = chan
    print(f"\n== 通道 {chan!r}")
    for name, base in COLORS.items():
        img, cnt = make(base)
        uv = PR._box_center_px(cnt, img)
        if uv is None:
            print(f"   {name:6s} → None（量不出来）")
            continue
        off = float(np.linalg.norm(np.asarray(uv, float) - TRUE_UV))
        print(f"   {name:6s} → ({uv[0]:6.1f}, {uv[1]:6.1f})"
              f"   真值 (99.5, 74.5)   差 {off:5.1f}px   "
              f"{'✓' if off < 1.0 else '✗ 落在整块的重心上，不是顶面'}")
    PR.knobs.BOX_CENTER_CHAN = old


def real_frame() -> int:
    """拿真机跑过那一趟存下来的那一帧（`out/look_see.jpg`）再比一遍。"""
    from real import perception as P                              # noqa: PLC0415

    path = Path(__file__).resolve().parents[2] / "out" / "look_see.jpg"
    img = cv2.imread(str(path))
    if img is None:
        print(f"\n== 真帧\n   读不到 {path}，跳过。")
        return 0
    ranges, fill_min, min_area = P.load_cal("red")
    mask = P.brown_mask(img, ranges)
    passing = [d for d in P.candidates(mask, min_area)
               if d["rect_fill"] >= fill_min]
    print(f"\n== 真帧 {path.name}  {img.shape[1]}x{img.shape[0]}"
          f"   过矩形度门槛的候选 {len(passing)} 个")
    if not passing:
        print("   这一帧没有合门槛的红块，跳过。")
        return 0
    d = passing[0]
    cnt = d["contour"]
    ys, xs = np.nonzero(cv2.drawContours(
        np.zeros(mask.shape, np.uint8), [cnt], -1, 255, -1) > 0)
    print(f"   选中的那一块：面积 {d['area_px']}px  矩形度 {d['rect_fill']:.2f}"
          f"  轮廓外接框 y {ys.min()}~{ys.max()}"
          f"（**整块的重心** ≈ ({xs.mean():.1f}, {ys.mean():.1f})）")
    got = {}
    for chan in ("r", "v"):
        old = PR.knobs.BOX_CENTER_CHAN
        PR.knobs.BOX_CENTER_CHAN = chan
        uv = PR._box_center_px(cnt, img)
        PR.knobs.BOX_CENTER_CHAN = old
        got[chan] = uv
        if uv is None:
            print(f"   通道 {chan!r} → None（量不出顶面中心）")
        else:
            print(f"   通道 {chan!r} → ({uv[0]:7.1f}, {uv[1]:7.1f})")
    if got.get("r") is not None and got.get("v") is not None:
        off = float(np.linalg.norm(np.asarray(got["r"], float)
                                   - np.asarray(got["v"], float)))
        print(f"   两条通道差 {off:.1f}px"
              f"（红块真机上跑通的是 'r' 那条；'v' 差多少就是这一改带来的位移量）")
    return 0


def main() -> int:
    print(f"当前 `BOX_CENTER_CHAN` = {PR.knobs.BOX_CENTER_CHAN!r}")
    print("（这一项只影响 `_box_center_px` 拿哪个通道做 Otsu；整块重心是 (100, 100)。）")
    run("r")
    run("v")
    rc = real_frame()
    print("\n结论：R 通道在绿块/蓝块上两个半块的读数一样（都是 0），Otsu 劈不出边界，"
          "整块都被当成「亮的那一半」→ 报的是整块重心 (100,100)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
