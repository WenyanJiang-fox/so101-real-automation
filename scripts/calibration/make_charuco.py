"""出一张 ChArUco 标定板，给平板全屏显示用。

**这张图本身没有尺寸。** `squareLength` 只在检测时起比例作用（码/格 = 0.75），
真正的物理边长是**板子显示出来之后拿尺子量出来的那个数**，到标定那一步才填进去。
所以图上不写毫米，写了也是假的 —— 同一个 PNG 在 11 寸和 13 寸平板上是不一样的板。

**为什么不是纯棋盘格。** ChArUco 每个格子有身份（黑白码），板子被挡住一角，
缺的角点能靠剩下的码补出来。机械臂标定时爪子就挡在画面里，这点很实用。

不用联网、不用装东西，cv2 自带（这台机上 5.0.0.93 实测通过）。

跑法（在 G:\\sim2real\\so101-real-harness 下）：
    G:/anaconda/envs/sim2real/python.exe make_charuco.py
    G:/anaconda/envs/sim2real/python.exe make_charuco.py --px 2360x1640 --out out/charuco_ipad.png

出图：G:\\sim2real\\so101-real-harness\\out\\charuco_8x6.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 码字典。**要 5x5 那一档，而且 ID 要够多** —— 4x4_50 只有 50 个码，
# 8x6 的板子要用 24 个，看着够，但留不下换板型的余地。
DICT_ID = cv2.aruco.DICT_5X5_1000

# 码占格子的比例。0.75 是 OpenCV 和 calib.io 都用惯的值，
# 四周留一圈白边，码才不会被相邻格子贴脸。
MARKER_RATIO = 0.75


def build(squares_x: int, squares_y: int):
    """按格数出板子。squareLength 随便给一个 —— 见文件头，它只是比例。"""
    d = cv2.aruco.getPredefinedDictionary(DICT_ID)
    board = cv2.aruco.CharucoBoard(
        (int(squares_x), int(squares_y)), 1.0, MARKER_RATIO, d)
    return d, board


def render(board, px: tuple[int, int], margin: int) -> np.ndarray:
    """把板子画成 px 这么大的图。板子按自己的长宽比缩进去，不拉伸。"""
    img = board.generateImage((int(px[0]), int(px[1])), marginSize=int(margin), borderBits=1)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def check(board, img) -> tuple[int, int]:
    """拿检测器回过头认一遍自己画的图 —— 认不全就说明这张图不能用。

    返回 (认出几个角点, 板子一共几个角点)。
    """
    det = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = det.detectBoard(img)
    total = board.getChessboardSize()
    want = (int(total[0]) - 1) * (int(total[1]) - 1)
    got = 0 if ch_corners is None else len(ch_corners)
    return got, want


def square_px(board, img) -> float:
    """画出来之后一个格子有多少像素 —— 用来反推屏幕上的实际边长。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    det = cv2.aruco.CharucoDetector(board)
    ch_corners, _, _, _ = det.detectBoard(gray)
    if ch_corners is None or len(ch_corners) < 4:
        return float("nan")
    p = np.asarray(ch_corners, dtype=np.float64).reshape(-1, 2)
    d = np.linalg.norm(p[None, :, :] - p[:, None, :], axis=2)
    d[d < 1e-6] = np.inf
    return float(np.median(d.min(axis=1)))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="出 ChArUco 标定板图（平板显示用）")
    ap.add_argument("--px", default="2400x1800",
                    help="图的像素尺寸 WxH。**跟平板原生分辨率对齐最清楚**，"
                         "不知道就先用默认，全屏显示时别拉伸。")
    ap.add_argument("--squares", default="8x6", help="横 x 竖 的格子数")
    ap.add_argument("--margin", type=int, default=40, help="四周留白像素")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "charuco_8x6.png")
    args = ap.parse_args()

    w, h = (int(v) for v in args.px.lower().split("x"))
    sx, sy = (int(v) for v in args.squares.lower().split("x"))
    d, board = build(sx, sy)

    img = render(board, (w, h), args.margin)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img)

    got, want = check(board, img)
    spx = square_px(board, img)

    print(f"图    {args.out}")
    print(f"       {img.shape[1]}x{img.shape[0]} px，格子 {sx}x{sy}，码 {len(board.getIds())} 个")
    print(f"自检  认出 {got}/{want} 个角点" + ("  认证通过" if got == want else "  **认不全，这张图不能用**"))
    print(f"格子  每个约 {spx:.1f} px")
    print()
    print("下一步：平板全屏打开这张图（保持长宽比，别拉伸），")
    print("       拿尺子量一个格子的实际边长（毫米），把那个数报回来。")
    return 0 if got == want else 1


if __name__ == "__main__":
    raise SystemExit(main())
