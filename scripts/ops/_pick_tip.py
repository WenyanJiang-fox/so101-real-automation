"""在画面里点一下爪尖，把那个像素坐标记下来。

**为什么点一下就够了。** 腕相机和固定爪都长在 `gripper` 那个体上，两者相对位置是
死的（手眼标定 `real_wrist_mount.json` 就是把相机定在 `gripperframe` 上）。所以
**固定爪尖在画面里永远同一个像素**，不管臂怎么动、腕子怎么转。那就不用算投影了 ——
直接把这个像素当准星用。

鼠标：左键 = 点爪尖（可点多次，最后一次算数）；`r` = 清掉重点；`q`/`ESC` = 存盘退出。
点错也没事，会画在上面，不满意就重来。

跑：`G:/anaconda/envs/sim2real/python.exe _pick_tip.py out/close_f100.jpg`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(r"G:\sim2real\so101-real-harness")
OUT_JSON = _ROOT / "assets" / "camera_calibration" / "real_wrist_tip_px.json"

# 显示尺寸。**不能用 WINDOW_AUTOSIZE**：屏幕 125% 缩放，1080p 会被撑成 2400x1350
# 顶出屏幕（见 memory `opencv-window-dpi-scaling`）。这条留着的另一个原因是手臂上的
# 键盘/鼠标事件只在 `waitKey` 泵消息时进来。
DISP_W, DISP_H = 1280, 720
NAME = "click fixed-jaw tip, then press q"


def main() -> int:
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "out" / "close_f100.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"读不到 {img_path}")
        return 1
    h, w = img.shape[:2]
    print(f"这一帧 {w}x{h}：{img_path}")
    print(f"窗口按 {DISP_W}x{DISP_H} 显示，点下去记的是**原图**坐标。")

    pts: list[tuple[float, float]] = []
    view = img.copy()
    cursor: list[tuple[int, int]] = []

    def redraw() -> None:
        nonlocal view
        view = img.copy()
        u, v = cursor[0] if cursor else (-99, -99)
        cv2.putText(view, f"({u}, {v})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, f"({u}, {v})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(view, (0, v), (w, v), (0, 255, 255), 1)
        cv2.line(view, (u, 0), (u, h), (0, 255, 255), 1)
        for i, (pu, pv) in enumerate(pts):
            last = (i == len(pts) - 1)
            col = (255, 0, 255) if last else (128, 128, 128)
            cv2.drawMarker(view, (int(round(pu)), int(round(pv))), col,
                           cv2.MARKER_CROSS, 40 if last else 24, 3 if last else 1)
            cv2.circle(view, (int(round(pu)), int(round(pv))), 14, col, 1)
            cv2.putText(view, f"{i + 1}", (int(pu) + 17, int(pv) - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        shown = cv2.resize(view, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)
        cv2.imshow(NAME, shown)

    def on_mouse(event, x, y, flags, _param) -> None:
        # cv2 给的 x, y 已经是**原图**坐标（它自己按窗口缩放换回来了），不用再乘比例。
        if event == cv2.EVENT_MOUSEMOVE:
            cursor[:] = [(x, y)]
            redraw()
        elif event == cv2.EVENT_LBUTTONDOWN:
            pts.append((float(x), float(y)))
            print(f"  第 {len(pts)} 个点：原图 ({x}, {y})", flush=True)
            save()
            redraw()

    def save() -> None:
        if not pts:
            return
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        mean = np.mean(np.asarray(pts, dtype=np.float64), axis=0)
        OUT_JSON.write_text(json.dumps(
            {"tip_px": [round(float(mean[0]), 1), round(float(mean[1]), 1)],
             "clicks_px": [[round(p[0], 1), round(p[1], 1)] for p in pts],
             "image": str(img_path), "image_size": [w, h],
             "what": "固定爪尖在腕相机画面里的像素。相机和固定爪都长在 gripper 体上，"
                     "两者相对位置是死的 —— 所以这个像素跟臂怎么动无关，是常数。"},
            ensure_ascii=False, indent=2), encoding="utf-8")

    cv2.namedWindow(NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NAME, DISP_W, DISP_H)
    cv2.setMouseCallback(NAME, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("r"):
            pts.clear()
            redraw()
        if cv2.getWindowProperty(NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    save()
    print(f"存到 {OUT_JSON}")
    print("点：", [(round(p[0], 1), round(p[1], 1)) for p in pts])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
