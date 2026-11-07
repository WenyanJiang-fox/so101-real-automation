"""采侧相机（固定支架）的内参视图：**相机不动，手挪平板**。

和 `real/calib_views.py` 的区别：那边是腕相机，臂带着相机走；这边相机钉死在支架上，
只能反过来挪标定板。所以这个脚本**不连串口、不动臂**，只开相机 + 认板子 + 存图。

自动收的条件（三条同时满足才存）：
  1. 认出的角点够 `--min-corners`（太少喂进 `calibrateCamera` 会把病态解带进来）；
  2. 板子**连续几帧基本不动**（人松手停稳了，避免运动模糊）；
  3. 和已经存过的每一张都**不一样**（中心/尺度/倾角差得够开）—— 同一姿态拍十张
     等于只拍了一张，解不出焦距和距离的那条耦合。

拍摄端到端要做的事：拿着平板在画面里换位置、换距离、换倾角，覆盖画面四个角和中间，
既要有正对镜头的，也要有 30~45° 斜的。**别让板子边缘出画面**，角点会丢。

跑法（在 G:\\sim2real\\so101-real-harness 下）：
    G:/anaconda/envs/sim2real/python.exe real/calib_side_views.py
    G:/anaconda/envs/sim2real/python.exe real/calib_side_views.py --max-views 30

用 `real/calib_side_views.json` 接 `calibrate_camera.py --views ... --out ...`。
出图：G:\\sim2real\\so101-real-harness\\out\\side_calib\\sideview_000.jpg ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from color_check import open_camera, rotate_frame        # noqa: E402

CFG = _ROOT / "configs" / "real.yaml"
OUT_DIR = _ROOT / "out" / "side_calib"
JSON_PATH = _ROOT / "out" / "side_calib_views.json"

CAM_W, CAM_H = 1920, 1080
FOURCC = "MJPG"


def signature(ch_c) -> tuple[float, float, float, float]:
    """板子当前长什么样：中心 x/y、等效尺寸、倾角（度）。用来判「和前面那张不一样」。"""
    p = np.asarray(ch_c, dtype=np.float64).reshape(-1, 2)
    cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
    (_, _), (w, h), ang = cv2.minAreaRect(p.astype(np.float32))
    return cx, cy, math.sqrt(max(w, 1.0) * max(h, 1.0)), float(ang)


def differs(a, b) -> bool:
    """和已存的那张比，位置/尺度/倾角任一项差得够开就算新姿态。"""
    dx, dy, da, dang = a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]
    dang = abs((dang + 90.0) % 180.0 - 90.0)          # minAreaRect 的角度是 [0,90) 带跳变
    return (math.hypot(dx, dy) > 60.0                 # px
            or (da / b[2] > 0.12 if b[2] > 0 else True)   # 尺度差 12%
            or dang > 8.0)                             # 度


def main() -> int:                                             # noqa: C901
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="采侧相机内参视图（手挪平板）")
    ap.add_argument("--index", type=int, default=None, help="相机序号，默认读配置的 side_cam_index")
    ap.add_argument("--focus", type=int, default=None, help="默认读配置的 side_cam_focus")
    ap.add_argument("--rotate", type=int, default=None, help="默认读配置的 side_cam_rotate")
    ap.add_argument("--min-corners", type=int, default=12,
                    help="一张至少认多少个角点才收（默认 12；<=8 会太松）")
    ap.add_argument("--stable-frames", type=int, default=6, help="连续几帧不动才算停稳")
    ap.add_argument("--max-views", type=int, default=30, help="收够这么多张就停")
    ap.add_argument("--seconds", type=float, default=0.0, help=">0 时跑这么多秒自动退出")
    ap.add_argument("--window", default="side calib", help="窗口标题")
    args = ap.parse_args()

    real = (yaml.safe_load(CFG.read_text(encoding="utf-8")) or {}).get("real") or {}
    index = args.index if args.index is not None else int(real.get("side_cam_index", 0))
    focus = args.focus if args.focus is not None else real.get("side_cam_focus")
    rotate = args.rotate if args.rotate is not None else int(real.get("side_cam_rotate") or 0)

    import make_charuco as mc                                    # noqa: PLC0415
    _, board = mc.build(8, 6)
    det = cv2.aruco.CharucoDetector(board)

    cap = open_camera(index, size=(CAM_W, CAM_H), fourcc=FOURCC, focus=focus)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window, 540, 960)

    t0 = time.time()
    prev_sig = None
    stable = 0
    print(f"侧相机 index {index}  焦距 {focus}  转 {rotate}°", flush=True)
    print("拿着平板换位置/距离/倾角，停稳两秒自动存；q 退出，s 手动存一张。", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("读帧失败", flush=True)
            break
        frame = rotate_frame(frame, rotate)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ch_c, ch_i, m_c, m_i = det.detectBoard(gray)
        n_c = 0 if ch_c is None else len(ch_c)

        sig = None
        if n_c >= args.min_corners:
            sig = signature(ch_c)
            if prev_sig is not None and differs(sig, prev_sig) is False:
                stable += 1
            else:
                stable = 0
        else:
            stable = 0
        prev_sig = sig

        fresh = True
        if sig is not None:
            for r in records:
                if not differs(sig, tuple(r["sig"])):
                    fresh = False
                    break

        take = False
        if sig is not None and fresh and stable >= args.stable_frames:
            take = True

        hud = f"corners {n_c:2d}  saved {len(records):2d}/{args.max_views}"
        if n_c < args.min_corners:
            hud += "   MOVE BOARD INTO VIEW"
        elif not fresh:
            hud += "   SAME POSE - MOVE IT"
        elif stable < args.stable_frames:
            hud += "   HOLD STILL"
        else:
            hud += "   OK"

        view = frame.copy()
        if ch_c is not None:
            cv2.aruco.drawDetectedCornersCharuco(view, ch_c, ch_i, (0, 255, 0))
        cv2.putText(view, hud, (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(args.window, view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            take = sig is not None

        if take:
            p = OUT_DIR / f"sideview_{len(records):03d}.jpg"
            cv2.imwrite(str(p), frame)
            records.append({
                "file": str(p.relative_to(_ROOT / "out")).replace("\\", "/"),
                "corners": int(n_c),
                "markers": 0 if m_c is None else int(len(m_c)),
                "sig": [float(v) for v in sig],
                "image_size": [int(frame.shape[1]), int(frame.shape[0])],
            })
            print(f"  [{len(records):2d}] {p.name}  角点 {n_c:2d}", flush=True)
            stable = 0
            if len(records) >= args.max_views:
                print(f"收够 {args.max_views} 张，停。", flush=True)
                break

        if args.seconds > 0 and (time.time() - t0) >= args.seconds:
            print(f"到 {args.seconds:g} 秒，停。", flush=True)
            break

    cap.release()
    cv2.destroyAllWindows()

    JSON_PATH.write_text(json.dumps({
        "camera": "side",
        "index": index,
        "focus": focus,
        "rotate": rotate,
        "rectified_size": [int(frame.shape[1]), int(frame.shape[0])],
        "views": records,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n收了 {len(records)} 张，读数 {JSON_PATH}", flush=True)
    if len(records) < 10:
        print("**少于 10 张解不稳。** 多换几个倾角和距离。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
