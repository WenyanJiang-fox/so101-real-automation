"""实时看一个 USB 相机的画面。

    python view_cam.py                 # 默认 index 1（从臂相机）
    python view_cam.py --camera 0      # 笔记本内置

按 q 退出。相机默认出 160x120，太小看不清，所以默认放大 4 倍显示
（`--scale`）；放大用的是最近邻，屏幕上还是像素本身，没有插值糊。

取帧沿用 `color_check.py` 里那套（连读几帧丢缓存）。
"""

import argparse
import sys
import time

import cv2
import numpy as np

from color_check import open_camera, grab_frame

# 试的顺序：先不管后端，再单独逼 DSHOW / MSMF。便宜 UVC 相机在不同后端下
# 能谈下来的分辨率不一样。
_BACKENDS = [("ANY", cv2.CAP_ANY), ("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]

# 要试的分辨率，从大到小。
_SIZES = [(1920, 1080), (1280, 720), (1024, 768), (800, 600), (640, 480), (320, 240), (160, 120)]

# MJPG 是 UVC 相机在高分辨率下唯一现实的格式；不设的话多数相机在
# 640x480 以上会自动掉回 YUY2 然后被拒。
_FOURCCS = ["MJPG", ""]


def _probe(camera: int, drain: int = 12) -> int:
    """把相机开着挨个要分辨率，报告每个后端实际给回来的尺寸。

    两件事容易骗自己，这里都堵上了：
      1. `cap.set` 换格式是异步的，头几帧经常还是旧尺寸 —— 所以连读 `drain` 帧，
         取最后一张的尺寸为准，同时回读相机自己报的尺寸。
      2. OpenCV 在 DSHOW 上，「打开后 set」和「打开时把参数传给构造函数」走的是
         两条不同的路 —— 所以两种都试。
    """
    print(f"相机 index={camera} 支持的分辨率（每档连读 {drain} 帧取最后一张）：\n")
    best = None
    for be_name, be in _BACKENDS:
        for fourcc in _FOURCCS:
            for mode in ("set", "ctor"):
                hits = []
                for w, h in _SIZES:
                    if mode == "ctor":
                        params = [cv2.CAP_PROP_FRAME_WIDTH, float(w),
                                  cv2.CAP_PROP_FRAME_HEIGHT, float(h)]
                        if fourcc:
                            params += [cv2.CAP_PROP_FOURCC,
                                       float(cv2.VideoWriter_fourcc(*fourcc))]
                        try:
                            cap = cv2.VideoCapture(camera, be, params)
                        except Exception:
                            break
                    else:
                        cap = cv2.VideoCapture(camera, be)
                        if cap.isOpened() and fourcc:
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    if not cap.isOpened():
                        cap.release()
                        break
                    claimed = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    got = None
                    for _ in range(max(1, drain)):
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            got = (frame.shape[1], frame.shape[0])
                    cap.release()
                    if got is None:
                        continue
                    hits.append((w, h, got, claimed))
                    if best is None or got[0] * got[1] > best[0] * best[1]:
                        best = (got[0], got[1], f"{be_name}/{mode}", fourcc or "default")
                if hits:
                    tag = f"{be_name} / {fourcc or 'default fourcc'} / {'构造函数传参' if mode == 'ctor' else '打开后 set'}"
                    print(f"  {tag}")
                    for w, h, got, claimed in hits:
                        mark = "" if got == (w, h) else f"  <- 要 {w}x{h} 只给到 {got[0]}x{got[1]}"
                        print(f"      要 {w}x{h} -> 实得 {got[0]}x{got[1]}（相机自称 {claimed[0]}x{claimed[1]}）{mark}")
                else:
                    print(f"  {be_name} / {fourcc or 'default fourcc'} / "
                          f"{'构造函数传参' if mode == 'ctor' else '打开后 set'}: 打不开或读不出帧")
    if best is None:
        print("\n一个分辨率都没谈下来。")
        return 1
    print(f"\n能拿到的最大：{best[0]}x{best[1]}（后端 {best[2]}，fourcc {best[3]}）")
    return 0


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    """绕画面中心转 `deg` 度（逆时针为正），画布放大到装得下整张图，四角补黑。"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(round(h * sin + w * cos))
    nh = int(round(h * cos + w * sin))
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="实时预览 USB 相机")
    ap.add_argument("--camera", type=int, default=1, help="相机 index，默认 1（从臂相机）")
    ap.add_argument("--width", type=int, default=640, help="要的宽度，默认 640")
    ap.add_argument("--height", type=int, default=480, help="要的高度，默认 480")
    ap.add_argument("--scale", type=float, default=4.0, help="显示放大倍数，默认 4")
    ap.add_argument("--window", default=None, help="窗口标题")
    ap.add_argument("--seconds", type=float, default=0.0, help=">0 时跑这么多秒自动退出")
    ap.add_argument("--crop", default="",
                    help="固定裁剪 x0,y0,x1,y1（原图坐标）；空 = 不裁")
    ap.add_argument("--rotate", type=float, default=0.0,
                    help="裁剪后整体转多少度（逆时针为正）；画布自动放大，不切画面")
    ap.add_argument("--probe", action="store_true", help="只问相机支持哪些分辨率，不开窗口")
    args = ap.parse_args()

    crop = None
    if args.crop.strip():
        crop = tuple(int(v) for v in args.crop.replace(" ", "").split(","))
        if len(crop) != 4:
            print("--crop 要四个数：x0,y0,x1,y1")
            return 2
    rot = float(args.rotate)

    if args.probe:
        return _probe(args.camera)

    title = args.window or f"camera {args.camera}"
    # **必须带 DSHOW 打开**（2026-10-20）：不给后端时 index 0 会落到 MSMF 上卡死 ——
    # 同一只相机用 DSHOW 3 秒内出帧，用默认后端 90 秒一个字节都没有。分辨率也在
    # 打开时传，打开后再 `set` 在 DSHOW 上经常返回 True 但没生效。
    cap = open_camera(args.camera, size=(args.width, args.height), fourcc="MJPG")

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        print(f"相机 index={args.camera} 打开了但读不出帧。")
        return 1
    h, w = frame.shape[:2]
    note = "" if crop is None else f"，裁 {crop[2] - crop[0]}x{crop[3] - crop[1]}"
    if rot:
        note += f"，转 {rot:g}°"
    print(f"相机 index={args.camera}：{w}x{h}{note}，放大 {args.scale:g}x 显示。"
          f"按 q 退出。")

    # **`WINDOW_AUTOSIZE` 在 125% 缩放的 Windows 上会把 1080p 的画面撑到屏幕外**
    # （2026-10-20）。改成可缩放的普通窗口，尺寸就是画面本身的尺寸，用户自己能拖。
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    t0 = time.time()
    n = 0
    best = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("读帧失败，退出。")
                break
            # **固定裁剪**：侧视相机只留白纸和机械臂。剪裁框写死在这里，和以后
            # 写数据用的是同一组数（2026-10-20）。
            if crop is not None:
                frame = frame[crop[1]:crop[3], crop[0]:crop[2]]
            # **转一下**：纸面和桌沿在画面里是斜的，转正了之后横竖就是场景的横竖。
            # 画布先放大到能装下整张转完的图，四角补黑，画面内容不丢。
            if rot:
                frame = _rotate(frame, rot)

            view = frame
            if args.scale != 1.0:
                view = cv2.resize(frame, None, fx=args.scale, fy=args.scale,
                                  interpolation=cv2.INTER_NEAREST)

            # 对焦判据：拉普拉斯方差，越大越锐利。拧镜头时盯着这个数往上走。
            sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            best = max(best, sharp)
            cv2.putText(view, f"sharp {sharp:6.1f}   best {best:6.1f}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # **窗口怎么拖都不变形**（2026-10-20 用户报「画面畸变」）：`WINDOW_NORMAL`
            # 允许自由拖，OpenCV 会把画面直接拉满窗口 —— 宽高比一改就拉伸。这里按窗口
            # 大小等比缩放、四周补黑边，画面内容一点不动。
            rect = cv2.getWindowImageRect(title)
            ww, wh = ((rect[2], rect[3])
                      if rect and rect[2] > 0 and rect[3] > 0 else (0, 0))
            shown = view
            if ww and wh:
                s = min(ww / view.shape[1], wh / view.shape[0])
                nw = max(1, int(round(view.shape[1] * s)))
                nh = max(1, int(round(view.shape[0] * s)))
                shown = cv2.resize(view, (nw, nh), interpolation=cv2.INTER_LINEAR)
                if (nw, nh) != (ww, wh):
                    canvas = np.zeros((wh, ww, 3), dtype=np.uint8)
                    x = (ww - nw) // 2
                    y = (wh - nh) // 2
                    canvas[y:y + nh, x:x + nw] = shown
                    shown = canvas
            cv2.imshow(title, shown)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break

            n += 1
            if args.seconds > 0 and (time.time() - t0) >= args.seconds:
                print(f"到 {args.seconds:g} 秒，退出。实测 {n / max(1e-6, time.time() - t0):.1f} fps。")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
