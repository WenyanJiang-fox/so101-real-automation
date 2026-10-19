"""点两下把**合爪轴**重标一遍的窗口（离线，不碰臂、不碰相机）。

用法：
    G:/anaconda/envs/sim2real/python.exe _click_grasp_dir.py
    G:/anaconda/envs/sim2real/python.exe _click_grasp_dir.py --img out/其他帧.jpg

两个窗口（都是 `WINDOW_NORMAL`，DPI 缩放不会撑爆屏幕）：
  · `full` —— 整帧，只作对照，也能点（精度差）
  · `zoom` —— 以轴心为中心 240x240 原始像素放大 3 倍，**点这个准**

点法：**第一下点轴心**（方块该被夹住的那个中心），**第二下点方向**（从轴心往外一指）。
点完窗口上立刻重画：洋红箭头 = 新的合爪轴，并当场算出这帧的「缝」「轴偏」两条读数。

按键：
  `r` 重点（保留当前设置不动）   `u` 撤销一下   `s` 存盘（先备份旧的）
  `f` 整帧窗口 / `z` 放大窗口里点谁，只是切换显示，点击两边都认
  `q` / `Esc` 退出

存盘写两处：
  · `assets/camera_calibration/real_wrist_jaw_shape.json`（线上就读这一份，改了下一眼生效）
  · `_backup/<时间戳>-real_wrist_jaw_shape.json`（旧的那一份）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from real.perception import brown_mask, candidates, load_cal   # noqa: E402
from skills.pick_real import (_box_center_px, _contour_pts,     # noqa: E402
                              _exit_px, _gap_axis_px)

SHAPE = ROOT / "assets" / "camera_calibration" / "real_wrist_jaw_shape.json"
CAND = ROOT / "out" / "jaw_dir_candidate.json"
WANT_GAP, TOL_AXIS = 20.0, 20.0        # 跟 `skills/pick_real.py` 那两条门槛一个口径

FULL_W, FULL_H = 1100, 620             # 整帧窗口的显示尺寸
ZOOM_N, ZOOM_S = 240, 3                # 放大窗口：240 原始像素 ×3 = 720

C_BOX = (0, 255, 0)
C_FIX = (255, 255, 0)
C_OPEN = (0, 165, 255)
C_CLOSED = (0, 255, 255)
C_AXIS = (255, 0, 255)
C_REF = (255, 255, 255)
OKC, BADC = (0, 255, 0), (0, 0, 255)

zoom_c = np.array([1008.0, 388.5])     # 放大窗口盯着哪一点（原始像素）


def _put(img, text, org, color, scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def _draw_scene(img, cnt, jaw, ref, d, axis, gap, exitpx, scale=1.0, off=(0.0, 0.0)):
    """把轮廓、三圈爪、轴线画进 `img`。`scale`/`off` 是画布坐标 → 原始像素的逆映射。"""
    def P(uv):
        v = (np.asarray(uv, dtype=np.float64) - np.asarray(off, dtype=np.float64)) * scale
        return tuple(np.round(v).astype(int))

    cv2.drawContours(img, [cnt], -1, C_BOX, max(1, int(4 * scale)), cv2.LINE_AA)
    for key, col in (("fixed_jaw", C_FIX), ("moving_open", C_OPEN),
                     ("moving_closed", C_CLOSED)):
        p = jaw.get(key)
        if p:
            q = (np.asarray(p, dtype=np.float64) - np.asarray(off, dtype=np.float64)) * scale
            cv2.polylines(img, [np.round(q).astype(np.int32)], True, col,
                          max(1, int(3 * scale)), cv2.LINE_AA)
    half = 0.5 * float(jaw.get("axis_len_px", 76.5))
    a, b = P(ref - d * half), P(ref + d * half)
    cv2.line(img, a, b, (255, 255, 255), max(2, int(7 * scale)), cv2.LINE_AA)
    cv2.line(img, a, b, C_AXIS, max(1, int(3 * scale)), cv2.LINE_AA)
    tip_len = half + 170.0
    cv2.arrowedLine(img, P(ref), P(ref + d * tip_len), C_AXIS, max(2, int(5 * scale)),
                    cv2.LINE_AA, tipLength=min(0.9, 26.0 / tip_len))
    cv2.circle(img, P(ref), max(3, int(12 * scale)), C_REF, max(1, int(3 * scale)),
               cv2.LINE_AA)
    gc = OKC if gap >= WANT_GAP else BADC
    ac = OKC if abs(axis) <= TOL_AXIS else BADC
    _put(img, f"gap {gap:+.0f}px (want >={WANT_GAP:.0f})", (10, 26), gc, 0.72, 2)
    _put(img, f"axis {axis:+.0f}px (tol {TOL_AXIS:.0f})", (10, 54), ac, 0.72, 2)
    _put(img, f"exit {exitpx:+.0f}px", (10, 82), (200, 200, 200), 0.72, 2)
    _put(img, f"ref ({ref[0]:.0f},{ref[1]:.0f})  dir ({d[0]:+.3f},{d[1]:+.3f})",
         (10, img.shape[0] - 14), C_AXIS, 0.72, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=str(ROOT / "out" / "jaw_raw.jpg"))
    a = ap.parse_args()

    bgr = cv2.imread(a.img)
    if bgr is None:
        print(f"读不到 {a.img}")
        return 1
    jaw = json.loads(SHAPE.read_text(encoding="utf-8"))
    ranges, fill_min, min_area = load_cal("red")
    cands = [d for d in candidates(brown_mask(bgr, ranges), min_area)
             if d["rect_fill"] >= fill_min]
    if not cands:
        print("这一帧里没有过得去的红区，量不出读数（照样能点，只是没数可看）")
        cnt = np.zeros((1, 1, 2), np.int32)
    else:
        cnt = cands[0]["contour"]
    pts = _contour_pts(cnt)
    box_uv = _box_center_px(cnt, bgr) if cnt.size else None
    polys = {k: np.asarray(jaw[k], dtype=np.float32).reshape(-1, 1, 2)
             for k in ("fixed_jaw", "moving_open", "moving_closed") if k in jaw}

    global zoom_c
    ref = np.asarray(jaw["axis_ref_px"], dtype=np.float64)
    d = np.asarray(jaw["axis_dir_px"], dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    if box_uv is not None:
        zoom_c = 0.5 * (np.asarray(box_uv, dtype=np.float64) + ref)
    else:
        zoom_c = ref.copy()

    clicks: list = []

    def metrics(ref, d):
        n = np.array([-d[1], d[0]])
        gap = _gap_axis_px(list(polys.values()), pts, d) if polys else float("nan")
        ex = _exit_px(list(polys.values()), pts, d) if polys else float("nan")
        ax = (float(np.dot(np.asarray(box_uv, dtype=np.float64) - ref, n))
              if box_uv is not None else float("nan"))
        return ax, gap, ex

    hover = {"full": None, "zoom": None}      # 鼠标在各自窗口里的**显示坐标**，只看不动

    def on_mouse(event, x, y, flags, param):
        win, scale, off = param
        if event == cv2.EVENT_MOUSEMOVE:
            hover[win] = (x, y)
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        uv = (np.asarray([x, y], dtype=np.float64) + np.asarray(off)) / scale
        clicks.append(uv)

    # **`WINDOW_AUTOSIZE` + 自己缩好的图**，不叫 OpenCV 去缩放。`WINDOW_NORMAL` +
    # `resizeWindow` 那一套在 125% 缩放的屏上，窗口客户区跟 `resizeWindow` 要的尺寸
    # 对不上，鼠标坐标就按「显示出来的那个尺寸」给 —— 于是点出来的点飘出去很远
    # （2026-10-16 用户实测）。这里两边都是固定像素：图是我自己缩到 1100x620 / 720x720
    # 的，窗口就按图的原尺寸显示，鼠标坐标跟图上像素一比一。
    cv2.namedWindow("full", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("zoom", cv2.WINDOW_AUTOSIZE)
    s_full = FULL_W / bgr.shape[1]
    cv2.setMouseCallback("full", on_mouse, ("full", s_full, (0.0, 0.0)))

    print("点两下：第一下轴心，第二下方向。`s` 存盘，`r` 重点，`q` 退出。")
    save_msg = ""
    while True:
        if len(clicks) >= 2:
            ref = clicks[-2].copy()
            v = clicks[-1] - clicks[-2]
            nv = float(np.linalg.norm(v))
            if nv > 1e-6:
                d = v / nv
                jaw["axis_ref_px"] = [float(ref[0]), float(ref[1])]
                jaw["axis_dir_px"] = [float(d[0]), float(d[1])]
                jaw["axis_len_px"] = float(round(nv, 1))
            zoom_c = ref.copy()
            clicks = []
        ax, gap, ex = metrics(ref, d)

        full = cv2.resize(bgr, (FULL_W, FULL_H), interpolation=cv2.INTER_AREA)
        _draw_scene(full, cnt, jaw, ref, d, ax, gap, ex, scale=s_full, off=(0.0, 0.0))
        if save_msg:
            _put(full, save_msg, (10, FULL_H - 40), (0, 255, 255), 0.72, 2)
        if clicks:
            for c in clicks:
                cv2.circle(full, tuple(np.round((c) * s_full).astype(int)), 6,
                           (0, 0, 255), -1, cv2.LINE_AA)

        x0 = int(round(zoom_c[0] - ZOOM_N / 2))
        y0 = int(round(zoom_c[1] - ZOOM_N / 2))
        x0 = max(0, min(bgr.shape[1] - ZOOM_N, x0))
        y0 = max(0, min(bgr.shape[0] - ZOOM_N, y0))
        crop = bgr[y0:y0 + ZOOM_N, x0:x0 + ZOOM_N]
        zoom = cv2.resize(crop, (ZOOM_N * ZOOM_S, ZOOM_N * ZOOM_S),
                          interpolation=cv2.INTER_NEAREST)
        offz = np.array([float(x0), float(y0)])
        _draw_scene(zoom, cnt, jaw, ref, d, ax, gap, ex, scale=float(ZOOM_S), off=offz)
        cv2.setMouseCallback("zoom", on_mouse, ("zoom", float(ZOOM_S), offz))
        if clicks:
            for c in clicks:
                cv2.circle(zoom, tuple(np.round((c - offz) * ZOOM_S).astype(int)), 8,
                           (0, 0, 255), -1, cv2.LINE_AA)

        # **鼠标在哪画个小十字**（在图上按光标自己的显示坐标画）。一比一的话它就在
        # 光标底下；飘开了就是映射又不对，一眼能看出来。
        for win, im in (("full", full), ("zoom", zoom)):
            if hover[win] is not None:
                hx, hy = hover[win]
                cv2.drawMarker(im, (hx, hy), (0, 0, 0), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
                _put(im, f"({hx},{hy})", (hx + 12, hy - 8), (255, 255, 255), 0.6, 1)

        cv2.imshow("full", full)
        cv2.imshow("zoom", zoom)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            clicks, save_msg = [], ""
        if k == ord("u") and clicks:
            clicks = clicks[:-1]
        if k == ord("s"):
            ts = time.strftime("%Y%m%d-%H%M%S")
            bak = ROOT / "_backup" / f"{ts}-real_wrist_jaw_shape.json"
            bak.parent.mkdir(exist_ok=True)
            if SHAPE.exists():
                shutil.copy2(SHAPE, bak)
            SHAPE.write_text(json.dumps(jaw, ensure_ascii=False, indent=2),
                             encoding="utf-8")
            CAND.write_text(json.dumps(jaw, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            save_msg = (f"saved  ref=({ref[0]:.0f},{ref[1]:.0f}) "
                        f"dir=({d[0]:+.3f},{d[1]:+.3f})  backup={bak.name}")
            print(save_msg, flush=True)
            print(f"  这一帧的读数：gap {gap:+.1f}px  axis {ax:+.1f}px  exit {ex:+.1f}px",
                  flush=True)

    cv2.destroyAllWindows()
    print(f"当前设置 ref=({ref[0]:.1f},{ref[1]:.1f}) dir=({d[0]:+.4f},{d[1]:+.4f})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
