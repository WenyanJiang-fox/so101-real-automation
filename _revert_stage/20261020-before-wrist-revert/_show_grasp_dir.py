"""把腕相机那一帧上的**红区轮廓**和**合爪轴**画出来，给人重新标方向用。

输入（都只读）：
  · `out/jaw_raw.jpg` —— 最近一次 `_jaw_check` 那一帧的原始画面（一个像素都没画过）
  · `assets/camera_calibration/real_wrist_jaw_shape.json` —— 现在在用的爪轮廓和合爪轴

输出：
  · `out/grasp_dir_marked.jpg` —— 同一帧上叠：红区轮廓（绿）、三圈爪轮廓（青/橙/黄）、
    合爪轴参考点（白圈）、合爪轴（洋红实线 + 箭头，箭头指的就是 `axis_dir_px` 的正方向）
  · 标准输出一段数，给人改 json 用

只画图，不碰机械臂、不碰相机、不写任何被线上读的文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from real.perception import brown_mask, candidates, load_cal   # noqa: E402

RAW = ROOT / "out" / "jaw_raw.jpg"
SHAPE = ROOT / "assets" / "camera_calibration" / "real_wrist_jaw_shape.json"
OUT = ROOT / "out" / "grasp_dir_marked.jpg"

C_BOX = (0, 255, 0)          # 绿：红区轮廓
C_FIX = (255, 255, 0)        # 青：固定爪
C_OPEN = (0, 165, 255)       # 橙：动爪张到底
C_CLOSED = (0, 255, 255)     # 黄：动爪合到底
C_AXIS = (255, 0, 255)       # 洋红：合爪轴
C_REF = (255, 255, 255)      # 白：轴参考点
C_TOP = (255, 128, 0)        # 深青：方块顶面中心


def _label(img, text, org, color, scale=0.9, thick=2):
    """带黑边的白字，画在彩色场景上也看得清。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def _legend(img, lines, org=(16, 16)):
    """左上角一块黑底白字的读数板，把标签从几何上挪开，不挡盒子。"""
    x0, y0 = org
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)[0][0]
            for t in lines) + 24
    h = 34 * len(lines) + 16
    panel = img.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.62, img, 0.38, 0, img)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (x0 + 12, y0 + 30 + 34 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)


def _arrow(img, start, vec, color, thick=4, head=26):
    """从 `start` 沿 `vec` 画一条带箭头的线（`vec` 的模就是长度）。"""
    s = np.asarray(start, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    e = s + v
    cv2.arrowedLine(img, tuple(np.round(s).astype(int)),
                    tuple(np.round(e).astype(int)), color, thick,
                    cv2.LINE_AA, tipLength=min(0.9, head / max(np.linalg.norm(v), 1e-6)))
    return e


def main() -> int:
    bgr = cv2.imread(str(RAW))
    if bgr is None:
        print(f"读不到 {RAW} —— 先跑一趟让 `_jaw_check` 写一张")
        return 1
    jaw = json.loads(SHAPE.read_text(encoding="utf-8"))
    h, w = bgr.shape[:2]
    print(f"画面 {w}x{h}；爪轮廓那份标在 {jaw['image_size'][0]}x{jaw['image_size'][1]}")

    ranges, fill_min, min_area = load_cal("red")
    mask = brown_mask(bgr, ranges)
    cands = [d for d in candidates(mask, min_area) if d["rect_fill"] >= fill_min]
    print(f"过得去的红区 {len(cands)} 个"
          + (f"（最近那个面积 {cands[0]['area_px']}px 矩形度 {cands[0]['rect_fill']:.2f}）"
             if cands else ""))
    if not cands:
        print("没有红区 —— 这张画面上量不出轮廓，换一帧再看")
        return 1
    cnt = cands[0]["contour"]

    img = bgr.copy()
    cv2.drawContours(img, [cnt], -1, C_BOX, 4, cv2.LINE_AA)
    x, y, bw, bh = cv2.boundingRect(cnt)

    for key, col in (("fixed_jaw", C_FIX), ("moving_open", C_OPEN),
                     ("moving_closed", C_CLOSED)):
        p = jaw.get(key)
        if not p:
            continue
        poly = np.asarray(p, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, col, 3, cv2.LINE_AA)
        c = poly.reshape(-1, 2).mean(axis=0)
        _label(img, {"fixed_jaw": "fixed_jaw", "moving_open": "moving_open",
                     "moving_closed": "moving_closed"}[key],
               (int(c[0]) + 12, int(c[1])), col, 0.8, 2)

    ref = np.asarray(jaw["axis_ref_px"], dtype=np.float64)
    d = np.asarray(jaw["axis_dir_px"], dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    half = 0.5 * float(jaw.get("axis_len_px", 76.5))

    cv2.line(img, tuple(np.round(ref - d * half).astype(int)),
             tuple(np.round(ref + d * half).astype(int)), (255, 255, 255), 7, cv2.LINE_AA)
    cv2.line(img, tuple(np.round(ref - d * half).astype(int)),
             tuple(np.round(ref + d * half).astype(int)), C_AXIS, 3, cv2.LINE_AA)
    tip = _arrow(img, ref, d * (half + 170.0), C_AXIS, 5)
    cv2.circle(img, tuple(np.round(ref).astype(int)), 12, C_REF, 3, cv2.LINE_AA)
    _label(img, "GRASP DIR", (int(tip[0]) - 90, int(tip[1]) - 24), C_AXIS, 1.1, 3)

    m = cv2.moments(cnt)
    uv = (np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
          if abs(float(m["m00"])) > 1e-9 else None)
    if uv is not None:
        cv2.circle(img, tuple(np.round(uv).astype(int)), 10, C_TOP, 3, cv2.LINE_AA)

    _legend(img, [
        f"red contour  bbox ({x},{y}) {bw}x{bh}",
        f"axis_ref_px  ({ref[0]:.0f}, {ref[1]:.0f})",
        f"axis_dir_px  ({d[0]:+.3f}, {d[1]:+.3f})   <- GRASP DIR arrow",
        f"axis_len_px  {jaw.get('axis_len_px')}",
        f"arrow tip    ({tip[0]:.0f}, {tip[1]:.0f})",
        f"red centroid ({uv[0]:.0f}, {uv[1]:.0f})" if uv is not None else "red centroid  -",
    ])

    cv2.imwrite(str(OUT), img)
    print(f"写了 {OUT}")
    print(f"现在这份：axis_ref_px={jaw['axis_ref_px']} axis_dir_px={jaw['axis_dir_px']} "
          f"axis_len_px={jaw.get('axis_len_px')}")
    print(f"红区外框 x={x} y={y} w={bw} h={bh}；外框中心 "
          f"({x + bw / 2:.0f},{y + bh / 2:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
