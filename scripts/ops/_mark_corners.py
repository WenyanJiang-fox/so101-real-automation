"""把 `_box_center_px` 认的那个**顶面中心**在图上标出来 —— 离线，只看图，不动臂。

口径不许另写一份：红区的掩膜、候选筛选、亮暗分界、凸包简化，全部调
`skills/pick_real.py` 里正在跑的那几个函数（`brown_mask` / `candidates` / `_box_center_px`），
阈值现读（`real.perception.load_cal`）。这里只多做一件事：把中间那张「亮块」也画出来，
好让人拿眼睛核。

输入是 `out/jaw_raw.jpg`（那一眼没画过的原帧），叠在 `out/jaw_check.jpg` 那份标注图上输出
`out/box_corners.jpg`。

    G:/anaconda/envs/sim2real/python.exe _mark_corners.py [颜色]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from real.perception import brown_mask, candidates, load_cal          # noqa: E402
from skills.pick_real import _box_center_px                           # noqa: E402

RAW = _ROOT / "out" / "jaw_raw.jpg"          # 干净的原帧（红区轮廓从它上面取）
BASE = _ROOT / "out" / "jaw_check.jpg"       # 已经画过爪子轮廓/合爪轴的那张
DST = _ROOT / "out" / "box_corners.jpg"


def main() -> int:
    color = sys.argv[1] if len(sys.argv) > 1 else "red"
    raw = cv2.imread(str(RAW))
    if raw is None:
        print(f"读不到 {RAW} —— 这一眼看的是哪一帧？")
        return 1
    ranges, fill_min, min_area = load_cal(color)
    mask = brown_mask(raw, ranges)
    cands = [d for d in candidates(mask, min_area) if d["rect_fill"] >= fill_min]
    if not cands:
        print(f"画面里没有过得去的{color}区（{len(cands)} 个候选）")
        return 1
    print(f"候选 {len(cands)} 个；用第 1 个（跟 `_jaw_check` 同一条）"
          f"  矩形度 {cands[0]['rect_fill']:.3f}")
    cnt = cands[0]["contour"]

    # --- 只为了画出来 / 打数字，把 `_box_center_px` 中间那两步复算一遍 -----------
    h, w = raw.shape[:2]
    full = np.zeros((h, w), np.uint8)
    cv2.drawContours(full, [np.asarray(cnt).reshape(-1, 1, 2).astype(np.int32)], -1, 255, -1)
    area0 = int((full > 0).sum())
    r = raw[:, :, 2]
    t, _ = cv2.threshold(r[full > 0].reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    top = np.zeros((h, w), np.uint8)
    top[(full > 0) & (r >= float(t))] = 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(top, 8)
    print(f"Otsu 分界 R ≥ {float(t):.0f}（红区 {area0} 像素里亮块占 "
          f"{100.0 * int(stats[1:, cv2.CC_STAT_AREA].sum()) / max(area0, 1):.0f}%），"
          f"共 {n - 1} 块")
    one = np.zeros((h, w), np.uint8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        one[lab == k] = 255
        print(f"取最大那块：{int(stats[k, cv2.CC_STAT_AREA])} 像素，"
              f"外接框 x {int(stats[k, cv2.CC_STAT_LEFT])}.."
              f"{int(stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH])}、"
              f"y {int(stats[k, cv2.CC_STAT_TOP])}.."
              f"{int(stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT])}")
    cs, _ = cv2.findContours(one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea)) if cs else None

    # --- 真正在跑的那条路 -------------------------------------------------------
    c = _box_center_px(cnt, raw)
    m = cv2.moments(cnt)
    cen = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])

    if c is None:
        print("新规则量不出来（返回 None）")
        return 1
    print(f"顶面中心 = ({c[0]:.1f}, {c[1]:.1f})")
    print(f"整块重心（不用它）= ({cen[0]:.1f}, {cen[1]:.1f})"
          f"   差 {np.linalg.norm(cen - c):.1f} 像素")

    img = cv2.imread(str(BASE))
    if img is None:
        img = raw.copy()
    cv2.drawContours(img, [np.asarray(cnt).reshape(-1, 1, 2)], -1, (255, 0, 0), 2)
    if one.any():
        cs2, _ = cv2.findContours(one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, [max(cs2, key=cv2.contourArea)], -1, (255, 0, 255), 2)
    if hull is not None:
        cv2.polylines(img, [hull.astype(np.int32)], True, (0, 255, 255), 2)
    cv2.drawMarker(img, (int(round(c[0])), int(round(c[1]))), (0, 255, 0),
                   cv2.MARKER_CROSS, 48, 4)
    txt = f"TOP FACE CENTER ({c[0]:.0f},{c[1]:.0f})"
    for dk, col, th in ((5, (0, 0, 0), 5), (2, (0, 255, 0), 2)):
        cv2.putText(img, txt, (int(round(c[0])) + 22, int(round(c[1])) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
    cv2.circle(img, (int(round(cen[0])), int(round(cen[1]))), 10, (128, 128, 128), 2,
               cv2.LINE_AA)
    cv2.imwrite(str(DST), img)
    print(f"-> {DST}   （蓝 = 红区轮廓、洋红 = 亮块（顶面）、黄 = 亮块凸包、"
          f"绿十字 = 顶面中心、灰圈 = 整块重心）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
