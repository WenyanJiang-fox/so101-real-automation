"""实时看腕相机 + 准星 + 红色区域。只读、不动臂。

画面走的是线上同一条路（`real.cameras.RealWristCamera.shot()`，去畸变那一套一样），
红区掩膜和准星也照抄 `skills/pick_real.py::_crosshair()`：固定像素（
`assets/camera_calibration/real_wrist_tip_px.json`）+ `pointPolygonTest`。

准星读数是**红区外面多少个像素**（正 = 在红区里面，负 = 在外面）。
按 q 或 ESC 退出。

    G:/anaconda/envs/sim2real/python.exe view_crosshair.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(_ROOT))

from real.cameras import RealWristCamera                          # noqa: E402
from real.perception import brown_mask, candidates, load_thresholds  # noqa: E402

NAME = "wrist cam + crosshair"
DISP_W, DISP_H = 1280, 720
MIN_AREA = 2000


def main() -> int:
    tip = np.asarray(json.loads(
        (_ROOT / "assets" / "camera_calibration" / "real_wrist_tip_px.json")
        .read_text(encoding="utf-8"))["tip_px"], dtype=np.float64)
    ranges = np.array(load_thresholds()["color"]["hsv"]["red"], dtype=np.uint8)
    print(f"准星 {tip}   红区 HSV {ranges.tolist()}   退出按 q / ESC")

    cam = RealWristCamera(index=1)
    cv2.namedWindow(NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NAME, DISP_W, DISP_H)
    last = None
    try:
        while True:
            rgb = np.asarray(cam.shot())
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            mask = brown_mask(img, ranges)
            cands = [c for c in candidates(mask, MIN_AREA) if c["rect_fill"] >= 0.78]
            txt = "no red region"
            if cands:
                main_c = max(cands, key=lambda c: c["area_px"])
                cv2.drawContours(img, [main_c["contour"]], -1, (255, 255, 0), 3)
                px = cv2.pointPolygonTest(main_c["contour"], tip, True)
                txt = f"crosshair -> red {px:+.0f} px"
                if last is None or abs(px - last) >= 3:
                    print(f"  {txt}   (area {main_c['area_px']})", flush=True)
                    last = px
            u, v = int(round(tip[0])), int(round(tip[1]))
            cv2.drawMarker(img, (u, v), (255, 0, 255), cv2.MARKER_CROSS, 60, 4)
            cv2.circle(img, (u, v), 24, (255, 0, 255), 2)
            cv2.putText(img, txt, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5,
                        cv2.LINE_AA)
            cv2.putText(img, txt, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow(NAME, cv2.resize(img, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA))
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
            if cv2.getWindowProperty(NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
        cam.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
