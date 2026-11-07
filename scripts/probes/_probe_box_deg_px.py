"""离线：**画面里**「方块侧面」和「合爪轴」到底差多少度。只读 `out/jaw_raw.jpg`。

不用相机、不用串口、不动臂。掩膜那条链跟 `_jaw_check` 逐字一样（同一份 HSV 区间、
同一个最小面积、同一个矩形度门槛），不然量的是另一块红区。

合爪轴那条线来自 `assets/camera_calibration/real_wrist_jaw_shape.json` —— 相机和两爪
在同一个刚体上，所以它在画面里的角度跟臂怎么转都无关。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

from real.perception import brown_mask, candidates, load_thresholds   # noqa: E402


def main() -> int:
    cal = load_thresholds()
    ranges = ((cal.get("color") or {}).get("hsv") or {})["red"]
    rr = cal.get("brown_rect") or {}
    fill, area = float(rr["rect_fill_min"]), int(rr["min_area_px"])

    jaw = json.loads((ROOT / "assets" / "camera_calibration"
                      / "real_wrist_jaw_shape.json").read_text(encoding="utf-8"))
    ax = np.asarray(jaw["axis_dir_px"], dtype=np.float64).reshape(-1)[:2]
    ax_deg = float(np.degrees(np.arctan2(float(ax[1]), float(ax[0]))))
    print(f"合爪轴 axis_dir_px={ax}  -> {ax_deg:+.2f}°")

    bgr = cv2.imread(str(ROOT / "out" / "jaw_raw.jpg"))
    mask = brown_mask(bgr, ranges)
    cands = [d for d in candidates(mask, area) if d["rect_fill"] >= fill]
    print(f"过得去的红区 {len(cands)} 块（门槛 rect_fill>={fill}，面积>={area}）")
    for d in cands:
        (cx, cy), (rw, rh), ang = d["rect"]
        # 方块是方的：两条边差 90° 等价，折到 [-45, 45]
        diff = (float(ang) - ax_deg + 45.0) % 90.0 - 45.0
        print(f"  uv=({cx:6.1f},{cy:6.1f}) 最小外接矩形 {rw:5.1f}x{rh:5.1f}"
              f" ang={float(ang):6.2f}°  离合爪轴 {diff:+6.2f}°"
              f"  rect_fill={d['rect_fill']:.3f}  面积={d['area_px']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
