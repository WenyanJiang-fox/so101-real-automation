"""离线探针：从旧日志里量「画面像素 / 底座毫米」。

只用一条事实：同一趟里爪子在动、方块不动。所以拿**相邻两眼**的
「TCP 实际位置」（`走到 [x y z]mm`，那是读回来的真值）和「方块中心像素」
（`轮廓：… 方块中心 (u,v)`）配对，`|Δ像素| / |Δ毫米|` 就是这两眼之间的比例。

方块中心那个像素会被爪子挡，个别眼会跳；所以取**中位数**，并把四分位一起打出来。
"""
from __future__ import annotations

import glob
import re
import sys
import numpy as np

PAT_TCP = re.compile(r"走到 \[ *(-?[\d.]+) +(-?[\d.]+) +(-?[\d.]+)\]mm")
PAT_BOX = re.compile(r"轮廓：.*?方块中心 \((\d+),(\d+)\)")

files = [sys.argv[1]] if len(sys.argv) > 1 else sorted(glob.glob("out/run_real_*.log"))
ratios: list[float] = []
for f in files:
    try:
        lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        continue
    tcp = None
    box = None
    box_tcp = None
    for ln in lines:
        m = PAT_TCP.search(ln)
        if m:
            tcp = np.array([float(m.group(i)) for i in (1, 2)]) / 1000.0
        b = PAT_BOX.search(ln)
        if b:
            new_box = np.array([float(b.group(1)), float(b.group(2))])
            if box is not None and tcp is not None and box_tcp is not None:
                dmm = float(np.linalg.norm(tcp - box_tcp))
                dpx = float(np.linalg.norm(new_box - box))
                if 0.5e-3 < dmm < 0.03 and dpx > 1.0:
                    ratios.append(dpx / (dmm * 1000.0))
            box = new_box
            box_tcp = tcp

if not ratios:
    print("没有可用的配对")
    sys.exit(1)
r = np.asarray(ratios)
print(f"配对 {len(r)} 组，来自 {len(files)} 个日志")
print(f"  中位数 {np.median(r):.2f} px/mm")
print(f"  四分位 {np.percentile(r, 25):.2f} ~ {np.percentile(r, 75):.2f}")
print(f"  5%~95% {np.percentile(r, 5):.2f} ~ {np.percentile(r, 95):.2f}")
lo, hi, want = 20.0, 70.0, 45.0
for name, v in (("下限 20px", lo), ("上带 70px", hi), ("带中值 45px", want)):
    print(f"  {name} ÷ 中位数 = {v / np.median(r):.2f}mm"
          f"（四分位 {v / np.percentile(r, 75):.2f} ~ {v / np.percentile(r, 25):.2f}）")
