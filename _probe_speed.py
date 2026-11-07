# -*- coding: utf-8 -*-
"""探针：夹爪从张着到合到底要多久（每一步都被测出来）。

只动 6 号舵机。跑完把爪子张回去。
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, r"G:\sim2real\so101-real-harness")

from servo_bridge import open_controller                          # noqa: E402

c = open_controller("COM10", fast=True)
try:
    p_open = c.read_position(6)
    print(f"起始 pos={p_open}", flush=True)
    t0 = time.time()
    c.sync_set_positions({6: 1450}, speed=0)
    last = p_open
    marks = []
    while time.time() - t0 < 3.0:
        time.sleep(0.02)
        p = c.read_position(6)
        marks.append((time.time() - t0, p))
        if abs(p - last) <= 1 and len(marks) > 10:
            break
        last = p
    print("  时间(s)  pos", flush=True)
    for t, p in marks[:: max(1, len(marks) // 15)]:
        print(f"   {t:5.2f}  {p:4d}", flush=True)
    print(f"  停在 pos={marks[-1][1]}，用时 {marks[-1][0]:.2f}s", flush=True)
    c.sync_set_positions({6: int(p_open)}, speed=0)
    time.sleep(1.0)
    print(f"  张回 pos={c.read_position(6)}（原 {p_open}）", flush=True)
finally:
    c.close()
