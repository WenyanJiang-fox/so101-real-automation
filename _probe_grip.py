# -*- coding: utf-8 -*-
"""探针：6 号舵机到底听不听写命令。只动夹爪，不动 1~5 号。"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, r"G:\sim2real\so101-real-harness")

from servo_bridge import deg_from_position, open_controller  # noqa: E402

fast = "--fast" in sys.argv
c = open_controller("COM10", fast=fast)
try:
    p0 = c.read_position(6)
    print(f"起始 pos={p0} deg={deg_from_position(p0):.1f}  fast={fast}", flush=True)
    for target in (p0 - 60, p0 - 60, p0):
        c.sync_set_positions({6: int(target)}, speed=0)
        time.sleep(0.6)
        p = c.read_position(6)
        print(f"  写 {target:4d} → 0.6s 后读 {p:4d} deg={deg_from_position(p):.1f}",
              flush=True)
    # 同步读那条路（arm.read_ticks 用的）
    got = c.sync_read_positions([6])
    print(f"  sync_read_positions → {got}", flush=True)
finally:
    c.close()
