# -*- coding: utf-8 -*-
"""探针：照 `_close_until_stall` 的写法推 10 步，把每一步「写什么、读回什么」打出来。

只动 6 号舵机（1~5 号每步写回原值）。
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, r"G:\sim2real\so101-real-harness")

from core.params import GENTLE                                   # noqa: E402
from real.arm import GRIP_SCALE, RealArm                         # noqa: E402

fast = "--fast" in sys.argv
arm = RealArm(port="COM10", speed=0, fast=fast)
try:
    ticks0 = {int(k): int(v) for k, v in arm.read_ticks().items()}
    print(f"ticks0={ticks0}", flush=True)
    q0 = float(arm.grip_q())
    arm._grip_cmd = float(np.clip(q0, arm.grip_lo, arm.grip_hi))
    print(f"q0={q0:+.5f}  _grip_cmd={arm._grip_cmd:+.5f}  "
          f"lo/hi={arm.grip_lo:+.4f}/{arm.grip_hi:+.4f}  step={GENTLE * GRIP_SCALE}",
          flush=True)
    for step in range(1, 11):
        arm._grip_cmd = float(np.clip(arm._grip_cmd - GENTLE * GRIP_SCALE,
                                      arm.grip_lo, arm.grip_hi))
        _qc = arm._ticks_to_qpos(ticks0)
        _qc[5] = arm._grip_cmd
        _write = dict(ticks0)
        _write[6] = int(arm._qpos_to_ticks(_qc)[6])
        arm.write_ticks(_write)
        time.sleep(0.15)
        t = arm.read_ticks()
        print(f"  第 {step:2d} 步 写 tick 6={_write[6]:4d}（_grip_cmd={arm._grip_cmd:+.5f}）"
              f"  0.15s 后读 tick 6={t[6]:4d}  q={arm.grip_q():+.5f}", flush=True)
finally:
    arm.close() if hasattr(arm, "close") else None
