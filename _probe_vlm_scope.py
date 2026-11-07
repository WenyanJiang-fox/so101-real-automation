# -*- coding: utf-8 -*-
"""验 2026-10-23 的两条口径，**不动臂**：

1. VLM 只能动感知偏移 / 安全限幅，悬停高度 / 下探深度一律拒收；
2. 变量只作用一集，下一集开头无条件还原。

    G:/anaconda/envs/sim2real/python.exe -u _probe_vlm_scope.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

import numpy as np                                                 # noqa: E402

from core import geom                                              # noqa: E402
from ppo import vlm                                                # noqa: E402
from skills.pick_real import knobs as KNOBS                        # noqa: E402

import collect_real_dataset as C                                   # noqa: E402

BASE = np.array([0.0358, 0.0], dtype=np.float64)


def dec(move: str, params: dict | None = None) -> dict:
    return {"cause": "slip", "evidence": "探针", "action": "baseline",
            "move": move, "params": params or {}, "closeness": 0.3,
            "confidence": 0.5, "contour_ok": None}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    geom.REAL_PERCEP_OFFSET = BASE.copy()
    KNOBS.JOINT1_STEP_DEG, KNOBS.JOINT_STEP_DEG = 30.0, None
    C._BASE_OFF = BASE.copy()

    print("[1] 非白名单动作")
    for mv in ("hover_up", "hover_down", "deeper", "shallower", "x_plus"):
        good, why = vlm.check(dec(mv, {"nudge_dx_mm": 10.0}), {})
        print(f"  {mv:10s} -> " + (f"通过 {good['params']}" if good
                                   else f"拒收（{why}）"))

    print("\n[2] 应用：不换档 + 两个旋钮都写进去")
    params: dict = {}
    act = C._apply_vlm(dec("x_minus", {"nudge_dx_mm": -12.0,
                                       "joint1_step_deg": 20.0}),
                       None, {}, params, tag="探针", code="slip")
    print(f"  返回档={act}  参数={params}")
    print(f"  偏移={np.round(geom.REAL_PERCEP_OFFSET, 4)}  "
          f"joint1={KNOBS.JOINT1_STEP_DEG}")

    print("\n[3] 只作用一集")
    C._vlm_episode_begin(11, params)
    print(f"  ep11 用完：pending.used_ep={C._VLM_PENDING.get('used_ep')}  "
          f"偏移={np.round(geom.REAL_PERCEP_OFFSET, 4)}")
    C._vlm_episode_begin(12, params)
    print(f"  ep12 开头：pending={C._VLM_PENDING}  参数={params}  "
          f"偏移={np.round(geom.REAL_PERCEP_OFFSET, 4)}  "
          f"joint1={KNOBS.JOINT1_STEP_DEG}")

    ok = (act is None and not params
          and abs(geom.REAL_PERCEP_OFFSET[0] - BASE[0]) < 1e-12
          and KNOBS.JOINT1_STEP_DEG == 30.0 and C._VLM_PENDING is None)
    print(f"\n还原成功：{ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
