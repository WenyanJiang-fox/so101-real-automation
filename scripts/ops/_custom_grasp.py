# -*- coding: utf-8 -*-
"""一次性：给桌上这只绿方块单开一条抓法。**抓成了就删掉这个文件。**

跟主技能只差一处：**合爪前的水平位置只认画面那把尺子**（`jaw_shape.json` 里描死的
爪轮廓 + 红区轮廓，走 `_set_clearance` 那套闭环），不认「对准后重量」的 3D 中心。
主技能这一版把微调整段跳过（`knobs.SKIP_CLEARANCE_REAL = True`），落点全押在 3D
中心上；实测腕子转 42° 之后那个中心自己漂了 28mm
（`out/panel_20261019_measure_label.log:51`）。

流程全部走技能里现成的那几段，不另写运动：
  回初始位置 → 进场看一眼 → 纠位/转正 → 合爪前微调（画面尺子，最多 8 步）
  → 竖直下探 → **合爪一次** → 判 → 抬。
**只合一次爪，不重试。**

    G:/anaconda/envs/sim2real/python.exe -u _custom_grasp.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 闸口一律关掉：不停下来等人。
for _k in ("PICK_CONFIRM_STEP", "PICK_CONFIRM", "PICK_CONFIRM_DESCEND",
           "PICK_RESUME", "PICK_START_AT", "PICK_MANUAL"):
    os.environ.pop(_k, None)

import numpy as np                                               # noqa: E402

from core import geom                                            # noqa: E402
from core.registry import make                                   # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg            # noqa: E402

cfg = load_cfg(DEFAULT_CONFIG)
rig = cfg["real"]
geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                             dtype=np.float64) / 1000.0
geom.REAL_PERCEP_OFFSET = np.array(
    [float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
    dtype=np.float64) / 1000.0
geom.OBJECTS = list(cfg.get("objects") or [])

import impls.judge_held                                          # noqa: E402,F401
import impls.policy_real                                          # noqa: E402,F401
import real.arm                                                   # noqa: E402,F401
import real.cameras                                               # noqa: E402,F401
import real.perception                                            # noqa: E402,F401
import skills.home_real                                           # noqa: E402,F401
import skills.pick_real                                           # noqa: E402,F401
from skills.pick_real import align as ALIGN                       # noqa: E402
from skills.pick_real import knobs                                # noqa: E402

# 只改这一处：这一趟不跳过微调，合爪前的水平位置由画面尺子闭环决定。
ALIGN.SKIP_CLEARANCE_REAL = False
print(f"[定制] 合爪前微调：开（主技能这一版是关的，SKIP_CLEARANCE_REAL="
      f"{knobs.SKIP_CLEARANCE_REAL}）", flush=True)

print("=" * 66)
print("CUSTOM GRASP：画面尺子定水平位置，只合一次爪")
print("=" * 66, flush=True)

arm = make("arm", "real", port=str(rig.get("port", "COM10")),
           speed=int(rig.get("speed", 0)))
arm.terminated = False
cams = {"real_wrist": make("camera", "real_wrist", source=arm,
                           index=int(rig.get("cam_index", 0)))}
percep_kw = ({"cam_name": str(cfg["perception_cam"])}
             if cfg.get("perception_cam") else {})
percep = make("perception", "brown_rect", cams=cams, ref=None,
              objects=geom.object_specs(), **percep_kw)

try:
    home = make("skill", "home_real")
    print("[回初始位置]", home.run(arm, cams), flush=True)

    from core.round_state import RoundState                       # noqa: PLC0415

    state = RoundState()
    skill = make("skill", "pick_real")
    out = skill.run(arm, cams, state=state, index=0, percep=percep,
                    instruction="pick up the green box")
    print("[结果]", out, flush=True)
    ok = bool(out.get("ok"))
    print(f"[定制] {'抓成了' if ok else '没抓成'}：verdict={out.get('verdict')}"
          f"  why={out.get('why') or '--'}", flush=True)
finally:
    try:
        cams["real_wrist"].close()
    except Exception as exc:                                      # noqa: BLE001
        print("关相机：", exc, flush=True)
    try:
        arm.close()
    except Exception as exc:                                      # noqa: BLE001
        print("关串口：", exc, flush=True)
