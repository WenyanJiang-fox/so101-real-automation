# -*- coding: utf-8 -*-
"""一次性实验：**跳过合爪前微调**，让固定爪尖直接到算出来的落点再合爪。

不修改任何现有文件。只在内存里把 `skills.pick_real.skill` 这个名字空间里的
`_set_clearance` 换成一个直接放行的桩 —— 走到那儿不挪水平、不量画面，紧接着由
`_descend_to_grasp` 竖直扎到「方块中心 - 1mm」，然后合爪、判、抬。

落点本身仍是技能自己的算式（`skill.py` 第 566-568 行）：
    固定爪尖 = 中心 + side·(半跨度 + 爪面探出 + 缝 3.3mm)
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
           "PICK_RESUME", "PICK_START_AT"):
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
from skills.pick_real import gates as GATES                      # noqa: E402
from skills.pick_real import skill as SKILLMOD                    # noqa: E402

# 只改这一处：竖直下探的终点从「中心 -1mm」抬到「中心 +2mm」。
# 上一趟实测：指令中心 -1mm、走完读回中心 -3.9mm（多走 2.9mm），爪尖离桌面只剩 4mm。
GATES.GRASP_Z_FINAL_REAL = 0.002
print(f"[实验] 竖直下探终点 = 中心 {GATES.GRASP_Z_FINAL_REAL * 1000:+.0f}mm"
      f"（原 -1mm）", flush=True)

print("=" * 66)
print("ONE-SHOT：跳过合爪前微调，固定爪尖直接到算出来的落点")
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

    def _skip_fine(arm_, cams_, percep_, center_, tag, **kw):
        print(f"    [ONE-SHOT] 跳过微调「{tag}」：水平不动，直接往下扎", flush=True)
        return {"ok": True}

    SKILLMOD._set_clearance = _skip_fine

    from core.round_state import RoundState                       # noqa: PLC0415

    state = RoundState()
    skill = make("skill", "pick_real")
    out = skill.run(arm, cams, state=state, index=0, percep=percep,
                    instruction="pick up the brown box")
    print("[结果]", out, flush=True)
finally:
    try:
        cams["real_wrist"].close()
    except Exception as exc:                                      # noqa: BLE001
        print("关相机：", exc, flush=True)
    try:
        arm.close()
    except Exception as exc:                                      # noqa: BLE001
        print("关串口：", exc, flush=True)
