# -*- coding: utf-8 -*-
"""先故意触发安全闸 → 让 VLM 给限制 → 再用我们自己的三档策略各跑一趟。

三档的顺序就是 `ppo.policy.ACTIONS`：先量一眼 → 不量直接下扎 → 完整微调。
每档之前都回初始位置，**不录数据集**。三趟各合一次爪（用户 2026-10-22 明确要试）。
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, r"G:\sim2real\so101-real-harness")

import collect_real_dataset as C                                  # noqa: E402
import impls.judge_held                                           # noqa: E402,F401
import impls.policy_real                                          # noqa: E402,F401
import real.arm                                                   # noqa: E402,F401
import real.cameras                                               # noqa: E402,F401
import real.perception                                            # noqa: E402,F401
import skills.home_real                                           # noqa: E402,F401
import skills.pick_real                                           # noqa: E402,F401
from core import geom                                             # noqa: E402
from core.params import speed_vmax                                # noqa: E402
from core.round_state import RoundState                           # noqa: E402
from ppo.policy import ACTIONS, RetryPolicy, apply_action         # noqa: E402
from skills.pick_real import knobs as K                           # noqa: E402
from skills.pick_real import motion                               # noqa: E402

TRIP_GOAL_MM = (-150.0, 80.0, 80.0)


def main() -> int:
    cfg = C.load_cfg(C.DEFAULT_CONFIG)
    C._setup_geom(cfg)
    rig = cfg["real"]
    C._BASE_OFF = np.array(geom.REAL_PERCEP_OFFSET, dtype=np.float64, copy=True)
    C._WORKSPACE = C._workspace(cfg)
    color = (cfg.get("objects") or [{"color": "green"}])[0].get("color", "green")
    task = f"pick up the {color} box"

    arm = C.make("arm", "real", port=str(rig.get("port", "COM10")),
                 speed=int(rig.get("speed", 0)), fast=bool(rig.get("fast", False)))
    arm.terminated = False
    arm.safety_why = None
    cams = {"real_wrist": C.make("camera", "real_wrist", source=arm,
                                 index=int(rig.get("cam_index", 1)))}
    percep_kw = ({"cam_name": str(cfg["perception_cam"])}
                 if cfg.get("perception_cam") else {})
    percep = C.make("perception", "brown_rect", cams=cams, ref=None,
                    objects=geom.object_specs(), **percep_kw)
    home = C.make("skill", "home_real")
    skill = C.make("skill", "pick_real")

    try:
        # ---- 1. 故意触发安全闸 ----
        home.run(arm, cams)
        before = K.JOINT1_STEP_DEG
        K.JOINT1_STEP_DEG = 180.0
        print(f"[1] 故意走 {TRIP_GOAL_MM}mm（1 号限幅临时放到 180°）", flush=True)
        try:
            motion._goto(arm, np.array(TRIP_GOAL_MM) / 1000.0,
                         vmax=speed_vmax(), grip=1.0, tries=1)
        finally:
            K.JOINT1_STEP_DEG = before
        if not getattr(arm, "safety_why", None):
            print("[1] 没触发安全闸，试验终止", flush=True)
            return 2
        print(f"[1] 触发：{arm.safety_why}", flush=True)

        # ---- 2. 让 VLM 给限制，并写回策略 ----
        pol = RetryPolicy(max_attempts=None)
        stats: dict = {}
        params: dict = {}
        C._safety_episode(arm, cams["real_wrist"], None, ep=0, round_no=0,
                          strategy=ACTIONS[1], out={}, stats=stats,
                          pol=pol, params_out=params)
        print("[2] VLM 变量：" + json.dumps(params, ensure_ascii=False), flush=True)
        print(f"[2] 生效：JOINT1_STEP_DEG={K.JOINT1_STEP_DEG}°", flush=True)

        # ---- 3. 三档各跑一趟 ----
        for i, act in enumerate(ACTIONS, 1):
            h = home.run(arm, cams)
            if not h.get("ok"):
                print(f"[3-{i}] 回初始位置没到位：{h.get('why')}", flush=True)
                break
            arm.terminated = False
            arm.safety_why = None
            apply_action(act, **C._apply_kw(params))
            print(f"\n[3-{i}] 「{act.label}」开始", flush=True)
            out = skill.run(arm, cams, state=RoundState(), index=0,
                            percep=percep, instruction=task)
            pol.observe(out, act)
            print(f"[3-{i}] 「{act.label}」结果 ok={out.get('ok')} "
                  f"verdict={out.get('verdict')} dropped={out.get('dropped')} "
                  f"why={out.get('why')}", flush=True)
            if getattr(arm, "safety_why", None):
                print(f"[3-{i}] 这一趟被安全闸拦下：{arm.safety_why}", flush=True)
                arm.safety_why = None
                arm.terminated = False
        return 0
    finally:
        try:
            home.run(arm, cams)
        except Exception as exc:                                   # noqa: BLE001
            print("回初始位置失败：", exc, flush=True)
        try:
            cams["real_wrist"].close()
        except Exception:                                          # noqa: BLE001
            pass
        arm.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
