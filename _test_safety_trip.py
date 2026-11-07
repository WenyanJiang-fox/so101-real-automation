# -*- coding: utf-8 -*-
"""故意让安全闸拦一次，验证「安全闸 → 存档 → VLM 给变量 → 写回策略」整条路。

做法：临时把 1 号关节限幅放开（`knobs.JOINT1_STEP_DEG`，运行时值，不改文件），
命令一个要横跨基座的点 `(-150, 80, 80)mm`。`_goto` 在**写舵机之前**会过
`safety.check_step`：一步要动 >45° 就立 `terminated`、一个字节都不写。

拦下之后调 `collect_real_dataset._safety_episode` —— 和采集循环走的是同一条路：
存档 → 问 VLM → 把它的 `joint1_step_deg` / 其它旋钮写进策略。
"""
from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, r"G:\sim2real\so101-real-harness")

import collect_real_dataset as C                                  # noqa: E402
import real.arm                                                   # noqa: E402,F401  注册臂
import real.cameras                                               # noqa: E402,F401  注册相机
import skills.home_real                                           # noqa: E402,F401  注册 home
import skills.pick_real                                           # noqa: E402,F401  注册抓取技能
from core.params import speed_vmax                                # noqa: E402
from ppo.policy import ACTIONS, RetryPolicy                       # noqa: E402
from skills.pick_real import knobs as K                           # noqa: E402
from skills.pick_real import motion                               # noqa: E402

GOAL_MM = (-150.0, 80.0, 80.0)


def main() -> int:
    cfg = C.load_cfg(C.DEFAULT_CONFIG)
    C._setup_geom(cfg)
    rig = cfg["real"]
    C._BASE_OFF = np.array(__import__("core.geom", fromlist=["x"]).REAL_PERCEP_OFFSET,
                           dtype=np.float64, copy=True)
    C._WORKSPACE = C._workspace(cfg)

    arm = C.make("arm", "real", port=str(rig.get("port", "COM10")),
                 speed=int(rig.get("speed", 0)), fast=bool(rig.get("fast", False)))
    arm.terminated = False
    wrist = C.make("camera", "real_wrist", source=arm,
                   index=int(rig.get("cam_index", 1)))
    try:
        C.make("skill", "home_real").run(arm, {"real_wrist": wrist})
        before = K.JOINT1_STEP_DEG
        K.JOINT1_STEP_DEG = 180.0          # 只在这一趟放开，让闸能看到原始的步长
        print(f"[试验] 1 号关节限幅 {before}°，本趟临时放到 180°")
        try:
            out = motion._goto(arm, np.array(GOAL_MM, dtype=np.float64) / 1000.0,
                               vmax=speed_vmax(), grip=1.0, tries=1)
        except Exception as exc:                                   # noqa: BLE001
            print("[试验] _goto 抛异常：", type(exc).__name__, exc)
            out = "exception"
        finally:
            K.JOINT1_STEP_DEG = before
        print(f"[试验] _goto 结果={out}  safety_why={getattr(arm, 'safety_why', None)!r}")
        if not getattr(arm, "safety_why", None):
            print("[试验] 没触发安全闸 —— 这个目标解出来没超 45°，换目标再试")
            return 2

        pol = RetryPolicy(max_attempts=None)
        stats: dict = {}
        params: dict = {}
        cur = next(a for a in ACTIONS if a.name == "no_measure")
        C._safety_episode(arm, wrist, None, ep=0, round_no=0, strategy=cur,
                          out={}, stats=stats, pol=pol, params_out=params)
        print("[试验] VLM 写回的变量：", json.dumps(params, ensure_ascii=False))
        print("[试验] 策略侧生效值：JOINT1_STEP_DEG =", K.JOINT1_STEP_DEG)
        print("[试验] arm 标志：terminated =", arm.terminated,
              " safety_why =", arm.safety_why)
        return 0
    finally:
        try:
            wrist.close()
        except Exception:                                          # noqa: BLE001
            pass
        arm.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
