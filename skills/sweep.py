"""摆场：把三个方块摆到拾取工位上。

**这一步不是机械臂干的，是「操作员」干的** —— 直接写场景状态，机械臂不动。
真机上就是人把方块放上去：位置放得准、朝向放不准，所以位置写死、朝向随机。

为什么要有这一步：每轮开头得让方块确实在工位附近，机械臂才有得抓。摆场的动画做成
一条抬起来的弧线，免得平移时从别的方块身上刮过去。

`OFFSET_XY` 是给对照实验用的：真机上人摆不到那么准，用它试试偏开工位以后机械臂还
找不找得到。默认 None，就是老老实实摆在工位坐标上。
"""

from __future__ import annotations

import numpy as np

from core import follow
from core.geom import PICK_XY, object_rest_z
from core.interfaces import Skill
from core.params import (PARK_FRAMES, PARK_JITTER_YAW, PARK_LIFT, PARK_SNAP,
                         OPEN_STEP)
from core.registry import register

# 固定的随机源：同一个种子每轮摆出来的朝向一样，出问题能复现
_RNG = np.random.default_rng(20261005)

# 给对照实验用的朝向覆盖，单位度；None = 照常按 PARK_JITTER_YAW 随机摇。
# 有了它才能一次换一个角度，看机械臂的转正是不是真跟着方块朝向走。
YAW_DEG: float | None = None

# 给对照实验用的位置偏移（米，x 和 y）。None = 就摆在工位坐标上。
# 真机上人摆不到那么准，所以拿它试「偏开工位之后机械臂还找不找得到」。
OFFSET_XY: tuple[float, float] | None = None


def _target(i: int) -> np.ndarray:
    """这一轮第 i 块该摆到哪儿：工位坐标 + 实验指定的偏移。"""
    p = np.asarray(PICK_XY[i], dtype=float)
    return p if OFFSET_XY is None else p + np.asarray(OFFSET_XY, dtype=float)


@register("skill", "park")
class ParkSkill(Skill):
    """把方块摆回工位。"""

    name = "park"

    def run(self, arm, cams, state=None, **kw) -> dict:
        if state.held is not None:
            follow.hold(arm, 60, grip=OPEN_STEP)   # 上一轮没放下去，操作员从爪子里取走
            state.clear_hand()

        state.seg, state.ignore = "摆件（操作员）", set(range(arm.n_cubes))
        start = np.array([arm.cube_pos(i) for i in range(arm.n_cubes)])
        tgt = np.array([_target(i) for i in range(arm.n_cubes)])
        move = [i for i in range(arm.n_cubes)
                if float(np.linalg.norm(start[i][:2] - tgt[i])) > PARK_SNAP]
        if not move:
            print("  摆件：方块都已在工位上，不动")
            return {"moved": []}

        # 朝向每块只摇一次，整段动画里保持不变。以前是每帧摇一次，方块一边被搬
        # 一边自己转，看着像在飘 —— 真机上人放下去了就不会再转。
        yaw = ({i: float(np.radians(YAW_DEG)) for i in move}
               if YAW_DEG is not None
               else {i: float(_RNG.uniform(-PARK_JITTER_YAW, PARK_JITTER_YAW))
                     for i in move})
        for k in range(1, PARK_FRAMES + 1):
            t = k / PARK_FRAMES
            for i in move:
                xy = (start[i][0] + (tgt[i][0] - start[i][0]) * t,
                      start[i][1] + (tgt[i][1] - start[i][1]) * t)
                arm.place_cube(i, xy,
                               object_rest_z(i) + PARK_LIFT * np.sin(np.pi * t),
                               yaw[i])
            arm.step_idle()

        now = np.array([arm.cube_pos(i) for i in range(arm.n_cubes)])
        print("  摆件：" + " / ".join(
            f"{arm.cube_order[i]}({start[i][0] * 1000:.0f},{start[i][1] * 1000:.0f})"
            f"→({now[i][0] * 1000:.0f},{now[i][1] * 1000:.0f})" for i in move)
            + ("   其余已在工位" if len(move) < arm.n_cubes else "")
            + (f"   指定朝向 {YAW_DEG:+.0f}°" if YAW_DEG is not None else "")
            + (f"   偏开工位 {OFFSET_XY[0] * 1000:+.0f},{OFFSET_XY[1] * 1000:+.0f}mm"
               if OFFSET_XY is not None else ""))
        return {"moved": move}
