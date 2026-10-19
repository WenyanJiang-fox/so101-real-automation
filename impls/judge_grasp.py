"""抓取判官：只判「抓起来了、没掉」。

**为什么要单有一个。** `judge_truth` 判的是「拿起来**放到**指定位置」，两条都满足
才算成。盲测那一支的任务就一句「把它抓起来」—— 不往下放，也就没有落点可判；拿
`judge_truth` 去判，抓得再稳也是「还挂在天上，没放下去」，成功永远是 0。

判据两条，都用真值（仿真里真值就在手上，不用猜）：
    爪里还是它   `evidence["held"]` 等于这一轮的目标序号。夹爪一松、物件一滑出去，
                 这个数当场变 None —— `pick` 全程守着（`_gone_guard`），不是这里才查。
    真的离了桌   物件中心高出它自己躺平的高度 `LIFT_TOL` 以上。跟 `judge_truth` 的
                 `GROUND_TOL` 是一对：那边高过它算「没放下」，这边高过它算「拿起来了」。

两条都成立才算成。**只看这两件事**：夹在哪个方位、夹得正不正、路上晃没晃，都不判。
"""
from __future__ import annotations

import numpy as np

from core.geom import object_rest_z
from core.interfaces import Judge
from core.registry import register
from core.types import Round, Verdict

LIFT_TOL = 0.015   # 米。比这个高出多少才算真离了桌面（抬升段实测能到 60mm 往上）


@register("judge", "grasp")
class GraspJudge(Judge):
    """看这一轮的目标最后是不是被爪提着、离了桌面。"""

    name = "grasp"

    def judge(self, r: Round) -> Verdict:
        ev = r.evidence or {}
        idx = ev.get("target_index")
        if idx is None or "cube_pos" not in ev:
            return Verdict(success=False, source="grasp", color=r.target,
                           reason="没有拿到这一轮的落点证据")

        idx = int(idx)
        here = np.asarray(ev["cube_pos"], dtype=float)[idx]
        rest = object_rest_z(idx)
        clear = float(here[2]) - rest
        held = ev.get("held")
        in_jaws = held is not None and int(held) == idx
        lifted = clear > LIFT_TOL
        ok = bool(in_jaws and lifted)

        if ok:
            reason = f"抓起来了，离桌 {clear * 1000:.0f}mm，还在爪里"
        elif not in_jaws:
            reason = "爪里没有它（没夹住，或者半路掉了）"
        else:
            reason = f"还在爪里但没离桌（只高出 {clear * 1000:.0f}mm）"

        return Verdict(success=ok, source="grasp", color=r.target, reason=reason,
                       detail={"target_index": idx,
                               "final_xyz": [float(v) for v in here],
                               "rest_z": float(rest),
                               "clear_m": clear,
                               "held_at_end": held})
