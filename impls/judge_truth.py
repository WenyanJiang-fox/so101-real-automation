"""真值判官：直接看方块最后停在哪，不看画面。

仿真里判成败不需要猜 —— 方块的位置和身份都是已知的。所以这个判官只做一件事：
**这一轮的目标方块，最后是不是落在它自己的放置标识点上、并且已经落地**。

两条都满足才算成：
    落点      方块中心到放置标识点的水平距离，小于 PLACE_TOL
    落地      方块中心高度贴近地面（松手之后是躺着的，不是被爪子提着）

**为什么不用「抬起来就算成」**：那是环境自己的成功判据（lift_threshold），只说明
抓起来了；这一轮的任务是「拿起来放到指定位置」，没放到就不算完。

**拿哪个坐标当「指定位置」**：优先用 `evidence["goal_xy"]` —— 那是这一轮真正要去的
那一点，可能是外面给的（`core.geom.PLACE_TARGET`），也可能就是表里的放置标识点。
证据里没有才退回表。判官不自己决定落点，只认证据。

真机上没有真值（没人知道方块实际在哪），那种情况换 `judge_threshold`：拿界线
（爪子里有没有东西、方块在不在视野里的判定区）来判。这就是「判官能换」的意思。
"""

from __future__ import annotations

import numpy as np

from core.geom import PLACE_XY, object_rest_z
from core.interfaces import Judge
from core.registry import register
from core.types import Round, Verdict

PLACE_TOL = 0.02   # 放置点判定半径（米）。实测放置残差约 12mm，取 20mm 留一倍余量；
                   # 再放大就会把「放到隔壁标识点上」也判成成功
GROUND_TOL = 0.010  # 落地判定：中心比这个形状**自己**躺平的高度高出这么多以上，算还在半空


@register("judge", "truth")
class TruthJudge(Judge):
    """看方块最后停在哪。"""

    name = "truth"

    def judge(self, r: Round) -> Verdict:
        ev = r.evidence or {}
        idx = ev.get("target_index")
        if idx is None or "cube_pos" not in ev:
            return Verdict(success=False, source="truth", color=r.target,
                           reason="没有拿到这一轮的落点证据")

        final = np.asarray(ev["cube_pos"], dtype=float)
        given = ev.get("goal_xy") is not None
        goal = np.asarray(ev["goal_xy"] if given else PLACE_XY[idx], dtype=float)
        here = final[idx]
        off = float(np.linalg.norm(here[:2] - goal))
        # **躺平高度按这个形状自己算，不能一律按 30mm 方块那 15mm。** 一根 60mm
        # 高的圆柱躺在地上，中心就在 30mm，拿 15mm 去比会把它一直判成「还挂在天上」。
        on_ground = float(here[2]) < object_rest_z(idx) + GROUND_TOL
        ok = bool(off < PLACE_TOL and on_ground)
        where = "给的坐标点" if given else "自己的放置标识点"

        if ok:
            reason = f"落到了{where}上"
        elif not on_ground:
            reason = "还挂在天上，没放下去"
        else:
            reason = f"放偏了，没落到{where}上"

        return Verdict(success=ok, source="truth", color=r.target, reason=reason,
                       detail={"target_index": int(idx),
                               "final_xyz": [float(v) for v in here],
                               "goal_xy": [float(v) for v in goal],
                               "goal_given": bool(given),
                               "offset_m": off,
                               "on_ground": on_ground,
                               "held_at_end": ev.get("held")})
