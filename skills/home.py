"""就位：把爪子收回中立位。

**为什么要先回中立位再去下一个方块**：直接从这个方块上方直奔下一个，两点的水平
方位角可能差一百多度，直连的弦会从底座上方切进去（`geom.round_route` 开头写了
实测数字）。先回到中立位再出发，等于把「横穿整张工位」拆成两段短的。

回中立位只走位置、不管朝向 —— 朝向由下一段的 `pick` 自己去对。
"""

from __future__ import annotations

from core import follow
from core.interfaces import Skill
from core.params import SPEED, WAYPOINT, acc_amax, speed_vmax
from core.registry import register


@register("skill", "home")
class HomeSkill(Skill):
    """回中立位。"""

    name = "home"

    def run(self, arm, cams, state=None, **kw) -> dict:
        if state is not None:
            state.seg, state.ignore = "回中立位", set()
        reason = follow.follow(arm, [WAYPOINT], vmax=speed_vmax(SPEED),
                               amax=acc_amax(), grip=1.0, final_tol=0.010)
        if state is not None and state.check_bump(arm):
            pass
        return {"ok": reason == "reached", "why": reason}
