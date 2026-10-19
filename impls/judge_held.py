"""判官 `held`：真机上判「抓起来了、抬起来没掉」。

两条判据，都用真机读得到的东西：

    6 号舵机角度   抬完之后嘴里还有没有东西（`grasp_check.classify_ref`）
    TCP 高度       从桌面上抬起来了没有

**不读方块真值。** `impls/judge_grasp.py` 是读的（`ev["cube_pos"]`、`object_rest_z`），
真机上没有 `cube_pos` 这个查询。证据不是自己读的，是主循环在轮末塞进
`Round.evidence` 的几个数：

    grip_deg  轮末现读的 6 号舵机角度（度）。**在轮末读一次**，不去信技能自己报的
              —— 判官量的是现在的现场，不是听人说刚才怎么样
    empty_deg 技能开跑时实测的空抓基准（度，`grasp_check.measure_empty`）。判据是
              「grip_deg 比它高了多少」，不问这只盒子该落在哪个角度区间 —— 换盒子
              不用重标。技能没跑起来就是 None，那时判不了，如实报
    tcp_z     轮末 TCP 的高度（米）
    lift_z    超过这个高度才算「离了桌面」（米，= 桌面 z + 摆件半高）

这一轮**不判放置**（放没放到点上），那是下一轮的事。
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import grasp_check                                                # noqa: E402
from core.interfaces import Judge                                 # noqa: E402
from core.registry import register                                # noqa: E402
from core.types import Round, Verdict                             # noqa: E402


@register("judge", "held")
class HeldJudge(Judge):
    """靠夹爪角度 + TCP 高度判「抓起来没掉」。"""

    name = "held"

    def judge(self, r: Round) -> Verdict:
        ev = dict(r.evidence or {})
        deg = ev.get("grip_deg")
        empty_deg, open_deg = ev.get("empty_deg"), ev.get("open_deg")
        tcp_z, lift_z = ev.get("tcp_z"), ev.get("lift_z")
        if (deg is None or empty_deg is None or open_deg is None
                or tcp_z is None or lift_z is None):
            # **读不到就说读不到，不猜。** 尤其 `grip_deg` 是 None 的时候 —— 那是
            # 「这一轮没读到舵机」，不是「爪里没东西」，两者天差地别。
            # `open_deg` 同样必填：没有张开那头，「读数比空抓基准高」就分不清是
            # 「夹着东西」还是「爪张着」—— 2026-10-14 首跑就是这么把全开的 168°
            # 判成 held 的。
            return Verdict(False, self.name, color=r.target,
                           reason=f"证据不全，判不了（grip_deg={deg} "
                                  f"empty_deg={empty_deg} open_deg={open_deg} "
                                  f"tcp_z={tcp_z} lift_z={lift_z}）",
                           detail=ev)

        verdict = grasp_check.classify_ref(deg, empty_deg,
                                           grasp_check.load_thresholds(),
                                           open_deg)
        lift_m = float(tcp_z) - float(lift_z)
        detail = {**ev, "verdict": verdict, "lift_m": lift_m}
        if verdict != grasp_check.HELD:
            return Verdict(False, self.name, color=r.target,
                           reason=f"爪里没有东西（6 号舵机 {deg:.1f}°，"
                                  f"行程 {empty_deg:.1f}°~{open_deg:.1f}°，判 {verdict}）",
                           detail=detail)
        if lift_m <= 0.0:
            return Verdict(False, self.name, color=r.target,
                           reason=f"夹住了但没抬起来（TCP 离桌面 {lift_m * 1000:.1f}mm）",
                           detail=detail)
        return Verdict(True, self.name, color=r.target,
                       reason=f"夹住了、抬起来了（6 号舵机 {deg:.1f}°，"
                              f"行程 {empty_deg:.1f}°~{open_deg:.1f}°，"
                              f"离桌面 {lift_m * 1000:.1f}mm）",
                       detail=detail)
