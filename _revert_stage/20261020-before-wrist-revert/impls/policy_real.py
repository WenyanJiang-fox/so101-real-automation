"""真机策略：只走抓取这一段。

和 `impls/policy_expert.py` 的差别不是「人写的顺序换成模型写的」，是**那一支有三件事
真机上没有对应物**，所以没带过来：

    `_out_of_reach`  读 `arm.cube_pos`（方块真值坐标）—— 真机上只有画面，没这个查询。
                     够不够得着由 `follow` 自己走出来的结果说话（走不到就 stall）。
    `_landed`        建一个 `Round` 去问判官「放没放到点上」—— 这一轮不判放置。
    `park`           操作员摆件是仿真的动作 —— 真机上摆件就是人把盒子放到桌上。

**`home` 换成了 `home_real`。** 仿真那支每轮结尾回 `core.params.WAYPOINT`，那是仿真
工位的坐标，真机够不够得着不知道。真机自己的初始点位在 `real_home.py`（`RAW_HOME`），
而且**每轮开头也先回一次** —— 不这么干，臂就停在上一轮收工时的地方（实测停到桌面
高度 `z=-0.010`、腕相机横着，后面每一眼都看不见盒子）。

留下的就一件事：**回初始位置 → 抓一次 → 没抓着就再试，试满 `max_attempts` 次收工
→ 回初始位置**。
"""

from __future__ import annotations

import time
from typing import Any

from core.interfaces import Policy
from core.registry import make, register


@register("policy", "real")
class RealPolicy(Policy):
    """真机上按写死的顺序调小模块。只有抓取这一段。"""

    name = "real"

    ORDER = ("home_real", "pick_real")

    def __init__(self, arm, cams, state, order=None, percep=None, judge=None,
                 max_attempts=3, attempt_seconds=180.0, round_seconds=600.0,
                 goal="grasp"):
        self.arm = arm
        self.cams = cams
        self.state = state
        self.order = tuple(order or self.ORDER)
        self.percep = percep
        self.judge = judge
        self.goal = str(goal or "grasp").lower()
        # 0 = 不管这一条。**真机的秒数要比仿真给得宽**：一个控制步 ≈ 读 66ms + 写 21ms
        # + 解算，贴近 0.1 秒，仿真那 60 秒在这里只够走六百步。默认值只是「配置里没写」
        # 时的兜底，正常从 `configs/real.yaml` 来。
        self.max_attempts = int(max_attempts or 0)
        self.attempt_seconds = float(attempt_seconds or 0.0)
        self.round_seconds = float(round_seconds or 0.0)
        self.index: int | None = None
        self.log: list[tuple[str, dict]] = []
        self.t0 = 0.0
        self.attempt = 0
        self.attempt_started = 0.0
        self.stop_why = ""

    def reset(self, info: dict[str, Any]) -> None:
        self.index = None if info is None else info.get("target_index")
        self.log = []
        self.t0 = 0.0
        self.attempt = 0
        self.attempt_started = 0.0
        self.stop_why = ""

    def deadline(self) -> tuple[float, str] | None:
        """时间上限。两条都算出来取**先到的那个** —— 整轮的总上限永远兜得住单次的。"""
        if self.t0 <= 0.0:
            return None
        cands = []
        if self.round_seconds > 0.0:
            cands.append((self.t0 + self.round_seconds,
                          f"整轮跑满 {self.round_seconds:.0f}s（总上限）"))
        if self.attempt_seconds > 0.0 and self.attempt > 0:
            cands.append((self.attempt_started + self.attempt_seconds,
                          f"第 {self.attempt} 次尝试跑满 {self.attempt_seconds:.0f}s"))
        return min(cands) if cands else None

    def plan(self, obs: dict[str, Any]) -> list:
        obs = obs or {}
        task = obs.get("task")
        self.t0 = time.time()
        self.attempt = 0
        self.stop_why = ""

        # **每轮开头先回初始位置。** 仿真那支是靠「上一轮结尾调了 home」让这一轮自然
        # 从那儿出发的，真机这一支原来两头都没有，于是臂停在上一轮收工的地方。摆件那
        # 一步（`park`）真机上没有对应物，所以只有回位这一步。
        self._call("home_real", task)

        while True:
            # **先看试满了没有，再数这一次。** 反过来写（先 +1 再比）的话，第 3 次跑完
            # 回头一看 4 > 3 才收工，日志和记录里就成了「试了 4 次」。
            if self.max_attempts > 0 and self.attempt >= self.max_attempts:
                self._stop(f"抓了 {self.max_attempts} 次都没抓起来，收工")
                break
            self.attempt += 1
            self.attempt_started = time.time()
            got = self._call("pick_real", task)
            if got.get("ok"):
                self._stop(f"第 {self.attempt} 次抓起来了（这一轮只判抓起来没掉）")
                break
            # **「找不着」是收工，不是再来一次。** 同一个现场、同一句话，再来一遍还是
            # 找不着 —— 抓取那边不是「没抓住」，是「压根没看清盒子在哪/朝哪」。
            if got.get("give_up"):
                self._stop(f"找不着目标（{got.get('why')}）")
                break
            print(f"    ↻ 没抓起来（第 {self.attempt}/{self.max_attempts or '∞'} 次），"
                  f"回去再来")
        # 收工回位，照仿真那支（`policy_expert` 第 162 行）。**试着抓着东西也照回** ——
        # 初始点位的夹爪是张开的，抓着东西回去等于把它放下，这是有意的。
        self._call("home_real", task)
        return []

    def _call(self, name: str, task) -> dict:
        """调一个小模块，把结果记进日志。日志由主循环统一打印。"""
        out = make("skill", name).run(self.arm, self.cams, state=self.state,
                                      index=self.index, percep=self.percep,
                                      instruction=task)
        self.log.append((name, out))
        return out

    def _stop(self, why: str) -> None:
        self.stop_why = why
        print(f"    ⏹ 收工：{why}")
