"""专家策略：人把顺序写死，照着调小模块。

**它是策略的一种，不是「没有策略」。** 大模型策略也是这个接口，区别只在顺序是谁
定的 —— 这里是人写在 `ORDER` 里，那边是模型当场看着画面决定。所以「换成大模型
参与控制」这件事，改的就是这一个文件，机械臂、相机、判官一个字不动。

一轮怎么走：
    摆件（操作员把方块摆到工位，机械臂不动）  —— **一轮只摆一次**
    然后**一次尝试 = 抓一次 + 放一次**，最多重来 `max_attempts` 次：
        抓  → 没抓起来，这一次就没成，回去再来
        放  → 放下去了但没落到点上（判官说的），也算这一次没成，回去再来

**三种情况当场收工，不再试**：
    出界   目标方块出了可达环（`core.geom.out_of_reach`）—— 够不着，不可逆
    试满   连着 `max_attempts` 次都没成
    找不着 抓取那边换机位扫了一圈、又换了俯视相机，还是认不出方块在哪/朝哪。
           它是在那边试完了才回来报的（`{"give_up": True}`）—— 同一个现场、同一句话，
           再来一次还是认不出来，重试只是把同一段路再走一遍。

**判官在轮中也要问一次。** 判官判的是「这一局成没成」，不是「该不该再试」；轮末
那一次照旧由主循环问，这里问的是中间那一次尝试到底落地了没有。两次用的是**同一个
判官**，所以「落点偏多少算没落到点上」始终只有一个说法（`judge_truth.PLACE_TOL`）。

**返回空序列 = 这一轮干完了。** 这个策略不像模型那样「吐一串动作」：它调的小模块
每个自己会走几十上百步（每一步都发一维动作），走完了才回来。所以 `plan()` 返回
空表，动作是在小模块里逐步发出去的 —— 发出去的每一步都会被主循环录下来。
"""

from __future__ import annotations

import time
from typing import Any

from core import geom
from core.interfaces import Policy
from core.registry import make, register
from core.round_state import evidence
from core.types import Round


@register("policy", "expert")
class ExpertPolicy(Policy):
    """按写死的顺序调小模块。"""

    name = "expert"

    ORDER = ("park", "pick", "place", "home")

    def __init__(self, arm, cams, state, order=None, percep=None, judge=None,
                 max_attempts=3, attempt_seconds=30.0, round_seconds=120.0,
                 goal="place"):
        self.arm = arm
        self.cams = cams
        self.state = state
        self.order = tuple(order or self.ORDER)
        self.percep = percep
        self.judge = judge
        # `goal="grasp"`：这一轮只到「抓起来、没掉」为止，不往下放（判官换 `judge_grasp`）。
        # 盲测那一支考的就是抓不抓得住，落点和放置都不在这一次的问题里。
        self.goal = str(goal or "place").lower()
        # 0 = 不管这一条。默认值只是「配置里没写」时的兜底，正常都从配置来。
        self.max_attempts = int(max_attempts or 0)
        self.attempt_seconds = float(attempt_seconds or 0.0)
        self.round_seconds = float(round_seconds or 0.0)
        self.index: int | None = None
        self.log: list[tuple[str, dict]] = []
        self.t0 = 0.0             # 这一轮开始的墙钟时刻
        self.attempt = 0          # 现在第几次尝试（1 起数）
        self.attempt_started = 0.0
        self.stop_why = ""        # 这一轮是为什么收的工，给日志和记录用

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

        if self.order != self.ORDER:
            # `--steps` 指定了只走其中几步（调试用）：照原样一遍走完，不重试、不判收工。
            # 那是个调试开关，不是「这一轮怎么跑得更好」。
            for name in self.order:
                self._call(name, task)
            return []

        # 摆件只做一次。重试的时候方块就躺在它掉下去 / 放偏的那个位置上 —— 再摆一次
        # 等于把场地清干净重来，试的就不是同一个现场了。
        self._call("park", task)

        # **只判「抓起来没掉」的那一支**（`goal="grasp"`）：抓成了这一次就成了，不往下
        # 放，也不问判官 —— 判据是抓取自己报的（抬到搬运高度、全程守着呢）。
        if self.goal == "grasp":
            while True:
                if self.max_attempts > 0 and self.attempt >= self.max_attempts:
                    self._stop(f"抓了 {self.max_attempts} 次都没抓起来，收工")
                    break
                self.attempt += 1
                self.attempt_started = time.time()
                if self._out_of_reach():
                    break
                got = self._call("pick", task)
                if got.get("ok"):
                    self._stop(f"第 {self.attempt} 次抓起来了（这一支只判抓起来没掉）")
                    break
                if got.get("give_up"):
                    self._stop(f"找不着目标（{got.get('why')}）")
                    break
                print(f"    ↻ 没抓起来（第 {self.attempt}/{self.max_attempts or '∞'} 次），"
                      f"回去再来")
            self._call("home", task)
            return []

        while True:
            # **先看试满了没有，再数这一次。** 反过来写的话（先 +1 再比）：
            # 第 3 次跑完回头一看 4 > 3 才收工，日志和记录里就成了「试了 4 次」。
            if self.max_attempts > 0 and self.attempt >= self.max_attempts:
                self._stop(f"试满 {self.max_attempts} 次（抓+放）还没成，收工")
                break
            self.attempt += 1
            self.attempt_started = time.time()
            if self._out_of_reach():
                break

            got = self._call("pick", task)
            if not got.get("ok"):
                # **「找不着」是收工，不是再来一次。** 抓空、掉了这些是这一次没成，
                # 回去再抓就是；可「画面里认不出方块在哪/朝哪」同一个现场再来一遍还是
                # 认不出 —— 抓取那边把换机位扫一圈、换相机都试过了才报这个（`give_up`）。
                if got.get("give_up"):
                    self._stop(f"找不着目标方块（{got.get('why')}）")
                    break
                continue                      # 这一次没成，回去再来
            if not self._call("place", task).get("ok"):
                continue                      # 掉了没捡回来 / 没降到贴地，也算这一次没成
            if self._landed():
                self._stop(f"第 {self.attempt} 次尝试放下去了，成")
                break
            print(f"    ↻ 放下去了但没落到点上（第 {self.attempt}/"
                  f"{self.max_attempts or '∞'} 次），回去再来")

        self._call("home", task)
        return []

    def _call(self, name: str, task) -> dict:
        """调一个小模块，把结果记进日志。日志由主循环统一打印。"""
        if name == "place" and self.state.held is None:
            out = {"ok": False, "why": "爪里没东西，跳过"}
        else:
            kw = {}
            if name == "pick":
                # **不许小模块自己偷偷再抓一次。** 抓了几次归这一层数 —— 里面再自救
                # 一把，「抓满 3 次」就数不准了（那个自救是给「顺序跑一遍」那版用的）。
                kw["retry"] = False
            out = make("skill", name).run(self.arm, self.cams, state=self.state,
                                          index=self.index, percep=self.percep,
                                          instruction=task, **kw)
        self.log.append((name, out))
        return out

    def _stop(self, why: str) -> None:
        self.stop_why = why
        print(f"    ⏹ 收工：{why}")

    def _out_of_reach(self) -> bool:
        """目标方块出了可达环就当场收工 —— 够不着、也回不来，再试是白试。"""
        i = self.index
        if i is None:
            return False
        xy = self.arm.cube_pos(int(i))[:2]
        if not geom.out_of_reach(xy):
            return False
        self._stop(f"{self.arm.cube_order[int(i)]} 方块出了可达环（离肩部轴 "
                   f"{geom.reach_radius(xy) * 1000:.0f}mm，可及 "
                   f"{geom.REACH_MIN_R * 1000:.0f}~{geom.REACH_MAX_R * 1000:.0f}mm），"
                   f"够不着了")
        return True

    def _landed(self) -> bool:
        """这一次尝试放下了没有。**问判官** —— 「偏多少算没落到点上」只有一个说法。"""
        if self.judge is None:
            return True      # 没有判官就没法在轮中判；轮末那一次照旧由主循环问
        i = int(self.index)
        r = Round(index=0, task="", target=self.arm.cube_order[i], seed=0,
                  evidence=evidence(self.arm, self.state, i, geom.PLACE_TARGET))
        v = self.judge.judge(r)
        if not v.success:
            print(f"    ✗ 判官：{v.reason}"
                  + (f"   落点偏离 {v.detail['offset_m'] * 1000:.1f}mm"
                     if "offset_m" in (v.detail or {}) else ""))
        return bool(v.success)
