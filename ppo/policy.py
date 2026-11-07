"""抓取重试策略：不限次数，每次只换一个变量，按实测精进。

**这一层不碰硬件、不碰技能。** 它只回答三件事：

    choose()       下一趟用哪一组参数（相邻两档只差一个变量）
    observe()      上一趟量到了什么，定位锚点更新到哪
    should_stop()  该不该停

「精进」不是靠猜：每一趟技能都会交回实测数（6 号舵机停在几度、方块被推走多少、
下探之后方块中心在哪），锚点和「哪一档开口最大」都从这些数里算。

以后接 PPO：`choose()` 换成策略网络的输出、`observe()` 把同一份测量喂进 replay
buffer，`panel/rig.py` 里的调用点不用动。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """一趟抓取用的一组参数。"""

    name: str
    label: str
    strategy: str          # `configs/strategies.json` 里的名字
    measure: bool          # False = `PICK_CLEARANCE_MEASURE=0`，不量直接下扎
    #: `change` 只在「第一次用这一档」时拿来说明它从上一档改了什么，空着就现算
    change: str = ""


#: 有意义的组合就这三种，按「改动最小 → 最大」排：
#:   baseline → no_measure   只关掉「先量一眼」
#:   no_measure → full       换成完整微调（这一步同时把微调打开了，因为 full 本来就走）
ACTIONS: tuple[Action, ...] = (
    Action("baseline", "先量一眼", "oneshot", True),
    Action("no_measure", "不量直接下扎", "oneshot", False),
    Action("full", "完整微调", "full", True),
)


#: 进场 / 纠位的默认悬停高度（米）。
HOVER_DEFAULT_M = 0.08


def apply_action(action: Action, *, bias_mm: float = 0.0,
                 hover_m: float | None = None) -> None:
    """把一趟要用的参数写进技能读的那几个名字。**跑技能之前调。**

    · `action`   → 策略表那一行 + 合爪前微调的开关（`PICK_CLEARANCE_MEASURE`）；
    · `bias_mm`  → `decide.set_bias`，加在下探终点的高度上（正 = 扎得浅）；
    · `hover_m`  → `knobs.HOVER_ABOVE_M`。`None` = 回到默认 0.08。
    """
    import os

    from skills.pick_real import decide as DECIDE
    from skills.pick_real import knobs as KNOBS
    from skills.pick_real import strategy as STRAT

    STRAT.apply(action.strategy)
    os.environ["PICK_CLEARANCE_MEASURE"] = "1" if action.measure else "0"
    DECIDE.set_bias(float(bias_mm))
    KNOBS.HOVER_ABOVE_M = HOVER_DEFAULT_M if hover_m is None else float(hover_m)


def _diff(a: Action | None, b: Action) -> str:
    """这一趟相对上一趟换了哪个变量。"""
    if a is None:
        return "起始档"
    parts: list[str] = []
    if a.strategy != b.strategy:
        parts.append(f"策略「{a.label}」→「{b.label}」")
    elif a.measure != b.measure:
        parts.append("合爪前微调 " + ("开→关" if a.measure else "关→开"))
    if not parts:
        return "不换变量，只按实测精进（锚点 / 下探终点）"
    return "；".join(parts)


class RetryPolicy:
    """一个目标（同一只方块）的重试账本。**换了目标就 `begin_target()` 清空。**"""

    def __init__(self, max_attempts: int | None = None):
        #: `None` = 不限次数（用户 2026-10-20：「无限次尝试，每次精进」）
        self.max_attempts = max_attempts
        self.n = 0
        #: 上一次认到的方块中心 (x, y)，米。下一步用它卡定位；`None` = 还没锚点
        self.anchor: tuple[float, float] | None = None
        self.tried: set[str] = set()
        self.history: list[dict] = []
        self._cur: Action | None = None
        self._last: Action | None = None

    # ---- 换目标 ----

    def begin_target(self) -> None:
        """上一趟放好了 / 换了一只方块：次数、试过哪些、锚点全清。"""
        self.n = 0
        self.anchor = None
        self.tried.clear()
        self.history.clear()
        self._cur = None
        self._last = None

    # ---- 下一趟 ----

    def choose(self) -> tuple[Action, str]:
        """没试过的按顺序来；都试过了，回到实测开口最大的那一档。

        返回 `(这一趟的动作, 相对上一趟换了什么)`。开口大 = 爪子合得浅 = 离夹住近，
        这是技能每趟都会交回来的实测数，不是估的。
        """
        act = next((a for a in ACTIONS if a.name not in self.tried), None)
        if act is None:
            act = self._best()
        change = _diff(self._last, act)
        self._cur = act
        return act, change

    def _best(self) -> Action:
        best: tuple[float, Action] | None = None
        for a in ACTIONS:
            gs = [float(r["grip_deg"]) for r in self.history
                  if r["action"] == a.name and r.get("grip_deg") is not None]
            if not gs:
                continue
            g = max(gs)
            if best is None or g > best[0]:
                best = (g, a)
        return best[1] if best is not None else ACTIONS[0]

    # ---- 收上一趟的实测 ----

    def observe(self, out: dict, act: Action | None = None) -> str:
        """记下这一趟的结果，更新锚点。返回一句给人看的总结。

        `act` 是**这一趟实际用的那一档**。`choose()` 会自己记进 `_cur`，但凡是外部
        （比如 VLM）选档的调用方，必须把这个参数传进来 —— 不传的话账本会把每一趟
        都记成第一档，汇总就是假的（2026-10-20 第一次接 VLM 时踩到）。
        """
        act = act or self._cur or ACTIONS[0]
        self.n += 1
        self.tried.add(act.name)

        # **锚点取「下探之后」那一眼**：方块被推开之后的位置才是下一次要找的地方；
        # 没量到就退回下探之前那一显。两个都没有就保留旧锚点。
        ca = out.get("box_center_after")
        cb = out.get("center_before")
        p = ca if ca is not None else cb
        g = out.get("grip_deg")
        rec = {
            "n": self.n,
            "action": act.name,
            "ok": bool(out.get("ok")),
            "verdict": out.get("verdict"),
            "grip_deg": None if g is None else float(g),
            "box_shift_mm": out.get("box_shift_mm"),
            "center": None if p is None else [float(p[0]), float(p[1])],
        }
        self.history.append(rec)
        if p is not None:
            self.anchor = (float(p[0]), float(p[1]))
        self._last = act
        self._cur = None

        said = f"第 {self.n} 次「{act.label}」：{'成' if rec['ok'] else '没成'}"
        if rec["grip_deg"] is not None:
            said += f"，6 号 {rec['grip_deg']:.1f}°"
        if rec["box_shift_mm"] is not None:
            said += f"，方块被推 {float(rec['box_shift_mm']):.1f}mm"
        if self.anchor is not None:
            said += (f"；锚点 → ({self.anchor[0] * 1000:.0f}, "
                     f"{self.anchor[1] * 1000:.0f})mm")
        return said

    # ---- 停不停 ----

    def should_stop(self, out: dict) -> tuple[bool, str]:
        """**只停这三种**：人不让跑了、目标根本看不见、自己设了次数上限。"""
        if out.get("give_up"):
            return True, "技能收工（停手 / 装配 / 找不到目标）"
        why = str(out.get("why") or "")
        if "找不着" in why or "没认出来" in why:
            return True, "找不到目标（再试也不会自己冒出来）"
        if self.max_attempts is not None and self.n >= int(self.max_attempts):
            return True, f"跑满上限 {self.max_attempts} 次"
        return False, ""

    # ---- 收工报告 ----

    def summary(self) -> str:
        if not self.history:
            return "这一轮没跑过"
        ok = sum(1 for r in self.history if r["ok"])
        parts: list[str] = []
        for a in ACTIONS:
            rs = [r for r in self.history if r["action"] == a.name]
            if not rs:
                continue
            gs = [float(r["grip_deg"]) for r in rs if r.get("grip_deg") is not None]
            one = f"{a.label} {len(rs)} 次（成 {sum(1 for r in rs if r['ok'])}"
            if gs:
                one += f"，开口最大 {max(gs):.1f}°"
            parts.append(one + "）")
        return f"共 {len(self.history)} 次，成 {ok} 次；" + "；".join(parts)
