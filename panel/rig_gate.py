"""断点：技能停在闸口上等点头的时候，面板这条线怎么接。

三件事：判断「现在是不是真停在断点上」（`_gate_state`）、把人的答复写出去
（`_gate_cmd`）、等待期间把面板按下的键做掉（`pump_wait`）。
"""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from panel.rig_common import GATE_FRESH_S, TICK_S             # noqa: E402


class GateMixin:
    """闸口那一套。"""

    def _gate_cmd(self, what: str, dz) -> None:
        """在断点上给技能一个答复：把 `out/clear_go.json` 写出去。

        **技能没停在断点上的时候不写。** 那个文件是「一次性批示」，写了没人读就一直躺在
        盘上 —— 技能走到下一个断点时先 `unlink` 它才等，所以不会串味；但人看到的是
        「按了允许却什么都没发生」，不如当场说清。
        """
        from panel import skill_driver                              # noqa: PLC0415

        if self._gate_state() is None:
            self._log("！技能没停在断点上 —— 这条批示没写（先按「跑一趟」）")
            return
        if what == "abort":
            # 「放弃」是人的明确意思：这一趟不干了。自动模式跟着关 —— 不然下一趟立刻
            # 又开起来，等于没放弃。
            self._auto = False
        self._log(skill_driver.open_gate(what, dz_mm=None if dz is None else float(dz)))

    def gate_idle(self) -> None:
        """技能**不再等点头**了（`skills/pick_real/gates.py::_wait_cmd` 返回时调一次）。

        把 `_gate_t` 清 0：`_gate_state()` 当场判 None，界面立刻把「允许」按钮关掉。
        原来只靠 `WAIT_TICK` 停止更新后等满 `GATE_FRESH_S`（0.6 秒），那段时间里按下的
        批示会被技能下一个断点 `unlink` 掉 —— 人觉得按了没反应，只好再按一次。
        """
        self._gate_t = 0.0

    def pump_wait(self) -> None:
        """技能停在断点上等点头的时候，**把这期间面板按下的键做掉**。

        挂在 `skills/pick_real` 的 `WAIT_TICK` 上，技能每等一轮（约 0.1 秒）调一次。
        它跑的就是硬件线程 —— 串口没有第二个人抢，跟平时按按钮走的是同一条路。

        **不许在这儿再起一趟技能。** 技能正停着，再进去一层就是两趟叠在一起：串口、
        姿态、闸口文件全乱。那种命令直接丢掉并说明。

        **「停手」在这儿的意思是「这一趟不干了」**，不是「把这一小段运动掐停」——
        技能正停着没在动，掐不到什么；真按下去照原样立 `terminated`，等点头等到一半的
        技能接着走第一步就会被它顶停，人看到的是「按了允许但不动」。所以在这儿翻成
        `{"abort": true}`，让技能自己从等点头里干净地退出来。
        """
        # **急停插在最前面。** 技能停在断点上等点头时，硬件线程就在这个函数里转；
        # 按下的那一下只能在这一点做掉（队列里的其它命令都排在它后面，而技能这会儿
        # 本来就该让位）。做完之后 `_estop_run` 会把 `terminated` 立回去，技能接着退。
        if getattr(self, "_estop", None) is not None and self._estop.is_set():
            self._estop_run()

        self._gate_t = time.time()
        from panel import skill_driver                              # noqa: PLC0415

        while True:
            try:
                cmd = self._q.get_nowait()
            except queue.Empty:
                break
            kind = cmd.get("kind")
            if kind in ("pick_run", "stage"):
                self._log("！技能正停在断点上 —— 这条先不接"
                          "（先给个「允许」或者「放弃这一趟」）")
                continue
            if kind == "quit":
                self._quit.set()
                self.arm.terminated = True
                continue
            if kind == "stop":
                self._auto = False
                self._log("⏹ 停手 —— 技能正停在断点上，这一趟按「放弃」收工")
                self._log(skill_driver.open_gate("abort"))
                continue
            try:
                self._do(cmd)
            except Exception as exc:                                # noqa: BLE001
                self._log(f"✗ 命令 {kind} 出错：{type(exc).__name__}: {exc}")
            finally:
                # 跟 `_loop` 里那两行一个道理：走完一趟就把「在动」放下来，
                # 不然画面上会一直挂着 `MOVING`（`view.draw` 见它就把读数冻住）。
                self._moving = False
        # 别让画面冻在断点那一帧。**节拍跟 `_loop` 一样**：`_tick` 自己不限速，
        # 抓一帧约 30ms、批量读舵机约 66ms，每一轮都刷的话等待那条循环会被拖慢一倍。
        now = time.time()
        if now - self._t_frame >= TICK_S:
            try:
                self._tick(now)
            except Exception as exc:                                # noqa: BLE001
                self._log(f"✗ 刷新出错：{type(exc).__name__}: {exc}")

    def _gate_state(self) -> dict | None:
        """现在是不是**正停在断点上等着**？等着就给一份状态，没等就是 `None`。

        两件事凑一起才算数：技能那一头 `WAIT_TICK` 刚被调过（`_gate_t` 新鲜），盘上
        又有读数文件（`skill_driver.gate_now`）。只看文件不行 —— 上一趟留下的那几份
        一直在盘上，面板会一直挂着「允许」按钮。
        """
        if time.time() - self._gate_t > GATE_FRESH_S:
            return None
        from panel import skill_driver                              # noqa: PLC0415

        got = skill_driver.gate_now()
        if got is None:
            return {"kind": "?", "rec": {}, "note": "技能停在闸口上，读数文件没读到",
                    "allow": ("go", "abort")}
        kind, rec = got
        return {"kind": kind, "rec": rec, "note": skill_driver.gate_note(kind),
                "allow": skill_driver.ALLOWED.get(kind, ("abort",))}
