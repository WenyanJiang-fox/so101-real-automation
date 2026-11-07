"""跨模块共用的两种数据形状：一轮的完整记录 Round，和一轮的结论 Verdict。

主循环、策略、判官、存盘全都用这两个，谁也不另造一套。Round.to_dict() 出来的
就是落盘时那一行记录；画面另外存视频，这里只留低维数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame:
    """一帧：这一步看到的东西 + 这一步发出的动作。

    `extra` 是主循环额外记的现场数据（方块真实位置、指垫接触数）。**只有主循环往
    这里写，技能一个都不读** —— 它是留证用的，不是给谁做判断用的。
    """

    cam: dict[str, np.ndarray]      # 相机名 -> 画面
    arm: np.ndarray                 # 六个关节角度
    action: np.ndarray              # 七个数：六个关节 + 夹爪
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    """一轮的结论。谁判的写在 source 里，原始证据写在 detail 里。"""

    success: bool
    source: str                              # "truth" / "threshold" / ...
    color: str | None = None                 # 真值颜色，或判出来的颜色
    reason: str = ""                         # 失败原因，人看得懂的一句话
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Round:
    """一轮的完整记录：从摆场到判完，所有要留的东西都在这。"""

    index: int
    task: str                       # 喂给模型的那句指令
    target: str                     # 这一轮要抓哪个颜色
    seed: int
    layout: dict[str, Any] = field(default_factory=dict)   # 这一轮的摆放
    frames: list[Frame] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)  # 给判官的现成证据
    verdict: Verdict | None = None
    meta: dict[str, Any] = field(default_factory=dict)      # 谁跑的、多久、几帧

    def add_frame(self, cam, arm, action, extra=None) -> None:
        self.frames.append(Frame(cam=dict(cam),
                                 arm=np.asarray(arm, dtype=float),
                                 action=np.asarray(action, dtype=float),
                                 extra=dict(extra or {})))

    def to_dict(self) -> dict:
        """落盘用。画面单独存视频，这里只记低维数据。"""
        return {
            "index": self.index,
            "task": self.task,
            "target": self.target,
            "seed": self.seed,
            "layout": self.layout,
            "n_frames": len(self.frames),
            "arm": [f.arm.tolist() for f in self.frames],
            "action": [f.action.tolist() for f in self.frames],
            "extra": [f.extra for f in self.frames],
            "evidence": self.evidence,
            "verdict": (None if self.verdict is None else {
                "success": self.verdict.success,
                "source": self.verdict.source,
                "color": self.verdict.color,
                "reason": self.verdict.reason,
                "detail": self.verdict.detail,
            }),
            "meta": self.meta,
        }
