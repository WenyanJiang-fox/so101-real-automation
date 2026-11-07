"""本轮现场：这一轮跑到哪了、爪里有没有东西、撞过谁。

技能之间要传话 —— 上一段夹住的方块，下一段要接着夹；合爪时记住的开口，抬升时
要守着。但技能在接口上各自独立，所以这些共有状态放在一个对象里，由策略建一个、
传给每个技能。

**为什么不挂全局变量**：一轮一个，轮与轮之间互不影响，将来并行跑几轮也不会串。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def evidence(arm, state: "RoundState", index, goal=None) -> dict:
    """给判官的现成证据：三个方块现在各在哪、爪里还有没有东西、要去的那一点。

    主循环（轮末判分）和策略（轮中判「这一次尝试到底落没落到点上」）都用这一份 ——
    两处各写一份的话，判官看到的证据会慢慢长成两个不一样的东西。
    """
    return {"target_index": index,
            "cube_pos": [[float(v) for v in arm.cube_pos(i)] for i in range(arm.n_cubes)],
            "goal_xy": None if goal is None else [float(v) for v in np.asarray(goal).ravel()[:2]],
            "held": state.held}


@dataclass
class RoundState:
    """一轮里的共有状态。"""

    held: int | None = None       # 现在爪里夹着哪个方块（下标），空手是 None
    grip_ref: float | None = None  # 合爪停住那一刻的夹爪关节角，抬升时守这个开口
    grasp_z: float = 0.02         # 合爪时 TCP 的离地高度；放的时候降到同一高度再松
    seg: str = ""                 # 现在在走哪一段，只给日志看
    ignore: set = field(default_factory=set)   # 这一段的「自己人」，碰了不算撞
    bumps: dict = field(default_factory=dict)  # 撞过的 (段, 方块, geom)，同一件只记一次

    def clear_hand(self) -> None:
        self.held = None
        self.grip_ref = None

    def check_bump(self, arm) -> bool:
        """走上一步之后调一次。撞到不该碰的方块就记一条（同一件只报一次）。"""
        order = arm.cube_order
        for ci, name, p in arm.bump_scan(self.ignore):
            key = (self.seg, ci, name)
            if key in self.bumps:
                continue
            self.bumps[key] = 1
            print(f"    ⚠ 撞到 {order[ci]} 方块：{self.seg}  撞它的 geom={name}"
                  f"  接触点 {p.round(3)}  TCP {arm.tcp().round(3)}"
                  f"  方块在 {arm.cube_pos(ci).round(3)}")
        return bool(self.bumps)
