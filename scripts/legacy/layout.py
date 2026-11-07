"""布局：每轮用什么颜色、把方块放哪、朝哪个方向。

颜色轮换的规则是「取已采条数最少的那个」，并列时随机 —— 这样数据不会偏科，
不需要事后配平。位置和偏航只在基础点周围抖动，抖动幅度由调用方给。

这个模块不 import 仿真，也不 import 真机。基础点位（PICK_X/PICK_Y）由调用方传，
仿真和真机各有各的。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 方块颜色 → MuJoCo geom_rgba。红/绿/蓝，饱和度拉到和真实方块一个量级，
# 免得 HSV 判据在仿真里因为颜色太淡而失效（那是仿真特有的假失败）。
CUBE_RGBA = {
    "red": (0.90, 0.15, 0.15, 1.0),
    "green": (0.15, 0.75, 0.20, 1.0),
    "blue": (0.15, 0.30, 0.90, 1.0),
}
COLORS = ("red", "green", "blue")


@dataclass(frozen=True)
class Layout:
    """一轮的全部场景参数。"""

    color: str
    xy: tuple[float, float]
    yaw: float

    def __str__(self) -> str:
        return (f"{self.color:5s} xy=({self.xy[0] * 1000:6.1f},"
                f"{self.xy[1] * 1000:6.1f})mm yaw={np.degrees(self.yaw):5.1f}°")


def next_layout(counts: dict, rng: np.random.Generator, base_xy: tuple[float, float],
                jitter_xy: float = 0.030, jitter_yaw: float = np.pi / 6) -> Layout:
    """挑下一个布局。counts = {颜色: 已采条数}，缺的颜色算 0。"""
    # 取采得最少的；并列的里面随机抽一个，避免固定顺序导致同一个色总是先采
    n = {c: int(counts.get(c, 0)) for c in COLORS}
    fewest = min(n.values())
    pool = [c for c in COLORS if n[c] == fewest]
    color = pool[int(rng.integers(len(pool)))]

    xy = (base_xy[0] + float(rng.uniform(-jitter_xy, jitter_xy)),
          base_xy[1] + float(rng.uniform(-jitter_xy, jitter_xy)))
    yaw = float(rng.uniform(-jitter_yaw, jitter_yaw))
    return Layout(color=color, xy=xy, yaw=yaw)
