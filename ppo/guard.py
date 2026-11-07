"""定位一致性：方块会被爪子碰一下挪一点，不可能一次挪出一大截。

用户 2026-10-20 的口径：「虽然方块可能被你的爪子移动碰了一下产生位移，但物体及其
3D 中心不可能产生大的位移，别定位差了好远。」

所以每一次重新认出来的中心都要跟**上一次的锚点**比一下：跳得不远就用新的（方块确实
被碰动了），跳得远就是这一眼认错了，退回锚点。
"""
from __future__ import annotations

import numpy as np

#: 两次估计之间允许的最大跳距（毫米）。
#: **按实测取**：2026-10-20 那一轮真机 8 趟，下探之后方块中心最多挪 33.3mm；
#: 60mm = 实测最大值的 1.8 倍。留这一档余量是因为下一次被碰的方向可能反着来，
#: 而认错东西的跳距通常远大于这个数。
MAX_JUMP_MM = 60.0


def clamp_center(new, anchor, max_mm: float = MAX_JUMP_MM,
                 tag: str = "") -> tuple[np.ndarray | None, float | None, bool]:
    """返回 `(可用的中心, 跳距mm, 是不是跳远了)`。

    · `anchor` 为空 → 原样返回，`跳距 = None`、`跳远了 = False`；
    · 跳距 ≤ `max_mm` → 用新的；
    · 跳距 > `max_mm` → **x/y 退回锚点**、z 用新的，`跳远了 = True`，打一行日志。
    """
    if new is None or anchor is None:
        return new, None, False
    n = np.asarray(new, dtype=np.float64).reshape(-1).copy()
    a = np.asarray(anchor, dtype=np.float64).reshape(-1)
    d = float(np.linalg.norm(n[:2] - a[:2])) * 1000.0
    if d <= max_mm:
        return n, d, False
    n[0], n[1] = a[0], a[1]
    print(f"    ！{tag}定位跳了 {d:.0f}mm（上限 {max_mm:.0f}mm）—— 方块不会被碰这么远，"
          f"这一眼按认错处理，退回上一次的中心 "
          f"({a[0] * 1000:.0f}, {a[1] * 1000:.0f})mm", flush=True)
    return n, d, True
