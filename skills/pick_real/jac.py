"""画面雅可比：探一次「关节度 → 方块中心像素」，之后按图像误差直接解关节增量。

**为什么不用「毫米/度」那套表。** 2026-10-18 那一趟微调：每一维都挂在一张经验表上
（1 号横移、4 号造缝、2.89mm/度、2° 最小步、方向余弦门槛……），结果 1 号的圆弧切向
和「离轴」方向差 36°，硬走 10mm 去纠 2.7mm 的残差，多出来的切向分量把臂沿合爪轴推
离方块（离轴 6.1→3.9mm、沿轴 12.9→22.5mm）。**方向不是猜出来的，是量出来的。**

这里量的是相机自己看到的：每个关节动已知度数，方块中心在画面里挪多少像素。之后
`dq = J⁺·dp`（阻尼最小二乘），一步把五个关节的增量一起解出来，不选关节、不查表。
"""
from __future__ import annotations

import numpy as np

from .knobs import (JAC_DAMP, JAC_PROBE_DEG, JAC_PROBE_JOINTS,
                    JAC_STEP_DEG)


def box_uv_now(cam, percep):
    """只量方块顶面中心（像素）。探针每一步都要量，比 `_jaw_check` 少一半活。"""
    import cv2                                                  # noqa: PLC0415

    from real.perception import brown_mask, candidates          # noqa: PLC0415

    from .rulers import _box_center_px                          # noqa: PLC0415

    if cam is None or percep is None:
        return None
    rgb = np.asarray(cam.shot())
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cands = [d for d in candidates(brown_mask(bgr, percep.ranges),
                                   percep.min_area)
             if d["rect_fill"] >= percep.fill_min]
    if not cands:
        return None
    return _box_center_px(cands[0]["contour"], bgr)


def probe(arm, cam, percep):
    """量 2×5 的 `∂(方块中心像素)/∂(关节度)`。量不出来返回 `None`。

    **正负交替着动**（+3°、−3°、+3°…）：一路正着走，五个关节的位移会攒到几十毫米，
    在抓取高度上就是拿爪尖扫桌面/方块。交替着动，臂只在出发点附近摆一下。

    每一步的分母用**读回来的实际角度**（舵机带载只走三到五成）；没动的那一维列留 0
    —— 那个关节在这个姿态上出不上力，解的时候自动不派它。量完回原位。
    """
    from real import joint_move as jm                           # noqa: PLC0415

    q0 = np.array(arm._ticks_to_qpos(arm.read_ticks()), dtype=np.float64)
    uv_prev = box_uv_now(cam, percep)
    if uv_prev is None:
        print("    ✗ 探雅可比：画面里量不到方块中心 —— 这一趟不走雅可比")
        return None
    J = np.zeros((2, 5))
    q_prev = q0.copy()
    for k, j in enumerate(JAC_PROBE_JOINTS):
        sign = 1.0 if k % 2 == 0 else -1.0
        q_cmd = q_prev.copy()
        q_cmd[j] = q_prev[j] + sign * np.radians(JAC_PROBE_DEG)
        q_act = np.asarray(jm.goto_q(arm, q_cmd, q_from=q_prev,
                                     settle=0.10, hold=3), dtype=np.float64)
        uv = box_uv_now(cam, percep)
        dq = float(np.degrees(q_act[j] - q_prev[j]))
        if uv is not None and abs(dq) >= 0.5:
            J[:, j] = (uv - uv_prev) / dq
        if uv is not None:
            uv_prev = uv
        q_prev = q_act
        if getattr(arm, "terminated", False):
            return None
    jm.goto_q(arm, q0, q_from=q_prev)
    return J


def dq_for(J, dp_px) -> np.ndarray | None:
    """阻尼最小二乘：`dq = J⁺·dp`，再把最大的一维夹到 `JAC_STEP_DEG`。

    **返回的是六个关节的增量**（和 `arm._ticks_to_qpos()` 同一个长度）。`J` 只有被探
    的那五列（`JAC_PROBE_JOINTS = 0~4`），没探的那一维（夹爪 5 号）补 0。

    2026-10-20 修：原来只返回那五个数，调用方 `align.py:723` 拿它去加六个数
    （`_q0 + _dq`），当场 `ValueError: operands could not be broadcast (6,) (5,)` ——
    整趟技能从这里抛出去。**不是补个 `[:5]`，是把缺失的那一维按 0 补齐**：夹爪那一步
    本来就不该动，后面 `_qc[5]` 另有赋值。
    """
    dp = np.asarray(dp_px, dtype=np.float64).reshape(-1)[:2]
    if not np.all(np.isfinite(dp)) or float(np.linalg.norm(dp)) < 1e-9:
        return None
    A = J @ J.T + (JAC_DAMP ** 2) * np.eye(2)
    try:
        dq5 = J.T @ np.linalg.solve(A, dp)
    except np.linalg.LinAlgError:
        return None
    m = float(np.max(np.abs(dq5))) if dq5.size else 0.0
    if m > JAC_STEP_DEG > 0.0:
        dq5 = dq5 * (JAC_STEP_DEG / m)
    dq = np.zeros(6, dtype=np.float64)
    dq[list(JAC_PROBE_JOINTS)] = dq5
    return dq
