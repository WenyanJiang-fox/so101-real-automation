# -*- coding: utf-8 -*-
"""推方块：合拢的爪子当挡板，把方块往指定方向推一段。

**为什么需要这一步**（2026-10-24 用户定）：合爪几次都不成的时候，方块往往已经
不在「抓得住」的位置上了 —— 贴到桌沿、偏到可达环外面、或者卡在一个腕子转不过去
的角上。继续在原地试同一套微调没有意义，先把方块推回来再抓。

**只做一次直线推，不做闭环规划**。推完把方块中心重新认一眼交回去，推几次、往哪推、
推多远由调用方定。理由和 `_set_clearance` 的一样：真机上臂走一步和指令差多少，只有
走完量了才知道，所以这里不预测，只报实际挪了多少。

**推的姿态**：爪子合到底当实心挡板，下到**方块中心高度**再横着走 —— 高于中心会从
方块顶上滑过去，低于中心会先蹭到桌面。起点在方块中心往回退「半个边长 + 净空」，
一路推到「中心 + 一步」，所以方块实际被推的距离接近 `step_m`。

**安全**：三步里任何一步没走到就抬回悬停点、返回失败，不硬走；`arm.terminated`
（看门狗）一置位也立刻退出。
"""
from __future__ import annotations

import numpy as np

from core.params import SLOW, speed_vmax

from .motion import _goto, _goto_reach
from .rulers import _box_half_m

#: 从方块**边缘**再往回退多少开始推（米）。退得比爪厚就够 —— 2026-10-24 实测：
#: 退 50mm 时起推点在两个方向上都撞关节限制（1 号 30° 钳制 / 5 号单步 45°），
#: 收到 20mm 让起推点离方块近一些。
PUSH_BACK_M = 0.020
#: 一次推多少（米）。**小步**：推过头比推不到更难收拾。
PUSH_STEP_M = 0.015
#: 起推 / 收工的悬停高度（米，方块中心往上）。
PUSH_HOVER_M = 0.080
#: 方块中心离目标这么近就不用推了（米）。
PUSH_NEAR_M = 0.020


def front_target() -> np.ndarray:
    """机械臂**正前方**、可达环中段那个点（基座系，米，两项）。

    2026-10-24 用户定：「尽量往机械臂正面去推」。正前方 = 从肩部轴
    （`geom.MARKER_CENTER`，就是底座在基座系的位置）沿 +x 出去一个环中段半径
    （`geom.MARKER_R`，130~260mm 的中值）。推到这里，方块在臂的正对面，
    腕子不用横着扭、1 号也不用转到钳制边上。
    """
    from core import geom                                          # noqa: PLC0415

    return np.array([float(geom.MARKER_CENTER[0]) + float(geom.MARKER_R),
                     float(geom.MARKER_CENTER[1])], dtype=np.float64)


def push_toward(arm, cams, percep, center, target_xy, *,
                text: str = "", step_m: float = PUSH_STEP_M) -> dict:
    """把方块从 `center` 往 `target_xy` 方向推一步。

    `center` 是方块中心（米，基座系，z 是中心高度）；`target_xy` 是希望它去的地方
    （米，两项）。`text` 是感知用的指令（和技能里同一个颜色）。

    返回 `{"ok", "why", "moved_mm", "center_after", "start_mm", "dir"}`：
    `moved_mm` 是推完重新认出来的实际位移；认不出来是 `None`（这一趟别当有效样本）。
    """
    c = np.asarray(center, dtype=np.float64).reshape(-1)
    tgt = np.asarray(target_xy, dtype=np.float64).reshape(-1)[:2]
    d = tgt - c[:2]
    dist = float(np.linalg.norm(d))
    if dist <= PUSH_NEAR_M:
        return {"ok": True, "why": "已经在目标附近，不用推", "moved_mm": 0.0,
                "center_after": None, "start_mm": None, "dir": None}
    u = d / dist
    back = float(_box_half_m(percep)) + PUSH_BACK_M
    start = np.array([c[0] - u[0] * back, c[1] - u[1] * back], dtype=np.float64)
    z = float(c[2])
    hover_z = z + PUSH_HOVER_M
    print(f"    ➜ 推：从 {np.round(start * 1000, 1)}mm 往 "
          f"({u[0]:+.2f}, {u[1]:+.2f}) 推到方块中心，爪合着、z={z * 1000:.0f}mm",
          flush=True)

    def _abort(why: str, gone: str) -> dict:
        _goto(arm, np.array([start[0], start[1], hover_z], dtype=np.float64),
              vmax=speed_vmax(SLOW), grip=-1.0)
        print(f"    ✗ 推没走成：{why}（{gone}）", flush=True)
        return {"ok": False, "why": why, "moved_mm": None, "center_after": None,
                "start_mm": [round(float(v) * 1000, 1) for v in start], "dir": u.tolist()}

    # 1) **先到方块正上方**，再横移到起点上方（2026-10-24 实测：一步从当前姿态
    # 直接走到起点，关节一次要转 40 多度，被安全闸 / 关节限幅拦下；拆成两小步）。
    # **走不动就降高再走**（`_goto_reach`）：这一档悬停点在这台机器上 4 号已经顶到
    # 88.6°（上限 95°），平着走根本到不了 66mm；技能自己进场也是降高走的。
    floor_z = z + 0.02
    r0, _p0 = _goto_reach(arm, np.array([c[0], c[1], hover_z], dtype=np.float64),
                          vmax=speed_vmax(SLOW), grip=-1.0, floor_z=floor_z)
    if getattr(arm, "terminated", False):
        return {"ok": False, "why": "看门狗叫停", "moved_mm": None,
                "center_after": None, "start_mm": None, "dir": u.tolist()}
    if r0 != "reached":
        return _abort("方块正上方没走到", r0)
    r1, _p1 = _goto_reach(arm, np.array([start[0], start[1], hover_z], dtype=np.float64),
                          vmax=speed_vmax(SLOW), grip=-1.0, floor_z=floor_z)
    if getattr(arm, "terminated", False):
        return {"ok": False, "why": "看门狗叫停", "moved_mm": None,
                "center_after": None, "start_mm": None, "dir": u.tolist()}
    if r1 != "reached":
        return _abort("起点上方没走到", r1)

    # 2) 下到方块中心高度（横着推之前的落点）
    r2 = _goto(arm, np.array([start[0], start[1], z], dtype=np.float64),
               vmax=speed_vmax(SLOW), grip=-1.0, tol=0.006)
    if getattr(arm, "terminated", False):
        return {"ok": False, "why": "看门狗叫停", "moved_mm": None,
                "center_after": None, "start_mm": None, "dir": u.tolist()}
    if r2 != "reached":
        return _abort("下到方块中心高度没走到", r2)

    # 3) 朝目标横推：走过的距离 = 退出来的净空 + 一步
    end = np.array([c[0] + u[0] * step_m, c[1] + u[1] * step_m, z], dtype=np.float64)
    r3 = _goto(arm, end, vmax=speed_vmax(SLOW), grip=-1.0, tol=0.006)
    if getattr(arm, "terminated", False):
        return {"ok": False, "why": "看门狗叫停", "moved_mm": None,
                "center_after": None, "start_mm": None, "dir": u.tolist()}
    # 4) 先抬回去再认方块 —— 爪子挡在画面正中间的时候，感知认不准。
    _goto(arm, np.array([end[0], end[1], hover_z], dtype=np.float64),
          vmax=speed_vmax(SLOW), grip=-1.0)

    c2 = None
    try:
        c2, _facts = percep.locate(text, guess=c)
    except Exception as exc:                                        # noqa: BLE001
        print(f"    推完认方块出错：{type(exc).__name__}: {exc}", flush=True)
    if c2 is None:
        print("    ✗ 推完没认到方块 —— 这一趟的位移量记不了", flush=True)
        return {"ok": False, "why": "推完没认到方块", "moved_mm": None,
                "center_after": None, "start_mm": [round(float(v) * 1000, 1) for v in start],
                "dir": u.tolist()}

    delta = np.asarray(c2, dtype=np.float64).reshape(-1)[:2] - c[:2]
    along = float(delta @ u) * 1000.0
    moved = float(np.linalg.norm(delta)) * 1000.0
    ok = (r3 == "reached") and (along > -2.0)
    why = ("" if ok else
           ("推的那一步没走到（" + r3 + "）" if r3 != "reached"
            else f"方块反而往回走了 {along:.1f}mm"))
    print(f"    ➜ 推完：方块 {np.round(delta * 1000, 1)}mm"
          f"（沿推的方向 {along:+.1f}mm，实际位移 {moved:.1f}mm）"
          f"{'' if ok else ' ← ' + why}", flush=True)
    return {"ok": bool(ok), "why": why, "moved_mm": round(moved, 1),
            "center_after": [round(float(v) * 1000, 1) for v in c2],
            "start_mm": [round(float(v) * 1000, 1) for v in start],
            "dir": u.tolist()}
