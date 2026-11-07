"""看一眼、扫一圈。"""
from __future__ import annotations

from core import follow, geom, shapes, vision
from core.params import (ALIGN_CHUNK, ALIGN_MAX_STEPS,            # noqa: E402
                         ALIGN_ROT_SLEW, ALIGN_TOL_DEG, CARRY,
                         GENTLE, GRIP_BIAS, GRIP_K, GRIP_RATE,
                         GRIP_SLACK, LIFT_Z, POS_SCALE,
                         SLOW, SPEED, STUCK_TOL,
                         acc_amax, speed_vmax)
import numpy as np
from .knobs import SWEEP_4, SWEEP_HEAD_FLEX, SWEEP_WAY
from .motion import _goto
from .pose import _view_radius
def _look(arm, cams, percep, instruction, guess):
    """看一眼：返回 (中心坐标, 朝向弧度, facts)；量不出来 center 是 None。

    直接走感知那条链（`real/perception.py` 的 `locate`）—— 它里面已经是「挑出棕色块
    → 估形状 → 投剪影」全套，在外面重做一遍等于两套口径。`guess` 传下去当搜索种子。
    """
    center, facts = percep.locate(instruction, guess=guess)
    print(f"      找到 {facts.get('found')}  矩形度 {facts.get('rect_fill')}"
          f"  重合度 {facts.get('iou')}")
    if center is None:
        why = facts.get("no_world_why") or facts.get("why") or "没认出来"
        print(f"      ✗ {why}")
        return None, None, facts
    return center, float(np.radians(facts["yaw_deg"])), facts

def _sweep(arm, cams, percep, instruction, state):
    """原地那一眼没认出来时，分两段找。

    **第 1 段 · 前后左右。** 就在现在站的地方铺四个点 —— 前、后、左、右，各挪一格，
    格子边长 = 腕相机当前能看清的那个圆的内切半径（`_view_radius` 现算）。这一段走
    `follow`（TCP + 逆解）：它就是平面平移，没有姿态变化。走完回中心再进第 2 段。

    **第 2 段 · 原地抬个头。** 就一个姿态：回到中心之后，1 号 pan 不动，4 号腕关节
    掰到 `SWEEP_HEAD_FLEX`。角度和姿态本身还是 `G:\\sim2real\\so101-real-harness\\real\\sweep_look.py`
    那张表（`poses(pan_deg=(0,), flex_deg=(SWEEP_HEAD_FLEX,))`），只是把范围收成一点。
    掰下去让光轴离开垂直方向，桌面落点跟着往外推几十厘米 —— 腕相机正下方只看得见约
    0.33m × 0.18m 一块，横移范围只有 ±0.18m，**掰腕子是唯一还能把视野推出这块地板的
    杠杆**。走之前先算好相机在哪、光轴打桌面哪一点（日志里打出来），并过一道安全带
    （TCP 不许比初始点位低 5mm）。这一段走关节空间，因为掰腕子那类姿态**逆解解不出来**
    （`sweep_look.py` 开头记着实测：解出来的构型是肩抬 -98.9°、抓取点丢 90mm，直接撞限位）。

    两段都是**认出目标就停**。返回 (中心坐标, 朝向弧度, facts)；都没认出来就是
    (None, None, 最后一眼的 facts)。
    """
    facts: dict = {}

    # ---- 第 1 段：前后左右 ----
    base = arm.tcp().copy()
    r = _view_radius(cams, float(base[2]))
    if r is None:
        print("    ！算不出画面在目标那个平面上覆盖多大，前后左右这一步跳过")
    else:
        print(f"    前后左右：一格 {r * 1000:.0f}mm（画面内切圆的半径），从"
              f"{np.round(base[:2] * 1000, 0)} 出发，四个方向")
        for i, (du, dv) in enumerate(SWEEP_4, 1):
            cand = base + np.array([du * r, dv * r, 0.0])
            way = SWEEP_WAY[(du, dv)]
            if geom.out_of_reach(cand[:2]):
                print(f"    · {way}（到 {np.round(cand[:2] * 1000, 0)}）"
                      f"出了可达环，够不着，跳过")
                continue
            state.seg = f"前后左右找目标（{i}/{len(SWEEP_4)}）"
            got = _goto(arm, cand, vmax=speed_vmax(SPEED))
            if got == "term":
                print(f"    ！时间上限掐到了（往{way}那一步），前后左右中断")
                return None, None, facts
            if got != "reached":
                print(f"    · 往{way}（到 {np.round(cand[:2] * 1000, 0)}）"
                      f"没走到（{got}），跳过")
                continue
            print(f"    ↻ 往{way}挪了一格，到 {np.round(arm.tcp()[:2] * 1000, 0)}，"
                  f"再看一眼")
            center, yaw, facts = _look(arm, cams, percep, instruction, None)
            if center is not None and yaw is not None:
                return center, yaw, facts
        # **回中心再抬头。** 不回的话第 2 段是从某个偏出去的点起算的，抬头推出去的
        # 视野方向也跟着偏 —— 覆盖哪一片就说不清了。
        if float(np.linalg.norm(arm.tcp()[:2] - base[:2])) > 1e-4:
            print(f"    回中心 {np.round(base[:2] * 1000, 0)} 再进第二段")
            _goto(arm, base, vmax=speed_vmax(SPEED))

    # ---- 第 2 段：原地抬个头 ----
    from real import joint_move as jm                              # noqa: PLC0415
    from real.sweep_look import poses                              # noqa: PLC0415

    # **就一个姿态**：pan 不动，4 号腕关节掰到底。表还是 `sweep_look.py` 那张表，只是
    # 把范围收成一点（`poses` 本来就吃 `pan_deg` / `flex_deg`），不另抄一份。
    grid, dropped = poses(arm, pan_deg=(0.0,), flex_deg=(SWEEP_HEAD_FLEX,))
    if grid is None:
        print("    ！模型里没有 `wrist_cam`，算不出相机朝哪，抬头这一步做不了")
        return None, None, facts
    if not grid:
        print(f"    ！4 号掰到 {SWEEP_HEAD_FLEX:+.0f}° 之后 TCP 比初始点位低，"
              f"被安全带滤掉了，不抬")
        return None, None, facts

    p = grid[0]
    state.seg = "原地抬头拉远"
    aim = "" if p["hit"] is None else (
        f"，光轴落桌面 {p['hit'][0] * 1000:+.0f},{p['hit'][1] * 1000:+.0f}mm")
    print(f"    ↻ 原地抬头：4 号掰到 {p['flex']:+.0f}°{aim}，再看一眼")
    jm.goto_q(arm, p["q"], q_from=arm._ticks_to_qpos(arm.read_ticks()))
    if getattr(arm, "terminated", False):
        print("    ！时间上限掐到了，抬头这一步没看成")
        return None, None, facts
    center, yaw, facts = _look(arm, cams, percep, instruction, None)
    return center, yaw, facts
