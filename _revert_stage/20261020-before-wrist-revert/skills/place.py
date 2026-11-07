"""放置：从搬运高度落到放置点上，贴地之后才松爪。

**落点可以是从外面给的一个坐标**（`core.geom.PLACE_TARGET`）—— 真机上到这一步没有
标识盘可看，只有一串数：「这个点离底座左右多少、前后多少」。给了就往那儿放，没给就
用表里那个放置标识点。这条路走得通，靠的是搬运本身是「按坐标走」（`_recenter` 那套
不在这一步），不需要看着落点找。

**路径是一折两段**：目标正上方 → 竖直落到抓取时那个高度，中间不刹车。分成「先横
移再下降」两段独立轨迹的话，横移段末尾速度归零，下降段再从头起步，视频里能看出
一顿；一折两段是一条轨迹，拐角处自然减速、拐过去继续走。

**贴地了才松爪。** 松爪是单程动作 —— 张开了方块就掉，没有回头路。所以张爪之前先
查两件事：落点高度到了没有（`AIR_TOL`）、方块还在不在爪里（指垫还贴着没有）。任
一条不满足就**不张爪**，保持原样并报出来。半空撒手是这一整个流程里唯一无法挽回的
错误，宁可停在半空让下一轮的摆件步骤去处理。

**松爪是一小步一小步张的**（`OPEN_STEP`），不是一步给满弹开 —— 弹开的那一下会把
方块推着滑一段。

**搬运途中掉了就回去再抓一次。** 搬运这一段挂着守卫（`_gone_guard`），方块一离开两
个指垫当场停下，不是等落完才查。掉了就按掉下去那一刻手臂自己的坐标
（`group="掉了 → 回来重抓"`）重走一遍抓取，捡回来了再接着搬运 —— 只给一次机会。
捡不回来就如实报失败，不硬撑、不半空撒手。这不是防滑，是「掉了能自己捡回来」。
"""

from __future__ import annotations

import numpy as np

from core import follow, geom
from core.interfaces import Skill
from core.params import (AIR_TOL, CARRY, LIFT_Z, OPEN_STEP, SLOW, acc_amax,
                         speed_vmax)
from core.registry import make, register

from .pick import _gone_guard, grip_hold


@register("skill", "place")
class PlaceSkill(Skill):
    """把手里的方块放到放置标识点上。"""

    name = "place"

    def run(self, arm, cams, state=None, index: int = 0, **kw) -> dict:
        i = index if index is not None else state.held
        if i is None:
            return {"ok": False, "why": "爪里没东西"}

        # 落点：外面给了坐标就用它（真机上只有这一对数），否则用表里的放置标识点
        pxy = geom.place_xy(i)
        dst = np.array([pxy[0], pxy[1], 0.0])
        state.seg, state.ignore = f"搬运 {arm.cube_order[i]}", {i}

        def bump():
            state.check_bump(arm)

        # 一折两段：先平移到放置点上方，再竖直落到抓取时的高度。
        # 挂着守卫走：方块半路掉了当场停，不是落完才发现。掉了按当时的 TCP 坐标回去
        # 重抓一次，捡回来就重走这一段；捡不回来就如实报失败。
        for attempt in (1, 2):
            down = follow.follow(arm, [[dst[0], dst[1], LIFT_Z],
                                       dst + [0, 0, state.grasp_z]],
                                 vmax=speed_vmax(CARRY), amax=acc_amax(),
                                 grip=lambda: grip_hold(arm, state), final_tol=0.004,
                                 on_step=bump, guard=_gone_guard(arm, state, i))
            if down != "guard":
                break
            p = arm.tcp()
            print(f"    ✗ 搬运途中方块掉了（停因 {down}）  TCP {np.round(p, 3)}"
                  f"  爪-方块接触 {arm.finger_cube_contacts(i)[:2]}")
            if attempt == 2:
                return {"ok": False, "why": "掉了，没捡回来"}
            print(f"      回去重抓一次：按掉下去那一刻自己的坐标 "
                  f"({p[0] * 1000:.0f},{p[1] * 1000:.0f})mm 一带去找")
            got = make("skill", "pick").run(arm, cams, state, i,
                                            src=[float(p[0]), float(p[1]),
                                                 geom.object_rest_z(i)],
                                            group="掉了 → 回来重抓", retry=False)
            if not got.get("ok"):
                return {"ok": False, "why": "掉了，没捡回来"}
            state.seg, state.ignore = f"搬运 {arm.cube_order[i]}", {i}

        nf, nm, _ = arm.finger_cube_contacts(i)
        z_tcp = float(arm.tcp()[2])

        # 贴地段几乎都是「高度到位、xy 差几毫米过不了 final_tol」，那不算没降下去。
        # 真正要拦的是「还挂在半空」和「方块早就不在爪里了」这两件事。
        if z_tcp > state.grasp_z + AIR_TOL or (nf == 0 and nm == 0):
            print(f"    ✗ 不松爪：贴地段停因 {down}"
                  f"  TCP z={z_tcp:.3f}（放置高度 {state.grasp_z:.3f}）"
                  f"  爪-方块接触 固定 {nf} / 活动 {nm}"
                  f"  方块在 {np.round(arm.cube_pos(i), 3)}")
            return {"ok": False, "why": "没降到贴地"}

        follow.hold(arm, 100, grip=OPEN_STEP, pos_slew=0.02, rot_slew=0.03, on_step=bump)
        state.clear_hand()

        state.seg = "放完退开"
        follow.follow(arm, [[dst[0], dst[1], LIFT_Z]], vmax=speed_vmax(SLOW),
                      amax=acc_amax(), grip=1.0, final_tol=0.015, on_step=bump)
        print(f"    放到 {np.round(arm.cube_pos(i)[:2], 3)}"
              f"（给的坐标 {np.round([dst[0], dst[1]], 3)}"
              f"{'，不是表里那个标识点' if geom.PLACE_TARGET is not None else ''}）")
        return {"ok": True, "at": [float(dst[0]), float(dst[1])]}
