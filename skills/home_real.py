"""回真机的初始位置 —— `real_home.RAW_HOME`。

**仿真的 `home` 不能用。** `skills/home.py` 回的是 `core.params.WAYPOINT = (0.20, 0, 0.16)`，
那是仿真工位的坐标，真机够不够得着不知道。真机自己的初始点位在
`G:\\sim2real\\so101-real-harness\\real_home.py`：2026-10-14 定的六个**舵机原始值**，
通电时走完读回 1.1mm 内一致。它的姿态是「手指朝下、腕相机正对桌面」，桌面和盒子都在
视野里。

**为什么每轮开头要先回这儿。** 仿真是每轮**结尾**调一次 `home`（`impls/policy_expert.py`
第 131 行和第 162 行各一次），下一轮自然就从那儿出发。真机这一支原来把 `home` 整个
跳过了，于是臂停在上一轮收工时的地方 —— 上一次实测停在 `z=-0.010`（桌面高度），腕
相机是横着的，后面每一眼都看不见盒子。

**走关节角、不走 `follow`。** 初始点位要求光轴正对下方，从别的姿态用逆解掰过去会撞
限位（`real/joint_move.py` 开头记着实测数字）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.interfaces import Skill                                 # noqa: E402
from core.registry import register                                # noqa: E402
from real import joint_move as jm                                 # noqa: E402

# 到位判据：六个关节里离目标最远的那个允许多少度。舵机是位置控制的，走完总有残余
# 误差；`real_home.py` 记的是「走完读回 1.1mm 内一致」，换算成关节角是个小值。
#
# **2026-10-22 按实测从 3 放到 5**（用户定）。连着两轮四次回位都停在 3.1 / 3.1 /
# 3.3 / 3.1°，只读一次关节角定位到 **3 号（elbow_flex）稳定差 +3.25°**，其余五轴
# ≤0.65°。起点一致性另有 `collect_real_dataset.HOME_DRIFT_MM = 10mm` 按 TCP 兜底，
# 这里只判「有没有真的开回家」，不放宽那一层。
HOME_TOL_DEG = 5.0


@register("skill", "home_real")
class HomeRealSkill(Skill):
    """把六个关节开回 `RAW_HOME`。判据是走完读回来的关节角离目标差多少。"""

    name = "home_real"

    def run(self, arm, cams, state=None, **kw) -> dict:
        if state is not None:
            state.seg, state.ignore = "回初始位置", set()
        # ---- 断点续跑时**别把臂开走** ----
        # 续跑的断点就是臂现在这个姿态，回一次初始位置等于把断点丢了。开关和断点文件都在
        # `skills/pick_real.py` 那一支（`PICK_RESUME` / `CHECKPOINT`）；认领的判据跟抓取
        # 那边**同一条**（`_arm_off_checkpoint`），两处不许各写一份 —— 不然会出现
        # 「回位跳过了、抓取却不认」或者反过来。
        from skills import pick_real as _pr                          # noqa: PLC0415

        if _pr._resume_on():
            ck = _pr._load_checkpoint()
            if ck is not None:
                off = _pr._arm_off_checkpoint(arm, ck)
                if off <= _pr.RESUME_TOL_DEG:
                    print(f"    回初始位置：跳过 —— 这一趟是断点续跑，断点就是臂现在这个"
                          f"姿态（最远的关节差 {off:.1f}°）")
                    if state is not None:
                        state.seg = "回初始位置（断点续跑，跳过）"
                    return {"ok": True, "why": f"跳过（断点续跑，臂在断点姿态上，差 {off:.1f}°）"}
        q_goal = jm.home_q(arm)
        q_now = jm.goto_q(arm, q_goal)
        err = float(np.max(np.abs(np.degrees(np.asarray(q_now) - q_goal))))
        print(f"    回初始位置：离目标最远的关节差 {err:.1f}°"
              f"（判据 {HOME_TOL_DEG:.0f}°）  TCP {np.round(arm.tcp(), 3)}")
        if err > HOME_TOL_DEG:
            # 走完读数离目标还差得远，多半是没走到（被什么挡住了 / 舵机没出力）。
            # 如实说没到，不要把「开过去了」当成「到了」。
            return {"ok": False, "why": f"回初始位置没到位（最远差 {err:.1f}°）",
                    "err_deg": err}
        return {"ok": True, "why": f"到位（最远差 {err:.1f}°）", "err_deg": err}
