"""关节空间走位：给一串目标关节角，按梯形速度排好，一步一步把关节角写下去。

**为什么真机上要有这条路径，不全走 `follow`（TCP + 逆解）。** 掰腕子那类姿态逆解
解不出来 —— `real/sweep_look.py` 开头记着实测：想靠 IK 掰姿态去找视野外的东西，解
出来的构型是肩抬 `-98.9°`、抓取点丢 90mm，直接撞限位。所以那一路只能直接给关节角。

**只有这一份。** 两个地方要用同一个：`real/sweep_look.py` 自己跑那条命令行，和
`skills/pick_real.py`（扫视）、`skills/home_real.py`（回初始位置）在 harness 里跑。
写成两份的话，速度、到点判据、安全判据就会各走各的。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.trajectory import build          # noqa: E402
from real_home import RAW_HOME             # noqa: E402

# 速度、频率、守候。跟 `real/sweep_look.py` 原来的命令行默认值同源，改就一起改。
HZ = 30.0
# **2026-10-14 提速 2.75 倍。** 原来 0.20 弧度/秒 = 11.5°/秒，进场那一段要动的最大关节
# 是 29°，光那一个关节就 2.5 秒；用户看录像的判断是「太慢、一顿一顿」。舵机是
# `speed=0`（自己最快，空载约 4.5 弧度/秒），0.55 还不到它的八分之一，离打滑很远。
# `JACC` 跟着抬，不然「快巡航」配「慢起步」看着是两截。
# **2026-10-15 用户：「每步时间太长了」。** 量出来的账：整趟 95.5 秒里走位占 52.7 秒，
# 走位里轨迹 59%、补写 22%、静置 15%、读回 4%。所以动这三个数，不动别的。
#   轨迹 59% → `JVEL` 0.55 → 0.95（仍不到舵机自己最快的四分之一，见上面那段）；
#   补写 22% → `HOLD` 8 → 4（`goto_q` 那条走位为主的调用也都传 3 了，默认值对齐）；
#   静置 15% → `SETTLE` 0.5 → 0.2（读回 31 趟只花 1.9 秒，说明舵机早就到位了）。
JVEL = 0.95        # 关节速度上限，弧度/秒（约 54°/秒）
JACC = 1.8         # 关节加速度上限，弧度/秒²
SETTLE = 0.2       # 走完之后再静一会儿，等舵机把残余误差吃掉
HOLD = 4           # 到点之后再写几遍同一个目标位置


def home_q(arm) -> np.ndarray:
    """初始点位（`real_home.RAW_HOME`）对应的关节角。"""
    return arm._ticks_to_qpos(RAW_HOME)


def goto_path(arm, q_list, *, hz=HZ, jvel=JVEL, jacc=JACC, settle=SETTLE,
              hold=HOLD) -> np.ndarray:
    """沿 `q_list` 这一串关节角串起来的轨迹走到底。返回走完之后**读回来**的关节角。

    写的是关节角、不是 TCP，理由见模块开头。`hold` 那几遍不能省：舵机是位置控制
    的，最后一帧写完不等它到位就往下走，读回来的会差一截。走的过程中
    `arm.terminated` 立起来就当场停手（时间上限掐到了），如实返回当时的位置。

    **为什么要有「一串点」这一支。** 直角走法是一步一步重解逆解的，每重解一次姿态
    就重新分配一遍关节 —— 实测下探那一段，中途重解让 2 号肩从 0.1° 自己沉到 8.4°，
    末端横着滑出去 111mm。一串点**一次解好、中途不再解**，就没有这个来源。
    `goto_q` 就是 `q_list` 只有两个点的这一条。
    """
    qs_in = [np.asarray(q, dtype=np.float64) for q in q_list]
    q_goal = qs_in[-1]
    traj = build(qs_in, vmax=jvel, amax=jacc, density=0.01)
    _ts, qs, _ = traj.sample(1.0 / hz)
    dt = 1.0 / hz
    t0 = time.time()
    for i, q in enumerate(qs):
        arm.write_ticks(arm._qpos_to_ticks(q))
        _tick(arm)
        nap = (i + 1) * dt - (time.time() - t0)
        if nap > 0:
            time.sleep(nap)
        if getattr(arm, "terminated", False):
            return arm._ticks_to_qpos(arm.read_ticks())
    t1 = time.time()
    for _ in range(int(hold)):
        arm.write_ticks(arm._qpos_to_ticks(q_goal))
        _tick(arm)
        time.sleep(dt)
    t2 = time.time()
    if settle > 0:
        time.sleep(float(settle))
    t3 = time.time()
    q_now = arm._ticks_to_qpos(arm.read_ticks())
    t4 = time.time()
    # **2026-10-15 用户：「每步时间太长了」。** 先量再改：轨迹、补写（`HOLD`）、静置
    # （`SETTLE`）各花多少秒分开打，才知道该动哪个数。四段加起来就是这一趟走位的总时长。
    print(f"      [计时] 轨迹 {len(qs)} 点 {t1 - t0:.2f}s + 补写 {int(hold)} 遍 "
          f"{t2 - t1:.2f}s + 静置 {float(settle):.2f}s + 读回 6 个舵机 {t4 - t3:.2f}s"
          f" = {t4 - t0:.2f}s")
    return q_now


def goto_q(arm, q_goal, *, q_from=None, hz=HZ, jvel=JVEL, jacc=JACC,
           settle=SETTLE, hold=HOLD) -> np.ndarray:
    """六个关节走到 `q_goal`（`goto_path` 只有起点和终点的那一条）。

    `q_from` 不给就从**读回来的**当前关节角出发。
    """
    q_prev = (np.asarray(q_from, dtype=np.float64) if q_from is not None
              else arm._ticks_to_qpos(arm.read_ticks()))
    return goto_path(arm, [q_prev, q_goal], hz=hz, jvel=jvel, jacc=jacc,
                     settle=settle, hold=hold)


def _tick(arm) -> None:
    """让主循环那一步照常发生：数步、录帧、看时间上限。

    `RealArm.drive` 每步都调 `on_step`，这条路径不经过 `drive`，不补这一下的话
    整段扫视在视频里是空的，时间上限也掐不动它（`terminated` 是 `on_step` 立的）。
    """
    arm.n_steps = int(getattr(arm, "n_steps", 0)) + 1
    cb = getattr(arm, "on_step", None)
    if cb is not None:
        cb()
