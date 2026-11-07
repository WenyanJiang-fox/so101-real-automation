"""合爪、判夹住、读 6 号舵机。"""
from __future__ import annotations

from core.params import (ALIGN_CHUNK, ALIGN_MAX_STEPS,            # noqa: E402
                         ALIGN_ROT_SLEW, ALIGN_TOL_DEG, CARRY,
                         GENTLE, GRIP_BIAS, GRIP_K, GRIP_RATE,
                         GRIP_SLACK, LIFT_Z, POS_SCALE,
                         SLOW, SPEED, STUCK_TOL,
                         acc_amax, speed_vmax)
import numpy as np
from .knobs import (CLOSE_MAX_STEPS, GRIP_APPROACH_MM, GRIP_FAST_FACTOR,
                    JAW_CLOSED_MM, JAW_OPEN_MM)


def _contact_deg(arm, half_m):
    """按「物体在合爪轴上的宽度」预判合到多少度会碰到它。

    返回 `(接触角, 每毫米多少度)`；预判不了返回 `None`。锚点全是现成的：
    本趟实测的 `arm.empty_deg` / `arm.open_deg`，配用户量过的爪尖开口
    `JAW_CLOSED_MM` / `JAW_OPEN_MM`：

        deg(W) = empty_deg + (W - 30.9) / (55.9 - 30.9) × (open_deg - empty_deg)

    **两个端点之间是线性插，只是「大概」**（没逐点校过），调用方自己留余量
    （`GRIP_APPROACH_MM` / `GRIP_RELEASE_MM`）。
    """
    empty = getattr(arm, "empty_deg", None)
    open_ = getattr(arm, "open_deg", None)
    if half_m is None or empty is None or open_ is None:
        return None
    span = float(open_) - float(empty)
    travel = float(JAW_OPEN_MM) - float(JAW_CLOSED_MM)
    if span <= 1e-6 or travel <= 1e-6:
        return None
    width_mm = 2.0 * float(half_m) * 1000.0
    frac = float(np.clip((width_mm - JAW_CLOSED_MM) / travel, 0.0, 1.0))
    return float(empty) + frac * span, span / travel


def _grip_hold(arm, state) -> float:
    """夹住之后给的那一维指令：恒定合爪一份，再按「被撑开多少」补回来。

    为什么不能给 0 见 `core/params.GRIP_BIAS`；开口为什么有下限见 `GRIP_SLACK`。
    在真机上多一层意思：这一维是**增量**，给 0 等于零刚度，舵机只是停在那个位置
    不出力，盒子靠自重就能滑下去。
    """
    ref = state.grip_ref
    if ref is None:
        return 0.0
    q = float(arm.grip_q())
    if q < ref - GRIP_SLACK:
        return GRIP_BIAS                       # 已经比下限更紧了：松回去，别再往里挤
    return float(np.clip((ref - q) / GRIP_K, -1.0, 0.0)) * GRIP_RATE - GRIP_BIAS

def _close_until_stall(arm, closing, max_steps: int = CLOSE_MAX_STEPS, *,
                       half_m=None):
    """一小步一小步合爪，压不动了就停。返回 (停因, 事实)。

    仿真那边靠**指垫接触点**判「两边都贴上了、再挤一点就停」。真机没有接触传感，
    只剩夹爪自己的角度：夹爪是位置控制的舵机，嘴里有东西就转不下去，角度会停在某个
    位置上不动 —— 那个「不动了」就是到位置了。

    **但停住这件事本身分不清「夹住盒子」和「自己合到底了」。** 差别只在角度大小上，
    那正是 `grasp_check.classify_ref` 管的事（跟本次实测的空抓基准比）。所以这里只管
    把爪合到停、把角度报出来，**判「夹住没有」交给调用方**，这一段不越权。

    **合爪这一段只让 6 号动（用户 2026-10-16 定）。** 原来每步走的是 `arm.drive`：写一
    个笛卡尔点、六维一起解逆解 —— 也就是说 1~5 号每一步都在被重算一遍，位置会跟着飘。
    既然这一步已经定了「就是合爪」，那 1~5 号就没有理由再动。现在开头量一次 tick，之后
    每一步把它们**原样写回去**（舵机是位置控制，写同一个位置就是钉住），只推进 6 号。
    """
    from real.arm import GRIP_SCALE as _ARM_GRIP_SCALE                   # noqa: PLC0415
    from servo_bridge import deg_from_position                           # noqa: PLC0415

    ticks0 = {int(k): int(v) for k, v in arm.read_ticks().items()}
    q0 = float(arm.grip_q())
    arm._grip_cmd = float(np.clip(q0, arm.grip_lo, arm.grip_hi))
    # **两段速度**（用户 2026-10-19）：按 `half_m` 预判接触角，离它还远就快、快到了退回
    # 原速。预判不了（没给 `half_m`、行程两端没量）就 `fast = False`，整段和原来一样。
    pred = _contact_deg(arm, half_m)
    if pred is None:
        deg_c = deg_per_mm = None
        print("    合爪：预判不了接触角（没有 half 或行程两端），整段走原速")
    else:
        deg_c, deg_per_mm = pred
        print(f"    合爪预判：物体宽 {2.0 * float(half_m) * 1000:.1f}mm → 大约在 "
              f"{deg_c:.0f}° 碰到（合到底 {float(arm.empty_deg):.0f}°、张到最大 "
              f"{float(arm.open_deg):.0f}°）；还差 {GRIP_APPROACH_MM:.0f}mm"
              f"（{GRIP_APPROACH_MM * deg_per_mm:.0f}°）以内退回原速")
    q_prev = None
    stuck = 0
    for step in range(1, int(max_steps) + 1):
        ticks_now = arm.read_ticks()
        q = float(arm._ticks_to_qpos(ticks_now)[5])
        deg_now = (deg_from_position(ticks_now[6]) if 6 in ticks_now else None)
        stuck = stuck + 1 if (q_prev is not None and q > q_prev - 1e-5) else 0
        q_prev = q
        if stuck > STUCK_TOL:
            return "stall", {"steps": step, "q0": q0, "q": q, "why": "压不动了"}
        # 6 号往前推一小步（和 `drive` 里那一维同一个增量、同一个行程夹），其余五个照
        # 开头量到的那串 tick 写回去 —— 一个字节都不挪。
        # 离预判接触点还远就乘 `GRIP_FAST_FACTOR`，进了慢区退回 `GENTLE`。
        inc = GENTLE
        if (deg_c is not None and deg_now is not None
                and deg_now > deg_c + GRIP_APPROACH_MM * deg_per_mm):
            inc = GENTLE * GRIP_FAST_FACTOR
        arm._grip_cmd = float(np.clip(arm._grip_cmd - inc * _ARM_GRIP_SCALE,
                                      arm.grip_lo, arm.grip_hi))
        _qc = arm._ticks_to_qpos(ticks0)
        _qc[5] = arm._grip_cmd
        _write = dict(ticks0)
        _write[6] = int(arm._qpos_to_ticks(_qc)[6])
        arm.write_ticks(_write)
        if arm.terminated:
            return "term", {"steps": step, "q0": q0, "q": float(arm.grip_q()),
                            "why": "到点收工"}
    return "steps", {"steps": int(max_steps), "q0": q0, "q": float(arm.grip_q()),
                     "why": "合爪步数用完（夹爪一直没停过）"}

def _gripper_deg(arm, ctrl):
    """读 6 号舵机角度，**单读失败就退到 `arm` 那条批量同步读**。返回 `(角度 or None, 走哪条)`。

    **为什么要有这一手**（2026-10-16）：`read_gripper_deg` 走 `get_servo_angle`（单舵机
    `INST_READ`，5 次取中位数），5 次全丢包返回 None。而 `out/run_real_1x2.log` 第 101 行
    就是这个情形：合爪 216 步、压不动了，紧接着单读 5/5 全失败 → `classify_ref` 得
    `no-reading` → 代码当场判「没夹住」、退回悬停点，**抬升那一段一步没走**；用户看画面
    说「没空抓啊，抓起来了啊」。

    同一步里 `arm.grip_q()` 走 `sync_read_positions`（一次批量读 6 个）从没丢过。两条读的
    是同一个寄存器、换算同一条式子（`servo_bridge.deg_from_position`），所以可以互相顶替。
    **不改判据、不改阈值** —— 只是把「读不到」这件事从「判失败」拉回「换个读法再读一次」。
    """
    from servo_bridge import deg_from_position, read_gripper_deg        # noqa: PLC0415

    d = read_gripper_deg(ctrl)
    if d is not None:
        return float(d), "单读"
    ticks = arm.read_ticks()
    pos = ticks.get(6)
    if pos is None:
        return None, "两条都读不到"
    return deg_from_position(pos), "arm 同步读（单读丢了）"
