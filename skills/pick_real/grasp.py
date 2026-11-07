"""合爪、判夹住、读 6 号舵机。"""
from __future__ import annotations

from core.params import (ALIGN_CHUNK, ALIGN_MAX_STEPS,            # noqa: E402
                         ALIGN_ROT_SLEW, ALIGN_TOL_DEG, CARRY,
                         GENTLE, GRIP_BIAS, GRIP_K, GRIP_RATE,
                         GRIP_SLACK, LIFT_Z, POS_SCALE,
                         SLOW, SPEED, STUCK_TOL,
                         acc_amax, speed_vmax)
import time

import numpy as np
from .knobs import (CLOSE_FAST_FRAC, CLOSE_FAST_GAIN, CLOSE_MAX_S,
                    CLOSE_MAX_STEPS, CLOSE_SLOW_GAIN, MAX_LEAD_RAD, STUCK_S)
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

def _close_until_stall(arm, closing, max_steps: int = CLOSE_MAX_STEPS,
                       max_s: float = CLOSE_MAX_S, empty_deg: float | None = None):
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

    **合爪分快慢两档**（2026-10-20 用户定：「合爪可以快，快抓到了要慢下来」）。行程按
    本次实测的两端算：起点 `q0`（张开）、终点 `empty_deg`（空抓基准）。走完
    `CLOSE_FAST_FRAC` 之前用 `CLOSE_FAST_GAIN` 倍的大步；之后压成 `CLOSE_SLOW_GAIN` 倍，
    贴上方块那一下就是慢的。**不给 `empty_deg` 就退回原来的一档匀速**（老调用方不受影响）。
    """
    from real.arm import GRIP_SCALE as _ARM_GRIP_SCALE                   # noqa: PLC0415
    from real.joint_move import _tick                                    # noqa: PLC0415

    ticks0 = {int(k): int(v) for k, v in arm.read_ticks().items()}
    q0 = float(arm.grip_q())
    arm._grip_cmd = float(np.clip(q0, arm.grip_lo, arm.grip_hi))
    # 空抓基准换算到关节角：`empty_deg` 是 6 号舵机的度，`_ticks_to_qpos` 走的是同一条
    # 标定，所以先把度还原成 tick 再换一次，别自己另写一条换算。
    q_empty = None
    if empty_deg is not None:
        try:
            _t = dict(ticks0)
            _t[6] = int(round(float(empty_deg) * 2048.0 / 180.0 + 1024.0))
            q_empty = float(arm._ticks_to_qpos(_t)[5])
        except Exception:                                            # noqa: BLE001
            q_empty = None
    said_slow = False
    q_prev = None
    t_move = time.time()
    t0 = time.time()
    for step in range(1, int(max_steps) + 1):
        # **按墙钟收，不按步数收**（2026-10-20）。步数那 300 是按一步约 200ms 定的，
        # 接上 `fast` 之后一步只剩 5~10ms，命令合到头了爪子还在路上，300 步就收工了 ——
        # 实测 7 趟里 6 趟停在 46~71°、嘴里什么都没有，判据却算成「夹住」。
        if time.time() - t0 > float(max_s):
            return "timeout", {"steps": step, "q0": q0, "q": float(arm.grip_q()),
                               "why": f"合爪超过 {float(max_s):.0f} 秒"}
        q = float(arm.grip_q())
        # **「压不动」也按墙钟判**：`STUCK_TOL` 那 25 步同样是按一步 200ms 定的（≈5 秒），
        # 现在 25 步只有约 50ms —— 夹爪从张到合完实测 1.12 秒，那时候它一动没动，
        # 判据先报了 stall（2026-10-20 实测：27 步判「没动过」，其实根本没给够时间）。
        if q_prev is not None and q <= q_prev - 1e-5:
            t_move = time.time()
        q_prev = q
        if time.time() - t_move > STUCK_S:
            return "stall", {"steps": step, "q0": q0, "q": q, "why": "压不动了"}
        # **指令最多领先实测 `MAX_LEAD_RAD`**：一步 5~10ms 时每步 0.573° 等于 60~110°/s，
        # 读数还没更新就又压一步，0.2 秒就把目标推到底 —— 爪子被一巴掌扇合上，压到盒子
        # 上时力也是这么砸上去的。卡一个领先量：跟着走时指令只超它一步，被盒子顶住时
        # 最多再往里挤这么多（这就是夹持力），挤到上限后位置不动、`STUCK_S` 到点收手。
        # ---- 快慢两档：按「本次行程走完多少」换 ----
        gain = 1.0
        frac = None
        if q_empty is not None and abs(q0 - q_empty) > 1e-6:
            frac = float(np.clip((q0 - q) / (q0 - q_empty), 0.0, 1.0))
            gain = CLOSE_FAST_GAIN if frac < CLOSE_FAST_FRAC else CLOSE_SLOW_GAIN
            if gain == CLOSE_SLOW_GAIN and not said_slow:
                said_slow = True
                print(f"    合爪：走完 {frac * 100:.0f}%（行程 {q0:+.3f} → "
                      f"{q_empty:+.3f} rad）→ 进慢档 ×{CLOSE_SLOW_GAIN}", flush=True)
        if q - arm._grip_cmd < MAX_LEAD_RAD:
            # 6 号往前推一小步（和 `drive` 里那一维同一个增量、同一个行程夹），其余五个照
            # 开头量到的那串 tick 写回去 —— 一个字节都不挪。
            arm._grip_cmd = float(np.clip(
                arm._grip_cmd - GENTLE * gain * _ARM_GRIP_SCALE,
                arm.grip_lo, arm.grip_hi))
        _qc = arm._ticks_to_qpos(ticks0)
        _qc[5] = arm._grip_cmd
        _write = dict(ticks0)
        _write[6] = int(arm._qpos_to_ticks(_qc)[6])
        arm.write_ticks(_write)
        # **合爪这一段原来不触发 `on_step`。** 它是整段数据的收尾（真机上 10~200 步、
        # 1~20 秒），不补这一下的话，录出来的数据集在「下探结束 → 抬升开始」之间是空的。
        # `last_action` 写的是这一步发出去的六个目标位置（度），不给就一直是下探那一步的
        # 旧值。
        arm.last_action = np.degrees(arm._ticks_to_qpos(_write))
        _tick(arm)
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
