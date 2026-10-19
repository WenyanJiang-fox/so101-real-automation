"""抓成之后放到哪儿、怎么张爪。"""
from __future__ import annotations

from core import follow, geom, shapes, vision
from core.params import (ALIGN_CHUNK, ALIGN_MAX_STEPS,            # noqa: E402
                         ALIGN_ROT_SLEW, ALIGN_TOL_DEG, CARRY,
                         GENTLE, GRIP_BIAS, GRIP_K, GRIP_RATE,
                         GRIP_SLACK, LIFT_Z, POS_SCALE,
                         SLOW, SPEED, STUCK_TOL,
                         acc_amax, speed_vmax)
import numpy as np
from .knobs import PLACE_AWAY_M, PLACE_DEG_WIN, PLACE_OPEN_STEPS, PLACE_RAISE_M, PLACE_R_MAX_M, PLACE_R_MIN_M, PLACE_SAMPLE_MAX, PLACE_Z_TOL_MM, WRIST_TWIST_DEG
from .rulers import _box_half_m
from .motion import _goto, _wrist_turn
from . import randlock
def _place_xy_real(center) -> tuple[np.ndarray, str]:
    """抽一个落点 `(x, y)`（米，基座系）。返回 `(xy, 一句话说明)`。

    **只在拾取工位那一圈里抽**（见上面那组常数的说明）。两个条件：`geom.out_of_reach`
    说够得着、离原来那个位置 ≥ `PLACE_AWAY_M` —— 原地放回去就不叫「随机位置」了。

    抽 `PLACE_SAMPLE_MAX` 次还没抽到就抛，**不硬放**：调用方如实报失败，方块留在爪里。
    """
    import random                                               # noqa: PLC0415
    import time                                                 # noqa: PLC0415

    # **第一次抽到的落点记下来，之后每趟都用它**（用户 2026-10-19）。关节回放要求
    # 下一轮开始时方块和上一轮一模一样，落点每趟重抽就永远对不上。
    frozen = randlock.get("place_xy")
    if frozen is not None:
        xy = np.asarray(frozen, dtype=np.float64).reshape(-1)[:2]
        return xy, "沿用记录下来的随机落点（第一次抓成那趟抽的）"

    base = np.asarray(geom.MARKER_CENTER, dtype=np.float64)
    pick = np.asarray(geom.real_pick_xy(), dtype=np.float64).reshape(-1)[:2]
    base_deg = float(np.degrees(np.arctan2(pick[1] - base[1], pick[0] - base[0])))
    seed = int(time.time() * 1000) % 100000
    rng = random.Random(seed)
    now = np.asarray(center, dtype=np.float64).reshape(-1)[:2]
    for _ in range(PLACE_SAMPLE_MAX):
        deg = base_deg + rng.uniform(-PLACE_DEG_WIN, PLACE_DEG_WIN)
        r = rng.uniform(PLACE_R_MIN_M, PLACE_R_MAX_M)
        xy = np.array([base[0] + r * np.cos(np.radians(deg)),
                       base[1] + r * np.sin(np.radians(deg))], dtype=np.float64)
        if geom.out_of_reach(xy):
            continue
        if float(np.hypot(xy[0] - now[0], xy[1] - now[1])) < PLACE_AWAY_M:
            continue
        randlock.put("place_xy", [float(xy[0]), float(xy[1])])
        return xy, (f"方位 {deg:+.0f}°（工位 {base_deg:+.0f}° ±{PLACE_DEG_WIN:.0f}）、"
                    f"半径 {r * 1000:.0f}mm（{PLACE_R_MIN_M * 1000:.0f}~"
                    f"{PLACE_R_MAX_M * 1000:.0f}），种子 {seed}"
                    f" —— 已记下，后面每趟都用它")

    # **抽不到就换一个，不是这一趟不放**（用户 2026-10-18：「如果放方块的位置结算不出来
    # 就换一个位置啊」）。随机抽空只说明这三个条件叠起来那个角太窄 —— round 2 实测命中率
    # 0.74%，200 次抽空有 22% 概率。网格扫同一圈，**在满足「离原处 ≥ `PLACE_AWAY_M`」的点里
    # 取最近的**：少走一段、也别顶到可达环外圈去（外圈手臂伸得最直、逆解残差最大）。
    # 一个满足的都没有，才退成整圈里最远的那个；整圈都在可达环外才算真没有位置。
    near_xy, near_d, near_tag = None, float("inf"), ""
    far_xy, far_d, far_tag = None, -1.0, ""
    for i in range(41):
        deg = base_deg - PLACE_DEG_WIN + 2.0 * PLACE_DEG_WIN * i / 40.0
        for j in range(29):
            r = PLACE_R_MIN_M + (PLACE_R_MAX_M - PLACE_R_MIN_M) * j / 28.0
            xy = np.array([base[0] + r * np.cos(np.radians(deg)),
                           base[1] + r * np.sin(np.radians(deg))], dtype=np.float64)
            if geom.out_of_reach(xy):
                continue
            d = float(np.hypot(xy[0] - now[0], xy[1] - now[1]))
            tag = f"方位 {deg:+.0f}°、半径 {r * 1000:.0f}mm"
            if d > far_d:
                far_xy, far_d, far_tag = xy, d, tag
            if PLACE_AWAY_M <= d < near_d:
                near_xy, near_d, near_tag = xy, d, tag
    if far_xy is None:
        raise RuntimeError(f"这一圈里没有一个够得着的落点（工位 {base_deg:+.0f}° "
                           f"±{PLACE_DEG_WIN:.0f}°、半径 "
                           f"{PLACE_R_MIN_M * 1000:.0f}~{PLACE_R_MAX_M * 1000:.0f}mm）"
                           f" —— 这一趟不放，方块还在爪里")
    if near_xy is not None:
        randlock.put("place_xy", [float(near_xy[0]), float(near_xy[1])])
        return near_xy, (f"{near_tag}（随机 {PLACE_SAMPLE_MAX} 次抽空，网格兜底取最近）；"
                         f"离原处 {near_d * 1000:.0f}mm —— 已记下，后面每趟都用它")
    randlock.put("place_xy", [float(far_xy[0]), float(far_xy[1])])
    return far_xy, (f"{far_tag}（随机 {PLACE_SAMPLE_MAX} 次抽空，网格兜底）；"
                    f"离原处 {far_d * 1000:.0f}mm，比 {PLACE_AWAY_M * 1000:.0f}mm 的"
                    f"「离原处」还近 —— 这一圈里就它最远；已记下，后面每趟都用它")

def _open_gently(arm, max_steps: int = PLACE_OPEN_STEPS):
    """一小步一小步张爪，张到不动了为止。返回 `(停因, 事实)`。

    `_close_until_stall` 同一个形状，方向反过来（`+ GENTLE`）。**不一步给满**：给满那
    一下会把方块推着滑一段（`skills/place.py` 开头记着这条）。1~5 号照开头量到的那串
    tick 原样写回去 —— 松爪这一步只有 6 号该动。

    张到头（舵机停在行程端点）和「被方块挡住」在这里都是「不动了」，**不需要分开** ——
    松爪这一步不判成败，判成败的是松爪**之前**那个高度。
    """
    from real.arm import GRIP_SCALE as _ARM_GRIP_SCALE          # noqa: PLC0415

    ticks0 = {int(k): int(v) for k, v in arm.read_ticks().items()}
    q0 = float(arm.grip_q())
    arm._grip_cmd = float(np.clip(q0, arm.grip_lo, arm.grip_hi))
    q_prev = None
    stuck = 0
    for step in range(1, int(max_steps) + 1):
        q = float(arm.grip_q())
        stuck = stuck + 1 if (q_prev is not None and q < q_prev + 1e-5) else 0
        q_prev = q
        if stuck > STUCK_TOL:
            return "stall", {"steps": step, "q0": q0, "q": q, "why": "张到头了"}
        arm._grip_cmd = float(np.clip(arm._grip_cmd + GENTLE * _ARM_GRIP_SCALE,
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
                     "why": "张爪步数用完（夹爪一直没停过）"}

def _place_real(arm, percep, center, *, grasp_z, state) -> dict:
    """抓成之后：搬到抽出来的落点、**手里最低那一点离桌面半个边长**的那一刻松爪。

    「最低那一点」两样都算、取低的那个：
      · **两个爪尖** —— 现在读（`arm.tcp()` 报的是固定爪尖，动爪可能更低）；
      · **方块下底面** —— 抓那一刻它离 TCP 多远（`center[2] - 半个边长 - grasp_z`），
        抬起来之后这一段跟着走，所以是个常量，不用现量。

    取低的那个，是因为撞桌子的会是它。取「最低点 ≥ 半个边长」这一条同时满足用户那句
    「爪子离地面要有半个边长的距离」—— 方块比爪尖低时爪子只会更高。

    **没落到那个高度就不松爪**（`skills/place.py` 那条规矩：半空撒手是这整个流程里唯一
    无法挽回的错误）。落过头不算问题：那说明方块已经压在桌面上了，松爪正好。
    """
    half = _box_half_m(percep)
    table = geom.real_table_z()
    c0 = np.asarray(center, dtype=np.float64).reshape(-1)[:3]
    try:
        xy, why_here = _place_xy_real(c0)
    except RuntimeError as exc:
        print(f"    ✗ 摆件：{exc} —— 这一趟不放，方块还在爪里")
        return {"ok": False, "why": str(exc)}

    # 手里最低那一点离 TCP 多远（米，负 = 在 TCP 下面）。
    tips0 = arm.tips()
    tcp0 = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    rel_tips = float(min(float(tips0[0][2]), float(tips0[1][2])) - float(tcp0[2]))
    rel_box = float((float(c0[2]) - half) - float(grasp_z))
    rel_low = float(min(rel_tips, rel_box))
    # **松爪高度 = 合爪高度 + `PLACE_RAISE_M`**（用户 2026-10-18）。原来按
    # `table + half - rel_low` 算「最低点离桌面半个边长」；现在照合爪那一刻的 TCP
    # 高度抬 5mm，比原来那一档低 5mm（方块底离桌面 5mm，松手掉下去）。
    z_rel = float(grasp_z) + PLACE_RAISE_M
    which = "爪尖" if rel_tips <= rel_box else "方块底"
    print(f"    摆件：落点 ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f})mm —— {why_here}")
    print(f"      半个边长 {half * 1000:.1f}mm、桌面 {table * 1000:+.1f}mm；"
          f"手里最低的是{which}（离 TCP {rel_low * 1000:+.1f}mm）"
          f" → 松爪时 TCP 要落在 {z_rel * 1000:+.1f}mm")

    state.seg = "摆件：搬过去"
    out = _goto(arm, np.array([xy[0], xy[1], LIFT_Z], dtype=np.float64),
                vmax=speed_vmax(CARRY), grip=None, tol=0.012)
    tcp1 = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    print(f"      搬过去：{out}  TCP ({tcp0[0] * 1000:+.1f}, {tcp0[1] * 1000:+.1f})"
          f" → ({tcp1[0] * 1000:+.1f}, {tcp1[1] * 1000:+.1f})mm"
          f"  差落点 {float(np.hypot(tcp1[0] - xy[0], tcp1[1] - xy[1])) * 1000:.1f}mm")

    # **转腕放在这儿**（用户 2026-10-18：「旋转爪子的指令应该在放置前，而不是在还没有
    # 把物体提起来那会儿」）。这一行往下走之前，臂已经在落点上方、方块离桌面还有一段；
    # 只动 5 号，位置不变。抽一个随机角度，抽到限位外由 `_wrist_turn` 夹住并返回实际值。
    import random                                               # noqa: PLC0415

    state.seg = "摆件：随机转腕"
    want = randlock.get("wrist_deg")
    if want is None:
        want = random.uniform(-WRIST_TWIST_DEG, WRIST_TWIST_DEG)
        randlock.put("wrist_deg", float(want))
        print(f"      转腕：抽到 {want:+.1f}° —— 已记下，后面每趟都用它")
    got = _wrist_turn(arm, float(want))
    print(f"      转腕：要 {want:+.1f}°，实际 {got:+.1f}°（5 号腕转；在落点上方、落下去之前）")

    state.seg = "摆件：落下去"
    down = _goto(arm, np.array([xy[0], xy[1], z_rel], dtype=np.float64),
                 vmax=speed_vmax(SLOW), grip=None, tries=1)
    tcp2 = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    tips2 = arm.tips()
    low = float(min(min(float(tips2[0][2]), float(tips2[1][2])),
                    float(tcp2[2]) + rel_box))
    clear = float(low - table)
    print(f"      落下去：{down}  TCP z {tcp1[2] * 1000:+.1f} → {tcp2[2] * 1000:+.1f}mm"
          f"（要 {z_rel * 1000:+.1f}）；水平 "
          f"({tcp2[0] * 1000:+.1f}, {tcp2[1] * 1000:+.1f})mm")
    print(f"      最低那一点离桌面 {clear * 1000:.1f}mm（要 ≥ {half * 1000:.1f}mm，"
          f"高过 {(half + PLACE_Z_TOL_MM / 1000.0) * 1000:.1f}mm 算没落下去）")
    if clear > half + PLACE_Z_TOL_MM / 1000.0:
        print(f"    ✗ 摆件：没落到半个边长上（离桌面还 {clear * 1000:.1f}mm）—— "
              f"**不松爪**，方块还在爪里，这一趟没摆成")
        return {"ok": False, "why": f"没落到位（最低点离桌面 {clear * 1000:.1f}mm）",
                "at": [float(xy[0]), float(xy[1])]}

    # **记下「方块是从哪儿、以什么姿势放下的」**（用户 2026-10-19）：下一轮方块就
    # 躺在这一点。松爪这一刻的两爪中点就是方块中心（水平），TCP 与它的相对位置就是
    # 抓它时要摆回去的位置；朝向、合爪高度、半边长一起记，下一轮照这条记录去抓，
    # 不用重新看一眼。方块是垂直落下去的，水平位置不跟着变。
    try:
        from real import joint_rec as _jrec                        # noqa: PLC0415
        _gap = np.asarray(arm.jaw_gap_center(), dtype=np.float64).reshape(-1)
        _jrec.mark("place_low", tcp=[float(v) for v in tcp2],
                   gap_center=[float(_gap[0]), float(_gap[1])],
                   quat=[float(v) for v in np.asarray(arm.quat()).reshape(-1)],
                   tool_x=[float(v) for v in
                           np.asarray(arm.tool_x(), dtype=np.float64).reshape(-1)[:3]],
                   half=float(half), grasp_z=float(grasp_z),
                   pick_center_z=float(c0[2]), table_z=float(table),
                   wrist_deg=float(got))
    except Exception:                                             # noqa: BLE001
        pass

    state.seg = "摆件：松爪"
    stop, info = _open_gently(arm)
    print(f"      松爪：{stop}（{info['why']}，{info['steps']} 步）"
          f"  开合角 {info['q0']:+.3f} → {info['q']:+.3f} rad")
    state.held = None
    state.grip_ref = None
    if arm.terminated:
        return {"ok": False, "why": "松爪中途到点收工", "at": [float(xy[0]), float(xy[1])]}

    state.seg = "摆件：退开"
    _goto(arm, np.array([xy[0], xy[1], LIFT_Z], dtype=np.float64),
          vmax=speed_vmax(SLOW), grip=1.0, tol=0.015)
    print(f"    摆件：放好了 —— 落点 ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f})mm，"
          f"松爪那一刻最低点离桌面 {clear * 1000:.1f}mm"
          f"（半个边长 {half * 1000:.1f}mm，方块从这一档掉到桌上）")
    return {"ok": True, "at": [float(xy[0]), float(xy[1])],
            "clearance_mm": clear * 1000.0, "half_edge_mm": half * 1000.0,
            "z_release_mm": float(tcp2[2]) * 1000.0,
            "wrist_twist_deg": got}
