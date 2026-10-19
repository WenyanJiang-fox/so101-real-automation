"""断点、确认、命令文件、存档。技能停下来的那几处全走这儿。"""
from __future__ import annotations

from core.params import (ALIGN_CHUNK, ALIGN_MAX_STEPS,            # noqa: E402
                         ALIGN_ROT_SLEW, ALIGN_TOL_DEG, CARRY,
                         GENTLE, GRIP_BIAS, GRIP_K, GRIP_RATE,
                         GRIP_SLACK, LIFT_Z, POS_SCALE,
                         SLOW, SPEED, STUCK_TOL,
                         acc_amax, speed_vmax)
import numpy as np
from . import knobs
from .knobs import AXIS_TOL_PX, CHECKPOINT, CHECKPOINT_VERSION, CLEAR_3D_TOL_MM, CLEAR_BAND_HI_PX, CLEAR_BAND_LO_PX, CLEAR_BAND_ON, CLEAR_FREE_ON, CLEAR_TARGET_PX, CONFIRM_DESCEND, CONFIRM_DESCEND_IMG, CONFIRM_GO, CONFIRM_IMG, CONFIRM_LIFT, CONFIRM_LIFT_IMG, CONFIRM_STEP, CONFIRM_WAIT_S, DESCEND_XY_TOL_MM, DESCEND_Z_TOL_MM, FREE_DESCENT, GRASP_Z_FINAL_REAL, MANUAL_CMD, MANUAL_STATE
from .rulers import _gap_want_px, _put_lines, _solve_delta
from .motion import _goto
def _stop_set() -> set[str]:
    """`STOP_AFTER` 拆成段名的集合。空字符串 / None 都是空集（＝不提前停）。"""
    return {s.strip() for s in str(knobs.STOP_AFTER or "").split(",") if s.strip()}

def _stop_here(seg: str, arm, note: str = "") -> dict | None:
    """这一段**走完了**。`STOP_AFTER` 点到它就返回一个收工的 dict，没点到就是 None。

    返回的 dict 一定带 `give_up: True` —— 策略层（`impls/policy_real.py`）见它就不重试，
    不然它会「没抓起来，回去再来」把这一趟原样再来两遍。
    """
    if seg not in _stop_set():
        return None
    print(f"    ⏹ `--stop-after` 点到「{seg}」，到此为止（不重试）")
    return {"ok": False, "why": f"只走到「{seg}」{note}".strip(),
            "stop_after": seg, "give_up": True, "tcp": arm.tcp().tolist()}

def _manual_on() -> bool:
    """微调这段是不是交给外面一步步给（环境变量 `PICK_MANUAL=1`）。"""
    import os                                                   # noqa: PLC0415

    return os.environ.get("PICK_MANUAL", "").strip().lower() not in ("", "0", "false", "no")

def _confirm_on() -> bool:
    """**合爪那一步**停下来等点头（环境变量 `PICK_CONFIRM=1`）。"""
    import os                                                   # noqa: PLC0415

    return os.environ.get("PICK_CONFIRM", "").strip().lower() not in ("", "0", "false", "no")

def _confirm_descend_on() -> bool:
    """**只在下探那一关**停下来等点头（环境变量 `PICK_CONFIRM_DESCEND=1`）。

    用户 2026-10-16：「下一轮下探前最后一帧给我看」—— 微调那几十眼不用停（他已经看过策略
    在动），但**下探之前那一帧必须让他看一眼**：那一眼就是「爪子相对方块摆成什么样」的终态，
    下探一落、跟着就合爪，没有第二次机会。`PICK_CONFIRM_STEP=1` 开着的时候这一关照旧停。
    """
    import os                                                   # noqa: PLC0415

    return (os.environ.get("PICK_CONFIRM_DESCEND", "").strip().lower()
            not in ("", "0", "false", "no")) or _confirm_step_on()

def _confirm_step_on() -> bool:
    """**微调每一步**也停下来等点头（环境变量 `PICK_CONFIRM_STEP=1`）。

    **2026-10-15 用户：「每步时间太长了」。** 原来两个闸口共用 `PICK_CONFIRM`，于是微调
    每一步都要走一趟「写 `out/clear_step.json` → 外面每 0.1 秒扫一次 → 写回
    `out/clear_go.json`」；实测每一眼多花几秒，而且没人守着就干等（那一趟白等了 7 分钟）。
    拆开之后：`PICK_CONFIRM=1` 只管合爪那一步（用户定过「一个实验只准合一趟爪」，那一步
    必须有人看过），微调不再停。
    """
    import os                                                   # noqa: PLC0415

    return os.environ.get("PICK_CONFIRM_STEP", "").strip().lower() not in (
        "", "0", "false", "no")

def _resume_on() -> bool:
    """这一趟是不是从断点接着走（环境变量 `PICK_RESUME=1`，`run_sim.py --resume-pick` 设）。"""
    import os                                                   # noqa: PLC0415

    return os.environ.get("PICK_RESUME", "").strip().lower() not in ("", "0", "false", "no")

def _start_at() -> str | None:
    """面板「从断点继续」给的入口（环境变量 `PICK_START_AT`）：`align` / `fine` / `close`。

    用户 2026-10-18：自动跑着停手之后，要能按停下来的那一段接着走。**跟 `PICK_RESUME`
    不是一回事** —— 那个是「进程被 kill 在等点头里」的续跑，这个是从当前段重新进。
    """
    import os                                                   # noqa: PLC0415

    got = os.environ.get("PICK_START_AT", "").strip().lower()
    return got or None

def _grasp_side_deg() -> float:
    """**换抓取方向**：把选面的参考轴绕竖直轴转这么多度（环境变量 `PICK_GRASP_SIDE_DEG`）。

    `0` = 现状（按形状自己挑那一对侧面）；`90` = 去对**另一对**侧面。
    只有两对跨度相同时（正方体）才会真的换 —— 长方体由 `Shape.normal` 按小跨度定，
    转多少度都还是原来那一对（`core/shapes.py:189` 那段口径）。
    """
    import os                                                   # noqa: PLC0415

    try:
        return float(os.environ.get("PICK_GRASP_SIDE_DEG", "0") or 0.0)
    except ValueError:
        return 0.0

def _raise_to_safe(arm) -> float:
    """**续跑第一步：竖直抬到安全高度**（`knobs.HOME_RAISE_Z_M`），返回抬完的 TCP z。

    为什么要它：技能从头起跑的前几步是「合一次空爪 → 进场」，而停手可能停在爪尖贴着
    方块的高度上 —— 在那儿合空爪就是拿爪去夹方块。所以续跑先原地上抬，每一步从读回来
    的姿态重解（一口气抬逆解残差会越容，一步不走）。已经在上面就一个字节不写。
    """
    tcp = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    target = float(knobs.HOME_RAISE_Z_M)
    dz = target - float(tcp[2])
    if dz <= 0.001:
        print(f"    续跑：已经在 {tcp[2] * 1000:+.0f}mm（安全高度 "
              f"{target * 1000:.0f}mm），不用抬")
        return float(tcp[2])
    print(f"    续跑：先竖直抬到安全高度 {target * 1000:.0f}mm"
          f"（现在 {tcp[2] * 1000:+.0f}mm，分 30mm 小步）")
    for _ in range(8):
        tcp = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
        dz = target - float(tcp[2])
        if dz <= 0.001:
            break
        _goto(arm, np.array([tcp[0], tcp[1], float(tcp[2]) + min(0.03, dz)],
                            dtype=np.float64), vmax=speed_vmax(SLOW), grip=None)
        after = float(np.asarray(arm.tcp(), dtype=np.float64)[2])
        if after - float(tcp[2]) < 0.001:
            print(f"    ！续跑抬不动了：停在 {after * 1000:+.0f}mm"
                  f"（要到 {target * 1000:+.0f}mm）—— 不再硬抬")
            break
    return float(np.asarray(arm.tcp(), dtype=np.float64)[2])

def _save_checkpoint(rec: dict) -> None:
    """把这一眼的状态写盘。**覆盖写** —— 盘上永远只留最近这一个断点。"""
    import json                                                 # noqa: PLC0415
    import time                                                 # noqa: PLC0415

    rec = {**rec, "version": CHECKPOINT_VERSION,
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"    ⤓ 断点已写盘：{CHECKPOINT.name}"
          f"（这一刻被 kill 的话，下一次带 `--resume-pick` 就从这一眼接着走）")

def _load_checkpoint() -> dict | None:
    """读盘上那份断点。没有 / 读不出来 / 版本不对，都当没有 —— 宁可从头走一趟。"""
    import json                                                 # noqa: PLC0415

    if not CHECKPOINT.exists():
        return None
    try:
        ck = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    except Exception as exc:                                    # noqa: BLE001
        print(f"    ! {CHECKPOINT.name} 读不出来（{type(exc).__name__}）—— 当没有断点")
        return None
    got = int(ck.get("version") or 0)
    if got != CHECKPOINT_VERSION:
        print(f"    ! {CHECKPOINT.name} 是版本 {got}，这一版认的是 {CHECKPOINT_VERSION}"
              f" —— 当没有断点，从头走这一趟")
        return None
    return ck

def _drop_checkpoint() -> None:
    """这一趟走完了（合爪了 / 判失败了），盘上那份断点作废。"""
    CHECKPOINT.unlink(missing_ok=True)

def _arm_off_checkpoint(arm, ck: dict) -> float:
    """臂现在离断点那个姿态差多少度（六个关节里离得最远的那个）。读不到就是 999。

    **这是续跑唯一认领不认领的判据**，`skills/home_real.py` 那边跳过回位也用它 ——
    两处不许各写一份，不然「回位跳过了、抓取却不认」会出现。
    """
    try:
        q_now = np.asarray(arm.angles(), dtype=np.float64).reshape(-1)
    except Exception:                                           # noqa: BLE001
        return 999.0
    q_ck = np.asarray(ck.get("angles") or [], dtype=np.float64).reshape(-1)
    if q_ck.size == 0 or q_now.shape != q_ck.shape:
        return 999.0
    return float(np.max(np.abs(q_now - q_ck)))

def _confirm_step(m: dict, dp, seg_axis, seg_gap, step_xy, tgt, gain: float, z0: float,
                  tries: int, *, mm: dict | None = None):
    """把这一步画出来、摆出去、等点头。

    返回 `(放不放行, 外面给的位移或 None)`：
      · `(False, None)` —— 叫停 / 等超时，这一趟不合爪，退回去；
      · `(True, None)` —— 点头了，走算出来的那一步；
      · `(True, (dx, dy))` —— **外面直接给位移**（米，底座坐标系，相对此刻的 TCP），
        这一步按它走、不走算出来的那一步。走完照常重量一眼、回到这个闸口。

    **位移怎么给**（`out/clear_go.json`，三种任选）：
      · `{"go": true}`            —— 走算出来的那一步；
      · `{"move": [dx, dy]}`      —— 毫米，相对此刻 TCP 的位移（z 照旧钉在 `z0`）；
      · `{"tgt": [x, y]}`         —— 毫米，底座系绝对坐标（z 照旧钉在 `z0`）；
      · `{"dz": -5}`              —— 毫米，把**钉住的那一档高度**挪一挪（可跟 `move` 一起给）；
      · `{"abort": true}`         —— 这一趟不合爪，退回去。

    用户 2026-10-16 定：**断点上要能「我说怎么移就怎么移」，不许只能退回重跑。**
    同一天又要「下一点高度看看能不能合上」—— 所以 `dz` 给出来：给了就把 `z0` 也改掉，
    后面每一眼都钉在**新的**那一档（只给一次、后面自动沿用，不然下一眼就弹回去）。

    画的是**当前那一帧**（`out/jaw_check.jpg`，`_jaw_check` 刚写的那张）上面再加箭头，
    都从画面里那个方块中心那一点出发：
      · 绿 = 轴偏那一段（`(−axis)·n_hat`，把方块中心拉回那条线上）；
      · 黄 = 缝那一段（`_gap_need_px(gap, gap_exit)·axis_dir`，把缝送回带里 —— 缝不够就
        把红区推离爪子轮廓，缝超了就把方块**拉回**两爪之间）；
      · 红 = 这一步要走的量。

    **`CLEAR_3D_ENFORCE` 开着的时候**（`mm` 不是 None）：拦门的是 3D 毫米那两条，绿黄两段
    是画面那条尺子的分解 —— **只给人看，不参与**；红箭头画的是这一步落到画面里的等效位移
    （`−jac @ step`）。画面那条量不出来（红区被爪子挡住）时 `dp` 是 None，就不画箭头。
    文字全 ASCII（`cv2.putText` 不认中文）。图写 `out/clear_step.jpg`，读数写
    `out/clear_step.json`，然后等 `out/clear_go.json`。
    """
    import json                                                 # noqa: PLC0415

    import cv2                                                  # noqa: PLC0415

    box = m.get("box_uv")
    box = None if box is None else np.asarray(box, dtype=np.float64)
    img = cv2.imread(str(knobs.JAW_CHECK_IMG))
    if img is not None and box is not None:
        p0 = (int(round(float(box[0]))), int(round(float(box[1]))))
        for seg, col, w in ((seg_axis, (0, 200, 0), 4), (seg_gap, (0, 220, 220), 4)):
            if seg is None:
                continue
            p1 = (int(round(float(box[0] + seg[0]))), int(round(float(box[1] + seg[1]))))
            cv2.arrowedLine(img, p0, p1, col, w, cv2.LINE_AA, tipLength=0.25)
        if dp is not None:
            end = (int(round(float(box[0] + dp[0]))), int(round(float(box[1] + dp[1]))))
            cv2.arrowedLine(img, p0, end, (0, 0, 255), 6, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(img, end, 7, (255, 255, 255), 2, cv2.LINE_AA)
    if img is not None:
        # 文字全 ASCII —— `cv2.putText` 不认中文。
        def _q(v, fmt="{:+.0f}"):
            return "?" if v is None else fmt.format(float(v))
        _want = (f"{CLEAR_BAND_LO_PX:.0f}..{CLEAR_BAND_HI_PX:.0f} = between jaws"
                 if CLEAR_BAND_ON else f">={CLEAR_TARGET_PX:.0f}")
        # 每行都短一点 —— `_put_lines` 按最长那行现算字号，压短一行就把整块字顶大一档。
        # 括号里那些常数（`want` / `tol`）原来挤在同一行，把字撑到画幅边上。
        px_line = (f"px (reference only):  gap {_q(m.get('gap'))} (want {_want})"
                   f"   axis {_q(m.get('axis'))} (tol {AXIS_TOL_PX:.0f})"
                   f"   along {_q(m.get('along'))}")
        step_line = (f"arm step ({step_xy[0] * 1000:+.1f}, {step_xy[1] * 1000:+.1f}) mm"
                     f"   gain {gain:.2f}   z pinned {z0 * 1000:+.1f} mm")
        lin = [f"tries {tries}   px and mm both on this frame"] if mm is None else [
            f"tries {tries}   RULER = 3D mm   box center FROZEN (looked before the descent)",
            f"off-axis {mm['p'] * 1000:+.1f} mm (tol {CLEAR_3D_TOL_MM:.0f})"
            f"   along-axis {mm['ds'] * 1000:+.1f} mm (report only)"
            f"   -> {'PASS' if mm['ok'] else 'FAIL'}",
            px_line,
        ]
        if mm is not None and dp is not None:
            step_line += f"   red arrow = that step ({dp[0]:+.0f}, {dp[1]:+.0f}) px"
        lines = lin + [step_line,
                       f"target {np.round(np.asarray(tgt) * 1000, 1)} mm (base frame)"]
        _put_lines(img, lines)
        cv2.imwrite(str(CONFIRM_IMG), img)

    rec = {"tries": tries,
           "ruler": "3d_mm_frozen" if mm is not None else "pixel",
           "gap": None if m.get("gap") is None else round(float(m["gap"]), 1),
           "want_gap": _gap_want_px(),
           "band_px": ([CLEAR_BAND_LO_PX, CLEAR_BAND_HI_PX] if CLEAR_BAND_ON else None),
           "axis": None if m.get("axis") is None else round(float(m["axis"]), 1),
           "tol_axis": AXIS_TOL_PX,
           "along": None if m.get("along") is None else round(float(m["along"]), 1),
           "box_uv": None if box is None else [round(float(v), 1) for v in box],
           "dp_px": None if dp is None else [round(float(dp[0]), 1), round(float(dp[1]), 1)],
           "seg_axis_px": None if seg_axis is None else [round(float(seg_axis[0]), 1),
                                                         round(float(seg_axis[1]), 1)],
           "seg_gap_px": None if seg_gap is None else [round(float(seg_gap[0]), 1),
                                                       round(float(seg_gap[1]), 1)],
           "step_mm": [round(float(step_xy[0]) * 1000, 2), round(float(step_xy[1]) * 1000, 2)],
           "tgt_mm": [round(float(v) * 1000, 1) for v in np.asarray(tgt).reshape(-1)],
           "gain": round(float(gain), 3),
           "z0_mm": round(float(z0) * 1000, 2),
           "img": str(CONFIRM_IMG)}
    if mm is not None:
        # 拦门的那两条，单位毫米。**看图就看这两个数。**
        rec["off_axis_mm"] = round(float(mm["p"]) * 1000, 2)
        rec["along_axis_mm"] = round(float(mm["ds"]) * 1000, 2)
        rec["tol_mm"] = CLEAR_3D_TOL_MM
        rec["frozen_center_mm"] = [round(float(v) * 1000, 1)
                                   for v in np.asarray(mm["center"]).reshape(-1)]
    CONFIRM_STEP.parent.mkdir(parents=True, exist_ok=True)
    CONFIRM_STEP.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    CONFIRM_GO.unlink(missing_ok=True)
    how = (f'给 {{"go": true}} 走算出来的这一步、{{"move": [dx, dy]}} 或 {{"tgt": [x, y]}}'
           f"（毫米，底座系）直接给位移、{{'abort': true}} 这一趟不合爪")
    if mm is None:
        print(f"    ⏸ 等点头：{CONFIRM_IMG.name} 是当前这一帧（绿=轴偏那一段、黄=缝那一段、"
              f"红=合起来），读数 {CONFIRM_STEP.name}；写 {CONFIRM_GO} {how}"
              f"（最多等 {CONFIRM_WAIT_S / 60:.0f} 分钟）")
    else:
        print(f"    ⏸ 等点头：{CONFIRM_IMG.name} 是当前这一帧（**尺子是 3D 毫米、方块中心"
              f"冻住**：离轴 {mm['p'] * 1000:+.1f}mm、沿轴差 {mm['ds'] * 1000:+.1f}mm，"
              f"两条都容 ≤{CLEAR_3D_TOL_MM:.0f}mm），读数 {CONFIRM_STEP.name}；写 "
              f"{CONFIRM_GO} {how}（最多等 {CONFIRM_WAIT_S / 60:.0f} 分钟）")
    cmd = _wait_cmd(CONFIRM_WAIT_S, CONFIRM_GO)
    if cmd is None:
        print("    ✗ 等点头等超时了 —— 这一趟不合爪，退回去")
        return False, None
    # **闸口叫「别微调了，直接下探」**（用户 2026-10-16：「可不可以结束微调直接下探」）。
    # 微调的进带条件是量出来的，它不进带就永远等不到下探；人看着爪子和方块觉得够正了，
    # 得有一条「收工」的口子。给 `{"descend": true}` 就从这儿跳出微调、照冻住的那个中心
    # 走竖直下探（`_descend_to_grasp`），水平一点不再挪。
    if cmd.get("descend"):
        print("    ✓ 外面叫直接下探 —— 微调到这儿收工，不再挪水平，去走竖直下探那一截")
        return True, ("descend", 0.0, 0.0)
    # **外面直接给位移**（用户 2026-10-16：「断点上我说怎么移就怎么移」）。给了就不走
    # 算出来的那一步 —— 走这个，走完照常重量一眼、再回到这个闸口。
    mv = cmd.get("move") if cmd.get("move") is not None else cmd.get("move_mm")
    dzc = cmd.get("dz")
    dz = None if dzc is None else float(dzc) / 1000.0
    if mv is not None:
        d = np.asarray(mv, dtype=np.float64).reshape(-1)
        if d.size >= 2:
            tail = "" if dz is None else f"、高度 {dz * 1000:+.1f}mm"
            print(f"    ✓ 外面给的位移 ({d[0]:+.1f}, {d[1]:+.1f})mm{tail} —— 按它走（不是算出来的 "
                  f"({step_xy[0] * 1000:+.2f}, {step_xy[1] * 1000:+.2f})mm）")
            return True, (float(d[0]) / 1000.0, float(d[1]) / 1000.0, dz)
        print(f"    ✗ 外面给的位移只有 {d.size} 个数，要两个 —— 这一趟不合爪，退回去")
        return False, None
    if dz is not None:
        # 只给高度：水平不动，把钉住的那一档挪一挪。
        print(f"    ✓ 外面只给了高度 {dz * 1000:+.1f}mm —— 水平不动，按它走")
        return True, (0.0, 0.0, dz)
    tg = cmd.get("tgt") if cmd.get("tgt") is not None else cmd.get("tgt_mm")
    if tg is not None:
        a = np.asarray(tg, dtype=np.float64).reshape(-1)
        if a.size >= 2:
            print(f"    ✓ 外面给的绝对目标 ({a[0]:+.1f}, {a[1]:+.1f})mm —— 按它走")
            return True, ("abs", float(a[0]) / 1000.0, float(a[1]) / 1000.0)
        print(f"    ✗ 外面给的绝对目标只有 {a.size} 个数，要两个 —— 这一趟不合爪，退回去")
        return False, None
    if not cmd.get("go"):
        print(f"    ✗ 外面叫停（{cmd}）—— 这一趟不合爪，退回去")
        return False, None
    print("    ✓ 点头了，走这一步")
    return True, None

def _descend_to_grasp(arm, center):
    """微调进带之后、合爪之前，**再竖直下去一截**（用户 2026-10-16 定）。

    返回 `(走没走成, 走完那一刻的 TCP)`。走不成（等超时 / 外面叫停 / 这一截顶住没到底）
    返回 `(False, ...)`，外面照旧退回悬停点、这一趟判失败 —— **没到底就不合爪**，这是
    原来 `_descend` 那一段就定的规矩（浅处合上去只合在方块顶上，抬起来必掉）。

    **只动 z。** 微调刚把水平那两维调进带里，这一步再碰一下水平就等于白调。
    终点 = 「方块中心 + `GRASP_Z_FINAL_REAL`」，起点读的是**此刻真实的 TCP**（不是微调前
    记的那个 —— 横移完高度会掉，`out/run_real_73.log` 第 62 行掉 0.9mm）。

    `PICK_CONFIRM_STEP=1` 时停下来等点头：图 `out/descend_step.jpg`（当前这一帧 + 两行字）、
    读数 `out/descend_step.json`，等的还是 `out/clear_go.json`：
      · `{"go": true}`    —— 按算出来的这一档往下走；
      · `{"dz": -3}`      —— 毫米，**把这一档再挪一挪**（负 = 再往下），只给一次、后面沿用；
      · `{"abort": true}` —— 不下去，这一趟不合爪。
    """
    import json                                                 # noqa: PLC0415

    import cv2                                                  # noqa: PLC0415

    tcp0 = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    z_tgt = float(center[2]) + GRASP_Z_FINAL_REAL
    drop = float(tcp0[2]) - z_tgt
    print(f"    微调完了，再竖直下去一截才合爪：从 {tcp0[2] * 1000:+.1f}mm 走到 "
          f"{z_tgt * 1000:+.1f}mm（方块中心 {float(center[2]) * 1000:+.1f}mm "
          f"{GRASP_Z_FINAL_REAL * 1000:+.0f}mm），下 {drop * 1000:.1f}mm，x/y 一点不动")
    if _confirm_descend_on():
        img = cv2.imread(str(knobs.JAW_CHECK_IMG))
        if img is not None:
            lines = [f"DESCEND before closing   z {tcp0[2] * 1000:+.1f} -> {z_tgt * 1000:+.1f} mm"
                     f"   (drop {drop * 1000:.1f} mm, x/y frozen)",
                     f"target = box center z {float(center[2]) * 1000:+.1f} mm"
                     f" + {GRASP_Z_FINAL_REAL * 1000:+.0f} mm"]
            _put_lines(img, lines)
            cv2.imwrite(str(CONFIRM_DESCEND_IMG), img)
        rec = {"z_now_mm": round(float(tcp0[2]) * 1000, 2),
               "z_target_mm": round(z_tgt * 1000, 2),
               "drop_mm": round(drop * 1000, 2),
               "center_z_mm": round(float(center[2]) * 1000, 2),
               "final_margin_mm": GRASP_Z_FINAL_REAL * 1000,
               "tcp_mm": [round(float(v) * 1000, 1) for v in tcp0],
               "img": str(CONFIRM_DESCEND_IMG)}
        CONFIRM_DESCEND.parent.mkdir(parents=True, exist_ok=True)
        CONFIRM_DESCEND.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        CONFIRM_GO.unlink(missing_ok=True)
        print(f"    ⏸ 等点头：{CONFIRM_DESCEND_IMG.name} 是当前这一帧 + 两行字，读数 "
              f"{CONFIRM_DESCEND.name}；写 {CONFIRM_GO} 给 {{\"go\": true}} 按这一档下、"
              f"{{\"dz\": -3}}（毫米，负 = 再往下）改这一档高度、{{'abort': true}} 不下去"
              f"（最多等 {CONFIRM_WAIT_S / 60:.0f} 分钟）")
        cmd = _wait_cmd(CONFIRM_WAIT_S, CONFIRM_GO)
        if cmd is None:
            print("    ✗ 等点头等超时了 —— 这一趟不合爪，退回去")
            return False, tcp0
        dzc = cmd.get("dz")
        if dzc is not None:
            z_tgt += float(dzc) / 1000.0
            print(f"    ✓ 外面把这一档改成 {z_tgt * 1000:+.1f}mm —— 按它走")
        elif not cmd.get("go"):
            print(f"    ✗ 外面叫停（{cmd}）—— 这一趟不合爪，退回去")
            return False, tcp0
        else:
            print("    ✓ 点头了，往下走")
    if drop <= 0.0:
        print(f"    ⚠ 读到的 TCP 已经在 {tcp0[2] * 1000:+.1f}mm、不比目标 {z_tgt * 1000:+.1f}mm "
              f"高 —— 不再往下走（免得变成往上抬），就这么合爪")
        return True, tcp0
    # **一气呵成**（用户 2026-10-16：「你下探的时候不准多抖几下，只准一气呵成」）：
    # `tries=1` = 解一次、走一趟，不重解、不补第二趟 —— 就是一段连续下降。
    out = _goto(arm, np.array([tcp0[0], tcp0[1], z_tgt], dtype=np.float64),
                vmax=speed_vmax(SLOW), grip=1.0,
                free=(FREE_DESCENT if CLEAR_FREE_ON else None),
                ow=(0.0 if CLEAR_FREE_ON else None), tries=1)
    tcp1 = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    print(f"    下探一截：{out}  TCP z {tcp0[2] * 1000:+.1f} → {tcp1[2] * 1000:+.1f}mm"
          f"（要 {z_tgt * 1000:+.1f}mm）；水平 ({tcp0[0] * 1000:+.1f}, {tcp0[1] * 1000:+.1f}) → "
          f"({tcp1[0] * 1000:+.1f}, {tcp1[1] * 1000:+.1f})mm")
    # **只验收 z + 横漂上限**（`DESCEND_XY_TOL_MM`，用户 2026-10-16「飘 <20mm 都没事」）：
    # `_goto` 那个 `out` 是三维误差判的，这一段横着飘是放开的，所以不能拿它当判据。
    _dz = abs(float(tcp1[2] - z_tgt))
    _dxy = float(np.hypot(tcp1[0] - tcp0[0], tcp1[1] - tcp0[1]))
    if _dz > DESCEND_Z_TOL_MM / 1000.0 or _dxy > DESCEND_XY_TOL_MM / 1000.0:
        print(f"    ✗ 这一截没到底（{out}；z 差 {_dz * 1000:.1f}mm 判据 "
              f"{DESCEND_Z_TOL_MM:.0f}mm、横漂 {_dxy * 1000:.1f}mm 判据 "
              f"{DESCEND_XY_TOL_MM:.0f}mm）—— 没到底就不合爪"
              f"（浅处合上去只合在方块顶上），退回去")
        return False, tcp1
    print(f"    ✓ 这一截认了（{out}；z 差 {_dz * 1000:.1f}mm、横漂 {_dxy * 1000:.1f}mm "
          f"≤ {DESCEND_XY_TOL_MM:.0f}mm）—— 合爪")
    return True, tcp1

def _confirm_lift(cam, tcp_now, z0: float, tries: int) -> bool:
    """**抬高度那一步也停下来等点头**（用户 2026-10-16 定）。

    横移完舵机在重力下会掉几毫米，原来这里直接自动走一趟竖直抬回去 —— 用户看到的是
    「一串没人看过的动作」，而且实测三次自动抬升读回来一次比一次低（−2.3 → −4.5 → −4.8mm，
    `out/run_real_68.log`）。现在跟 `_confirm_step` 一个口径：**先拍一张当前的整幅画面**
    （不是 `_jaw_check` 那张旧帧）、摆出去、等人点头，才动。图写 `out/clear_lift.jpg`、
    读数写 `out/clear_lift.json`，等的还是 `out/clear_go.json`。

    只在 `PICK_CONFIRM_STEP=1` 下走（跟微调那一步同一个开关）。
    """
    import json                                                 # noqa: PLC0415

    import cv2                                                  # noqa: PLC0415

    z_now = float(np.asarray(tcp_now, dtype=np.float64)[2])
    drop_mm = (float(z0) - z_now) * 1000.0
    img = None
    if cam is not None:
        try:
            img = cv2.cvtColor(np.asarray(cam.shot()), cv2.COLOR_RGB2BGR)
        except Exception as e:                                  # noqa: BLE001
            print(f"    （抬高度这一眼拍不到：{type(e).__name__}: {e}）")
            img = None
    if img is not None:
        # 文字全 ASCII —— `cv2.putText` 不认中文。
        lines = [
            f"LIFT STEP  tries {tries}  (NOT a lateral nudge)",
            f"TCP z now {z_now * 1000:+.1f} mm   floor {z0 * 1000:+.1f} mm"
            f"   dropped {drop_mm:+.1f} mm",
            f"goes straight UP to z = {z0 * 1000:+.1f} mm   xy stays "
            f"({float(tcp_now[0]) * 1000:+.1f}, {float(tcp_now[1]) * 1000:+.1f}) mm",
        ]
        _put_lines(img, lines)
        cv2.imwrite(str(CONFIRM_LIFT_IMG), img)

    rec = {"kind": "lift", "tries": tries,
           "z_now_mm": round(z_now * 1000, 2),
           "z_floor_mm": round(float(z0) * 1000, 2),
           "drop_mm": round(drop_mm, 2),
           "xy_now_mm": [round(float(tcp_now[0]) * 1000, 1),
                         round(float(tcp_now[1]) * 1000, 1)],
           "img": str(CONFIRM_LIFT_IMG)}
    CONFIRM_LIFT.parent.mkdir(parents=True, exist_ok=True)
    CONFIRM_LIFT.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    CONFIRM_GO.unlink(missing_ok=True)
    print(f"    ⏸ 等点头（**抬高度**）：{CONFIRM_LIFT_IMG.name} 是刚拍的整幅画面，读数 "
          f"{CONFIRM_LIFT.name}（TCP 读到 {z_now * 1000:+.1f}mm、比 {z0 * 1000:+.1f}mm 低 "
          f"{drop_mm:.1f}mm，要走一趟竖直抬回去）；写 {CONFIRM_GO} 给 "
          f'{{"go": true}} 就抬、给 {{"abort": true}} 这一趟不合爪（最多等 '
          f"{CONFIRM_WAIT_S / 60:.0f} 分钟）")
    cmd = _wait_cmd(CONFIRM_WAIT_S, CONFIRM_GO)
    if cmd is None:
        print("    ✗ 等点头等超时了 —— 这一趟不合爪，退回去")
        return False
    if not cmd.get("go"):
        print(f"    ✗ 外面叫停（{cmd}）—— 这一趟不合爪，退回去")
        return False
    print("    ✓ 点头了，抬回去")
    return True

def _confirm_pass(m: dict, mm: dict | None, tag: str, *, ruler: str = "3D mm") -> bool:
    """**判过了也停下来等点头**（只在 `PICK_CONFIRM=1` 下走）。

    原来判过就直接进合爪了。尺子换成 3D 那条之后不能这么走：万一它系统性偏到「过」那一侧，
    爪子会在没人看过一眼的情况下合上 —— 而用户定过「一个实验只准合一趟爪」。这一跳只摆
    当前那一帧和读数，等 `out/clear_go.json` 的 `{"go": true}`。

    `ruler` 是**这一次拦门的是哪条**，只影响印在图上和人看的那行字；`mm` 那一份（3D 读数）
    量不出来时是 None，那就不印它 —— 拦门的那条数不能因为旁支量不出来就丢了。
    """
    import json                                                 # noqa: PLC0415

    import cv2                                                  # noqa: PLC0415

    p = None if not mm else mm.get("p")
    ds = None if not mm else mm.get("ds")
    c3 = None if not mm else mm.get("center")
    g = m.get("gap")
    a = m.get("axis")
    img = cv2.imread(str(knobs.JAW_CHECK_IMG))
    if img is not None:
        lines = [f"tries READY TO CLOSE   RULER = {ruler}"]
        if p is not None and ds is not None:
            lines.append(f"off-axis {p * 1000:+.1f} mm (tol {CLEAR_3D_TOL_MM:.0f})"
                         f"   along-axis {ds * 1000:+.1f} mm (report only)")
        lines.append(f"gap {'--' if g is None else format(float(g), '+.1f')}px"
                     f"   axis {'--' if a is None else format(float(a), '+.1f')}px"
                     f"   -> PASS")
        if c3 is not None:
            lines.append(f"frozen center {np.round(np.asarray(c3) * 1000, 1)} mm")
        _put_lines(img, lines, (0, 255, 0))
        cv2.imwrite(str(CONFIRM_IMG), img)

    rec = {"pass": True, "ruler": ruler, "tag": tag,
           "off_axis_mm": None if p is None else round(float(p) * 1000, 2),
           "along_axis_mm": None if ds is None else round(float(ds) * 1000, 2),
           "tol_mm": CLEAR_3D_TOL_MM,
           "frozen_center_mm": None if c3 is None else
                               [round(float(v) * 1000, 1)
                                for v in np.asarray(c3).reshape(-1)],
           "gap_px": None if g is None else round(float(g), 1),
           "axis_px": None if a is None else round(float(a), 1),
           "img": str(CONFIRM_IMG)}
    CONFIRM_STEP.parent.mkdir(parents=True, exist_ok=True)
    CONFIRM_STEP.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    CONFIRM_GO.unlink(missing_ok=True)
    print(f"    ⏸ 判过了、**先不合**：{CONFIRM_IMG.name} 是当前这一帧（拦门的是 {ruler}；"
          f"缝 {'--' if g is None else format(float(g), '+.1f')} 像素、"
          f"轴偏 {'--' if a is None else format(float(a), '+.1f')} 像素），"
          f"读数 {CONFIRM_STEP.name}；写 {CONFIRM_GO} 给 "
          f'{{"go": true}} 就合爪、给 {{"abort": true}} 这一趟不合（最多等 '
          f"{CONFIRM_WAIT_S / 60:.0f} 分钟）")
    cmd = _wait_cmd(CONFIRM_WAIT_S, CONFIRM_GO)
    if cmd is None:
        print("    ✗ 等点头等超时了 —— 这一趟不合爪，退回去")
        return False
    if not cmd.get("go"):
        print(f"    ✗ 外面叫停（{cmd}）—— 这一趟不合爪，退回去")
        return False
    print("    ✓ 点头了，合爪")
    return True

def _publish(m: dict, arm, cam, center, step: int, tag: str, *, fresh: bool = False) -> None:
    """把这一眼的读数摆出去：往 `out/manual_state.jsonl` 追加一行。

    一行就是一步的全部 —— 判据读数、方块中心、该在的那一点、当前 TCP、以及**反解出来的
    建议位移**（毫米，底座坐标系）。操作员（我/人）读它决定下一步走多少，`read` 一次就
    能看最近几步，不用一步一步问。
    """
    import json                                                 # noqa: PLC0415
    import time                                                 # noqa: PLC0415

    rec = {"t": time.strftime("%H:%M:%S"), "step": int(step), "tag": tag,
           "gap": None if m.get("gap") is None else round(float(m["gap"]), 1),
           "want_gap": _gap_want_px(),
           "band_px": ([CLEAR_BAND_LO_PX, CLEAR_BAND_HI_PX] if CLEAR_BAND_ON else None),
           "which": m.get("which", ""),
           "axis": None if m.get("axis") is None else round(float(m["axis"]), 1),
           "along": None if m.get("along") is None else round(float(m["along"]), 1),
           "tol_axis": AXIS_TOL_PX,
           "ok": bool(m.get("ok")), "why": m.get("why", "")}
    if m.get("box_uv") is not None:
        rec["box_uv"] = [round(float(v), 1) for v in np.asarray(m["box_uv"]).reshape(-1)]
    if m.get("axis_ref_px") is not None:
        rec["ref_px"] = [round(float(v), 1) for v in np.asarray(m["axis_ref_px"]).reshape(-1)]
    if m.get("axis_dir_px") is not None:
        rec["axis_dir"] = [round(float(v), 4) for v in np.asarray(m["axis_dir_px"]).reshape(-1)]
    if getattr(arm, "tcp", None) is not None:
        try:
            rec["tcp_mm"] = [round(float(v) * 1000, 1) for v in np.asarray(arm.tcp()).reshape(-1)]
        except Exception:                                       # noqa: BLE001
            pass
    # 反解建议：同一个算式（`_solve_delta`），自动那一路拿它定方向，这里只是把它摆出来。
    got = _solve_delta(cam, center, m)
    if got is not None:
        rec["suggest_mm"] = [round(float(got[1][0]) * 1000, 1), round(float(got[1][1]) * 1000, 1)]
    MANUAL_STATE.parent.mkdir(parents=True, exist_ok=True)
    with MANUAL_STATE.open("w" if fresh else "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"    ⇢ 读数已摆到 {MANUAL_STATE.name}（第 {step} 步）："
          f"缝 {rec['gap']} / 轴偏 {rec['axis']} / 沿轴 {rec['along']}"
          f" / 建议 {rec.get('suggest_mm')}mm")

def _peek_cmd(path=None):
    """看一眼有没有命令。**文件存在且能解析成 JSON** 才算有 —— 写的那一瞬间文件可能只落了
    一半，解析不了就当还没写好，接着等。`path` 不给就是手动那一路的 `out/manual_cmd.json`；
    逐步点头那一路（`PICK_CONFIRM=1`）传 `CONFIRM_GO`。"""
    import json                                                 # noqa: PLC0415

    path = MANUAL_CMD if path is None else path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

def _take_cmd(path=None):
    """取走一条命令：读到就删掉文件（文件在不在 = 有没有待办）。没有返回 `None`。"""
    cmd = _peek_cmd(path)
    if cmd is None:
        return None
    try:
        (MANUAL_CMD if path is None else path).unlink()
    except OSError:
        pass
    return cmd

def _wait_cmd(timeout: float, path=None):
    """等一条命令，最多 `timeout` 秒。到点还没等到返回 `None`。

    每一轮先调一次 `WAIT_TICK`（默认 None，行为不变）—— 面板靠它在这一头把按键做掉，
    见那个变量的说明。**钩子抛异常不能把等待顶掉**：那一头的活儿（刷快照、关节点动）
    出点小错是常事，等点头本身不能因此中断。
    """
    import time                                                 # noqa: PLC0415

    t0 = time.time()
    cmd = None
    while time.time() - t0 < timeout:
        if knobs.WAIT_TICK is not None:
            try:
                knobs.WAIT_TICK()
            except Exception as exc:                            # noqa: BLE001
                print(f"    ！等点头时的顺手活儿出错（不影响继续等）："
                      f"{type(exc).__name__}: {exc}")
        cmd = _take_cmd(path)
        if cmd is not None:
            break
        # 面板接上 `WAIT_TICK` 的时候，一次点击从按下到被取走要等这一觉。原来固定 0.1 秒，
        # 加上面板那一头刷新读数（66~86ms），点一下最多要 ~0.17 秒才有反应。这一头只是
        # 查一下文件在不在，20ms 一遍足够便宜（用户 2026-10-18：「响应延迟依旧很高」）。
        time.sleep(0.02 if knobs.WAIT_TICK is not None else 0.1)
    # **不管是怎么结束的**（拿到答复 / 超时 / 被顶出去），都要告诉外面「不等了」——
    # 面板靠这一下当场把「允许」按钮关掉，免得人在已经走开的那 0.6 秒里白按一次。
    if knobs.WAIT_IDLE is not None:
        try:
            knobs.WAIT_IDLE()
        except Exception as exc:                                # noqa: BLE001
            print(f"    ！告诉面板「不等了」时出错（不影响继续）："
                  f"{type(exc).__name__}: {exc}")
    return cmd
