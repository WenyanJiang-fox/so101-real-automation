"""合爪前的微调：手动那一路和自动那一路。"""
from __future__ import annotations

from core import follow, geom, shapes, vision
from core.follow import POS_SLEW, ROT_SLEW
from core.params import (ALIGN_CHUNK, ALIGN_MAX_STEPS,            # noqa: E402
                         ALIGN_ROT_SLEW, ALIGN_TOL_DEG, CARRY,
                         GENTLE, GRIP_BIAS, GRIP_K, GRIP_RATE,
                         GRIP_SLACK, LIFT_Z, POS_SCALE,
                         SLOW, SPEED, STUCK_TOL,
                         acc_amax, speed_vmax)
import numpy as np
from .knobs import ALIGN_MAX_MOVES, ALIGN_PROBE_DEG, AXIS_TOL_PX, CLEAR_3D_ENFORCE, CLEAR_3D_TOL_MM, CLEAR_AXIS_AMP_MAX_MM, CLEAR_AXIS_BACK, CLEAR_AXIS_FAR, CLEAR_BAND_HI_PX, CLEAR_BAND_LO_PX, CLEAR_BAND_ON, CLEAR_CONTACT_GAIN, CLEAR_FLAT_STOP, CLEAR_GAIN, CLEAR_MIN_DEG_GAIN_MAX, CLEAR_MIN_JOINT_DEG, CLEAR_MIN_STEP_M, CLEAR_STEP_DZ_M, CLEAR_STEP_M, CLEAR_TARGET_PX, CLEAR_Z_KEEP_M, CLEAR_Z_RECOVER_M, CLEAR_Z_RECOVER_MIN, CLEAR_Z_FLOOR_M, FREE_DESCENT, MANUAL_CMD, MANUAL_GOTO_TOL, MANUAL_MAX_MOVES, MANUAL_WAIT_S, SKIP_CLEARANCE_REAL, _ax_free, _gap_free
from .pose import _align_deg, _px_per_mm, _wrist
from .rulers import _box_deg_px, _box_half_m, _gap_leaving, _gap_need_px, _gap_ok, _gap_px, _gap_want_px, _jaw_check, _jaw_check_3d, _jaw_mm
from .gates import _confirm_on, _confirm_pass, _confirm_step, _confirm_step_on, _drop_checkpoint, _manual_on, _peek_cmd, _publish, _save_checkpoint, _wait_cmd
from .jac import dq_for as _jac_dq, probe as _jac_probe
from .motion import _free_for, _goto, _joint_dir_xy, _step_min_deg, _wrist_turn
def _set_clearance_manual(arm, cam, percep, center, tag: str) -> dict:
    """手动那一路：量完把读数摆出去、等操作员给一串位移，走到进带里或者外面叫停。

    命令（`out/manual_cmd.json`）三样，全是可选的：
        `moves` 一串 `[dx, dy]` 或 `[dx, dy, dz]`（毫米，底座坐标系），按序走、每步量一次；
        `hold` 这一串走完继续等（不给就也继续等 —— 等命令本来就是这儿的常态）；
        `abort` 收工，这一趟不合爪。
    **走的当中外面又塞了新命令**，剩下的几步丢掉、按新的来。这就是「臂不停下来等一个
    来回」的做法：操作员读一次数、想一步要几十秒，臂停在那儿就是白等。

    兜底两条：等一条命令最多 `MANUAL_WAIT_S`，手动步数最多 `MANUAL_MAX_MOVES`。哪条
    到顶都返回 `ok=False`，调用方不合爪、退回去 —— 不把臂带电扔在方块上方。

    **基准是进来那一瞬间的 tcp，不是每一步量回来的 tcp。** 2026-10-15 `out/run_real_46.log`
    记着：每一步都是「当前实测 + 偏移」，臂压根到不了指令那个 z，差的那点每步攒一层，
    三步下来 z 从 +0.2 掉到 −6.4mm —— 下探高度是算出来的，它自己往下爬等于把那个数作废。
    现在 `base` 进循环前取一次，`off` 累加，指令给 `[dx, dy, dz]` 时 z 也跟着动。
    """
    MANUAL_CMD.unlink(missing_ok=True)          # 上一趟剩的命令不算数
    last = _jaw_check(cam, percep, arm, tag)
    base = np.asarray(arm.tcp(), dtype=np.float64)
    off = np.zeros(3)
    moves = 0
    _publish(last, arm, cam, center, 0, tag, fresh=True)
    while True:
        if last.get("ok"):
            print(f"    ✓ 手动微调第 {moves} 步进带里了 —— 照常合爪")
            return {**last, "tries": moves, "manual": True}
        if last.get("gap") is None:
            print(f"    ✗ 量不了：{last.get('why')} —— 这一趟不合爪，退回去")
            return {**last, "ok": False, "tries": moves, "manual": True}
        if moves >= MANUAL_MAX_MOVES:
            print(f"    ✗ 手动走满 {MANUAL_MAX_MOVES} 步还没进带里 —— 这一趟不合爪")
            return {**last, "ok": False, "why": f"手动走满 {MANUAL_MAX_MOVES} 步",
                    "tries": moves, "manual": True}
        print(f"    ⏸ 等命令：读 {MANUAL_CMD}（最多等 {MANUAL_WAIT_S / 60:.0f} 分钟）")
        cmd = _wait_cmd(MANUAL_WAIT_S)
        if cmd is None:
            print("    ✗ 等命令等超时了 —— 这一趟不合爪，退回去")
            return {**last, "ok": False, "why": "等命令超时", "tries": moves, "manual": True}
        if cmd.get("abort"):
            print("    ✗ 外面叫停 —— 这一趟不合爪，退回去")
            return {**last, "ok": False, "why": "外面叫停", "tries": moves, "manual": True}
        for pair in cmd.get("moves", []) or []:
            if moves >= MANUAL_MAX_MOVES:
                break
            d = np.array([float(pair[0]), float(pair[1]),
                          float(pair[2]) if len(pair) > 2 else 0.0]) / 1000.0
            off = off + d
            tgt = base + off
            moves += 1
            print(f"    ➜ 操作员第 {moves} 步：走 ({d[0] * 1000:+.1f}, {d[1] * 1000:+.1f},"
                  f" {d[2] * 1000:+.1f})mm（累计偏移 {np.round(off * 1000, 1)}mm，"
                  f"到 {np.round(tgt * 1000, 1)}mm，爪张着）")
            _goto(arm, tgt, vmax=speed_vmax(SLOW), grip=1.0, tol=MANUAL_GOTO_TOL)
            last = _jaw_check(cam, percep, arm, f"{tag}（手动第 {moves} 步后）")
            _publish(last, arm, cam, center, moves, tag)
            if last.get("ok") or last.get("gap") is None:
                break
            if _peek_cmd() is not None:
                print("    ↻ 有新命令进来了 —— 这一串剩下的不走了，按新的来")
                break


def _set_clearance(arm, cams, percep, center, tag: str, *, instruction=None,
                   guess=None, clear=0.0, resume=None, ctx=None,
                   axis_only: bool = False) -> dict:
    """合爪前把爪子摆到「爪子轮廓和红色有距离 **且** 合爪方向线压住方块中心」。
    **挪一眼、量一眼，直到画面里那两条同时过。**

    **两条门槛，都是画面像素**（`_jaw_check`，2026-10-15 用户定：「我现在就需要在爪子轮廓
    和红色有距离 + 合爪方向线和目标中心对齐，只有满足这两个标准才能合爪」）：
      · 缝落在带里（`CLEAR_BAND_LO_PX <= gap <= CLEAR_BAND_HI_PX`；`CLEAR_BAND_ON = False`
        时退回旧的 `gap >= CLEAR_TARGET_PX`）：红区离**整圈**爪子轮廓最近有多远 ——
        「有距离」，而且**不许退过头**（2026-10-16：退过头 = 方块已经在两爪外面，空抓）；
      · `|axis| <= AXIS_TOL_PX`：方块顶面中心到合爪轴那条线的垂距 —— 「线和中心对齐」。
    3D 那一份（`_jaw_check_3d`）照量照打、**不拦门**（它跟画面在这台机器上差十几毫米，
    见文件头 `CLEAR_3D_ENFORCE` 上面那段）。

    怎么走（**次数和步长都不写死**，2026-10-15 用户定「你的策略要能按照实际情况来」）：
      1. 两条误差**分开算、各管一维**（见文件头那段）：`dp = (−axis)·n_hat +
         _gap_need_px(gap, gap_exit)·axis_dir`；
      2. 用 `B`（「TCP 走 1mm → 方块中心动多少像素」）换成毫米，乘 `CLEAR_GAIN`：
         `δ = CLEAR_GAIN · B⁻¹ · dp`，z 不动、爪保持张着；
      3. 走一步再量一眼，拿「TCP 实际走的毫米」和「方块中心实际动的像素」把 `B` 更新一次
         （Broyden），臂比指令走得多、走不动、方向不对都会被学进 `B`。

    **什么时候停**：两条都过；或连着 `CLEAR_FLAT_STOP` 眼误差不再变小；或这一步已经不足
    `CLEAR_MIN_STEP_M`。**都不是次数上限**，是量出来的。停不进门槛返回 `ok=False`，
    调用方不合爪、退回去（2026-10-15 用户定：「一直调不进带里先判失败」）。

    **微调只挪水平两维，高度一律钉在下探那一档**（2026-10-15 用户定：「禁止微调下高度」）。
    `z0` 进循环前取一次，之后每一步的 `tgt` 都用它当 z，**不取每一步读回来的那个** ——
    `out/run_real_54.log`：命令里 z 分量本来就是 0，但起点拿的是「刚读回来、已经掉下去的
    TCP」，而 `_goto` 每趟落点又比指令低 0.5～2.3mm，4 步下来 z 从 0.0 爬到 −7.4mm。
    水平两维仍按读回来的值闭环（那是要修上一步的残差），只有 z 是绝对量。

    **修正的时候不能碰方块**：上一步把缝压回带里以下、而且还在掉，步子乘
    `CLEAR_CONTACT_GAIN` 压下来。

    **逐步点头那一路**（环境变量 `PICK_CONFIRM=1`）走法跟上面**一模一样**，只是每一次
    `_goto` 之前停下来：这一步画进 `out/clear_step.jpg`、读数写 `out/clear_step.json`，
    等 `out/clear_go.json` 点头（`_confirm_step`）。

    **手动那一路**（环境变量 `PICK_MANUAL=1`）不自己挪 —— 交给 `_set_clearance_manual`：
    量完把读数摆到 `out/manual_state.jsonl`、等 `out/manual_cmd.json`，步长和步数都由
    外面一步一步给。

    **断点续跑那一趟**（`resume` 不是 None，见文件头 `CHECKPOINT` 那一段）走法一模一样，
    只是 `z0` / `B` / `gain` / `prev` / 已走次数 / `flat` 全从盘上那份接过来、臂一步不动
    —— 断点就是臂现在这个姿态。那一趟 `PICK_MANUAL` 不再生效（手动那一路不写断点）。

    `ctx` 是「微调过了之后那几段还要用的东西」（`hover` / `closing` / 夹爪行程两端），
    循环里一个数都不用，只是**原样写进断点**，续跑那一趟再读回来。

    `axis_only=True` 是**下探之后那一趟**（用户 2026-10-16：「下探以后再加一个微调，但是
    只能动轴偏方向，其他方向不能动」）：竖直那一截已经走完、两个爪子正夹在方块两侧，
    这时候**只修轴偏**（`|axis| <= AXIS_TOL_PX`），缝那条不当门槛、沿合爪轴那一段一律写 0
    —— 在这个高度上沿合爪轴挪就是拿爪面去怼方块。缝照样量、照样打，只是不拦门。
    """
    # **微调整段跳过**（用户 2026-10-19：就要 `_oneshot_grasp.py` 那两趟的结果）。
    # 加在这里而不是各个调用点：合爪前、下探之后那两趟都走这个函数，一个口子管住全部。
    if SKIP_CLEARANCE_REAL:
        print(f"    跳过微调「{tag}」：水平不动，直接往下扎")
        return {"ok": True}
    try:
        return _clearance_loop(arm, cams, percep, center, tag, instruction=instruction,
                               guess=guess, clear=clear, resume=resume, ctx=ctx,
                               axis_only=axis_only)
    finally:
        # **停手不删断点**（用户 2026-10-18）：`arm.terminated` 为真 = 用户按了停手，
        # 盘上那一份正是「停下来这一眼」，面板的「从断点继续」要靠它接。正常走完
        # （合爪了 / 判失败了）才作废 —— 从断点接上那一趟也在这里收尾。
        if not getattr(arm, "terminated", False):
            _drop_checkpoint()

def _clearance_loop(arm, cams, percep, center, tag, *, instruction=None, guess=None,
                    clear=0.0, resume=None, ctx=None, axis_only: bool = False) -> dict:
    """`_set_clearance` 的正文。说明见它上面（以及文件头 `CHECKPOINT` 那一段）。"""
    cam = _wrist(cams)
    if resume is None and _manual_on():
        return _set_clearance_manual(arm, cam, percep, center, tag)

    seed3d = np.asarray(center if guess is None else guess, dtype=np.float64).reshape(-1)
    # **冻住的那个方块中心**（`CLEAR_3D_ENFORCE` 那条尺子用的就是它）。来源是下探**之前**
    # 那一眼量到的那一个 —— `run` 里 `_measure` 给的 `center`（`guess` 有就用 `guess`，
    # 两个是同一眼的量，`guess` 只是给 `percep.locate` 当种子的那一份）。方块在桌上不动，
    # 这一趟从头到尾都用它，**不再每眼重量一次**。
    frozen3d = seed3d.copy()
    # **下探那一档高度，微调全程钉住**（`_set_clearance_manual` 的 `base` 同一个道理）：
    # 之后每一步都往这个 z 上走，臂掉下去了也拉回来，不让它一层层往下攒。
    z0 = float(np.asarray(arm.tcp(), dtype=np.float64)[2])
    # **桌面这条底线**（用户 2026-10-18：「你看你现在爪子都戳到地面了」）：微调是在合爪
    # 高度上横着挪的，臂在负载下会往下沉 —— 沉到桌面附近还接着挪，就是拿爪尖在桌面上扫。
    try:
        z_floor = float(geom.real_table_z()) + CLEAR_Z_FLOOR_M
    except RuntimeError:
        z_floor = None
    # **半个边长**（米）：缝不够时那一步的下限用（见下面 `CLEAR_GAP_MIN_HALF` 那一段）。
    half_m = _box_half_m(percep)
    B = None                        # 「TCP 走 1mm → 画面里方块中心动多少像素」，实测出来的
    b_src = "投出来"                # `B` 是哪来的 —— 只影响日志里那句「步子 × 谁的响应」
    gain = CLEAR_GAIN
    # **画面雅可比**（2026-10-18）：微调第一眼之后探一次，之后每步用 `dq = J⁺·dp`。
    J = None
    _probed = False
    prev = None                     # 上一眼：误差、`box_uv`、那一刻的 `tcp`、那一刻的 `gap`
    flat = 0
    n = 0
    # **高度这一条只在第一趟抬一次**（2026-10-18）：这个姿态下抬不回来就认了，
    # 后面每一步都再冲一次就是用户说的「机械臂动作反复」。
    z_lift_ok = True
    if resume is not None:
        # ---- 从盘上那一份接上：z0 / B / 步长 / 次数接过来，臂不再重走前几段 ----
        z0 = float(resume["z0"])
        gain = float(resume.get("gain") or CLEAR_GAIN)
        n = int(resume.get("tries") or 0)
        flat = int(resume.get("flat") or 0)
        if resume.get("B") is not None:
            B = np.asarray(resume["B"], dtype=np.float64)
        b_src = "断点那一趟实测的"
        # **`prev` 不恢复。** 它是**上一个进程**那一眼记的 `tcp` / `box_uv`；两个进程之间
        # 臂会因为断电重读、重力塌落挪掉零点几毫米（实测 2026-10-15：z 掉 0.37mm、x 偏
        # 0.53mm），而方块没动 —— Broyden 会把这一段**不是我们指令出来的**位移当成响应学
        # 进去，一步就把 `B` 学坏（实测 [[1484,−5030],[−4521,−1350]] → [[246,−5090],
        # [−884,−1174]]，同一条残差算出来的步子从 1.5mm 变成 8.7mm）。**从这一眼重新起
        # 一个 `prev`**：下一步走完再拿「指令的位移 ↔ 方块动的像素」更新，那才是这一趟的。
        ruler = ("尺子是 3D 毫米（中心用盘上那份、轴走 FK）" if CLEAR_3D_ENFORCE
                 else f"响应矩阵是 {b_src}")
        print(f"    ⤒ 从断点接上：微调第 {n} 次之前那一眼（z 钉在 {z0 * 1000:+.1f}mm、"
              f"步子 {gain:.2f} 倍、{ruler}、已走 {n} 次），"
              f"臂一步不动，直接去量这一眼")
    last: dict = {}
    # **缝这一维连着打转就不再管它，转去纠轴偏**（用户 2026-10-18）。置起来之后这一趟
    # 的拦门只看离轴（同 `axis_only`），走位也只走轴偏那一维。
    skip_gap = False
    # 上一眼走的是哪一维（`True` = 轴偏）。换维时把「不降计数」清零 —— 缝那一段攒下的
    # 打转次数不该记到轴偏头上。
    last_phase: bool | None = None
    while True:
        # **停手就收工**（用户 2026-10-18）：`arm.terminated` 只让轨迹停住，循环原来
        # 会带着停下来的姿态继续量、继续挪。断点在 `finally` 里保留（停手不删），
        # 面板的「从断点继续」接着走这一眼。
        if getattr(arm, "terminated", False):
            print("    ✗ 停手了 —— 微调不再往下走（断点留在盘上）")
            return {**last, "ok": False, "why": "停手", "give_up": True}
        sub = tag if n == 0 else f"{tag}（微调第 {n} 次后）"
        # **只修轴偏那一趟：缝那条不拦门**（用户 2026-10-16：「只能动轴偏方向」、2026-10-17
        # 「下探后的微调无需调整缝隙」）。这一半现在由 `_jaw_check` 自己按 `axis_only` 判 ——
        # 原来是在这里把 `ok` 改回来，于是**打出去的那一行跟拦门的结论对不上**（缝在替轴偏
        # 拦门，轴偏明明在容内却报「不过」），见 `_jaw_check` 里那段。
        # `skip_gap` 之后同样只认轴偏（缝已经放弃）。
        last = _jaw_check(cam, percep, arm, sub,
                          axis_only=(axis_only or skip_gap))
        if axis_only or skip_gap:
            last["axis_only"] = True
        if CLEAR_3D_ENFORCE:
            # **拦门这条尺子**：3D 毫米。中心是冻住的那一个，合爪轴走 FK —— 两条都不看
            # 红区轮廓，爪子挡在上面也一样准。
            mm = _jaw_mm(arm, frozen3d, clear, sub, extra="（冻住的，不现量）", cam=cam)
            if mm.get("no_world"):
                print(f"    ✗ 量不了：{mm.get('why')} —— 这一趟不合爪，退回去")
                return {**last, **mm, "ok": False, "why": mm.get("why"), "tries": n,
                        "pixel": False}
            # **`axis_only` 那一趟只拿离轴那条拦门**（跟画面那一支同一个口径：这一趟沿轴
            # 不动，沿轴差多少都拦不住、也不该拦）。
            _off_ok = bool(abs(float(mm["p"])) <= CLEAR_3D_TOL_MM / 1000.0)
            # **画面那条轴偏也得过**（用户 2026-10-18：「轴偏不过肯定抓不到」）。
            # 3D 那条容 5mm、画面那条容 35 像素（≈3mm）—— 实测同一眼能出现 3D 报
            # +4.7mm「过」、画面报 −53 像素「不过」，然后合爪抓空。两条各用各的容，
            # **都过才算过**；画面量不出来（`axis is None`）才只认 3D 那条。
            _axis_px = last.get("axis")
            _axis_px_ok = bool(_axis_px is None
                               or abs(float(_axis_px)) <= AXIS_TOL_PX)
            # **缝那一半用画面那条带**（用户 2026-10-17：「新版离轴+旧版的缝」、
            # 「缝是 20~75 像素」）：`CLEAR_BAND_LO_PX <= gap <= CLEAR_BAND_HI_PX`。
            # 3D 的「沿轴」那维照量照打、不在这里拦门。
            _gap_px = last.get("gap")
            _gap_ok = bool(_gap_px is not None
                           and CLEAR_BAND_LO_PX <= float(_gap_px) <= CLEAR_BAND_HI_PX)
            # 缝那一维已经放弃（连着打转）→ 只看离轴，同 `axis_only`。
            mm_ok = bool(_off_ok and _axis_px_ok
                         and (True if (axis_only or skip_gap) else _gap_ok))
            if _off_ok and not _axis_px_ok:
                print(f"    ⚠ 3D 离轴在容里（{mm['p'] * 1000:+.1f}mm / 容 "
                      f"{CLEAR_3D_TOL_MM:.0f}mm），画面那条不过（轴偏 "
                      f"{float(_axis_px):+.0f} 像素 / 容 {AXIS_TOL_PX:.0f}）—— "
                      f"按不过算，接着纠")
            if mm_ok:
                if axis_only or skip_gap:
                    _only_txt = ("这一趟只修轴偏" if axis_only
                                 else "缝那一维连着打转、已放弃，只修轴偏")
                    print(f"    ✓ 离轴进容里了（离轴 {mm['p'] * 1000:+.1f}mm / 容 "
                          f"{CLEAR_3D_TOL_MM:.0f}mm；画面那条 "
                          + ("—（量不出来，只认 3D）" if _axis_px is None
                             else f"{float(_axis_px):+.0f} 像素 / 容 {AXIS_TOL_PX:.0f}")
                          + f"；{_only_txt}，沿轴差 "
                          f"{mm['ds'] * 1000:+.1f}mm 只报数）—— 合爪")
                else:
                    print(f"    ✓ 离轴和缝都过（离轴 {mm['p'] * 1000:+.1f}mm / 容 "
                          f"{CLEAR_3D_TOL_MM:.0f}mm；画面轴偏 "
                          + ("—" if _axis_px is None else f"{float(_axis_px):+.0f}")
                          + f" 像素 / 容 {AXIS_TOL_PX:.0f}；缝 {float(_gap_px):+.0f} 像素 / "
                          f"要落在 [{CLEAR_BAND_LO_PX:.0f}, {CLEAR_BAND_HI_PX:.0f}] 里）"
                          f"—— 合爪")
                # **`ok` 必须显式覆盖**：`last`（画面那条）这一刻多半是 False（红区被爪子
                # 挡着，缝过不了门槛），拦门的换成毫米那条之后它不再代表这一趟的结论。
                # **判过了这一跳也写断点，而且无条件写**（用户 2026-10-15 要求：不许有
                # 「停下来了、却接不上」的口子；2026-10-18：自动模式闸口关着也要写）。
                # 盘上这份就是「继续：下探一截并合爪」的那个入口。
                _save_checkpoint({
                    "why": "合爪前微调判过了、停在等你点头（盘上写的就是这一眼）",
                    # **哪种微调判过的**：续跑「close」那一支靠它决定走不走竖直下探。
                    "axis_only": bool(axis_only),
                    "instruction": instruction, "tag": tag, "ctx": ctx,
                    "center": [float(v) for v in np.asarray(center).reshape(-1)],
                    "guess": [float(v) for v in np.asarray(seed3d).reshape(-1)],
                    "clear": float(clear),
                    "angles": [float(v) for v in np.asarray(arm.angles()).reshape(-1)],
                    "tcp": [float(v) for v in np.asarray(mm["tcp"]).reshape(-1)],
                    "z0": float(z0), "gain": float(gain), "tries": int(n),
                    "flat": int(flat), "B": None if B is None else np.asarray(B).tolist(),
                    "prev": None,
                    "look": {"frozen_mm": [float(v) * 1000 for v in frozen3d.reshape(-1)],
                             "mm_p": float(mm["p"]) * 1000,
                             "mm_ds": float(mm["ds"]) * 1000,
                             "tol_mm": CLEAR_3D_TOL_MM},
                })
                if _confirm_on() and not _confirm_pass(last, mm, sub):
                    return {**last, "ok": False, "why": "合爪前等点头时被叫停", "tries": n,
                            "pixel": False}
                return {**last, "ok": True, "mm": mm, "tries": n, "pixel": False}
        else:
            # 老口径：拦门的是画面像素那两条，3D 那份照量照打、只报数。
            mm = _jaw_check_3d(arm, cam, percep, instruction, seed3d, clear, sub)
            if last.get("ok"):
                if axis_only:
                    print(f"    ✓ 轴偏进带里了（轴偏 {last['axis']:+.0f} 像素 / 容 "
                          f"{AXIS_TOL_PX:.0f}；这一趟只修轴偏，缝 "
                          + ("—（合爪轴没穿过方块）" if last.get("gap") is None
                             else f"{last['gap']:+.0f} 像素")
                          + " 只报数） —— 合爪")
                else:
                    print(f"    ✓ 两条都过（缝 {last['gap']:+.0f} 像素、"
                          f"轴偏 {last['axis']:+.0f} 像素）—— 合爪")
                # **判过了也停下来等点头。** 拦门的换回画面之后，这个口子原来只在 3D 那一支
                # 上；不补的话「画面判过」会直接合爪 —— 而用户定过「一个实验只准合一趟爪」，
                # 合爪这一下必须有人看过这一帧再点头。摆的是**冻住的**那个中心（抓取目标），
                # 不是这一眼现量的那个。
                if _confirm_on() and not _confirm_pass(
                        last, {"center": frozen3d}, sub, ruler="picture (gap/axis)"):
                    return {**last, "ok": False, "why": "合爪前等点头时被叫停", "tries": n,
                            "pixel": True}
                return {**last, "tries": n, "pixel": True}
        if last.get("gap") is None and not CLEAR_3D_ENFORCE:
            print(f"    ✗ 量不了（画面那条）：{last.get('why')} —— 这一趟不合爪，退回去")
            return {**last, "ok": False, "why": last.get("why"), "tries": n, "pixel": True}
        if last.get("gap") is None:
            # 拦门不看它了，红区被挡住量不出来只留一句 —— 这一趟照走毫米那条。
            print(f"    ⚠ 画面那条量不了（{last.get('why')}）—— 这条不拦门，照走毫米那条")

        # ---- 该挪多少 ----
        tcp = np.asarray(arm.tcp(), dtype=np.float64)
        box_uv = (None if last.get("box_uv") is None
                  else np.asarray(last["box_uv"], dtype=np.float64))
        if CLEAR_3D_ENFORCE:
            # **拦门这条尺子**：`p` 是离轴、`ds` 是沿轴，两条都得进容里
            # （`_off_ok and _gap_ok`）。**但一次只走一维、每维走自己的关节**（用户
            # 2026-10-17：「你要分清造缝隙的方向，造缝隙别造错方向」）：
            #   · 轴偏那一维（⊥合爪轴）→ **只放 1 号**（底座转，横移靠它）；
            #   · 缝那一维（沿合爪轴）→ **只放 4 号**。
            # 两维混成一步是走不动的。
            _u3 = np.asarray(mm["u"], dtype=np.float64).reshape(-1)[:2]
            _n3 = np.asarray(mm["n"], dtype=np.float64).reshape(-1)[:2]
            # 缝进没进带。拦门的就是画面那条带（`_gap_ok` 在上面算过）。
            _gap_in = bool(_gap_ok)
            # **缝没有读数就不许造缝**（用户 2026-10-18：「微调舵机完全用错了，没有起到
            # 效果」）。`gap is None` 的意思是**合爪轴整条没穿过方块**（日志原文：「方块
            # 横跨 −216~−78 像素，合爪轴在 0 那一条」）—— 臂在方块侧面十几毫米外。这一刻
            # 3D 的「沿轴差」是拿一条不穿过方块的轴投影出来的，数看着在容里（±几毫米），
            # 按它往 −u 推，就是推着臂远离方块：2026-10-18 那趟离轴 9.0 → 12.6 → 16.6 →
            # 21.5 → 23.8 → 27.1mm，八步全白走。造缝这一维**只有画面给得回读数**
            # （`_gap_ok`），给不出来就先拿离轴那条把轴对上。
            if _gap_px is None and not (axis_only or skip_gap):
                _gap_in = True
                print("    ⚠ 画面量不出缝（合爪轴没穿过方块）—— 这一维没有读数，"
                      "不做「造缝」，先拿离轴那条把轴对上")
            if axis_only or skip_gap or _gap_in:
                # 缝已经在带里（或这一趟压根不许碰沿轴、或缝那维已经放弃）→
                # **只走离轴那一维，给 4 号**。
                move_m = float(mm["p"]) * _n3
                deep = 0.0                                          # 这一趟不碰沿轴
                axis_phase = True
                _seg_txt = "轴偏（⊥合爪轴 → 1 号）"
            else:
                # 缝没进带 → **只走沿合爪轴那一维，给 1 号**。
                # 幅度取 3D 的 `|ds|`；**「往哪边」以画面为准** —— 缝拦门的是画面那条带，
                # 3D 的沿轴这维照量照打、不拦门。两条尺子的定义都写死了：
                #   · 画面 `_gap_need_px`：**正 = 沿合爪轴推出去**、负 = 把方块拉回来；
                #   · 3D：`u` 从固定爪尖指向动爪尖、方块该待在 +u 侧 `s_want` 处，
                #     所以 **推出去 = 沿 −u 走 = `ds < 0`**。
                # 两条不一致就是「造错方向」：按画面走，并把两边都打出来。
                _mag = abs(float(mm["ds"]))
                if last.get("gap") is None:
                    _push = float(mm["ds"]) < 0.0                   # 画面量不出来，退回 3D 的符号
                else:
                    _push = _gap_need_px(float(last["gap"]),
                                         float(last.get("gap_exit") or 0.0)) > 0.0
                    if _push != (float(mm["ds"]) < 0.0):
                        print(f"    ⚠ 缝的方向两条尺子不一致：画面说"
                              f"{'推出去' if _push else '拉回来'}，3D 沿轴差 "
                              f"{float(mm['ds']) * 1000:+.1f}mm 说"
                              f"{'推出去' if float(mm['ds']) < 0 else '拉回来'}"
                              f" —— 按画面那条走")
                move_m = (-_mag if _push else _mag) * _u3
                deep = float(mm["ds"])                              # 沿轴差（<0 = 进太深）
                axis_phase = False
                _seg_txt = "缝（沿 −u 推出去 → 4 号）" if _push else "缝（沿 +u 拉回来 → 4 号）"
            # **这一步走哪一维、往哪个方向，摆在日志里。** 方向按**实际派上去的那个关节**
            # 现算（`_joint_dir_xy`，只多一次串口读）；原来写死算 1 号的切向 —— 2026-10-17
            # 两段对调之后，那行会一边指着 1 号、一边让 4 号去动，读数跟动作对不上。
            _dir_req = move_m / max(float(np.linalg.norm(move_m)), 1e-12)
            _free_now = _ax_free() if axis_phase else _gap_free()
            _who = int(_free_now[0]) if _free_now else -1
            try:
                _q_now = np.array(arm._ticks_to_qpos(arm.read_ticks()), dtype=np.float64)
                _d0, _mm0 = _joint_dir_xy(arm, _q_now, _who)
            except Exception:                                       # noqa: BLE001
                _d0, _mm0 = None, 0.0
            _cos1 = 0.0 if _d0 is None else float(_dir_req @ _d0)
            print(f"    ⇢ 这一段走「{_seg_txt}」：({_dir_req[0]:+.2f}, {_dir_req[1]:+.2f})，"
                  f"合爪轴 u=({_u3[0]:+.2f}, {_u3[1]:+.2f})")
            if _d0 is not None:
                print(f"       {_who + 1} 号能走的方向 ({_d0[0]:+.2f}, {_d0[1]:+.2f})，"
                      f"夹角 {np.degrees(np.arccos(np.clip(_cos1, -1.0, 1.0))):.0f}°，"
                      f"cos = {_cos1:+.2f}（{_mm0:.2f}mm/度）"
                      + ("  ← 几乎出不上力" if abs(_cos1) < 0.5 else ""))
            err = float(np.linalg.norm(move_m)) * 1000.0            # 毫米
            axis_dir = np.asarray(last.get("axis_dir_px", (1.0, 0.0)), dtype=np.float64)
            n_hat = np.array([-axis_dir[1], axis_dir[0]], dtype=np.float64)
            seg_axis = seg_gap = dp = None
        else:
            # ---- **分两段走：先退干净，再对轴**（2026-10-16 用户定）----
            # 原来两条误差加在一起一次走：垂直那条（`n_hat`）把中心拉回轴上、沿轴那条
            # （`axis_dir`）把缝退到门槛外。**在画面上这两条是反方向的** —— 爪轮廓占
            # y 285~420，合爪轴 y 388.5 横穿它，把方块中心拉回轴上就是把方块往轮廓里塞。
            # 第一眼实测（`out/clear_step.json` 2026-10-16）：轴差 67 像素要求往上冲、
            # 缝只差 9.6 像素要求往左挪，合起来就是「往上冲 67 像素」—— 用户的判词是
            # 「你在左右偏移太少了，你需要把爪子轮廓完整移出去」。
            # 所以：`gap_exit`（整块红区退出去还要多少像素）没清掉之前，**这一趟只走
            # 横向、一点不追轴**；清掉了，才去把中心拉回轴上。
            axis_dir = np.asarray(last["axis_dir_px"], dtype=np.float64)
            n_hat = np.array([-axis_dir[1], axis_dir[0]], dtype=np.float64)
            gap_exit = float(last.get("gap_exit") or 0.0)
            # **缝的目标是一条带，不是「越远越好」**（见 `_gap_need_px`）：缝大于上限时返回
            # **负值**，把方块拉回两爪之间；小于下限才照老的往外推。
            need_gap = _gap_need_px(float(last["gap"]), gap_exit)
            seg_axis = np.zeros(2, dtype=np.float64)
            seg_gap = np.zeros(2, dtype=np.float64)
            if axis_only or skip_gap:
                # **只走轴偏那一维**（用户 2026-10-16「其他方向不能动」；2026-10-18 缝连着
                # 打转也走这一支）：沿合爪轴那一段一律写 0，不管缝差多少 —— 下探之后沿
                # 这条轴挪就是拿爪面怼方块。
                seg_axis = (-float(last["axis"])) * n_hat
                axis_phase = True
            elif _gap_leaving(need_gap):
                # 带开着的时候「缝超出上限」（`need_gap < 0`）也走这一段 —— 那一步是**把
                # 方块拉回两爪之间**，跟往外推是同一根轴、只是方向相反。
                seg_gap = need_gap * axis_dir
                axis_phase = False
            else:
                seg_axis = (-float(last["axis"])) * n_hat
                axis_phase = True
            dp = seg_axis + seg_gap
            err = float(np.linalg.norm(dp))                         # 像素
            deep = None

        # **换维了就把「不降计数」清零**（用户 2026-10-18）：缝那一段攒下的打转次数不该
        # 记到轴偏头上；`prev` 也一起丢掉 —— 上一眼的误差是另一个量，不能拿来比。
        if last_phase is not None and axis_phase != last_phase:
            flat = 0
            prev = None
        last_phase = axis_phase

        if prev is not None:
            flat = flat + 1 if err >= prev["err"] - 0.5 else 0
            # **轴偏来回摆就停**（用户 2026-10-18：「避免机械臂动作反复」）。一步 2° 是
            # 这台舵机能动的最小步；残差比它小的时候，这一步会越过目标，下一眼就反号、
            # 幅度还不小 —— 再走就是一来一回原地蹭，方块也跟着被爪面推。判据：反号，
            # 而且新的那一边不小于出发前的 1/2（明显越过、不是收敛到零附近）。
            if (axis_phase and CLEAR_3D_ENFORCE and prev.get("p") is not None
                    and float(prev["p"]) * float(mm["p"]) < 0.0
                    and abs(float(mm["p"])) >= abs(float(prev["p"])) * 0.5):
                print(f"    ✗ 轴偏在这一点上来回摆（{float(prev['p']) * 1000:+.1f} → "
                      f"{float(mm['p']) * 1000:+.1f}mm，反了号；最小一步 2° 跨不过这点残差）"
                      f"—— 停，不再走")
                return {**last, "ok": False, "why": "轴偏来回摆，最小步跨不过",
                        "tries": n, "pixel": not CLEAR_3D_ENFORCE}
            if CLEAR_3D_ENFORCE:
                # 这一步把沿轴压过头了（`ds < 0` 还在往下走）→ 把步子压下来
                if deep < 0.0 and deep < float(prev["ds"]):
                    gain *= CLEAR_CONTACT_GAIN
                    print(f"    ⚠ 上一步把沿轴压过头了（沿轴差 {float(prev['ds']) * 1000:+.1f} → "
                          f"{deep * 1000:+.1f}mm）—— 步子压到 {gain:.2f} 倍")
            else:
                # ---- 实测响应：TCP 实际走的毫米 ↔ 画面里方块中心实际动的像素 ----
                dx = tcp[:2] - prev["tcp"][:2]
                if float(np.linalg.norm(dx)) > 2e-4:
                    dy = box_uv - prev["box_uv"]
                    B = B + np.outer(dy - B @ dx, dx) / float(dx @ dx)  # Broyden
                    b_src = "实测的"
                # 这一步把缝压到**下限以下**、还在掉 → 把步子压下来（爪面在往方块里挤）
                if (last["gap"] < prev["gap"]
                        and last["gap"] < (CLEAR_BAND_LO_PX if CLEAR_BAND_ON
                                           else CLEAR_TARGET_PX)):
                    gain *= CLEAR_CONTACT_GAIN
                    print(f"    ⚠ 上一步把缝压回带里以下了（缝 {prev['gap']:+.0f} → "
                          f"{last['gap']:+.0f} 像素）—— 步子压到 {gain:.2f} 倍")

        if CLEAR_FLAT_STOP is not None and flat >= CLEAR_FLAT_STOP:
            # **连着打转就直接进下一步，不判失败**（用户 2026-10-18）：
            #   · 缝这一维打转        → 不管缝了，改成只纠轴偏（继续这一趟循环）；
            #   · 轴偏打转（下探之前） → 不再纠，直接下探（`ok=True`，外面照常走
            #                            `_descend_to_grasp`）；
            #   · 下探之后只修轴偏打转 → 直接合爪（`ok=True`，外面照常合）。
            _flat_txt = f"{err:.1f}{'mm' if CLEAR_3D_ENFORCE else ' 像素'}"
            if not axis_phase and not axis_only:
                print(f"    ⚠ 缝这一维连着 {flat} 眼没进展（还差 {_flat_txt}）—— "
                      f"不管缝了，改成只纠轴偏")
                skip_gap = True
                flat = 0
                continue
            # **轴偏纠不动就不合爪了**（用户 2026-10-18：「轴偏不过肯定抓不到」）。
            # 原来这里 `ok=True` 照走（合爪 / 继续下探），结果是带着 4~5mm 的离轴去合爪
            # —— 2026-10-18 那一趟就是这么抓空的，还把方块推走 46.6mm。打转 = 舵机在这个
            # 姿态上能动的最小步已经跨不过这点残差，再走只会来回蹭；如实报失败，等面板上的人。
            _why = (f"下探之后轴偏连着 {flat} 眼没进展（还差 {_flat_txt}）—— "
                    f"不合爪，停在这儿" if axis_only else
                    f"轴偏连着 {flat} 眼没进展（还差 {_flat_txt}）—— "
                    f"不再下探，停在这儿")
            print(f"    ✗ {_why}")
            return {**last, "ok": False, "gave_up": "axis", "why": _why,
                    "tries": n, "pixel": not CLEAR_3D_ENFORCE}
        if CLEAR_3D_ENFORCE:
            # **响应矩阵那套不走了**：残差本身就是毫米，`delta` 直接就是该走多少，
            # 不用再拿「TCP 走1mm ↔ 画面动多少像素」换算、也不用 Broyden 学。
            step_xy = move_m * gain
        else:
            if B is None:
                jac = _px_per_mm(cam, center)
                if jac is None:
                    print("    ✗ 投不出像素/毫米（这一步该往哪走算不出来）—— 无法微调，不合爪")
                    return {**last, "ok": False, "why": "投不出像素/米", "tries": n,
                            "pixel": True}
                B = -jac
            try:
                step_xy = np.linalg.solve(B, dp) * gain
            except np.linalg.LinAlgError:
                print("    ✗ 实测出来的响应矩阵是退化的（这一步该往哪走算不出来）—— 不合爪")
                return {**last, "ok": False, "why": "响应矩阵退化", "tries": n, "pixel": True}
        # **放大之前那一步**：雅可比那一路拿它换成「方块中心该挪多少像素」，不走下面
        # 那条「凑够 2°」的放大（2.7mm 的残差被放大到 10mm，正是 2026-10-18 越调越远的
        # 直接来源）。
        _step_raw = np.array(step_xy, dtype=np.float64, copy=True)
        # **一步走多长：由「至少让关节动 `CLEAR_MIN_JOINT_DEG` 度」定**（用户 2026-10-16
        # 「你应该定最小舵机度数」）。方向照算出来的，长度从 `CLEAR_STEP_M` 起算：这么长
        # 就够最小度数就按它走，不够就往长里解，一直解到最大的那个自由关节至少动
        # `CLEAR_MIN_JOINT_DEG` 度（上限 `CLEAR_MIN_DEG_GAIN_MAX` 倍）。
        _nrm0 = float(np.linalg.norm(step_xy))
        # **步长按「哪个关节最吃得动」估**（`_free_for` 先挑首选、再补一个能动的）。
        # 2026-10-18 起走位那一步改全臂解（见下面两处 `_goto`），这里只用来定这一步
        # 该走多长，不再代表「只放这几个关节」。
        _free_step = None
        if _nrm0 > 1e-9:
            _dir = step_xy / _nrm0
            # **这一维谁走得动就放谁**（`_free_for`）：底座那一维出不上力时补一个关节
            # 进来。写死的那两组（`FREE_GAP` / `FREE_AXIS`）现在是「首选」，不再是唯一。
            _free_step = _free_for(arm, (_dir_req if CLEAR_3D_ENFORCE else None),
                                   _ax_free() if axis_phase else _gap_free())
            _len = min(_nrm0, CLEAR_STEP_M)
            _deg, _j, _dz = 0.0, -1, 0.0
            for _ in range(4):
                _deg, _j, _dz = _step_min_deg(
                    arm, np.array([tcp[0] + _dir[0] * _len,
                                   tcp[1] + _dir[1] * _len, z0], dtype=np.float64),
                    _free_step)
                _next = _len
                if 1e-6 < _deg < CLEAR_MIN_JOINT_DEG:
                    _next = min(_len * (CLEAR_MIN_JOINT_DEG / _deg),
                                _nrm0 * CLEAR_MIN_DEG_GAIN_MAX)
                # **这一挪要掉多少高度，也进同一条算式。** 补进来的 2/3 号是竖直面里
                # 走弧的，横移的同时高度会跟着掉；不把它算进来，步子照横移量定、臂一边
                # 横移一边往下沉（上一趟横移完掉了 16.8mm）。掉得超过 `CLEAR_STEP_DZ_M`
                # 就把长度按比例压回来 —— 宁可这一趟少挪一点，也不拿高度换横移。
                if abs(_dz) > CLEAR_STEP_DZ_M > 0.0:
                    _next = min(_next, _len * (CLEAR_STEP_DZ_M / abs(_dz)))
                # **压缩要真的写回去。** 这个循环原来只会「往长里解」：`_next <= _len`
                # 直接 break，而 break 之前没把 `_len` 换成 `_next` —— 于是新加的压缩
                # 分支算出来 11.23mm、4.55mm 高度，写下去的仍是没压过的那一版
                # （2026-10-17 那趟日志：每步都印「带下来的高度 +4.55mm（上限 3.0）」
                # 却照走 11.23mm）。先处理压缩，再处理放大。
                if _next < _len * 0.99:
                    _len = _next
                    continue
                if _next <= _len * 1.01:        # 够了，或者乘不上去了：这一轮的 _deg 就是它
                    break
                _len = _next
            step_xy = _dir * _len
            print(f"    ⤢ 算出来该走 {_nrm0 * 1000:.2f}mm —— 这一步走 "
                  f"{_len * 1000:.2f}mm（{_j + 1} 号舵机要动 {_deg:.2f}°，"
                  f"至少要 {CLEAR_MIN_JOINT_DEG:.1f}°），带下来的高度 "
                  f"{_dz * 1000:+.2f}mm（每步上限 {CLEAR_STEP_DZ_M * 1000:.1f}mm），"
                  f"方向不变")
        # **闸口开着就别在这儿判死**（用户 2026-10-16：「断点上我说怎么移就怎么移」）。
        # 原来这条在闸口前面：算出来不足 0.5mm 当场判失败退回去，人连一眼都看不到 ——
        # 而这恰恰是最该让人接手的一刻（`out/run_real_73.log` 第 63 行就是这么结束的）。
        # 闸口开着就放它过去，等人在断点上给位移；人给 `go` 就是他自己要这一小步。
        if (float(np.linalg.norm(step_xy)) < CLEAR_MIN_STEP_M
                and not _confirm_step_on()):
            print(f"    ✗ 该走的这一步只剩 {float(np.linalg.norm(step_xy)) * 1000:.2f}mm"
                  f"（还差 {err:.1f}"
                  f"{'mm' if CLEAR_3D_ENFORCE else ' 像素'} 没进带里）—— 这一趟不合爪，退回去")
            return {**last, "ok": False, "why": "这一步不足 0.5mm", "tries": n,
                    "pixel": not CLEAR_3D_ENFORCE}
        if float(np.linalg.norm(step_xy)) < CLEAR_MIN_STEP_M:
            print(f"    ⚠ 算出来的这一步只剩 {float(np.linalg.norm(step_xy)) * 1000:.2f}mm"
                  f"（还差 {err:.1f} 像素 没进带里）—— 闸口开着，等你给位移")
        tgt = np.array([tcp[0] + step_xy[0], tcp[1] + step_xy[1], z0], dtype=np.float64)
        if CLEAR_3D_ENFORCE:
            # 画给人看的那支红箭头：这一步落到画面里等于把方块中心挪这么多像素。
            jac_draw = _px_per_mm(cam, seed3d) if cam is not None else None
            dp = None if jac_draw is None else (-jac_draw) @ step_xy
            print(f"    ➜ 该走 ({step_xy[0] * 1000:+.1f}, {step_xy[1] * 1000:+.1f})mm"
                  f"（离轴还差 {mm['p'] * 1000:+.1f}、沿轴还差 {mm['ds'] * 1000:+.1f}mm，"
                  f"容 ≤{CLEAR_3D_TOL_MM:.0f}；步子 {gain:.2f} 倍 × 冻住的中心 + FK 合爪轴）"
                  f"，到 {np.round(tgt * 1000, 1)}mm"
                  f"（z 钉在 {z0 * 1000:+.1f}、刚刚读回来 {tcp[2] * 1000:+.1f}，爪张着）再量一眼")
        else:
            print(f"    ➜ 该走 ({step_xy[0] * 1000:+.1f}, {step_xy[1] * 1000:+.1f})mm"
                  f"（方块中心在画面里该挪 ({dp[0]:+.0f}, {dp[1]:+.0f}) 像素、还差 {err:.0f}；"
                  f"其中轴偏还差 {last['axis']:+.0f}、缝还差 "
                  f"{_gap_want_px() - float(last['gap']):+.0f}，"
                  f"步子 {gain:.2f} 倍 × {b_src}响应）"
                  f"，到 {np.round(tgt * 1000, 1)}mm"
                  f"（z 钉在 {z0 * 1000:+.1f}、刚刚读回来 {tcp[2] * 1000:+.1f}，爪张着）再量一眼")
        prev = {"err": err, "box_uv": box_uv, "tcp": tcp, "ds": deep,
                "p": (float(mm["p"]) if CLEAR_3D_ENFORCE else None),
                "gap": None if last.get("gap") is None else float(last["gap"]), "B": B}
        # 逐步点头那一路（`PICK_CONFIRM_STEP=1`）：走之前把这一步画出来、摆出去、等点头。
        # **只开 `PICK_CONFIRM`（合爪那一步）时，断点照写、但不等人** —— 断点是「停下来
        # 也能接上」的那一半，跟「要不要停」不是一回事（`--resume-pick` 靠它）。
        over_mv = None          # 外面在闸口给的位移（给了就不走轴偏那套放大）
        # **断点写在等点头之前，而且每次都写**（用户 2026-10-18）。闸口关着（自动模式）
        # 也照写 —— 面板停手之后「从断点继续」就靠盘上这一份。以前只在确认闸口开着时写，
        # 自动模式一路跑下来盘上什么都没有，停下来就无处可接。
        _save_checkpoint({
            "why": "合爪前微调停在等你点头（盘上写的就是这一眼）",
            # **这一趟是哪种微调**：`False` = 下探之前那趟（两段都走），`True` = 下探
            # 之后那趟（只修轴偏）。续跑要看它，不然会在抓取高度上按两段走一遍。
            "axis_only": bool(axis_only),
            "instruction": instruction, "tag": tag, "ctx": ctx,
            "center": [float(v) for v in np.asarray(center).reshape(-1)],
            "guess": [float(v) for v in np.asarray(seed3d).reshape(-1)],
            "clear": float(clear),
            # **臂的姿态**：续跑认不认领就看这一串（`_arm_off_checkpoint`）。
            "angles": [float(v) for v in np.asarray(arm.angles()).reshape(-1)],
            "tcp": [float(v) for v in tcp.reshape(-1)],
            "z0": float(z0), "gain": float(gain), "tries": int(n), "flat": int(flat),
            # 响应矩阵那套在 3D 那条尺子上不用（`B` 就是 None）；照实写进去，续跑
            # 那一趟也照实读回来。
            "B": None if B is None else np.asarray(B).tolist(),
            "prev": {"err": float(err), "tcp": tcp.tolist(), "ds": deep,
                     "box_uv": None if box_uv is None else box_uv.tolist(),
                     "gap": None if last.get("gap") is None else float(last["gap"])},
            "look": {"gap": None if last.get("gap") is None else float(last["gap"]),
                     "axis": None if last.get("axis") is None else float(last["axis"]),
                     "along": None if last.get("along") is None else float(last["along"]),
                     "box_uv": None if box_uv is None else box_uv.tolist(),
                     "frozen_mm": [float(v) * 1000 for v in frozen3d.reshape(-1)],
                     "mm_p": float(mm["p"]) * 1000, "mm_ds": float(mm["ds"]) * 1000,
                     "tol_mm": CLEAR_3D_TOL_MM,
                     "step_mm": [float(step_xy[0]) * 1000, float(step_xy[1]) * 1000],
                     "tgt_mm": [float(v) * 1000 for v in np.asarray(tgt).reshape(-1)]},
        })
        if _confirm_on() or _confirm_step_on():
            # `mm` 只在**它拦门的时候**往下传。拦门的换回画面之后，`mm` 是这一眼现量的 3D
            # （不是冻住的那个），印出来会被读成「冻住的那个中心离轴差多少」—— 标签和数
            # 对不上，比不印更坏（`out/run_real_63.log` 第 44 行就是这么错的）。
            if _confirm_step_on():
                ok_step, over = _confirm_step(
                    last, dp, seg_axis, seg_gap, step_xy, tgt, gain, z0, n,
                    mm=mm if CLEAR_3D_ENFORCE else None)
                if not ok_step:
                    return {**last, "ok": False, "why": "微调等点头时没走成", "tries": n,
                            "pixel": not CLEAR_3D_ENFORCE}
                if over is not None:
                    if over[0] == "descend":
                        # **闸口收工**：不再 `_goto`、不再重量一眼，直接把这一趟判过，
                        # 让外面照冻住的那个中心去走竖直下探那一截（`_descend_to_grasp`）。
                        return {**last, "ok": True, "why": "闸口叫直接下探", "tries": n,
                                "pixel": not CLEAR_3D_ENFORCE}
                    # **外面给的位移顶掉算出来的那一步**（用户 2026-10-16）。不给 `dz` 时
                    # z 照旧钉在 `z0`；给了就**连 `z0` 一起改** —— 后面每一眼都钉在新的
                    # 那一档，不然下一眼立刻弹回旧高度，等于没下。
                    tcp_ov = np.asarray(arm.tcp(), dtype=np.float64)
                    dz = over[2] if len(over) > 2 else None
                    if over[0] == "abs":
                        tgt = np.array([over[1], over[2], z0], dtype=np.float64)
                    else:
                        tgt = np.array([tcp_ov[0] + over[0], tcp_ov[1] + over[1], z0],
                                       dtype=np.float64)
                    if dz is not None:
                        tgt[2] = float(tcp_ov[2]) + float(dz)
                        z0 = float(tgt[2])
                        print(f"    钉住的那一档高度改成 {z0 * 1000:+.1f}mm")
                    print(f"    走到 {np.round(tgt * 1000, 1)}mm（外面给的那一步）")
                    over_mv = over
        # ---- 轴偏那一段：**按算出来的 1 倍走，不冲不退**（用户 2026-10-16：「不准多补偿」）----
        # 原来这里是「10 倍冲过去、8 倍退回来」（净 2 倍）。现在默认关掉了：一步只走一下。
        # 要退回 10/8 那条路就设 `PICK_AXIS_AMP=1`。
        import os as _os                                             # noqa: PLC0415

        _amp_on = _os.environ.get("PICK_AXIS_AMP", "").strip().lower() in (
            "1", "true", "yes", "on")
        # ---- 画面雅可比（2026-10-18）：第一眼之后探一次，之后每步直接解关节增量 ----
        if not _probed and over_mv is None:
            _probed = True
            print("    探画面雅可比：五个关节各动一下，量「度 → 方块中心像素」…")
            J = _jac_probe(arm, cam, percep)
            if J is None:
                print("      ✗ 没探成 —— 这一步退回按关节走")
            else:
                print("      雅可比（像素/度）："
                      + "  ".join(f"J{j + 1}" for j in range(5)))
                for _r in J:
                    print("        " + "  ".join(f"{v:+7.2f}" for v in _r))
        # 这一步要在画面里把方块中心挪多少像素：3D 那条拿投影换成像素，画面那条现成。
        _dp_px = None
        if J is not None and over_mv is None:
            if CLEAR_3D_ENFORCE:
                _pj = _px_per_mm(cam, seed3d) if cam is not None else None
                if _pj is not None:
                    _dp_px = (-np.asarray(_pj, dtype=np.float64)
                              @ np.asarray(_step_raw, dtype=np.float64))
            elif dp is not None:
                _dp_px = np.asarray(dp, dtype=np.float64) * gain
        _dq = None if _dp_px is None else _jac_dq(J, _dp_px)
        if _dq is not None:
            _q0 = np.array(arm._ticks_to_qpos(arm.read_ticks()), dtype=np.float64)
            _qc = np.array(_q0, dtype=np.float64)
            _qc[:_dq.size] += np.radians(_dq)   # J 是像素/度，qpos 是弧度
            _qc[5] = _q0[5] if axis_phase else arm.grip_hi
            print(f"    雅可比：目标像素位移 ({_dp_px[0]:+.1f}, {_dp_px[1]:+.1f})px"
                  f" → 关节 {np.round(_dq, 2)}°（已换成弧度下发）")
            from real import joint_move as _jm                       # noqa: PLC0415
            _jm.goto_q(arm, _qc, q_from=_q0, settle=0.15, hold=4)
        elif axis_phase and over_mv is None and not _amp_on:
            _t = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
            one = np.array([_t[0] + step_xy[0], _t[1] + step_xy[1], z0], dtype=np.float64)
            print(f"    轴偏这一段：1 倍走一下 ({np.round(one * 1000, 1)}mm)")
            # **一步只准动一下，且只动这一段派到的那个关节**（2026-10-18 恢复）。
            # 5 个关节一起解出来的每维都不到 1°，多数落在舵机 2° 死区里：指令下去了、
            # 臂不走，日志上就是「微调没有效果」。单关节的毫米/度是实测过的（1 号横移、
            # 4 号造缝），度数能给准。夹爪那一维照此刻读数写回去（`grip=None`）。
            _goto(arm, one, vmax=speed_vmax(SLOW), grip=None,
                  free=(_free_step if _free_step is not None else _ax_free()),
                  tries=1)
        elif axis_phase and over_mv is None:
            tcp_a = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
            net = CLEAR_AXIS_FAR - CLEAR_AXIS_BACK                   # 净 2 倍
            if float(np.linalg.norm(step_xy)) * net * 1000.0 > CLEAR_AXIS_AMP_MAX_MM:
                # 净走 >15mm：不冲不绕，一步走到 2 倍。
                two = np.array([tcp_a[0] + net * step_xy[0],
                                tcp_a[1] + net * step_xy[1], z0], dtype=np.float64)
                print(f"    轴偏这一段：净 {net:.0f} 倍已超过 {CLEAR_AXIS_AMP_MAX_MM:.0f}mm，"
                      f"直接走 {net:.0f} 倍 ({np.round(two * 1000, 1)}mm)")
                _goto(arm, two, vmax=speed_vmax(SLOW), grip=1.0, free=_ax_free())
            else:
                amp = np.array([tcp_a[0] + CLEAR_AXIS_FAR * step_xy[0],
                                tcp_a[1] + CLEAR_AXIS_FAR * step_xy[1], z0], dtype=np.float64)
                print(f"    轴偏这一段：先走 {CLEAR_AXIS_FAR:.0f} 倍 ({np.round(amp * 1000, 1)}mm)")
                _goto(arm, amp, vmax=speed_vmax(SLOW), grip=1.0, free=_ax_free())
                tcp_b = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
                back = np.array([tcp_b[0] - CLEAR_AXIS_BACK * step_xy[0],
                                 tcp_b[1] - CLEAR_AXIS_BACK * step_xy[1], z0], dtype=np.float64)
                print(f"    再退回来 {CLEAR_AXIS_BACK:.0f} 倍 ({np.round(back * 1000, 1)}mm)"
                      f" —— 净 {net:.0f} 倍")
                _goto(arm, back, vmax=speed_vmax(SLOW), grip=1.0, free=_ax_free())
        else:
            # 外面在闸口给的位移**也是微调里的位移**，照旧守着这一段该放哪几个关节
            # （用户 2026-10-16 定的：1 号横移、4 号造缝）。2026-10-18 中途改成过全臂
            # 解，当天就撤回来了 —— 全臂每维不到 1°，落在 2° 死区里，臂不动。
            # **轴偏这一段：一步只准动一下**（`tries=1`，不重解、不补第二趟），
            # 夹爪那一维也不碰（`grip=None` = 照此刻读数写回去）。
            _goto(arm, tgt, vmax=speed_vmax(SLOW),
                  grip=(None if axis_phase else 1.0),
                  free=(_free_step if _free_step is not None
                        else (_ax_free() if axis_phase else _gap_free())),
                  tries=(1 if axis_phase else None))
        # **横移完高度掉了：抬一次，抬不动就不再试**（用户 2026-10-18：「还要精确能到达
        # 你想去的地方」「避免机械臂动作反复」）。原来这里是「一律不抬」（2026-10-16 的
        # 决定），起因是当时那条抬升用的是放大 + 多趟重解，冲 11~16mm、横甩 25~34mm、
        # 冲上去又落回来。现在只走**一趟、目标就是 `z0`**：抬回来一半以上算这条路能走，
        # 下一步掉了再抬；抬不回来一半就把这条关掉，后面每步只报数 —— 不来回冲。
        # 掉不到 `CLEAR_Z_RECOVER_M`（5mm，舵机 2° 死区那一档）就只报数，不动臂。
        tcp_now = np.asarray(arm.tcp(), dtype=np.float64)
        _dz_short = z0 - tcp_now[2]
        if _dz_short > CLEAR_Z_RECOVER_M:
            if not z_lift_ok:
                print(f"    ⚠ 横移完高度掉了 {_dz_short * 1000:.1f}mm（读到 "
                      f"{tcp_now[2] * 1000:+.1f}mm、该 ≥{z0 * 1000:+.1f}mm）"
                      f" —— 这一趟已经试过抬不起来，不再试")
            elif getattr(arm, "terminated", False):
                pass
            else:
                print(f"    ↑ 横移完高度掉了 {_dz_short * 1000:.1f}mm（读到 "
                      f"{tcp_now[2] * 1000:+.1f}mm、该 ≥{z0 * 1000:+.1f}mm）—— "
                      f"抬一次回 {z0 * 1000:+.1f}mm")
                _goto(arm, np.array([tcp_now[0], tcp_now[1], z0], dtype=np.float64),
                      vmax=speed_vmax(SLOW), grip=None, free=FREE_DESCENT, tries=1)
                _z_after = float(np.asarray(arm.tcp(), dtype=np.float64)[2])
                _left = z0 - _z_after
                _got = (_dz_short - _left) / max(_dz_short, 1e-9)
                if _got < CLEAR_Z_RECOVER_MIN:
                    z_lift_ok = False
                    print(f"    ⚠ 这一次只收回 {_got * 100:.0f}%（现在 z="
                          f"{_z_after * 1000:+.1f}mm、还差 {_left * 1000:.1f}mm）—— "
                          f"这个姿态抬不动，后面不再试（不来回冲）")
                else:
                    print(f"    ✓ 抬回来了：现在 z={_z_after * 1000:+.1f}mm、"
                          f"还差 {_left * 1000:+.1f}mm（收回 {_got * 100:.0f}%）")
        elif _dz_short > CLEAR_Z_KEEP_M:
            print(f"    ⚠ 横移完高度掉了 {_dz_short * 1000:.1f}mm"
                  f"（读到 {tcp_now[2] * 1000:+.1f}mm、该 ≥{z0 * 1000:+.1f}mm）"
                  f" —— 不到 {CLEAR_Z_RECOVER_M * 1000:.0f}mm，不动臂")
        # **沉到桌面附近就收工，不再横移**（用户 2026-10-18：「爪子都戳到地面了」）。
        # 这一条放在抬升之后量：抬回来了就还能接着走，抬不回来就不要在桌面上扫。
        if z_floor is not None:
            _z_now = float(np.asarray(arm.tcp(), dtype=np.float64)[2])
            if _z_now < z_floor:
                print(f"    ✗ 爪尖已经沉到桌面附近（读到 z={_z_now * 1000:+.1f}mm、"
                      f"桌面 {geom.real_table_z() * 1000:+.1f}mm、底线留 "
                      f"{CLEAR_Z_FLOOR_M * 1000:.0f}mm）—— 不再横移，这一趟停在这儿")
                return {**last, "ok": False,
                        "why": f"高度沉到桌面附近（{_z_now * 1000:+.1f}mm）",
                        "tries": n, "pixel": not CLEAR_3D_ENFORCE}
        n += 1

def _align(arm, closing, grip=-1.0, cam=None, percep=None, use_px: bool = False) -> tuple[int, float]:
    """原地把腕子转到开合轴对上盒子侧面。返回 (动了几个动作, 转完还差几度)。

    `grip=-1.0`：转正时爪子合着（同 `_goto`）。

    **2026-10-17 换做法。** 原来走 `follow.hold` 那条笛卡尔闭环：每 10 步把「腕子该朝哪」
    交给逆解，让 `drive` 一边压位置一边压姿态。实测三趟（改判据、改参考点、去掉步数上限）
    **一次都没转过去** —— 每趟都跑满 200 步、读数原地不动，模型和画面两条尺子都说没对上。
    位置被钉住的时候，那条路给不出纯旋转。

    现在直接走**关节空间纯旋转 + 画面闭环**：

      1. 先拧 `ALIGN_PROBE_DEG` 度，看模型那条是涨是落 → 定方向。
      2. 一次拧 |差|，单次不超过 45°（`align_action` 那条也在 45° 附近才稳）。
      3. 重复，直到 |差| ≤ `ALIGN_TOL_DEG`，最多 `ALIGN_MAX_MOVES` 趟。**不设步数上限。**

    画面那条量不出来（没相机 / 没色块）才退回模型那条，老口径一字不改。

    **2026-10-17 第二次改：判据换成模型那条。** 上一趟画面这条报「还差 1.0°」，模型那条
    同一时刻报 85.4°，于是转正一步没拧就放行；下探按 `closing` 锁姿态，5 号要转到
    −157.2°（自己的行程端点）还差 4.3mm，这条列解不出来，一步没走就 stall —— 根本没走到
    微调。差这 85° 不是抖动：`sh.normal` 对长方体取的是**跨度最小那一对**的面（15×29
    这只取短的那一维），画面那条按「矩形两条边差 90° 等价」折掉了这一层，所以「贴着长边」
    和「贴着短边」在它眼里都是 0°。**下探和落点用的就是模型这条**，这里就得跟它一条。
    """
    base = np.asarray(closing, dtype=np.float64).copy()

    def _model() -> float | None:
        v = _align_deg(arm, base)
        return None if v is None else float(v)

    # **旋转对称的形状，模型这条恒等于 0**：`Cylinder.normal` 直接回「当前开合轴」，
    # 圆周上哪个方向都能夹，模型就说「不用转」。估成圆柱/杯子的时候它给不出该转多少。
    # 画面那条不受这个约束（红区剪影的边就是世界系里的侧面）。`use_px=True` 时改用它。
    # 真圆柱转到哪都一样、转一下无害；估错了其实是个方块时，这一下正是要的。
    label = "画面量" if use_px else "模型量"

    def _meas() -> float | None:
        if use_px:
            v = _box_deg_px(cam, percep)
            if v is not None:
                return abs(float(v))
        return _model()

    cur = _meas()
    if cur is None:
        return _align_model(arm, closing, grip)
    if cur <= ALIGN_TOL_DEG:
        return 0, cur

    # `_align_deg` 是 arccos，只给 0~180，**不带正负** —— 方向当场试探一次再定。
    moves = 0
    sign = 1.0
    probe = _wrist_turn(arm, ALIGN_PROBE_DEG)
    moves += 1
    if abs(probe) < 1e-6:                          # 5 号一开始就在行程头上
        print(f"    转正：5 号一步没拧动（在行程端点），{label}还差 {cur:.1f}°")
        return moves, cur
    nxt = _meas()
    if nxt is None:
        return moves, cur
    if nxt > cur + 0.3:                            # 越拧越远 = 方向反了
        sign = -1.0
    # **每一步都打出来**（用户 2026-10-18 问「为什么停在 45°」）：拧了多少、量到多少、
    # 为什么停。上一轮日志只记了「拧了 4 次、还差 32.2°」，停在哪条分支看不出来。
    print(f"    转正：5 号试探 {probe:+.1f}° → {label} {cur:.1f}° → {nxt:.1f}°"
          f"（方向 {'+' if sign > 0 else '-'}）")
    best = min(cur, nxt)

    for _ in range(ALIGN_MAX_MOVES):
        cur = _meas()
        if cur is None:
            break
        if cur <= ALIGN_TOL_DEG:
            return moves, cur
        want = sign * float(np.clip(cur, -45.0, 45.0))
        got = _wrist_turn(arm, want)
        moves += 1
        nxt = None if abs(got) < 1e-6 else _meas()
        print(f"    转正：5 号拧 {want:+.1f}°（实际 {got:+.1f}°）→ {label} {cur:.1f}° → "
              f"{'—' if nxt is None else f'{nxt:.1f}°'}")
        if abs(got) < 1e-6:                        # 5 号顶到行程头
            print("    转正：停因 = 5 号已经顶到行程端点，一步没拧动")
            break
        if nxt is None:
            print(f"    转正：停因 = {label}不出来了")
            break
        if nxt > cur + 0.5:                        # 这一步拧反了，下一趟掉头
            sign = -sign
        elif nxt > cur - 0.5:                      # 拧了不动：到头了
            print(f"    转正：停因 = 拧了 {got:+.1f}° 但误差没降（{cur:.1f}° → {nxt:.1f}°）")
            break
        if nxt < best:
            best = nxt
    cur = _meas()
    return moves, (float("nan") if cur is None else cur)

def _align_model(arm, closing, grip=-1.0) -> tuple[int, float]:
    """老口径：模型里的 `tool_x` 对着 `closing`，笛卡尔闭环。画面那条量不出来时用。"""
    hold_at = arm.tcp().copy()
    base = np.asarray(closing, dtype=np.float64).copy()

    def _rot(deg: float) -> np.ndarray:
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ base

    prev = None
    for k in range(0, ALIGN_MAX_STEPS, ALIGN_CHUNK):
        left = _align_deg(arm, base)
        if left is not None and left <= ALIGN_TOL_DEG:
            return k, left
        prev = left
        tgt = base if prev is None else _rot(float(prev))
        follow.hold(arm, ALIGN_CHUNK, grip=grip, closing=tgt, at=hold_at,
                    pos_slew=POS_SLEW, rot_slew=ALIGN_ROT_SLEW)
    left = _align_deg(arm, base)
    return ALIGN_MAX_STEPS, (float("nan") if left is None else float(left))
