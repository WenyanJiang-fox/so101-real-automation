"""编排：按顺序调上面这些，注册成 `pick_real`。"""
from __future__ import annotations

from core import follow, geom, shapes, vision
from core.follow import POS_SLEW, ROT_SLEW
from core.interfaces import Skill
from core.params import (ALIGN_CHUNK, ALIGN_MAX_STEPS,            # noqa: E402
                         ALIGN_ROT_SLEW, ALIGN_TOL_DEG, CARRY,
                         GENTLE, GRIP_BIAS, GRIP_K, GRIP_RATE,
                         GRIP_SLACK, LIFT_Z, POS_SCALE,
                         SLOW, SPEED, STUCK_TOL,
                         acc_amax, speed_vmax)
from core.registry import register
from core.round_state import RoundState
import numpy as np
from . import knobs
from . import vlm_gate as _vlm_gate
from .knobs import APPROACH_MIN_ABOVE_M, CLOSE_START_STEPS, FIX_CLEAR_REAL, GRASP_Z_MARGIN_REAL, GRASP_Z_OVER_TOP_REAL, HOME_AFTER_PLACE, PLACE_AFTER_PICK, RESUME_TOL_DEG
from .align import _align, _set_clearance
from .pose import _align_deg, _declared_half, _fold90, _h, _vertical, _wrist
from .gates import (_arm_off_checkpoint, _descend_to_grasp, _drop_checkpoint,
                    _load_checkpoint, _raise_to_safe, _resume_on, _save_checkpoint,
                    _start_at, _stop_here)
from .vla_end import enabled as _vla_end_on
from .rulers import _box_deg_px, _jaw_axis_deg, _jaw_axis_model_deg, _jaw_shape, _save_look, _yaw_from_px
from .grasp import _close_until_stall, _grip_hold, _gripper_deg
from .motion import _descend, _goto, _goto_reach, _trace
from .look import _look, _sweep
from .place import _place_real

#: 末端交给 VLA 的岔口。空串 = 走脚本那一整套（默认）；`"after_align"` = 脚本只管到
#: 「腕子对上方块侧面」，之后的**下探 → 横向对齐 → 合爪**交给 π0.5。
#: 由 `skills/pick_real/strategy.py::apply()` 按 `configs/strategies.json` 里那一行覆盖。
#: 和环境变量 `PICK_VLA_END` 是两个岔口：那个在「下探到底」之后接，这个在**下探之前**接。
VLA_END = ""


def _descend_planned(arm, percep, center):
    """竖直扎到底那一截：**终点按 `decide.plan` 现算**，扎完把实测偏差记进账本。

    和直接调 `_descend_to_grasp` 的差别只有两处：
      · 扎之前把 `gates.GRASP_Z_FINAL_REAL` 换成「默认值 + 上一趟实测超深」的补正量；
      · 扎完读回 TCP，算这一趟实际比指令深/浅多少，记下来给下一趟用。

    **`GRASP_Z_FINAL_REAL` 扎完要还原**：断点、面板显示、别的调用方读的都是策略表
    那一份，这一趟的补正只对这一次下扎有效。
    """
    from . import decide, gates                                  # noqa: PLC0415

    shape = type(getattr(percep, "shape", None)).__name__
    p = decide.plan(shape)
    keep = gates.GRASP_Z_FINAL_REAL
    gates.GRASP_Z_FINAL_REAL = float(p["grasp_z_final_mm"]) / 1000.0
    try:
        ok, tcp = _descend_to_grasp(arm, center)
    finally:
        gates.GRASP_Z_FINAL_REAL = keep
    z = float(np.asarray(tcp, dtype=np.float64).reshape(-1)[2])
    off = (z - (float(center[2]) + float(p["grasp_z_final_mm"]) / 1000.0)) * 1000.0
    decide.remember(shape, overshoot_mm=off)
    print(f"    [现算] 下探终点 {float(p['grasp_z_final_mm']):+.1f}mm（{p['why']}）；"
          f"实测停 {z * 1000:+.1f}mm，比指令 {off:+.1f}mm", flush=True)
    return ok, tcp


def _keep_near_anchor(c, tag: str):
    """新认出来的中心离上一次的锚点太远 → 退回锚点。

    方块只会被爪子碰动一点（2026-10-20 实测 8 趟，单趟最多 33.3mm），跳得远的那一眼
    是认错了。锚点由重试策略写进 `knobs.ANCHOR_XY`；没锚点就原样返回。
    """
    if c is None or knobs.ANCHOR_XY is None:
        return c
    from ppo.guard import clamp_center                        # noqa: PLC0415

    out, _d, _jumped = clamp_center(c, knobs.ANCHOR_XY, tag=tag)
    return out


def _snap_after_close(cams, verdict) -> None:
    """合爪之后马上留一张原始帧（一个像素都不画），成败都留。

    下探那一张（`JAW_RAW_IMG`）是合爪**前**的，看不出爪里到底有没有、正不正；这一张
    才有那个信息，用来训「合爪后夹住没有」的判官。没相机就静默跳过。
    """
    if cams is None:
        return
    cam = _wrist(cams)
    if cam is None:
        return
    try:
        import json                                                   # noqa: PLC0415
        import time                                                   # noqa: PLC0415
        from pathlib import Path                                      # noqa: PLC0415

        import cv2                                                    # noqa: PLC0415

        rgb = np.asarray(cam.shot())
        cv2.imwrite(str(knobs.JAW_AFTER_IMG),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # 判据跟图一起落盘：归档按它命名，不靠整集成败去猜。
        Path(knobs.JAW_AFTER_JSON).write_text(
            json.dumps({"verdict": str(verdict), "t": time.time()}),
            encoding="utf-8")
    except Exception as exc:                                          # noqa: BLE001
        print(f"  [VLM·记] 合爪后那一帧没留下（{exc}）", flush=True)


def _close_and_lift(arm, ctrl, cal, state, index, *, grasp_z, empty_deg, open_deg,
                    center, hover, closing, percep=None, cams=None,
                    already_closed: bool = False) -> dict:
    """微调过了 → 合爪 → 判夹住没有 → 抬升到搬运高度。

    **两条路共用这一份**：一种是这一趟从头走完的（`PickRealSkill.run`），一种是从断点
    接上的（`PickRealSkill._resume_pick`，见文件头 `CHECKPOINT` 那一段）。判据、日志、
    返回的字段一个字不差，差别只在**进来之前那几步**走没走。

    参数全是前面算出来的：`center` 盒子中心、`hover` 悬停点（没夹住时退到这儿）、
    `closing` 合爪轴、`grasp_z` 下探停在哪、`empty_deg`/`open_deg` 这一趟的夹爪行程两端。
    **行程两端带上、不在这儿重量** —— 这时候两个爪子正夹在方块两侧，合一次空爪就是拿
    爪去夹方块。

    **`already_closed=True` 是 VLA 末端那一支进来的路**（`vla_end.run`）：爪子已经由
    模型合上了，这里**不再合一次**（再合会把「基本没动」误判成扭矩没使能），直接判。
    """
    from grasp_check import GraspMonitor, classify_ref                   # noqa: PLC0415
    from servo_bridge import read_gripper_deg                            # noqa: PLC0415

    # ---- 6. 合爪：合到压不动，再拿角度判夹住没有 ----
    # **合爪那一刻的位姿单独记一份**（下探到底之后还修过轴偏，`check` 记的那个数不是它）。
    # 「失败那几趟差多少偏移」全靠这个数和方块中心算相对位姿。
    try:
        _tcp_close = [round(float(v) * 1000.0, 1)
                      for v in np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)[:3]]
    except Exception:                                               # noqa: BLE001
        _tcp_close = None
    state.seg = "合爪"
    if already_closed:
        stop, info = "vla", {"why": "VLA 末端已经合过爪", "steps": 0,
                             "q0": float(arm.grip_q()), "q": float(arm.grip_q())}
    else:
        stop, info = _close_until_stall(arm, closing, empty_deg=empty_deg)
    grip_deg, _src = _gripper_deg(arm, ctrl)
    verdict = classify_ref(grip_deg, empty_deg, cal, open_deg)
    print(f"    合爪：{stop}（{info['why']}，{info['steps']} 步）"
          f"  开合角 {info['q0']:+.3f} → {info['q']:+.3f} rad"
          f"  6 号舵机 {grip_deg if grip_deg is None else f'{grip_deg:.1f}°'}（{_src}）"
          f"  → 判 {verdict}")
    _snap_after_close(cams, verdict)
    if verdict != "held":
        # **光看角度分不清「夹空了」和「压根没动」—— 两者都停在基准附近。**
        # 分得开的是**开口变没变**：真合爪是「从张开一路合过来」，`q` 离 `q0` 很远；
        # 一动没动说明舵机收到目标位置没出力 —— 最常见是扭矩没使能（`real_home.py`：
        # 使能位断电才清，harness 自己不开）。这一条单独报出来，别让人去怀疑盒子。
        didnt_move = (not already_closed
                      and abs(float(info["q"]) - float(info["q0"])) < 1e-6)
        # 退回去的那一段顺手把爪张开（`grip=1.0`）—— 现在是半合着的，直接横着退，
        # 指垫会贴着盒子侧面一路刮过去。
        _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
        why = (f"没夹住：开口从头到尾没变过（{info['q0']:+.3f} → {info['q']:+.3f} rad），"
               f"夹爪没动过（扭矩没使能？）" if didnt_move else
               f"没夹住（6 号舵机 {grip_deg}°，空抓基准 {empty_deg:.1f}°，判 {verdict}）")
        return {"ok": False, "why": why,
                "close": info, "grip_deg": grip_deg, "empty_deg": empty_deg,
                "verdict": verdict, "tcp_close_mm": _tcp_close}

    state.held = index
    state.grip_ref = float(arm.grip_q())
    print(f"    夹住了：开合角 {state.grip_ref:+.3f} rad，"
          f"按这个开口抬升（`_grip_hold` 守着它）")

    # ---- 7. 抬升 ----
    # **走关节空间 `_goto`，不再走 `follow.follow`**（用户 2026-10-18：「抓取成功后，
    # 往往需要很长时间才能重新有动作然后放置」）。笛卡尔那条一步要问三次舵机，实测约
    # 5 步/秒，抬 10cm 排了 332 步；`panel_20261018c.log` 那一趟从合爪到判 stall 白跑了
    # 约 16 秒，TCP 一点没动（z −17.5 → −19.0mm）。搬运、下探用的都是 `_goto`，同距离
    # 3 趟约 4 秒。`grip=None` = 保持此刻的开口不动。
    state.seg = "抬升到搬运高度"
    mon = GraspMonitor(ctrl, cal, samples=3, empty_deg=empty_deg,
                       open_deg=open_deg)
    mon.mark_peak()

    up = _goto(arm, np.array([center[0], center[1], LIFT_Z], dtype=np.float64),
               vmax=speed_vmax(CARRY), grip=None, tol=0.012)
    mon.poll()
    end_deg, _esrc = _gripper_deg(arm, ctrl)
    end_verdict = classify_ref(end_deg, empty_deg, cal, open_deg)
    dropped = bool(mon.dropped())
    z_now = float(arm.tcp()[2])
    print(f"    抬升：{up}  TCP z={z_now:.3f}（该到 {LIFT_Z:.2f}）"
          f"  6 号舵机 起 {mon.peak if mon.peak is None else round(mon.peak, 1)}°"
          f" → 末 {end_deg if end_deg is None else round(end_deg, 1)}°（{_esrc}）"
          f"  判 {end_verdict}  掉落判据 {dropped}")

    # **成没成只看「夹住没有、掉没掉」**（用户 2026-10-16：「这一趟抓成功了呀」）。
    # 原来还搭着 `up == "reached"` —— 抬升没走到 `LIFT_Z` 就把整趟判失败，可那是
    # 「抬到多高」的事，跟「抓没抓住」是两件事。那一趟（`…/bi79lknh7.output` 第 382~388 行）
    # 6 号 46.6°、判 held、掉落判据 False，却被这一条判死。抬升走到哪照旧打在下面。
    ok = bool(end_verdict == "held" and not dropped)
    if ok and up != "reached":
        print(f"    ⚠ 抓着没掉，但抬升只走到 {z_now * 1000:+.0f}mm（该到 "
              f"{LIFT_Z * 1000:+.0f}mm，停因 {up}）—— 这不算这一趟失败")
    out = {"ok": ok, "grasp_z": grasp_z, "grip_deg": end_deg,
           "empty_deg": empty_deg, "start_deg": mon.peak,
           "verdict": end_verdict, "dropped": dropped,
           "lift": up, "tcp_z": z_now, "tcp_close_mm": _tcp_close}
    if not ok:
        out["why"] = (f"没夹住/掉了（6 号舵机 {end_deg}°，判 {end_verdict}，"
                      f"掉落判据 {dropped}；抬升停因 {up}）")
        return out
    # ---- 抓成了：搬到随机落点放下去（用户 2026-10-17）----
    # **放在这里、不放在 `run` 里**：这一段两条路（从头走 / 从断点接上）共用，
    # 放进去两条就都有；`percep` 是给「半个边长」用的（`_box_half_m`）。
    if PLACE_AFTER_PICK and percep is not None:
        out["place"] = _place_real(arm, percep, center, grasp_z=grasp_z, state=state)
        # ---- 放好了就回初始位置（用户 2026-10-18）----
        # **只在真放成了才收臂**：没放成（落点抽不到 / 没落到位）方块可能还在爪里，
        # 这时开走会把方块带走或者甩掉。走 `skills/home_real.py` 同一条（`home_q`）。
        if HOME_AFTER_PLACE and out["place"].get("ok"):
            from real import joint_move as jm                       # noqa: PLC0415

            # **先抬高、再合爪、最后收臂**（用户 2026-10-18）。抬高就抬到搬运高度
            # `HOME_RAISE_Z_M`；已经在上面就不抬。合爪跟在抬高后面 —— 张着爪从桌面上
            # 收臂会一路蹭桌沿、蹭方块。
            tcp_now = np.asarray(arm.tcp(), dtype=np.float64)
            dz = float(knobs.HOME_RAISE_Z_M) - float(tcp_now[2])
            if dz > 0.001:
                state.seg = "回初始位置：先抬高"
                print(f"    回初始位置：先竖直抬高 {dz * 1000:.0f}mm（到 "
                      f"{knobs.HOME_RAISE_Z_M * 1000:.0f}mm），抬起来再合爪")
                # **分 30mm 一小步**（同面板 `_home`）：一口气抬到目标那一下逆解残差
                # 会越过容差、一步不走；每小步都从读回来的姿态重解。
                for _ in range(6):
                    tcp_now = np.asarray(arm.tcp(), dtype=np.float64)
                    dz = float(knobs.HOME_RAISE_Z_M) - float(tcp_now[2])
                    if dz <= 0.001:
                        break
                    _goto(arm, np.array([tcp_now[0], tcp_now[1],
                                         float(tcp_now[2]) + min(0.03, dz)],
                                        dtype=np.float64),
                          vmax=speed_vmax(SLOW), grip=None)
                    after = float(arm.tcp()[2])
                    if after - float(tcp_now[2]) < 0.001:
                        print(f"    ！抬不动了：停在 {after * 1000:+.0f}mm"
                              f"（要到 {knobs.HOME_RAISE_Z_M * 1000:+.0f}mm）—— 不再硬抬")
                        break
            q0 = arm._ticks_to_qpos(arm.read_ticks())
            q_close = np.array(q0, dtype=np.float64, copy=True)
            q_close[5] = float(arm.grip_lo)
            state.seg = "回初始位置：合爪"
            jm.goto_q(arm, q_close, q_from=q0)
            state.seg = "回初始位置"
            q_end = jm.goto_q(arm, jm.home_q(arm))
            print(f"    回初始位置：放好了，收臂 —— 走到 "
                  f"{np.round(np.degrees(np.asarray(q_end)), 1)}")
            out["home"] = True
    # **一趟完整抓放成功 → 盘上那份断点作废**（用户 2026-10-18）：不然下一次按
    # 「从断点继续」会认领一个早就走完的姿态。停手（`arm.terminated`）不动它 ——
    # 那正是要接着走的那一份。
    if out.get("ok") and (out.get("place") or {}).get("ok") \
            and not getattr(arm, "terminated", False):
        _drop_checkpoint()
    return out


def _poke_check_after_descend(percep, text: str, center):
    """下探刚走完，**再认一眼方块**：中心比下探之前冻住的那个挪了多少毫米。

    **能判的是「方块被推走了」**：方块在桌上不动，中心挪了就是爪子戳的。
    **判不了「只压住、没推动」**：没有力/接触传感器，舵机那点位置误差分不清是顶到东西
    还是重力下垂。所以这里只报数，不判死、不改后面怎么走。

    **返回 `(挪了多少毫米, 这一眼量到的中心)`**（2026-10-18）。第二个数是后面那一趟
    「只修轴偏」要用的方块中心：原来它拿的是**下探之前**冻住的那个，方块被推走之后
    那个数就指在空处 —— 实测挪了 46.6mm，那一趟还照旧把它当目标，轴偏「判过」了也
    是假过。认不出来（画面里没方块 / 感知抛了）返回 `(None, None)` 并说明，调用方
    沿用旧中心，不拿旧数糊弄成「没挪」。
    """
    try:
        c_after, _facts = percep.locate(text, guess=center)
    except Exception as exc:                                        # noqa: BLE001
        print(f"    下探之后认方块出错：{type(exc).__name__}: {exc}")
        return None, None
    if c_after is None:
        print("    下探之后没认到方块 —— 戳没戳到这一眼判不了")
        return None, None
    c_after = np.asarray(c_after, dtype=np.float64).reshape(-1)
    # **跳得太远就不是方块在动**（`ppo/guard.py`）：方块被爪子推一下最多挪几十毫米。
    # 这一眼按认错处理，返回 `None` 让调用方沿用下探之前冻住的那个中心。
    if knobs.ANCHOR_XY is not None:
        from ppo.guard import clamp_center                    # noqa: PLC0415

        c_after, _jump, jumped = clamp_center(c_after, knobs.ANCHOR_XY, tag="下探之后")
        if jumped:
            print("    · 这一眼不认，沿用下探之前冻住的那个中心")
            return None, None
    d = (c_after[:2] - np.asarray(center, dtype=np.float64).reshape(-1)[:2])
    mm = float(np.linalg.norm(d)) * 1000.0
    print(f"    下探之后方块中心 {np.round(c_after * 1000, 1)}mm，"
          f"比下探之前冻住的那个挪了 {mm:.1f}mm（只看数，阈值由人定）")
    return mm, c_after

def _stopped(arm) -> dict | None:
    """停手了没有（`arm.terminated`）。停了就交出这一趟的收工结果。

    **`terminated` 只让轨迹停住，编排原来会带着停下来的姿态继续往下走**（用户
    2026-10-18）：下探停下之后照旧去合爪、微调停下之后照旧再挪一步。每一处走位
    回来都问一遍它，停了就收工。`give_up` 让策略层别接着下一趟。
    """
    if getattr(arm, "terminated", False):
        print("    ✗ 停手了 —— 这一趟不再往下走")
        return {"ok": False, "why": "停手", "give_up": True}
    return None


@register("skill", "pick_real")
class PickRealSkill(Skill):
    """真机上抓那只棕色盒子。判据全在 `grasp_check` 和 `real/perception.py` 里。"""

    name = "pick_real"

    def run(self, arm, cams, state=None, index: int = 0, percep=None,
            instruction=None, **kw) -> dict:
        """抓一次。返回 `{'ok': 成没成, 'why': ..., 'grip_deg': ..., ...}`。

        只往下走一趟，**不重试** —— 抓几次归策略层数（`impls/policy_real.py`），
        里面再自救一把，「抓满 3 次」就数不准了。

        **带 `--resume-pick`（环境变量 `PICK_RESUME=1`）时会先找盘上的断点**（见文件头
        `CHECKPOINT` 那一段）：盘上有、臂也还在那个姿态上，就**从合爪前微调那一跳接着走**
        （`_resume_pick`），不再重走回位 → 起手合爪 → 进场 → 看一眼 → 纠位 → 转正 → 下探。
        """
        from grasp_check import (GraspMonitor, classify_ref, empty_margin,  # noqa: PLC0415
                                 load_thresholds, measure_empty, measure_open)
        from servo_bridge import read_gripper_deg                           # noqa: PLC0415

        cal = load_thresholds()
        ctrl = getattr(arm, "ctrl", None)
        if ctrl is None:
            raise RuntimeError("真机抓取要靠 6 号舵机读数判夹没夹住，"
                               "`arm.ctrl` 现在是 None（串口关了？）")
        if percep is None or not hasattr(percep, "shape"):
            return {"ok": False, "why": "真机抓取要感知给世界坐标（percep）",
                    "give_up": True}
        state = state if state is not None else RoundState()
        # **手里抓着东西就不续跑**（用户 2026-10-18）：方块还在爪里，重走抓取那一段
        # 会把它扔掉或者撞桌子。面板的按钮这时也灰着（`rig.resume_blocked`），这里是
        # 最后一道 —— 只拦续跑，正常「跑一趟」不受影响（那一趟自己会重判）。
        if state.held is not None and (_start_at() is not None or _resume_on()):
            return {"ok": False, "why": "手里抓着东西，不能续跑", "give_up": True}
        sh = percep.shape

        # **合爪前那两道判据要有东西可量。** 三圈爪子轮廓没描过（或者是在别的画幅上
        # 描的）时，`_set_clearance` 一定过不去 —— 那还不如在这儿就收工，别白跑一趟
        # 机械臂动作、白费一次尝试的时间预算。
        try:
            _jaw_shape(_wrist(cams))
        except RuntimeError as e:
            print(f"    ✗ {e}")
            return {"ok": False, "why": str(e), "give_up": True}

        # ---- 0. 断点续跑？----
        # **判在最前面**，因为下面每一步都在动臂：起手那一下 `follow.hold` 合爪、进场、
        # 纠位、转正、下探。断点就是臂现在这个姿态，这几步一步都不能走 —— 尤其是起手
        # 合爪，在这个高度上合上去就是直接去夹方块。
        if _resume_on():
            ck = _load_checkpoint()
            if ck is None:
                print("    ！这一趟带着 `--resume-pick`，但盘上没有断点（或者读不出来）"
                      " —— 从头走这一趟")
            else:
                off = _arm_off_checkpoint(arm, ck)
                if off > RESUME_TOL_DEG:
                    print(f"    ✗ 盘上那份断点是 {ck.get('created')} 写的，但臂已经不在"
                          f"那个姿态上了（最远的关节差 {off:.1f}° > 判据 "
                          f"{RESUME_TOL_DEG:.0f}°）—— 不续跑，如实收工")
                    return {"ok": False, "why": f"臂不在断点姿态上（最远的关节差 {off:.1f}°）",
                            "give_up": True}
                print(f"    ⤒ 盘上有断点（{ck.get('created')} 写的），臂还在那个姿态上"
                      f"（最远的关节差 {off:.1f}°）—— 从合爪前微调那一跳接着走")
                return self._resume_pick(arm, cams, percep, state, index, ctrl, cal, ck)

        # ---- 0.1 面板的「从断点继续」（`PICK_START_AT`）----
        # 用户 2026-10-18：自动跑着停手 / 关掉之后，手动动过臂，再按当前段接着走。
        #   `fine`  → 断点对得上就续微调，对不上退成「对准」；
        #   `close` → 微调已经判过，跳过微调、直接竖直下探一截再合爪；
        #   `align` → 抬到安全高度，然后从头走（进场 / 看一眼 / 纠位 / 转正 / 下探 …）。
        start_at = _start_at()
        if start_at in ("fine", "close"):
            ck = _load_checkpoint()
            off = None if ck is None else _arm_off_checkpoint(arm, ck)
            if ck is not None and off <= RESUME_TOL_DEG:
                print(f"    ↻ 续跑入口「{start_at}」：断点对得上（{ck.get('created')} 写的，"
                      f"最远的关节差 {off:.1f}°），从断点接着走")
                return self._resume_pick(arm, cams, percep, state, index, ctrl, cal, ck,
                                         close_only=(start_at == "close"))
            why = "盘上没有断点" if ck is None else f"断点姿态对不上（最远的关节差 {off:.1f}°）"
            print(f"    ↻ 续跑入口「{start_at}」：{why} —— 退成「对准」入口")
            start_at = "align"
        if start_at == "align":
            # **必须先抬到安全高度**：这会儿爪尖可能就在方块两侧，先合空爪会夹到方块。
            _raise_to_safe(arm)

        # ---- 0. 行程两端：合一次空爪、张一次到底，量出这一趟的两头在哪 ----
        # **这就是自适应的那一步。** 后面所有的「夹住没有」都只看**读数是不是卡在这两头
        # 中间**，不再问「这只盒子该落在哪个角度区间」—— 换盒子、换爪面都不用重标。
        # **两头先拿到，再动臂。** 拿不到就当场收工：不然等于白跑一整趟机械臂动作，
        # 还白耗掉一次尝试的时间预算。
        empty_deg = getattr(arm, "empty_deg", None)
        if empty_deg is None:
            state.seg = "行程两端 → 合一次空爪、再张到底"
            before = read_gripper_deg(ctrl)
            empty_deg = measure_empty(ctrl)
            open_deg = measure_open(ctrl, tick=arm.grip_open_tick)
            if empty_deg is None or open_deg is None:
                return {"ok": False, "why": "6 号舵机读不到数，行程两端建不起来",
                        "give_up": True}
            arm.empty_deg = empty_deg
            arm.open_deg = open_deg
            print(f"    行程两端：合到底 {empty_deg:.1f}°  /  张到最大 {open_deg:.1f}°"
                  f"（判「夹住」= 读数落在 "
                  f"{empty_deg + empty_margin(cal):.1f}° ~ "
                  f"{open_deg - empty_margin(cal):.1f}° 之间）")
            # **只提醒，不拦。** 合爪前后的角度几乎没变，多半是舵机没出力（扭矩没使能，
            # 见 `real_home.py`），那样后面每一趟都会判「没夹住」。但也有一种正当情况：
            # 爪本来就是合着的。所以这里只把话摆出来，收不收工由人看日志决定。
            if before is not None and abs(empty_deg - before) <= 2.0:
                print(f"    ！合爪指令下去了角度没动（{before:.1f}° → {empty_deg:.1f}°）"
                      f"—— 6 号舵机没出力（扭矩没使能？），或者爪本来就是合着的")
        else:
            open_deg = getattr(arm, "open_deg", None)
            print(f"    行程两端：沿用上一趟实测的 合到底 {empty_deg:.1f}°"
                  f" / 张到最大 "
                  f"{'没记下' if open_deg is None else f'{open_deg:.1f}°'}")

        # ---- 0.5 起步前先把爪子合上 ----
        # `measure_open()` 是张到底量行程的那一下，量完爪子就停在**张着**的位置；
        # 第二次、第三次尝试则是上一趟留下的姿态（合爪到死、或者张开着退回来）。
        # **不能靠 `_goto` 的 `grip=-1.0` 去合**：那只是在 `_grip_cmd` 上按 0.2 rad/步
        # 往回合，全程 1.86 rad 要 ~10 步 —— 而那十步正好是臂从初始位往工位挪的
        # 第一段。`out/run_real_9.log` 第 9~12 行拍到的就是这个（开合 +1.705 → -0.153）。
        # 张开的爪 55.9mm 宽、合到底 30.9mm，宽出来的那 25mm 全蹭在进场那一段上。
        # 这里先用 `drive()` 把它真正合死，臂再动。
        follow.hold(arm, CLOSE_START_STEPS, grip=-1.0, at=arm.tcp(),
                    pos_slew=POS_SLEW, rot_slew=ROT_SLEW)

        table_z = geom.real_table_z()
        # 进场用的起手坐标：**配置里那个「盒子大致摆在哪」**，不是量出来的。这时候
        # 还没看，不知道盒子在不在那儿 —— 后面看一眼量准了会纠回来。
        if knobs.REPLAY is not None:
            guess = np.asarray(knobs.REPLAY["guess"], dtype=np.float64)
            print(f"    回放：起搜中心用 {np.round(guess * 1000, 0)}mm"
                  f"（不按配置里的 {np.round(geom.real_pick_xy() * 1000, 0)}mm）")
        elif knobs.GUESS_XY is not None:
            # **目标就是上一趟放下的地方**（用户 2026-10-20）：面板在上一趟摆件成功之后
            # 把落点写进 `knobs.GUESS_XY`。这里只拿它当起搜中心 —— 下面「初始位置先看
            # 一眼」和进场之后那一量都照旧，画面说是哪儿就是哪儿。
            guess = np.array([float(knobs.GUESS_XY[0]), float(knobs.GUESS_XY[1]),
                              table_z + float(sh.rest_z)], dtype=np.float64)
            print(f"    起搜中心用上一趟放下的位置 {np.round(guess[:2] * 1000, 0)}mm"
                  f"（配置里写的是 {np.round(geom.real_pick_xy() * 1000, 0)}mm）")
        else:
            guess = np.array([*geom.real_pick_xy(), table_z + float(sh.rest_z)])

        # ---- 0.9 **先站着看一眼**（用户 2026-10-17：「初始位置的视野也要找目标」）----
        # 原来进场点就是配置里那个「盒子大致摆在哪」，走过去才看。实测那一轮（面板日志
        # 2026-10-17 03:27）：**站在初始位置时刷新里看得见盒子**（长方体 16×20×16，
        # 重合度 0.82），走到进场点反倒「找到 False」，接着前后左右四个方向扫、再抬头
        # 拉远，全空，19.3s 收工。
        #
        # 现在改成**先看不走**：初始位置这一帧认得出，就把进场点摆到量出来的那个位置上
        # （`guess` 换掉，下面的进场照旧走 `guess + 80mm`）；认不出就一个字节都不变、
        # 照旧用配置那个数。**这一步不动臂** —— 只是拿当前这一帧算一次，失败也不亏。
        text = instruction or "pick up the brown box"
        if knobs.REPLAY is None:
            state.seg = "初始位置先看一眼"
            print("    初始位置先看一眼（不动臂）：")
            c0, y0, f0 = _look(arm, cams, percep, text, guess)
            if c0 is not None and y0 is not None:
                c0 = _keep_near_anchor(c0, "初始位置")
                print(f"    ↻ 初始位置看得见：x={c0[0] * 1000:.0f} y={c0[1] * 1000:.0f}"
                      f" z={c0[2] * 1000:.0f}mm yaw {np.degrees(y0):+.1f}°"
                      f" → 进场点改用它（配置里写的是 "
                      f"{np.round(guess[:2] * 1000, 0)}mm）")
                guess = np.array([c0[0], c0[1], guess[2]], dtype=np.float64)
            else:
                print("    · 初始位置没认出来，照旧用配置里那个位置进场")

        # ---- 1. 进场：到大致位置上方 `HOVER_ABOVE_M`（默认 80mm）----
        state.seg = "进场 → 盒子大致位置上方"
        _enter = guess + np.array([0.0, 0.0, float(knobs.HOVER_ABOVE_M)])
        out, _enter = _goto_reach(arm, _enter, vmax=speed_vmax(SPEED),
                                  floor_z=float(guess[2]) + APPROACH_MIN_ABOVE_M)
        print(f"    进场：{out}  TCP {np.round(arm.tcp(), 3)}"
              f"（该到 {np.round(_enter, 3)}）")
        if out == "term" or getattr(arm, "terminated", False):
            # `_goto` 现在只交 "reached"/"stall"，停手要看 `terminated`（用户 2026-10-18）。
            return {"ok": False, "why": "停手", "give_up": True}
        hit = _stop_here("进场", arm, f"（进场 {out}）")
        if hit is not None:
            return hit

        # ---- 2. 看一眼：画面里那只盒子在基座系哪儿、朝哪 ----
        state.seg = "看一眼 → 量盒子的世界坐标"
        if knobs.REPLAY is not None:
            # **回放分支：不看**。中心与朝向用那一趟记下来的值，`_look` / `_sweep`
            # 一步都不走 —— 真机那一趟量出来的就是这两个数。
            center = np.asarray(knobs.REPLAY["center"], dtype=np.float64)
            yaw = float(np.radians(knobs.REPLAY["yaw_deg"]))
            facts = {"est": None, "shape": f"回放：{knobs.REPLAY.get('src', '')}",
                     "size": ""}
            print(f"    回放：不看了，直接用那一趟量出来的"
                  f" x={center[0] * 1000:.0f} y={center[1] * 1000:.0f}"
                  f" z={center[2] * 1000:.0f}mm  yaw {np.degrees(yaw):+.1f}°")
        else:
            center, yaw, facts = _look(arm, cams, percep, text, guess)
        if center is None:
            # **原地不成，先左右看看，再抬头把视野推出去。** 机位一动，画面里那坨色块
            # 的形状、遮挡、边角全变了。上一轮首跑就是卡在这儿（1 个棕色候选、矩形度
            # 0.50 < 门槛 0.78）。两段扫完还没认出来就收工 —— 不再原地重试。
            print("    原地那一眼没认出来，先前后左右，再抬头拉远")
            state.seg = "找目标：前后左右 → 抬头拉远"
            center, yaw, facts = _sweep(arm, cams, percep, text, state)
            hit = _stopped(arm)
            if hit is not None:
                return hit
        if center is None:
            why = facts.get("no_world_why") or facts.get("why") or "没认出来"
            print(f"    ✗ {why}")
            return {"ok": False, "why": f"找不着（{why}）", "give_up": True}
        center = _keep_near_anchor(center, "进场之后")
        print(f"    量出来：x={center[0] * 1000:.0f} y={center[1] * 1000:.0f}"
              f" z={center[2] * 1000:.0f}mm  yaw {np.degrees(yaw):+.1f}°"
              f"（起搜中心 {np.round(guess * 1000, 0)}mm，挪了 "
              f"{np.round((center - guess) * 1000, 0)}mm）")
        # **存下这一眼的画面。** 它就是这个 `center` 的来源 —— 后面冻住的那个 3D 中心、
        # 下探的 `aim`、`clear` 全是从它算的。回放分支没有真画面，`_save_look` 自己跳过。
        if knobs.REPLAY is None:
            _save_look(cams, facts)
        # 「看一眼」这一段走完了（`_look` 或 `_sweep` 出来都算）。停在这儿的话，臂上
        # 一步是什么姿态就是什么姿态 —— 只报这一眼量到了什么。
        hit = _stop_here("看一眼", arm,
                         f"（量到 x={center[0] * 1000:.0f} y={center[1] * 1000:.0f}"
                         f" z={center[2] * 1000:.0f}mm yaw {np.degrees(yaw):+.1f}°）")
        if hit is not None:
            return hit

        # **形状和尺寸这一眼已经估出来了，往下全用它**（`real/perception.py` 的
        # `estimate_shape` 那一支）：进场高度已经过去了，接下来转正挑哪一面
        # （`sh.normal`）、下探终点都吃这个形状。估不出来才接着用配置里那一行 ——
        # 哪个是哪个感知那边已经并排打出来了，这里只再说一遍往下用的是哪个。
        est = facts.get("est")
        if est is not None:
            sh = shapes.from_spec(est)
        print(f"    往下用这个形状：{facts.get('shape')} {facts.get('size')}"
              f"（{'画面估出来的' if est is not None else '配置里写的'}）")

        # ---- 3. 纠位：挪到**量出来的那个位置**上方，再停稳 ----
        # 进场是照配置那个坐标走的，盒子偏了就不在正上方了。不纠的话下探那一段
        # 就从「只剩纯下降」变成「一边横移一边下降」，横移是贴着盒顶扫过去的。
        state.seg = "挪到盒子正上方"
        hover = center + np.array([0.0, 0.0, float(knobs.HOVER_ABOVE_M)])
        # **先垂直、正上方，再转腕**（用户 2026-10-18）。转正那一步是绕工具轴拧 5 号：
        # 工具轴歪着的时候，两爪是在一个斜锥面上扫过去的 —— 实测进场这个姿态工具轴离
        # 垂直 43.8°，在它上面拧 65° 等于让两爪横扫一大圈。先把这个姿态摆成爪尖朝下
        # （`_vertical`：工具 z 钉成竖直、方位角保住），后面的转正才是绕竖直轴的纯偏航。
        # 位置和姿态**一次走位搞定**（`quat=`），不先挪上去再原地摆一次 —— 少一趟动作。
        q_down, off_v = _vertical(arm.quat())
        print(f"    纠位：去盒子正上方、爪尖摆垂直（现在离垂直 {off_v:.1f}°）")
        # **姿态权重给高**（用户 2026-10-18：「为啥不改？」—— 上一趟走完还剩 9.0°）。
        # 默认 `OW=0.05` 是拿位置换姿态：离线探针（`_probe_vertical.py`，拿日志读回的
        # 关节当起点）同一个点解出来 ow 0.05 → 位置 1.0mm、离垂直 6.4°；
        # ow 0.2 → 位置 3.6mm、离垂直 0.8°（4 号 93.0°，离 95° 上限还有 2°）。
        # 位置那 3.6mm 由 `_goto` 的第二趟补；姿态这一趟只在这里要求，别处照旧。
        # **多给几趟**（用户 2026-10-18：「垂直到目标正上方」）：默认 3 趟上一趟差 5.1mm
        # 就收在 `stall`，那 5mm 让后面微调从偏心起手。5 趟到得了就到，到不了照样往下走。
        out, hover = _goto_reach(arm, hover, vmax=speed_vmax(SPEED), quat=q_down,
                                 ow=0.2, tries=5,
                                 floor_z=float(center[2]) + APPROACH_MIN_ABOVE_M)
        print(f"    纠位：{out}  TCP {np.round(arm.tcp(), 3)}"
              f"（该到 {np.round(hover, 3)}），爪尖离垂直 "
              f"{_vertical(arm.quat())[1]:.1f}°")
        if out == "term" or getattr(arm, "terminated", False):
            # `_goto` 现在只交 "reached"/"stall"，停手要看 `terminated`（用户 2026-10-18）。
            return {"ok": False, "why": "停手", "give_up": True}
        hit = _stop_here("挪到盒子正上方", arm, f"（纠位 {out}）")
        if hit is not None:
            return hit

        # ---- 4. 转正：把开合轴转到贴着盒子某一对侧面 ----
        # 挑哪一面由形状自己算（`core.shapes.Box.normal`）：四条侧面法向里**最贴近
        # 现在这条开合轴**的那条 —— 腕子转得最少。盒子摆的 yaw 是随机的，不转正就是
        # 拿指尖去顶棱。
        # **先拿画面把 `yaw` 纠回来**（2026-10-17）。`yaw` 是模型量（俯视剪影拟合），
        # 它歪多少，`sh.normal` 算出来的侧面法向就跟着歪多少 —— 而转正挑哪一面、下探
        # 的落点全挂在它上面。画面这条不经过它：相机和两爪同一个刚体，红区剪影离合爪轴
        # 多少度，就是世界系里离侧面多少度。实测那一眼：画面 27.6°、模型 1.8°。
        fix = _yaw_from_px(sh, center, yaw, _wrist(cams), percep)
        if fix is not None:
            yaw_px, a_meas, a_pred = fix
            print(f"    yaw 按画面纠回来：{np.degrees(yaw):+.1f}° → "
                  f"{np.degrees(yaw_px):+.1f}°（红区剪影 {a_meas:+.1f}°，"
                  f"模型在纠过的 yaw 上投出来 {a_pred:+.1f}°）")
            yaw = yaw_px
        # **三个都该说同一条线。** 模型这边（两爪尖投到画面）和描死的那条（`axis_dir_px`）
        # 对不上，`_box_deg_px` 就不能当「世界系里差多少度」用 —— 先把这条摆出来。
        ax_m = _jaw_axis_model_deg(arm, _wrist(cams))
        ax_t = _jaw_axis_deg(_wrist(cams))
        if ax_m is not None and ax_t is not None:
            print(f"    合爪轴在画面里：模型（两爪尖投出来）{ax_m:+.1f}°，"
                  f"描死的 {ax_t:+.1f}°（差 {_fold90(ax_m - ax_t):+.1f}°）")
        # **旋转对称的形状（圆柱/杯子）模型那条恒为 0**（`normal` 直接回当前开合轴）——
        # 那就不问模型、改问画面（红区剪影的边）。`use_px` 那条只在形状对称时才走。
        _sym = getattr(sh, "yaw_period_deg", 1.0) is None
        closing = np.asarray(sh.normal(yaw, arm.tool_x()), dtype=np.float64)
        state.seg = "腕子对上方块侧面"
        steps, left = _align(arm, closing, cam=_wrist(cams), percep=percep, use_px=_sym)
        # 转完把 `closing` 跟到新朝向（圆柱的 `normal` 就是当前开合轴）。不跟的话
        # `_descend` 会按旧朝向锁腕子，把刚转的那一下原样转回去。
        if _sym:
            closing = np.asarray(sh.normal(yaw, arm.tool_x()), dtype=np.float64)
        hit = _stopped(arm)
        if hit is not None:
            return hit
        px_left = _box_deg_px(_wrist(cams), percep)
        m_left = _align_deg(arm, closing)
        print(f"    转正：拧了 {steps} 次 5 号；画面那尺子量还差 "
              + ("量不出来" if px_left is None else f"{px_left:+.1f}°")
              + "，模型那条 "
              + ("量不出来" if m_left is None else f"{m_left:.1f}°")
              + f"（`ALIGN_TOL_DEG` 是 {ALIGN_TOL_DEG:.0f}°）")

        # ---- 5. 下探：一路降到盒子的中心高度 ----
        # **落点不是盒子中心，是「固定爪面贴着盒子那个侧面」的那一点。** 照
        # `pick.py:679-697` 那一段（仿真那一支的落点算式）。`arm.tcp()` 现在是**固定
        # 爪尖**（`real/arm.py` 的 `TCP_DELTA_BODY`），把固定爪尖对到盒子中心就等于让
        # 固定爪插进盒子轮廓里 —— `out/run_real_22.log` 三趟量到的盒子位置 (248,32) /
        # (245,66) / (240,73)，每趟被固定爪从顶面捅着推走几十毫米，合爪时嘴里是空的。
        #
        # 算式：落点 = 盒子中心 + side·(半边长 + 缝)·开合轴。`side` 取固定爪**现在待的
        # 那一侧**（它本来就在某一侧），所以是把固定爪往外摆到贴着盒子那面、缝留在外面。
        # 半边长问形状自己（`sh.support`）：盒子斜着的时候沿法向比半边长宽。
        #
        # 下探终点原来就取中心高度。**2026-10-15 用户改成「中心 + `GRASP_Z_MARGIN_REAL`」**：
        # `out/run_real_36.log` 那一趟爪尖落在估计上表面之下 7mm，横向让开时读数
        # 一动没动、方块跟着走了，用户判定爪尖戳在方块上，先要求再留 10mm、后又加到 15mm
        # （`out/run_real_40.log` 那一趟留着 10mm 时量到爪子轮廓跟红区还是贴上了）。
        # **2026-10-16 加到 25mm**，见 `GRASP_Z_MARGIN_REAL` 那一节。
        xt = _h(arm.tool_x())
        f_tip = np.asarray(arm.tcp(), dtype=np.float64)
        gap = np.asarray(arm.jaw_gap_center(), dtype=np.float64)
        # `side` 取「固定爪相对开口中点在哪一侧」。**不能像 `pick.py` 那样拿
        # `fixed_tip - 盒子中心` 判** —— 那一支里 TCP 和固定爪尖是**两个点**，这个差
        # 就是固定爪横偏的方向；真机上 `tcp()` 报的就是固定爪尖，两者重合，这个差退化
        # 成「悬停点离盒子中心多少毫米」，那是纠位 stall 留下的残差噪声（`run_real_23`
        # 就栽在这儿：三趟 `side` 都是 −1，盒子被撂在两爪外面，合爪合了个空）。
        side = 1.0 if float(np.dot(f_tip - gap, xt)) >= 0.0 else -1.0
        half = float(sh.support(yaw, xt))
        # **画面估出来的半边长不单独用，跟配置里申报的那件比，取大的那个。**
        # 2026-10-15：`out/run_real_29.log` 那一轮，画面把方块估成 15×19×16mm，
        # `support` 给 8.5mm，而配置里申报的是边长 18mm（半边长 9mm，用户核过）——
        # 落点因此只离盒子中心 9.5mm（8.5 + 0 + 1mm 缝），爪尖基本贴着真侧面往下走，
        # 用户看到的就是「固定爪的尖端直接戳上去了」。
        # **估小是危险的那一侧**（爪尖戳上去、把方块推走、合爪时嘴里是空的），
        # 估大只是缝大一点：爪子张到底 55.9mm，让开十几毫米照样夹得住。所以取大。
        dh = _declared_half(yaw, xt)
        if dh is not None and dh > half:
            print(f"    半边长：画面估出 {half * 1000:.1f}mm < 配置申报的 "
                  f"{dh * 1000:.1f}mm —— 按大的那个让开")
            half = dh
        # **位移纠正：缝要从「爪面」量，不是从爪尖量。** 仿真那边是
        # `skills/pick.py:688-694` 的 `pad = arm.fixed_face_reach(xh, side)`，
        # 注释写得明白：「少算这一段，`FIX_CLEAR` 那 1mm 实际只剩 0.25mm」。
        # 这个模型里没有指垫球，所以 `RealArm.fixed_face_reach` 直接量固定指的网格：
        # 在**盒子那一截高度带**里，固定指的料比爪尖还往盒子里探多少。实测
        # （`_probe_jawbox.py`，这个姿态）探 1.0mm —— 也就是说原来那 5mm 缝实际是 4mm。
        rest_z = float(sh.rest_z)
        face = getattr(arm, "fixed_face_reach", None)
        pad = (float(face(xt, side, (float(center[2]) - rest_z,
                                     float(center[2]) + rest_z)))
               if face is not None else 0.0)
        clear = half + pad + FIX_CLEAR_REAL
        # 余量两条取大的：一条从**中心**量（`GRASP_Z_MARGIN_REAL`），一条从**顶面**量
        # （`sh.rest_z + GRASP_Z_OVER_TOP_REAL`）—— 见那两个常数的注释。
        z_margin = max(GRASP_Z_MARGIN_REAL, rest_z + GRASP_Z_OVER_TOP_REAL)
        grasp_z = float(center[2]) + z_margin
        aim = np.array([center[0] + side * clear * xt[0],
                        center[1] + side * clear * xt[1],
                        grasp_z])
        print(f"    落点按固定爪贴面算：固定爪在开合轴 {'+' if side > 0 else '-'} 侧，"
              f"从盒子中心让开 {half * 1000:.1f}mm（半边长）+ "
              f"{pad * 1000:.1f}mm（爪面探出爪尖的部分）"
              f"+ {FIX_CLEAR_REAL * 1000:.0f}mm（缝）"
              f"，开口中点离固定爪尖 {np.linalg.norm(gap - f_tip) * 1000:.1f}mm"
              f"；下探高度 {grasp_z * 1000:.1f}mm（中心 {float(center[2]) * 1000:.1f}"
              f" + 余量 {z_margin * 1000:.1f}mm = max({GRASP_Z_MARGIN_REAL * 1000:.0f}mm,"
              f" 半高 {rest_z * 1000:.1f} + {GRASP_Z_OVER_TOP_REAL * 1000:.0f}mm)）")
        entry = np.array([aim[0], aim[1], float(center[2]) + 0.05])
        # **末端交给 VLA 的第二个岔口**（2026-10-25，策略表里的 `vla_end`）。进到这里
        # 脚本已经做完「认方块 → 搬过去 → 腕子对上方块侧面」，落点 `aim` 也算好了；
        # 从这儿到合上爪原来是 下探（`_descend`）→ 横向对齐（`_set_clearance`）→
        # 合爪（`_close_until_stall`）三件事，现在一起交给策略服务。夹没夹住仍然由
        # `_close_and_lift(already_closed=True)` 判 —— 模型只出动作，不出判据。
        # 走完（或走不成）都不再回头走脚本那三段，避免一趟里合两次爪。
        if VLA_END == "after_align":
            from . import vla_end                                  # noqa: PLC0415

            r = vla_end.run(arm, cams, prompt=text, empty_deg=empty_deg,
                            open_deg=open_deg, cal=cal)
            if not r.get("ok"):
                _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
                print(f"    ✗ VLA 末端没走成（{r.get('why')}）：这一趟不合爪，退回去")
                return {"ok": False, "why": f"VLA 末端没走成（{r.get('why')}）",
                        "vla": r}
            return _close_and_lift(arm, ctrl, cal, state, index,
                                   grasp_z=grasp_z, empty_deg=empty_deg,
                                   open_deg=open_deg, center=center, hover=hover,
                                   closing=closing, percep=percep, cams=cams,
                                   already_closed=True)
        #
        # **下探这一段不走笛卡尔那条路。** 那一条是「走一小步、读一次、重解一次」，
        # 实测在真机上一边降一边被带偏（`out/run_real_18.log`：指令 x 恒定 0.239，
        # 实际从 237 滑到 204）。改成先把整列点解好、再一口气走完，见 `_descend`。
        # 代价是这一段不挂 `real/anti_jam.py` 那层守卫了 —— 它是挂在 `follow` 循环里
        # 按步判的，关节空间这条路没有「每步判一次」的位置。顶住了会表现为
        # 「指令下去了、读回来没跟上」，`_descend` 每趟都把两边打出来。
        state.seg = "下探到盒子"
        out, tcp = _descend(arm, aim[:2], entry[2], aim[2], closing=closing,
                            open_at=0.5)
        grasp_z = float(tcp[2])
        print(f"    下探：{out}  TCP z={grasp_z:.3f}（该到 {aim[2]:.3f}）")
        if out != "reached":
            # **最后一趟到不了也不再叫停**（用户 2026-10-18：「下探最后一步还不行就停在那儿
            # 期望后续微调吧，别直接叫停」）。以前这里退回悬停点、整趟判失败；现在停在走到
            # 的地方，把「没到底」交给后面那一截微调继续纠。
            print(f"    ⚠ 下探没到底（{out}，停在 z={grasp_z:.3f}、该到 {aim[2]:.3f}）"
                  f"—— 不叫停，就停在这儿，交给后面的微调")
        hit = _stop_here("下探到盒子", arm, f"（下探 {out}，停在 z={grasp_z:.3f}）")
        if hit is not None:
            return hit
        hit = _stopped(arm)
        if hit is not None:
            return hit
        # 合爪前这一眼是**拦门的判据**，拦门的是画面里那两条（用户 2026-10-15：「我现在就
        # 需要在爪子轮廓和红色有距离 + 合爪方向线和目标中心对齐，只有满足这两个标准才能
        # 合爪」）：缝落在带里 且 `|axis| <= AXIS_TOL_PX`。不过就挪一步、
        # 再量一眼（`_set_clearance`，次数和步长都不写死），挪到进带里为止；挪不进去就
        # 不合爪 —— 退回悬停点，这一趟判失败。
        # 段名要在**进微调之前**换成「合爪前微调」：面板的「从断点继续」靠它认入口
        # （`panel/rig.py::resume_stage`）。停在「下探到盒子」那一句上会被当成还没微调。
        state.seg = "合爪前微调"
        # **VLA 末端接手的岔口**（2026-10-23，`PICK_VLA_END=1` 才走）。进来这一刻：
        # 臂在下探到底、6 号舵机张到最大 —— 正是 `_build_endseg.py` 切的那一帧。
        # 后面「横向纠偏 → 最后竖直一截 → 合爪」三件事交给策略服务；夹没夹住仍然由
        # `_close_and_lift(already_closed=True)` 判。
        if _vla_end_on():
            from . import vla_end                                  # noqa: PLC0415

            r = vla_end.run(arm, cams, prompt=text, empty_deg=empty_deg,
                            open_deg=open_deg, cal=cal)
            if not r.get("ok"):
                _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
                print(f"    ✗ VLA 末端没走成（{r.get('why')}）：这一趟不合爪，退回去")
                return {"ok": False, "why": f"VLA 末端没走成（{r.get('why')}）",
                        "vla": r}
            return _close_and_lift(arm, ctrl, cal, state, index,
                                   grasp_z=grasp_z, empty_deg=empty_deg,
                                   open_deg=open_deg, center=center, hover=hover,
                                   closing=closing, percep=percep, cams=cams,
                                   already_closed=True)
        check = _set_clearance(arm, cams, percep, center, "下探到底（合爪前）",
                               instruction=text, clear=clear,
                               # 微调过了之后那几段还要用的东西，一起写进断点
                               # （`ctx=` 那一段的说明见 `_set_clearance`）。
                               ctx={"hover": [float(v) for v in hover],
                                    "closing": [float(v) for v in closing],
                                    "empty_deg": float(empty_deg),
                                    "open_deg": float(open_deg)})
        if not check.get("ok"):
            _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
            print(f"    ✗ 合爪前轮廓检查没过（{check.get('why') or '进不了带里'}）："
                  f"这一趟不合爪，退回去")
            return {"ok": False, "why": f"合爪前轮廓检查没过（{check.get('why')}）",
                    "clear": {k: v for k, v in check.items() if k != "near_uv"}}
        # **判过、还没下扎这一刻写一份断点**（用户 2026-10-18）：面板停手之后
        # 「继续：下探一截并合爪」就从这儿接。不能沿用 `_set_clearance` 判过那一跳
        # 写的那份 —— 它正常返回时 `finally` 已经把它删了（`:145`），这里按编排这边
        # 手里现成的数重写一份（续跑 close 那一支只认 `center` / `ctx` / `angles`）。
        state.seg = "微调判过（还没合爪）"
        _save_checkpoint({
            "stage": "before_close",
            "why": "合爪前微调判过了、还没竖直下扎（盘上写的就是这一眼）",
            "instruction": text, "tag": "下探到底（合爪前）",
            # 这一份永远是**下探之前**那趟判过的（只修轴偏那趟在下探之后）。
            "axis_only": False,
            "center": [float(v) for v in np.asarray(center).reshape(-1)],
            "guess": [float(v) for v in np.asarray(center).reshape(-1)],
            "clear": float(clear),
            # 续跑认领姿态看这一串（`_arm_off_checkpoint`）。
            "angles": [float(v) for v in np.asarray(arm.angles()).reshape(-1)],
            "tcp": [float(v) for v in np.asarray(arm.tcp()).reshape(-1)],
            "z0": float(np.asarray(arm.tcp(), dtype=np.float64)[2]),
            "ctx": {"hover": [float(v) for v in hover],
                    "closing": [float(v) for v in closing],
                    "empty_deg": float(empty_deg),
                    "open_deg": float(open_deg)},
            "prev": None,
        })
        # **进带了才往下扎这一截**（用户 2026-10-16：「微调结束后加一个垂直下探，然后再合爪」）。
        # 放在微调后面是有意的：微调全程 z 钉死、只挪水平，扎下去之后就没法再横着调了。
        # 下探终点原来只到「中心 + 25mm」，爪尖停在方块顶面上方 —— `out/run_real_74.log`
        # 就是在这儿空抓的。这一截只降 z，走完才合爪。
        # **下探之前留一张原始帧**：VLM 在下探之后要判「方块还在不在两爪之间」，
        # 靠的是「下扎前后哪儿变了」。没装 VLM 时这一步直接返回 None、不写盘。
        pre_img = _vlm_gate.save_pre_descend(cams)
        state.seg = "合爪前竖直下探"
        ok_d, tcp_d = _descend_planned(arm, percep, center)
        hit = _stopped(arm)
        if hit is not None:
            return hit
        if not ok_d:
            _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
            print("    ✗ 合爪前那一截竖直下探没走成：这一趟不合爪，退回去")
            return {"ok": False, "why": "合爪前竖直下探没走成"}
        # **下探有没有把方块戳走**（用户 2026-10-18）：再认一眼，比冻住的那个中心。
        box_shift_mm, box_now = _poke_check_after_descend(percep, text, center)
        # **方块被推走了就按这一眼的中心修轴偏**（2026-10-18）：`_set_clearance` 里
        # 3D 那条尺子用的是传进去的 `center`；拿旧中心等于对着一块空桌面纠轴偏。
        center_now = center if box_now is None else box_now
        # **VLM 在下探之后看一眼**（2026-10-23 用户定）：现量一次离轴，再问它
        # 「方块还在两爪之间吗」。判「不在」只做一件事 —— 抬回悬停点重下一次；
        # 不改档、不改旋钮、不外溢到下一集。一集仍然只合一趟爪。
        jg = _vlm_gate.check(arm, cams, percep, tag="下探之后（只修轴偏）",
                             pre_img=pre_img, center=center_now)
        if jg.get("veto"):
            print("    ⟲ VLM 判「方块不在两爪之间」—— 抬回悬停点、重下一次"
                  "（这一集仍然只合一趟爪）", flush=True)
            _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
            hit = _stopped(arm)
            if hit is not None:
                return hit
            state.seg = "VLM 否决后重下"
            _shift2, box_now2 = _poke_check_after_descend(percep, text, center_now)
            if box_now2 is not None:
                center_now = box_now2
                box_now = box_now2          # 交回策略的「下探之后方块在哪」用最后一次
            ok_d, tcp_d = _descend_planned(arm, percep, center_now)
            hit = _stopped(arm)
            if hit is not None:
                return hit
            if not ok_d:
                _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
                print("    ✗ VLM 否决重下之后还是没走成：这一趟不合爪，退回去")
                return {"ok": False, "why": "VLM 否决重下之后没到底"}
            # 重下之后再量一次离轴（只量、不再问、不再否决）—— 这是数据里那对
            # 「重下前后」的第二个数。
            _vlm_gate.check(arm, cams, percep, tag="VLM 否决后重下（只量）",
                            ask=False, center=center_now)
        # **下探之后再修一次轴偏**（用户 2026-10-16：「下探以后再加一个微调，但是只能动
        # 轴偏方向，其他方向不能动」）。竖直那一截把相机跟着带下去了一截，画面里量出来的
        # 轴偏跟在上面那一档量的不是同一个数 —— 所以要在这个高度上再对一次。**只走轴偏
        # 那一维**（`axis_only=True`）：这个高度上沿合爪轴挪就是拿爪面去怼方块。
        state.seg = "下探之后（只修轴偏）"
        ctx2 = {"hover": [float(v) for v in hover],
                "closing": [float(v) for v in closing],
                "empty_deg": float(empty_deg), "open_deg": float(open_deg)}
        check2 = _set_clearance(arm, cams, percep, center_now, "下探之后（只修轴偏）",
                                instruction=text, clear=clear, ctx=ctx2, axis_only=True)
        if not check2.get("ok"):
            _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
            print(f"    ✗ 下探之后轴偏没修进带里（{check2.get('why') or '进不了带里'}）："
                  f"这一趟不合爪，退回去")
            return {"ok": False, "why": f"下探之后轴偏没修进带里（{check2.get('why')}）",
                    "clear": {k: v for k, v in check2.items() if k != "near_uv"}}
        tcp_d = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
        res = _close_and_lift(arm, ctrl, cal, state, index, grasp_z=float(tcp_d[2]),
                              empty_deg=empty_deg, open_deg=open_deg,
                              center=center, hover=hover, closing=closing,
                              percep=percep, cams=cams)
        res["box_shift_mm"] = box_shift_mm
        # **交回实测**（`ppo/policy.py` 用）：这一趟对着哪个中心下的爪、下探之后方块在
        # 哪儿。重试策略拿后一个当下一趟的定位锚点。
        res["center_before"] = [float(center[0]), float(center[1]), float(center[2])]
        res["box_center_after"] = (None if box_now is None else
                                   [float(box_now[0]), float(box_now[1]), float(box_now[2])])
        return res

    def _resume_pick(self, arm, cams, percep, state, index, ctrl, cal, ck: dict,
                     close_only: bool = False) -> dict:
        """从盘上那份断点接着走：进微调循环 → 合爪 → 判 → 抬升。

        **前面那几段一步不动**（`follow.hold` 起手合爪、进场、看一眼、纠位、转正、下探）
        —— 断点就是臂现在这个姿态。
        **夹爪行程两端也不重量**：从盘上那份接过来。这个高度上两个爪子正夹在方块两侧，
        合一次空爪就是拿爪去夹方块（`_close_and_lift` 那段注释同一个道理）。
        """
        ctx = ck.get("ctx") or {}
        empty_deg, open_deg = ctx.get("empty_deg"), ctx.get("open_deg")
        if empty_deg is None or open_deg is None:
            print("    ✗ 盘上那份没记夹爪行程两端 —— 不敢在这个高度上合空爪，不续跑")
            return {"ok": False, "why": "断点里没有夹爪行程两端", "give_up": True}
        arm.empty_deg, arm.open_deg = float(empty_deg), float(open_deg)
        print(f"    行程两端沿用断点那一趟实测的：合到底 {float(empty_deg):.1f}°"
              f" / 张到最大 {float(open_deg):.1f}°（**不再合一次空爪** —— "
              f"这个高度上合上去就夹到方块了）")

        center = np.asarray(ck["center"], dtype=np.float64)
        guess = np.asarray(ck.get("guess") or ck["center"], dtype=np.float64)
        hover = np.asarray(ctx["hover"], dtype=np.float64)
        closing = np.asarray(ctx["closing"], dtype=np.float64)
        text = ck.get("instruction") or "pick up the brown box"
        if close_only:
            # **微调已经判过了**（断点写在「判过、还没下扎」那一刻）：跳过所有微调，
            # 直接竖直下探一截再合爪 —— 用户 2026-10-18「微调后就下探」。
            state.seg = "从断点接上 → 下探一截并合爪"
            print("    续跑入口 close：微调已判过，跳过微调，直接竖直下探一截再合爪")
            ok_d, tcp_d = _descend_planned(arm, percep, center)
            hit = _stopped(arm)
            if hit is not None:
                return hit
            if not ok_d:
                _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
                print("    ✗ 合爪前那一截竖直下探没走成：这一趟不合爪，退回去")
                return {"ok": False, "why": "合爪前竖直下探没走成"}
            box_shift_mm, _box_now = _poke_check_after_descend(percep, text, center)
            res = _close_and_lift(arm, ctrl, cal, state, index,
                                  grasp_z=float(tcp_d[2]),
                                  empty_deg=float(empty_deg), open_deg=float(open_deg),
                                  center=center, hover=hover, closing=closing,
                                  percep=percep, cams=cams)
            res["box_shift_mm"] = box_shift_mm
            res["center_before"] = [float(center[0]), float(center[1]), float(center[2])]
            res["box_center_after"] = (None if _box_now is None else
                                       [float(_box_now[0]), float(_box_now[1]),
                                        float(_box_now[2])])
            return res
        if bool(ck.get("axis_only")):
            # **断点是在「下探之后那趟只修轴偏」里停的**：竖直那一截上一趟已经走过了
            # （盘上 `axis_only` 记着），这一趟**不再往下扎**，直接接着修轴偏、然后合爪。
            state.seg = "从断点接上 → 下探之后（只修轴偏）"
            check2 = _set_clearance(arm, cams, percep, center,
                                    ck.get("tag") or "下探之后（只修轴偏）",
                                    instruction=text, guess=guess,
                                    clear=float(ck.get("clear") or 0.0),
                                    resume=ck, ctx=ctx, axis_only=True)
            if not check2.get("ok"):
                _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
                print(f"    ✗ 下探之后轴偏没修进带里（{check2.get('why') or '进不了带里'}）："
                      f"这一趟不合爪，退回去")
                return {"ok": False,
                        "why": f"下探之后轴偏没修进带里（{check2.get('why')}）",
                        "clear": {k: v for k, v in check2.items() if k != "near_uv"}}
            tcp_d = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
            return _close_and_lift(arm, ctrl, cal, state, index,
                                   grasp_z=float(tcp_d[2]),
                                   empty_deg=float(empty_deg), open_deg=float(open_deg),
                                   center=center, hover=hover, closing=closing,
                                   percep=percep, cams=cams)
        state.seg = "从断点接上 → 合爪前微调"
        check = _set_clearance(arm, cams, percep, center,
                               ck.get("tag") or "下探到底（合爪前）",
                               instruction=text, guess=guess,
                               clear=float(ck.get("clear") or 0.0), resume=ck, ctx=ctx)
        if not check.get("ok"):
            _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
            print(f"    ✗ 合爪前轮廓检查没过（{check.get('why') or '进不了带里'}）："
                  f"这一趟不合爪，退回去")
            return {"ok": False, "why": f"合爪前轮廓检查没过（{check.get('why')}）",
                    "clear": {k: v for k, v in check.items() if k != "near_uv"}}
        # 断点接上这条也一样：进带了再往下扎那一截（跟 `run` 那条同一个道理）。
        state.seg = "合爪前竖直下探"
        ok_d, tcp_d = _descend_planned(arm, percep, center)
        hit = _stopped(arm)
        if hit is not None:
            return hit
        if not ok_d:
            _goto(arm, hover, vmax=speed_vmax(SLOW), grip=1.0)
            print("    ✗ 合爪前那一截竖直下探没走成：这一趟不合爪，退回去")
            return {"ok": False, "why": "合爪前竖直下探没走成"}
        # **下探有没有把方块戳走**（用户 2026-10-18）：断点接上这条也一样量。
        box_shift_mm, _box_now = _poke_check_after_descend(percep, text, center)
        res = _close_and_lift(arm, ctrl, cal, state, index, grasp_z=float(tcp_d[2]),
                              empty_deg=float(empty_deg), open_deg=float(open_deg),
                              center=center, hover=hover, closing=closing,
                              percep=percep, cams=cams)
        res["box_shift_mm"] = box_shift_mm
        res["center_before"] = [float(center[0]), float(center[1]), float(center[2])]
        res["box_center_after"] = (None if _box_now is None else
                                   [float(_box_now[0]), float(_box_now[1]),
                                    float(_box_now[2])])
        return res
