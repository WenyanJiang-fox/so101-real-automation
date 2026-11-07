"""末端交给 VLA：脚本把臂带到指定位置，从这里到合上爪由模型做。

**只是「接一段」，不是「换掉技能」。** 两个入口，接进来的时刻不同：

- 环境变量 `PICK_VLA_END=1`（2026-10-23）：在**下探到底、6 号舵机张到最大**之后接。
  剩下的是 `_set_clearance`（横向纠偏）、`_descend_planned`（最后竖直一截）、
  `_close_until_stall`（合爪）三件事。
- 策略表里的 `vla_end`（2026-10-25，`strategies.json` → `skill.py::VLA_END`）：
  在**腕子对上方块侧面之后、下探之前**接，三件事变成 `_descend`（下探）、
  横向对齐、合爪。

合没合上仍然由脚本判（`_close_and_lift` 的 `already_closed=True` 那一支）——
模型只负责动作，不负责判据。

**为什么接在这儿。** `_build_endseg.py` 切出来的训练段起点，就是「6 号舵机第一次
张到最大」那一帧；这一段的观测（腕相机 + 侧相机 + 六个关节角）和训练时同一口径，
所以模型看到的第一个输入落在它见过的分布里。

**关掉的方式**：环境变量 `PICK_VLA_END` 不给（或给 `0`）就整支不装，技能照旧走
脚本那三个人。开了之后服务连不上、没侧相机、模型一直不合爪 —— 三种都退回悬停点、
这一趟判失败，不自动重试。
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np

PROMPT = "pick up the green box"

#: 6 号舵机低于这个数就算「合上了」。空抓基准约 −9°，张开 19.5°，取 5 当分界。
CLOSE_DEG = 5.0

#: 一次跑多少个 action chunk（每个 chunk 10 步）。训练段中位 28 帧 ≈ 7 秒，
#: 一个 chunk ≈ 2.5 秒 → 8 个给到 20 秒，够模型走完还留余量。
MAX_CHUNKS = 8


def enabled() -> bool:
    """`PICK_VLA_END=1` 才接。"""
    return os.environ.get("PICK_VLA_END", "").strip() not in ("", "0", "false", "False")


def _args(**over) -> SimpleNamespace:
    """`_pi05_run._follow_chunk` 要的那几个参数，和 `_pi05_run.py` 的默认值一致。"""
    base = dict(vmax=0.35, amax=0.6, ctl_hz=30.0, max_lag=30.0, max_jump=45.0,
                max_deg_step=1.5, sub_dt=0.05, tol_deg=0.5)
    base.update(over)
    return SimpleNamespace(**base)


def run(arm, cams, *, prompt: str = PROMPT, port: int = 8000,
        max_chunks: int = MAX_CHUNKS,
        empty_deg: float | None = None, open_deg: float | None = None,
        cal: dict | None = None) -> dict:
    """从「下探到底、爪张开」开始，让模型走到合爪为止。

    返回 `{"ok": 模型走到合爪没有, "why": ..., "chunks": n, "grip": 6 号舵机度数}`。
    `ok=True` 只代表**模型合了爪**，夹没夹住由调用方 `_close_and_lift` 判。
    """
    from .pose import _wrist                                        # noqa: PLC0415

    side = cams.get("side") if hasattr(cams, "get") else None
    if side is None:
        return {"ok": False, "why": "cams 里没有 side（模型是两路相机训练的）"}
    wrist = _wrist(cams)

    # 这两个都在 `_pi05_run.py` 里调通了，直接复用 —— 不另写一份观测编码和轨迹下发。
    # **`_pi05_run` 模块顶上有一句 `os.chdir(ROOT)`**（它是独立脚本），import 完把
    # 工作目录还回去，不然会给技能和面板换根目录。
    _cwd = os.getcwd()
    try:
        from _pi05_link import connect                              # noqa: PLC0415
        from _pi05_run import _follow_chunk, _obs                   # noqa: PLC0415
    finally:
        os.chdir(_cwd)
    from real import safety                                         # noqa: PLC0415

    try:
        client = connect("127.0.0.1", port, open_timeout=10.0)
    except Exception as exc:                                        # noqa: BLE001
        return {"ok": False, "why": f"策略服务连不上（{type(exc).__name__}: {exc}）"}

    a = _args()
    chunks, grip = 0, None
    try:
        print(f"    → VLA 接手末端：最多 {max_chunks} 个 chunk，"
              f"6 号舵机 < {CLOSE_DEG:.0f}° 算合上", flush=True)
        for _ in range(int(max_chunks)):
            if getattr(arm, "terminated", False):
                return {"ok": False, "why": arm.safety_why or "terminated",
                        "chunks": chunks, "grip": grip}
            obs = _obs(arm, wrist, side)
            t0 = time.time()
            out = client.infer(obs)
            acts = np.asarray(out["actions"], dtype=np.float64)
            if acts.ndim != 2 or len(acts) == 0:
                return {"ok": False, "why": "模型给了空的 actions",
                        "chunks": chunks, "grip": grip}
            print(f"      chunk {chunks + 1}：{len(acts)} 步，第一步 "
                  f"{np.round(acts[0], 1)}，最后一步 {np.round(acts[-1], 1)}"
                  f"（推理 {time.time() - t0:.1f}s）", flush=True)
            why = _follow_chunk(arm, acts, a)
            chunks += 1
            grip = float(np.asarray(arm.angles(), dtype=np.float64)[5])
            print(f"        走完：实测 {np.round(arm.angles(), 1)}"
                  f"（6 号 {grip:+.1f}°）", flush=True)
            if why:
                return {"ok": False, "why": why, "chunks": chunks, "grip": grip}
            if safety.check_pose(arm, f"VLA 末端第 {chunks} 个 chunk 之后"):
                return {"ok": False, "why": getattr(arm, "safety_why", "安全闸"),
                        "chunks": chunks, "grip": grip}
            # **合上就收，但「合上」得是夹住东西，不是在空中合到底。**
            # 原来这一句是 `if grip < CLOSE_DEG or want_close:`，`want_close` 看的是
            # **模型指令**：模型在还够不着的高度合一次爪，整支就当场收工。2026-10-25
            # 实测（`_vla_end_try.py` 第一趟）：chunk 1 末点 6 号 −7.7°，臂还在
            # z=56.9mm、方块顶在 z≈−14mm，脚本判 empty 收场，下探一步都没走。
            # 现在只看**实测**，而且用脚本自己那条判据（`classify_ref`，比的是这一趟
            # 实测的空抓基准）：空中合到底会被判 empty，继续走下一个 chunk。
            if empty_deg is not None and cal is not None:
                from grasp_check import HELD, classify_ref            # noqa: PLC0415
                from servo_bridge import read_gripper_deg             # noqa: PLC0415

                from .grasp import _close_until_stall                 # noqa: PLC0415

                g = read_gripper_deg(arm.ctrl, samples=3)
                verdict = classify_ref(g, empty_deg, cal, open_deg)
                print(f"        6 号 {g if g is None else f'{g:.1f}°'}"
                      f"（空抓基准 {empty_deg:.1f}°）→ 判 {verdict}", flush=True)
                # **角度像夹住还不够，得再压一下**（用户 2026-10-25：「窄缝隙要是还能再
                # 收紧就是没夹住，不能收紧就是夹住了」）。中间那一段缝隙和真夹住东西在
                # 角度上是同一档 —— 13:41 那趟 6 号 55.9° 判 held、方块其实没进爪。
                # 压不动 = 嘴里有东西；还能朝空抓基准那头收 = 只是缝小。
                # 复用合爪那套 `_close_until_stall`（自带 `MAX_LEAD_RAD` 挤压量和
                # `STUCK_S` 停手判据），不另写一份。
                if verdict == HELD:
                    stop, info = _close_until_stall(arm, None, empty_deg=empty_deg)
                    g2 = read_gripper_deg(arm.ctrl, samples=3)
                    verdict = classify_ref(g2, empty_deg, cal, open_deg)
                    print(f"        压一下：{stop}（{info['why']}，{info['steps']} 步）"
                          f"  6 号 {g:.1f}° → "
                          f"{'读不到' if g2 is None else f'{g2:.1f}°'}"
                          f"  → 判 {verdict}", flush=True)
                    grip = float(np.asarray(arm.angles(), dtype=np.float64)[5])
                    g = g2 if g2 is not None else g
                if verdict == HELD:
                    return {"ok": True, "why": "", "chunks": chunks, "grip": grip,
                            "grip_deg": g, "verdict": verdict}
            elif grip < CLOSE_DEG:
                # 老调用方没给空抓基准：退回「实测角 < 5°」这一档（分不出夹住/空中合到底）
                return {"ok": True, "why": "", "chunks": chunks, "grip": grip}
        return {"ok": False, "why": f"{max_chunks} 个 chunk 走完还没合爪",
                "chunks": chunks, "grip": grip}
    finally:
        try:
            client.close()
        except Exception:                                           # noqa: BLE001
            pass
