# -*- coding: utf-8 -*-
"""用云端 π0.5（AutoDL）驱动真机：观测 → 动作块 → 写舵机。

链路：本机 `_tunnel.py`（SSH 转发 127.0.0.1:8000）→ 服务器 `scripts/serve_policy.py`。

    G:/anaconda/envs/sim2real/python.exe -u _pi05_run.py --dry-run --steps 3
    G:/anaconda/envs/sim2real/python.exe -u _pi05_run.py --steps 200 --hz 4

**像素口径和采集时一致**：`real/recorder.py:250-253` 存的是 cv2 原样的 BGR 帧
（lerobot 不做通道转换），这里也不换，直接发同一路图。
**服务端不含 repack**，所以直接发 repack 之后的结构（`images` / `state` / `prompt`）。

**安全策略（三层，任何一层单独都能挡住一次瞬移）**
  1. 伺服速度档：`speed=0` 是舵机最高速（`servo_control_standalone.py:742`），这里
     改成一个有限值，物理速度上限卡在舵机里，软件这边睡过头也冲不出去。
  2. 闭环限速：每一小段的起点是**刚读回来的实际角度**，目标 = 实测 + 最多
     `--max-deg-step`，所以指令永远离实测不超过一小段，不存在「攒一大截再放」。
  3. 硬闸：单关节一步要动超过 `real/safety.MAX_STEP_DEG`（45°）就不写、立
     `terminated` 停手；每步走完再用 `safety.check_pose` 看整条臂还在不在工作包络里。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402

from core import geom                                            # noqa: E402
from core.registry import make                                   # noqa: E402
from core.trajectory import build                                # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg            # noqa: E402

PROMPT = "pick up the green box"
WRIST_SIZE = (960, 540)      # (宽, 高)，和采集时一样


#: 图先压成 JPEG 再发。**链路实测只有 0.06 MB/s**：两张 960x540 的裸数组共 3.1MB，
#: 单程就要 50 秒 —— 每块 40~88 秒的等待全在这儿，服务端本地推理只要 0.21 秒。
#: 压成 JPEG 后约 140KB，同样的链路 2~3 秒。服务端 `_decode_images` 用 cv2 解回来，
#: 和这里 `cv2.imencode` 是同一条通道编码，进模型的数组和发裸数组时逐位相同。
JPEG_QUALITY = 85


def _jpg(a: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.asarray(a, dtype=np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def _obs(arm, wrist, side) -> dict:
    img = wrist.shot()
    img = cv2.resize(np.asarray(img), WRIST_SIZE, interpolation=cv2.INTER_AREA)
    images = {"left_wrist_0_rgb": _jpg(img)}
    if side is not None:
        images["base_0_rgb"] = _jpg(side.shot())
    return {
        "images": images,
        "state": np.asarray(arm.angles(), dtype=np.float32),
        "prompt": PROMPT,
    }


#: 舵机速度档的换算：位置 0~4095 对应 0~360°，见 `servo_bridge.deg_from_position`。
TICKS_PER_DEG = 2048.0 / 180.0

#: 合爪那一刻的 TCP 高度写到这儿，给摆件用（`_pi05_place.py` 读它）。
#: **必须是合爪那一刻的**：`place._place_real` 的松爪高度 = 合爪高度 + 5mm，拿抬起来
#: 之后的高度去算，就会在半空松手。
GRASP_REC = ROOT / "out" / "pi05_last_grasp.json"
_GRASP_Z: float | None = None
_LAST_CMD6: float | None = None


def _note_close(arm, cmd6: float) -> None:
    """6 号的**指令**从张开（≥ `SQUEEZE_BELOW_DEG`）落到合爪（< 那个值）的那一下，
    记下此刻的 TCP。只看下降沿：开机时 6 号本来就停在 −7.8°（初始位置），
    按读数判会把开机那一下当成合爪。"""
    global _GRASP_Z, _LAST_CMD6
    prev, _LAST_CMD6 = _LAST_CMD6, float(cmd6)
    if _GRASP_Z is not None or prev is None:
        return
    if not (prev >= SQUEEZE_BELOW_DEG > cmd6):
        return
    tcp = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    _GRASP_Z = float(tcp[2])
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "tcp_m": [float(v) for v in tcp], "grasp_z_m": _GRASP_Z,
           "q_deg": [float(v) for v in np.asarray(arm.angles()).reshape(-1)]}
    GRASP_REC.parent.mkdir(parents=True, exist_ok=True)
    GRASP_REC.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [记] 合爪那一刻 TCP z {_GRASP_Z * 1000:+.1f}mm → {GRASP_REC}", flush=True)


def _write_deg(arm, target_deg: np.ndarray) -> np.ndarray:
    """度数目标 → 刻 → 舵机。返回真正写下去的那一组度数。

    **不夹到 [0,180]**：模型输出的度和 `arm.angles()` 同口径，关节角可以是负的
    （实测在位的读数就有 -35 / -93.9）。夹了会把 1/2/5 号一次推到 0°，看上去就是瞬移。
    行程由 `_qpos_to_ticks` 按标定管；6 号的 20° 张开上限由 `write_ticks` 拦。
    """
    q = np.radians(np.asarray(target_deg, dtype=np.float64))
    _note_close(arm, float(np.degrees(q)[5]))
    ticks = arm._qpos_to_ticks(q)
    arm.write_ticks(ticks)
    arm.last_action = np.degrees(q).astype(np.float32)
    return arm.last_action


#: 模型把 6 号合到多少度以下，就额外再往里压一份。5° 和 `vla_end.CLOSE_DEG` 同一个分界：
#: 张开的读数是 19.5°，空抓基准约 −9°，5° 以下只可能是「在合爪」，不会是「在开爪」。
SQUEEZE_BELOW_DEG = 5.0

_SQUEEZE_DEG: float | None = None


def _squeeze_deg() -> float:
    """合爪时额外往里压多少度。**直接用 `knobs.MAX_LEAD_RAD`**，不另起一个数字。

    那个 0.1 rad ≈ 5.7° 是脚本路径已经压在方块上验过的夹持力（见 `knobs.py` 里那段
    注释：舵机是位置控制，指令和实测之间那点角度差就是力）。`skills.pick_real` 的包
    `__init__` 很长，所以这里懒 import 一次、缓存下来。
    """
    global _SQUEEZE_DEG
    if _SQUEEZE_DEG is None:
        from skills.pick_real.knobs import MAX_LEAD_RAD           # noqa: PLC0415

        _SQUEEZE_DEG = float(np.degrees(MAX_LEAD_RAD))
    return _SQUEEZE_DEG


def _clip_target(arm, target_deg) -> np.ndarray:
    """把模型给的目标夹进这台臂真能到的范围。

    * 1~5 号：MJCF 的关节限位；
    * 6 号：`[grip_lo, GRIP_OPEN_DEG]`。**目标必须先夹**，否则 `write_ticks` 会把它夹回
      20°，闭环永远到不了目标、一路拖到超时停手；
    * 6 号合爪时**再往里多压 `_squeeze_deg()`**。模型给的最后一个点常常正好落在方块
      表面上，位置误差归零 → 手上没有劲，抬起来就滑（2026-10-25 try6 方块掉在地上）。
      这一份额外的位置误差就是夹持力，压到 `grip_lo` 硬限位为止。
    """
    from real.arm import GRIP_OPEN_DEG                            # noqa: PLC0415

    t = np.asarray(target_deg, dtype=np.float64).copy()
    t = np.clip(t, np.degrees(arm.q_lo), np.degrees(arm.q_hi))
    t[5] = min(t[5], float(GRIP_OPEN_DEG))
    t[5] = max(t[5], float(np.degrees(arm.grip_lo)))
    if t[5] < SQUEEZE_BELOW_DEG:
        t[5] = max(t[5] - _squeeze_deg(), float(np.degrees(arm.grip_lo)))
    return t


def _drive_to(arm, target_deg, args) -> str | None:
    """闭环走到 `target_deg`：每一小段从**实测**出发，最多再走 `--max-deg-step`。

    返回 None = 到了；返回字符串 = 安全闸拦下来的原因（已立 `terminated`）。
    """
    from real import safety                                        # noqa: PLC0415

    tgt = _clip_target(arm, target_deg)
    cur = np.asarray(arm.angles(), dtype=np.float64)
    need = float(np.abs(tgt[:5] - cur[:5]).max())
    n_est = max(1, int(np.ceil(need / max(args.max_deg_step, 1e-6))))
    # 预算：按最多 4 倍的理论时间（带载实测只走指令的三到五成，闭环要几轮才补得齐）。
    t_end = time.time() + max(1.5, 4.0 * n_est * args.sub_dt + 0.5)
    # **6 号不要求走到位**：合爪夹到方块就停了，那是正常结果，不是卡住。
    # 只按「还欠多少 / 每小段多少」数它的段数，数完就放行。
    grip_left = int(np.ceil(abs(float(tgt[5] - cur[5]))
                            / max(args.max_deg_step, 1e-6)))

    prev = None
    n = 0
    while True:
        if getattr(arm, "terminated", False):
            return arm.safety_why or "terminated"
        cur = np.asarray(arm.angles(), dtype=np.float64)
        # **实测速度也要看**：指令端限住了，但如果舵机自己在冲（速度档失效、外力），
        # 这里要能发现。阈值给到限速的 3 倍再加 2°，躲开读数噪声。
        if prev is not None:
            dv = float(np.abs(cur[:5] - prev[:5]).max())
            if dv > args.max_deg_step * 3.0 + 2.0:
                return safety._trip(arm, f"实测一步动了 {dv:.1f}°"
                                         f"（限速 {args.max_deg_step:.1f}°/小段）")
        prev = cur

        err = tgt - cur
        body_done = float(np.abs(err[:5]).max()) <= args.tol_deg
        if body_done and grip_left <= 0:
            return None
        step = np.clip(err, -args.max_deg_step, args.max_deg_step)
        if body_done:
            step[:5] = 0.0
        if grip_left <= 0:
            step[5] = 0.0
        else:
            grip_left -= 1
        _write_deg(arm, cur + step)
        n += 1
        if not body_done and time.time() > t_end:
            return safety._trip(arm, f"走不到目标：还差 {np.round(err, 1)}，"
                                     f"{n} 小段 / {4.0 * n_est * args.sub_dt + 0.5:.1f}s 用完")
        time.sleep(args.sub_dt)


def _follow_chunk(arm, acts_deg, args) -> str | None:
    """把一个 action chunk 变成一条 toppra 轨迹，按 `--ctl-hz` 连续下发。

    位置连续（轨迹起点就是实测）、速度连续（`--vmax`）、加速度连续（`--amax`）三件事
    都交给 toppra（`core/trajectory.py`，和 `real/joint_move.py` 同一套），不手写速度曲线。
    返回 None = 走完；字符串 = 安全闸原因（已立 `terminated`）。
    """
    from real import safety                                        # noqa: PLC0415

    acts = np.asarray(acts_deg, dtype=np.float64)
    if acts.ndim != 2 or len(acts) == 0:
        return None
    tgt = np.stack([_clip_target(arm, a) for a in acts])
    q_meas = np.asarray(arm.angles(), dtype=np.float64)
    d0 = float(np.abs(tgt[0] - q_meas).max())
    if d0 > args.max_jump:
        i = int(np.argmax(np.abs(tgt[0] - q_meas)))
        return safety._trip(arm, f"模型这一步要动 {d0:.1f}°（{i + 1} 号关节），"
                                 f"上限 {args.max_jump:.0f}°")

    wps = np.vstack([q_meas[None, :], tgt])
    keep = [0]                      # 原地不动的相邻路点会让样条退化，去掉
    for i in range(1, len(wps)):
        if float(np.abs(wps[i] - wps[keep[-1]]).max()) > 1e-3:
            keep.append(i)
    wps = wps[keep]
    if len(wps) < 2:
        return None
    try:
        traj = build(np.radians(wps), vmax=args.vmax, amax=args.amax, density=0.01)
    except Exception as exc:                                       # noqa: BLE001
        print(f"      轨迹建不出来（{exc}），退回逐点限速", flush=True)
        for a in tgt:
            why = _drive_to(arm, a, args)
            if why:
                return why
        return None

    dt = 1.0 / args.ctl_hz
    t0 = time.time()
    t_hard = t0 + traj.duration * 1.5 + 2.0
    m_next = t0 + 0.25            # 实测盯梢的节拍
    prev_meas = q_meas.copy()
    n = 0
    while True:
        if getattr(arm, "terminated", False):
            return arm.safety_why or "terminated"
        now = time.time()
        if now - t0 >= traj.duration:
            break
        if now > t_hard:
            return safety._trip(arm, f"轨迹跑了 {now - t0:.1f}s 还没完"
                                     f"（预计 {traj.duration:.1f}s）")
        q_traj, _v = traj.at(now - t0)
        q_traj = np.clip(q_traj, arm.q_lo, arm.q_hi)
        _write_deg(arm, np.degrees(q_traj))
        n += 1
        if now >= m_next:
            m_next = now + 0.25
            m = np.asarray(arm.angles(), dtype=np.float64)
            span = args.vmax * 0.25 * 180.0 / np.pi * 3.0 + 2.0
            dv = float(np.abs(m[:5] - prev_meas[:5]).max())
            if dv > span:
                return safety._trip(arm, f"实测 0.25s 走了 {dv:.1f}°，限速 "
                                         f"{args.vmax * 180 / np.pi:.0f}°/s")
            lag = float(np.abs(m - np.degrees(q_traj)).max())
            if lag > args.max_lag:
                return safety._trip(arm, f"实测落后轨迹 {lag:.1f}°"
                                         f"（上限 {args.max_lag:.0f}°）")
            prev_meas = m
        nap = (t0 + n * dt) - time.time()
        if nap > 0:
            time.sleep(nap)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100, help="总共走多少个控制步")
    ap.add_argument("--hz", type=float, default=4.0, help="控制频率，和采集的 fps 一致")
    ap.add_argument("--dry-run", action="store_true", help="只推理，不写舵机")
    ap.add_argument("--port", type=int, default=8000, help="本机隧道端口")
    ap.add_argument("--record", default="",
                    help="每步把腕相机和侧相机的原图存到这个目录（给复盘看）")
    ap.add_argument("--max-deg-step", type=float, default=1.5,
                    help="每个小段单个关节最多动多少度（1.5° / 50ms ≈ 30°/s）")
    ap.add_argument("--sub-dt", type=float, default=0.05,
                    help="小段之间的间隔（秒）")
    ap.add_argument("--tol-deg", type=float, default=0.5,
                    help="离目标多近算到位（度）")
    ap.add_argument("--servo-speed", type=int, default=400,
                    help="舵机速度档（刻/秒，0 = 舵机自己最快，不建议）。400 ≈ 35°/s，"
                         "比软件限速 20°/s 高，正常不会碰到它")
    ap.add_argument("--vmax", type=float, default=0.35,
                    help="轨迹的关节速度上限，弧度/秒（0.35 ≈ 20°/s）")
    ap.add_argument("--amax", type=float, default=0.6,
                    help="轨迹的关节加速度上限，弧度/秒²")
    ap.add_argument("--ctl-hz", type=float, default=30.0,
                    help="轨迹下发的频率（Hz）；4 Hz 是模型的输出节奏，不是写舵机的节奏")
    ap.add_argument("--max-lag", type=float, default=30.0,
                    help="实测落后轨迹超过这么多度就停手（舵机卡住 / 被挡住）")
    ap.add_argument("--max-jump", type=float, default=45.0,
                    help="单个关节一步要动超过这么多度就停手不写（默认和 "
                         "real/safety.py 的 MAX_STEP_DEG 一致）")
    ap.add_argument("--home", action="store_true",
                    help="开跑前先回初始位置（走 toppra 轨迹，限速）")
    ap.add_argument("--home-jvel", type=float, default=0.35,
                    help="回位时的关节速度上限，弧度/秒（0.35 ≈ 20°/s）")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args()

    from _pi05_link import connect                                    # noqa: PLC0415

    cfg = load_cfg(args.config)
    rig = cfg["real"]
    geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
    geom.REAL_PICK_XY = np.array(rig["pick_xy_mm"], dtype=np.float64) / 1000.0
    geom.REAL_PERCEP_OFFSET = np.array(rig.get("percep_offset_mm") or (0.0, 0.0),
                                       dtype=np.float64) / 1000.0
    geom.OBJECTS = list(cfg.get("objects") or [])

    side = None
    side_cfg = (cfg.get("collect") or {}).get("side_cam") or {}
    arm = wrist = client = None
    try:
        import real.arm                                               # noqa: F401,PLC0415
        import real.cameras                                           # noqa: F401,PLC0415
        from real.side_cam import SideCamera                          # noqa: PLC0415

        # **速度档不走 config 的 `speed`（那是 0 = 最高速）**，由 `--servo-speed` 给有限值。
        arm = make("arm", "real", port=str(rig.get("port", "COM10")),
                   speed=int(args.servo_speed), fast=bool(rig.get("fast", False)))
        arm.terminated = False
        wrist = make("camera", "real_wrist", source=arm,
                     index=int(rig.get("cam_index", 1)))
        if side_cfg:
            side = SideCamera(index=int(side_cfg.get("index", 0)),
                              crop=side_cfg.get("crop"),
                              rotate=float(side_cfg.get("rotate", 0.0)))

        if args.home:
            from real import joint_move as jm                         # noqa: PLC0415

            q_goal = jm.home_q(arm)
            print(f"回初始位置（jvel {args.home_jvel} rad/s ≈ "
                  f"{np.degrees(args.home_jvel):.0f}°/s）…", flush=True)
            q_now = jm.goto_q(arm, q_goal, jvel=args.home_jvel, jacc=1.0)
            err = float(np.max(np.abs(np.degrees(np.asarray(q_now) - q_goal))))
            print(f"回位完成：最远关节差 {err:.1f}°，现在 {np.round(arm.angles(), 1)}",
                  flush=True)

        client = connect("127.0.0.1", args.port)
        print(f"策略服务已连上；{args.steps} 步 @ {args.hz}Hz，"
              f"{'dry-run（不动臂）' if args.dry_run else '会驱动机械臂'}", flush=True)
        print(f"安全：舵机速度档 {args.servo_speed} 刻/秒 ≈ "
              f"{args.servo_speed / TICKS_PER_DEG:.0f}°/s"
              f"{'（0 = 舵机最高速，没上限）' if args.servo_speed <= 0 else ''}；"
              f"轨迹 vmax {args.vmax:.2f} rad/s ≈ {args.vmax * 180 / np.pi:.0f}°/s、"
              f"amax {args.amax:.2f} rad/s²；"
              f"硬闸 {args.max_jump:.0f}°；下发 {args.ctl_hz:.0f} Hz", flush=True)

        from real import safety                                    # noqa: PLC0415

        rec_dir = Path(args.record) if args.record else None
        if rec_dir is not None:
            rec_dir.mkdir(parents=True, exist_ok=True)
            print(f"录帧 → {rec_dir}", flush=True)

        step = 0
        while step < args.steps and not getattr(arm, "terminated", False):
            t0 = time.time()
            obs = _obs(arm, wrist, side)
            if rec_dir is not None:
                # 存的是**原样整帧**（腕部 960×540、侧视 860×900），不裁不缩。
                for key, name in (("left_wrist_0_rgb", "wrist"),
                                  ("base_0_rgb", "side")):
                    im = obs["images"].get(key)
                    if im is None:
                        continue
                    # 现在发出去的是 JPEG 字节；原样落盘就是那一帧，不用再转通道。
                    cv2.imwrite(str(rec_dir / f"{step:04d}_{name}.jpg"),
                                cv2.imdecode(np.frombuffer(im, dtype=np.uint8),
                                             cv2.IMREAD_COLOR))
            t1 = time.time()
            out = client.infer(obs)
            acts = np.asarray(out["actions"], dtype=np.float64)
            acts = acts[: max(1, args.steps - step)]
            t2 = time.time()
            if args.dry_run:
                step += len(acts)
                print(f"[步 {step:4d}] dry-run：{len(acts)} 个动作 "
                      f"{np.round(acts[0], 1)}…  不动臂", flush=True)
                continue
            # **一个 chunk 一条轨迹**：toppra 过 10 个路点，按 `--ctl-hz` 连续下发。
            why = _follow_chunk(arm, acts, args)
            if why:
                break
            step += len(acts)
            # **6 号把模型给的末点和脚本额外压的 5.7° 分开报**：实测停在哪儿、模型想去
            # 哪儿、多压了多少，三件事一行看完，才知道「夹不紧」是模型没合到位还是没劲。
            print(f"[步 {step:4d}] 走完 {len(acts)} 个动作  实测 {np.round(arm.angles(), 1)}  "
                  f"（观测 {(t1 - t0) * 1e3:.0f}ms + 推理 {(t2 - t1) * 1e3:.0f}ms）  "
                  f"6 号：模型末点 {acts[-1, 5]:+.1f}° → 下发 "
                  f"{_clip_target(arm, acts[-1])[5]:+.1f}°", flush=True)
            if safety.check_pose(arm, f"第 {step} 步之后"):
                break
        if getattr(arm, "terminated", False):
            print(f"已停手：{getattr(arm, 'safety_why', '')}", flush=True)
            return 2
        return 0
    finally:
        for name, obj in (("策略连接", client), ("侧相机", side), ("腕相机", wrist),
                          ("机械臂", arm)):
            if obj is not None:
                try:
                    obj.close()
                except Exception as exc:                              # noqa: BLE001
                    print(f"关{name}：", exc, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
