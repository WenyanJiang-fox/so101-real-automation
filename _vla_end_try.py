# -*- coding: utf-8 -*-
"""跑一趟「腕子对齐后交给 VLA」：脚本管到腕子对上方块侧面，剩下的交给 π0.5。

**只抓一次、不重试。** 装配照 `panel/rig_assembly.py`（腕相机 + 侧相机 + 感知），
回位照 `_oneshot_grasp.py`，策略照 `skills/pick_real/strategy.py` 的新条目。

    G:/anaconda/envs/sim2real/python.exe -u _vla_end_try.py

要先有两条隧道在跑（策略服务 8000、VLM 8100）：
    G:/anaconda/envs/sim2real/python.exe -u _tunnel.py 8000 8000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# **别省这一行**：控制台是 GBK，技能里会打 `↻`（纠位那一行），GBK 编不出来就当场
# `UnicodeEncodeError`，整趟停在那儿。2026-10-25 实测。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 闸口一律关掉：不停下来等人。**`PICK_VLA_END` 也不能有** —— 那个是「下探到底之后」
# 的另一个岔口，开着就会在更晚的地方再走一次 VLA，一趟合两次爪。
for _k in ("PICK_CONFIRM_STEP", "PICK_CONFIRM", "PICK_CONFIRM_DESCEND",
           "PICK_RESUME", "PICK_START_AT", "PICK_VLA_END"):
    os.environ.pop(_k, None)

import threading                                                 # noqa: E402
import time                                                      # noqa: E402

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402

from core import geom                                            # noqa: E402
from core.registry import make                                   # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg            # noqa: E402

STRATEGY = "vla_end"
TAG = time.strftime("%m%d_%H%M")


def _start_recorder(side, fps: float = 10.0):
    """侧相机整段录下来，最后一次性编码成 mp4。

    `side.shot()` 只交缓存帧，不抢它自己的读帧线程；先在内存里存 JPEG（一帧约 100KB），
    直接存 RGB 数组的话几分钟就是上 G。**不裁不缩**，和相机配置里那个视野一致。

    每帧连**取帧那一刻的墙钟**一起存（用户 2026-11-05：烧时间码）。时钟不是事后按帧号
    算出来的，是逐帧记的 —— 中间卡了、丢帧了，时间码会把真实间隔照出来。
    """
    frames: list[tuple[float, bytes]] = []
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            f = side.shot()
            if f is not None:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok:
                    frames.append((time.time(), buf.tobytes()))
            stop.wait(1.0 / fps)

    thr = threading.Thread(target=loop, name="side-rec", daemon=True)
    thr.start()
    return frames, stop, thr


def _save_side_video(frames: list[tuple[float, bytes]], seconds: float) -> Path | None:
    if not frames:
        print("！侧相机一帧都没录到，不写视频", flush=True)
        return None
    from real.video import stamp, write_mp4                    # noqa: PLC0415
    t0 = frames[0][0]
    rgb = []
    for i, (ts, b) in enumerate(frames):
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        el = ts - t0
        stamp(img, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
              + f".{int(ts % 1 * 10)}", org=(14, 40))
        stamp(img, f"T+{int(el // 60):02d}:{el % 60:05.2f}  #{i:05d}", org=(14, 82))
        rgb.append(img)
    fps = len(rgb) / seconds if seconds > 0 else 10.0
    path = ROOT / "out" / f"vla_end_{TAG}_side.mp4"
    write_mp4(path, rgb, fps)
    # 逐帧墙钟另存一份：画面上烧的钟点能和这个文件逐行对上，一镜到底就是可核对的，
    # 不用只靠肉眼。列 = 帧号, Unix 秒（3 位小数）。
    path.with_suffix(".csv").write_text(
        "frame,unix_ts\n" + "".join(f"{i},{ts:.3f}\n" for i, (ts, _) in enumerate(frames)),
        encoding="utf-8")
    return path

cfg = load_cfg(DEFAULT_CONFIG)
rig = cfg["real"]
geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                             dtype=np.float64) / 1000.0
geom.REAL_PERCEP_OFFSET = np.array(
    [float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
    dtype=np.float64) / 1000.0
geom.OBJECTS = list(cfg.get("objects") or [])

import impls.judge_held                                           # noqa: E402,F401
import impls.policy_real                                          # noqa: E402,F401
import real.arm                                                   # noqa: E402,F401
import real.cameras                                               # noqa: E402,F401
import real.perception                                            # noqa: E402,F401
import skills.home_real                                           # noqa: E402,F401
import skills.pick_real                                           # noqa: E402,F401
from core.round_state import RoundState                           # noqa: E402
from skills.pick_real import strategy as STRAT                    # noqa: E402
from skills.pick_real import skill as SKILLMOD                    # noqa: E402

color = str((cfg.get("objects") or [{"color": "green"}])[0].get("color", "green"))
task = f"pick up the {color} box"

print("=" * 66)
print(f"VLA 末端：脚本管到「腕子对上方块侧面」，之后交给 π0.5（策略 {STRATEGY}）")
print("=" * 66, flush=True)

arm = make("arm", "real", port=str(rig.get("port", "COM10")),
           speed=int(rig.get("speed", 0)),
           fast=bool(rig.get("fast", False)))
arm.terminated = False
cams = {"real_wrist": make("camera", "real_wrist", source=arm,
                           index=int(rig.get("cam_index", 1)))}
percep_kw = ({"cam_name": str(cfg["perception_cam"])}
             if cfg.get("perception_cam") else {})
percep = make("perception", "brown_rect", cams=cams, ref=None,
              objects=geom.object_specs(), **percep_kw)

side_cfg = (cfg.get("collect") or {}).get("side_cam") or {}
side = None
_rec = None
_t_rec = 0.0
try:
    if side_cfg:
        from real.side_cam import SideCamera                      # noqa: PLC0415
        side = SideCamera(index=int(side_cfg.get("index", 0)),
                          crop=side_cfg.get("crop"),
                          rotate=float(side_cfg.get("rotate", 0.0)))
        cams["side"] = side
        print(f"[侧相机] {side.size()[0]}x{side.size()[1]}", flush=True)
        _rec = _start_recorder(side)
        _t_rec = time.time()
    else:
        print("！配置里没有 collect.side_cam —— π0.5 要两路相机，这一趟必失败",
              flush=True)

    sp = STRAT.apply(STRATEGY)
    print(f"[策略] {sp.get('label')}（{STRATEGY}）：VLA_END={SKILLMOD.VLA_END!r}",
          flush=True)

    home = make("skill", "home_real")
    print("[回初始位置]", home.run(arm, cams), flush=True)

    state = RoundState()
    skill = make("skill", "pick_real")
    out = skill.run(arm, cams, state=state, index=0, percep=percep,
                    instruction=task)
    print("[结果]", out, flush=True)
finally:
    if _rec is not None:
        _rec[1].set()
        _rec[2].join(timeout=2.0)
        _vid = _save_side_video(_rec[0], time.time() - _t_rec)
        print(f"[视频] {_vid}", flush=True)
    for name, obj in (("侧相机", side), ("腕相机", cams.get("real_wrist")),
                      ("机械臂", arm)):
        if obj is None:
            continue
        try:
            obj.close()
        except Exception as exc:                                   # noqa: BLE001
            print(f"关{name}：", exc, flush=True)
