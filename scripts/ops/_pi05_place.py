# -*- coding: utf-8 -*-
"""VLA 已经夹住方块之后，**把手里这块轻轻放下**（不重新夹、不重新感知定位）。

    G:/anaconda/envs/sim2real/python.exe -u _pi05_place.py

为什么要单独一支：`_pi05_run.py` 只管把模型的动作块写下去，模型这一趟没有抬升/摆件
这一段。所以方块夹住之后要么再合一趟爪（用户 2026-10-15 明令「一个实验只准合一趟
爪」），要么就照脚本专家那条摆件路子把它放下去。

**摆件这一段一行新逻辑都不写**：直接调 `skills.pick_real.place._place_real`，搬过去、
随机转腕、落到「合爪高度 + `PLACE_RAISE_M`」、**最低点离桌面超过半个边长就拒绝松爪**、
最后慢慢张爪，全是它原有的行为。

四个入参从**此刻的臂**上取，不另设常量：
  · `grasp_z` —— 现在 TCP 的 z。这一趟 VLA 没抬升，此刻的高度就是合爪高度；
  · `center`  —— 现在 TCP 的 x, y 和上面那个 z。它只用来算「离原处多远」和方块下底面；
  · `percep`  —— 只被 `_box_half_m` 用来取配置里那一行形状/尺寸，**不调 `locate`**
                 （方块正夹在爪里，这时候定位是自找麻烦）；
  · `state`   —— 空的 `RoundState`，摆件只用它把 `held` / `grip_ref` 清掉。

**松爪之后不再动臂**（`_place_real` 末尾那次退开照走），本脚本随即退出 —— 下一次
上电时舵机保持在那儿，用户自己去看。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np                                               # noqa: E402

import real.arm                                                  # noqa: E402,F401
import real.cameras                                              # noqa: E402,F401
import real.perception                                           # noqa: E402,F401
from core import geom                                            # noqa: E402
from core.registry import make                                   # noqa: E402
from core.round_state import RoundState                          # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg            # noqa: E402
from skills.pick_real.place import _place_real                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp-json", default=str(ROOT / "out" / "pi05_last_grasp.json"),
                    help="`_pi05_run.py` 记下的合爪高度；不给就用默认那个路径")
    args = ap.parse_args()

    cfg = load_cfg(DEFAULT_CONFIG)
    rig = cfg["real"]
    geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
    geom.REAL_PICK_XY = np.array(rig["pick_xy_mm"], dtype=np.float64) / 1000.0
    geom.REAL_PERCEP_OFFSET = np.array(rig.get("percep_offset_mm") or (0.0, 0.0),
                                       dtype=np.float64) / 1000.0
    geom.OBJECTS = list(cfg.get("objects") or [])

    # 速度档和 `_pi05_run.py` 同一个默认值（400 刻/秒 ≈ 35°/s）；摆件走 `joint_move`
    # 自己的 JVEL，这里给有限的档是为了别让舵机在全速档上被 `write_ticks` 直接拉过去。
    arm = make("arm", "real", port=str(rig.get("port", "COM10")),
               speed=400, fast=bool(rig.get("fast", False)))
    arm.terminated = False

    percep = make("perception", "brown_rect", cams={}, ref=None,
                  objects=geom.object_specs())
    state = RoundState()

    # **松爪高度按「合爪那一刻」的 TCP 算，不按此刻的。** `_pi05_run.py` 在 6 号指令
    # 落到 5° 以下的那一下把当时的 TCP 写进 `out/pi05_last_grasp.json`；抬起来之后再问
    # 「现在多高」，放的时候就会在半空松手 —— 用户 2026-10-25 明令不许高空抛物。
    rec_p = Path(args.grasp_json)
    if not rec_p.is_file():
        print(f"✗ 没找到 {rec_p} —— 不知道合爪时 TCP 多高，不能算松爪高度，不动", flush=True)
        return 2
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    grasp_z = float(rec["grasp_z_m"])
    tcp_g = np.asarray(rec["tcp_m"], dtype=np.float64).reshape(-1)
    center = np.array([tcp_g[0], tcp_g[1], grasp_z], dtype=np.float64)

    tcp = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)
    print(f"放下手里这块：此刻 TCP ({tcp[0] * 1000:+.1f}, {tcp[1] * 1000:+.1f}, "
          f"{tcp[2] * 1000:+.1f})mm，关节 {np.round(arm.angles(), 1)}", flush=True)
    print(f"  合爪记录 {rec.get('ts')}：TCP z {grasp_z * 1000:+.1f}mm "
          f"→ 松爪高度按这一档算", flush=True)

    out = _place_real(arm, percep, center, grasp_z=grasp_z, state=state)
    print(f"摆件结果：{out}", flush=True)
    if not out.get("ok"):
        print("✗ 没放成 —— 方块还在爪里，臂停在原地，等人看", flush=True)
        return 1
    print(f"✓ 放好了：落点 {out.get('at')}，松爪时最低点离桌面 "
          f"{out.get('clearance_mm', 0):.1f}mm", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
