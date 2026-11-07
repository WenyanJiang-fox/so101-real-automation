# -*- coding: utf-8 -*-
"""只读探针：看舵机总线还在不在、哪几个舵机应答。

**不写任何寄存器、不动力矩、不发位置指令。** 只做三件事：
  1. 列出这台机器上现有的串口（USB 转接板还在不在）；
  2. 打开 `COM10`，逐个舵机单独读三次位置；
  3. 再走一次同步读（`sync_read_positions`，`RealArm.read_ticks` 用的就是它）。

    G:/anaconda/envs/sim2real/python.exe -u _probe_bus.py
    G:/anaconda/envs/sim2real/python.exe -u _probe_bus.py COM12
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

from serial.tools import list_ports                             # noqa: E402

from servo_bridge import open_controller                        # noqa: E402

SERVOS = (1, 2, 3, 4, 5, 6)


def main() -> int:
    port = sys.argv[1] if len(sys.argv) > 1 else "COM10"
    print("现有串口：", flush=True)
    for p in list_ports.comports():
        print(f"  {p.device}  {p.description}  (hwid={p.hwid})", flush=True)

    print(f"\n打开 {port} …", flush=True)
    c = open_controller(port)
    print(f"  is_open={c.is_open}  baud={getattr(c, 'baudrate', '?')}", flush=True)

    print("\n逐个舵机单独读（每次 `read_position`）", flush=True)
    for sid in SERVOS:
        vals = []
        for _ in range(3):
            try:
                vals.append(c.read_position(sid))
            except Exception as exc:                            # noqa: BLE001
                vals.append(f"ERR {type(exc).__name__}: {exc}")
        print(f"  {sid} 号: {vals}", flush=True)

    print("\n同步读（RealArm.read_ticks 走的就是这一条）", flush=True)
    try:
        got = c.sync_read_positions(list(SERVOS))
        print(f"  返回 {len(got)} 个：{got}", flush=True)
        print(f"  缺的：{[s for s in SERVOS if s not in got]}", flush=True)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  ERR {type(exc).__name__}: {exc}", flush=True)

    if "--scan" in sys.argv:
        print("\n扫 0~20 号（看有没有舵机换了 ID）", flush=True)
        for sid in range(0, 21):
            v = c.read_position(sid)
            if v is not None:
                print(f"  {sid} 号 应答：{v}", flush=True)
        print("  扫完", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
