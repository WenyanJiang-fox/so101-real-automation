"""起面板：`G:/anaconda/envs/sim2real/python.exe -m panel`

    python -m panel                     用 configs/real.yaml 里的串口 / 相机
    python -m panel --port COM10 --cam 1
    python -m panel --pic-h 720         画面那半边的初始高度（不裁，等比缩）

**一个原生 Windows 窗口**（tkinter），左边画面、右边按钮 —— 详见 `panel/ui.py` 开头。
**它占着串口和相机。** `run_sim.py` / `view_cam.py` / `mcp_server.py` 开着的时候起不来
—— 会当场报错，不会半死不活地跑。

退出（关窗口 / `q` / `ESC` / Ctrl-C）走三步：**先回初始位置**（`real_home.RAW_HOME`，
和按 `h` 那一下同一条路），再关串口和相机，**力矩始终不松** —— 松了这条臂会塌下来。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="panel", description="真机 SO-101 手动面板")
    ap.add_argument("--config", default=str(_ROOT / "configs" / "real.yaml"))
    ap.add_argument("--port", default=None, help="串口，默认取配置里的")
    ap.add_argument("--cam", type=int, default=None, help="相机 index，默认取配置里的")
    ap.add_argument("--pic-h", type=int, default=620, help="画面那半边的初始高度（像素）")
    ap.add_argument("--run", action="store_true",
                    help="开窗就自己跑一趟（等于按 F1）。跑起来之后断点上照样停着等人点头")
    a = ap.parse_args(argv)

    # **控制台是 GBK 的时候，任何一个非 GBK 字符都会抛 `UnicodeEncodeError`，
    # 而异常是从技能里冒出来的 —— 那一趟直接收工，臂一步都没走。** 2026-10-17 那一趟
    # 就是这么死的：`skills/pick_real.py:3797` 的 `↻ 初始位置看得见`。技能里这种字符
    # 还有十几处（`↻`、`✗`、`→`），堵一处没用。输出一律按 UTF-8 走，编不了的换成
    # `?`；只改这一个入口，`run_sim.py` 那条不动。
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                          # noqa: BLE001
            pass

    from panel.rig import Rig                                       # noqa: PLC0415
    from panel.ui import Panel                                      # noqa: PLC0415

    print("正在装配：串口 + 腕相机 + 感知（这一步要几秒）…", flush=True)
    try:
        rig = Rig(cfg_path=a.config, port=a.port, cam_index=a.cam)
    except Exception as exc:                                        # noqa: BLE001
        print(f"\n✗ 装配失败：{type(exc).__name__}: {exc}\n"
              f"  常见原因：串口被别的进程占着（run_sim.py / view_cam.py / "
              f"mcp_server.py），或者相机 index 不对。", flush=True)
        return 2

    rig.start()
    if a.run:
        # `submit` 只是往进程内的队列里塞一条；`Panel.run()` 里那个 tkinter 主循环跟它
        # 没关系，所以窗口还没画出来就能塞（rig 那条线程自己在泵队列）。
        print("    `--run`：窗口一开就跑一趟。", flush=True)
        rig.submit(kind="pick_run")
    panel = Panel(rig, pic_h=a.pic_h)
    try:
        panel.run()
    except KeyboardInterrupt:
        print("\n    Ctrl-C，收工。", flush=True)
    finally:
        print("    停线程…", flush=True)
        stopped = rig.stop()
        # **回初始位置放在 `stop()` 之后、`close()` 之前**（2026-10-17 用户要的）：
        # `stop()` 已经把硬件线程停了、这一刻臂上没有别人；串口和相机还开着，走得动。
        # 走完再关，这样下次开机不是从「刚才随手停在哪」起算的。
        #
        # **技能没停就别走这一步。** `stop()` 只等 3 秒；技能还在跑的时候，
        # `go_home()` 会和它抢同一条总线，而且它第一句就把 `arm.terminated` 清回
        # False（`out/panel_20261018_target.log` 19:48:03 那次：home 和技能同时写，
        # 臂最后没回家）。那种情况如实说，不装作回过了。
        if stopped:
            print("    回初始位置（走完再关串口）…", flush=True)
            rig.go_home()
        else:
            print("    技能还在跑（3 秒内没停），跳过回初始位置 —— 不跟它抢舵机",
                  flush=True)
        print("    关串口和相机（力矩不松）…", flush=True)
        rig.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
