"""硬件层的公共叶子：常数、计时、配置读取。**谁都能 import 它，它不 import 别人。**

放这儿是为了不让 `rig_*.py` 那几个 mixin 互相 import 出环：节拍和路径常数只有一份，
`rig.py` 照样把它们转出去，旧的 `from panel.rig import TICK_S` 不会断。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml                                                     # noqa: E402

DEFAULT_CONFIG = _ROOT / "configs" / "real.yaml"

# 快照刷新周期。**这个数就是「实时」的那一格。** 抓一帧约 30ms、走一次批量读舵机
# 约 66ms，加起来 100ms 就是一档；再快也只是把总线占满，读数反而更飘。
TICK_S = 0.10
# 重量一眼的间隔。感知那条链（估形状 + 投剪影）比抓帧贵得多，臂不动的时候读数也不变，
# 所以它按自己的节拍走。
PERCEP_S = 1.50
# 「技能正停在断点上等着」这个判断的新鲜度（秒）。技能等点头的时候每约 0.1 秒调一次
# `WAIT_TICK`（就是 `pump_wait`），它记下时刻；超过这个数没再被调过，就说明技能已经
# 从断点上走开了 —— 盘上那几份读数文件还在，但人不该再看到「允许」按钮。
GATE_FRESH_S = 0.6
# 只出画面、不读舵机的那一档的间隔。跟界面那一层的 `UI_S`（80ms）对齐就够——
# 再快界面也画不了那么快。
FRAME_S = 0.08

# ---- 临时计时（`PANEL_PROFILE=1` 才打）----
_PROF = bool(os.environ.get("PANEL_PROFILE"))
_PROF_T0 = time.perf_counter()


def _prof(tag: str, msg: str) -> None:
    if not _PROF:
        return
    print(f"[prof {time.perf_counter() - _PROF_T0:7.2f}s] {tag}: {msg}", flush=True)


def load_cfg(path=DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
