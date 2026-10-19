"""写 mp4。和 `run_sim.py:95` 的 `write_video` 是同一套编码参数，只是不拼多路画面。

**为什么不直接 import 仿真那个**：`run_sim.py` 一 import 就把 `impls/arm_sim.py`、
mujoco 场景、nexus 路径全拉起来（四百多行），真机这一路为了一个十几行的写文件助手
去背那坨依赖不划算，也违背本目录「只往上借 core/」的约定。参数照抄，保持两边
出来的 mp4 用同一个播放器看观感一致。
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np


def write_mp4(path: Path, frames: list, fps: float) -> None:
    """RGB 帧列表 -> mp4。`frames` 里每项是 HxWx3 的 uint8（RGB，不是 BGR）。

    码率固定 crf 23。不做丢帧 —— 按差异丢会让回放忽快忽慢。
    """
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", format="mp4") as c:
        s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        s.width = int(frames[0].shape[1])
        s.height = int(frames[0].shape[0])
        s.pix_fmt = "yuv420p"
        s.options = {"crf": "23", "preset": "veryfast"}
        for img in frames:
            for pk in s.encode(av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(np.asarray(img).astype(np.uint8)), format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
