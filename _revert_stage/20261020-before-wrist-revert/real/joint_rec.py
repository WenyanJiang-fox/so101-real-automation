# -*- coding: utf-8 -*-
"""关节回放：把一趟真机跑下来的**指令**按时间原样再走一遍。

记什么：`RealArm.write_ticks` 每次下发的六个舵机原始位置 + `perf_counter` 时间戳。
这是指令、不是读数 —— 每个控制步写什么就是什么，不额外占串口。回放时按记录的时间
间隔把同一串位置再写一遍。

不记什么：感知、判据、画面。回放不看方块在不在，走的是同一条关节轨迹。所以
「回放到底抓没抓住」只能在切点上现读夹爪：`skills/pick_real/skill.py` 在抬升判完
那一刻打一个 `lifted` 标记（`mark`），回放走到那里先停下判，判不过就不往下走、
退回正常逻辑（`panel/rig.py::_run_replay`）。

随机量（放置落点、腕转角度）的冻结不在这里，见 `skills/pick_real/randlock.py`。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "out" / "joint_rec"
REC_NPZ = OUT_DIR / "last.npz"
REC_JSON = OUT_DIR / "last.json"

# 舵机号。跟 `real/arm.py` 的 `SERVO_IDS` 同源，这里写死是为了让存盘文件自解释。
IDS = (1, 2, 3, 4, 5, 6)

# 回放时单步最多等这么久（秒）。手动那一趟在断点上等待的时间会被录进点间隔里，
# 不封顶就会把几分钟原样再等一遍。正常控制步的间隔在 0.03~0.25s，碰不到它。
MAX_GAP = 0.5

# 当前这次录制的 Recorder。`real/arm.py` 的 `WRITE_HOOK` 指向它的 `feed`。
REC: "Recorder | None" = None

# 这一次进程里录成过至少一趟（`panel/rig.py` 用它决定自动模式下一轮敢不敢回放）。
# **不认磁盘上的旧文件**：换了一天、方块早就不在原处了，回放会白走一趟。
HAVE = False


def _jsonable(o):
    """元数据里混进来的 numpy 标量 / 数组转成 JSON 能写的。"""
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Recorder:
    """收 `write_ticks` 的指令流。时间从第一次写入算 0。"""

    def __init__(self) -> None:
        self.t: list[float] = []
        self.rows: list[list[int]] = []
        self.marks: list[dict] = []
        self.meta: dict = {}
        self._last: dict[int, int] = {}
        self._t0: float | None = None

    def feed(self, ticks: dict[int, int]) -> None:
        self._last.update({int(k): int(v) for k, v in ticks.items()})
        if any(i not in self._last for i in IDS):
            # 第一次写就缺舵机号 —— 这一行不能记，记下来回放时会往 0 写。
            return
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self.t.append(now - self._t0)
        self.rows.append([int(self._last[i]) for i in IDS])

    def mark(self, name: str, **kw) -> None:
        """在「当前已经写了多少个点」这个位置打一个切点。"""
        self.marks.append({"name": str(name), "i": len(self.rows),
                           "t": (self.t[-1] if self.t else 0.0), **kw})

    def save(self, meta: dict | None = None) -> Path | None:
        if len(self.rows) < 2:
            return None
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            REC_NPZ,
            t=np.asarray(self.t, dtype=np.float64),
            ticks=np.asarray(self.rows, dtype=np.int32),
        )
        info = {"ids": list(IDS), "n": len(self.rows),
                "seconds": float(self.t[-1]), "marks": self.marks,
                **(self.meta), **(meta or {})}
        REC_JSON.write_text(json.dumps(info, ensure_ascii=False, indent=1,
                                       default=_jsonable), encoding="utf-8")
        global HAVE
        HAVE = True
        return REC_NPZ


def begin(arm=None) -> Recorder:
    """开始录。挂到 `real.arm.WRITE_HOOK` 上，返回 Recorder。"""
    global REC
    import real.arm as _arm                                        # noqa: PLC0415

    rec = Recorder()
    REC = rec
    _arm.WRITE_HOOK = rec.feed
    return rec


def end() -> None:
    """停止录：摘掉钩子。**必须在同一线程里配对调用**（面板的硬件线程）。"""
    global REC
    import real.arm as _arm                                        # noqa: PLC0415

    _arm.WRITE_HOOK = None
    REC = None


def mark(name: str, **kw) -> None:
    """技能里打切点。没在录时什么都不做。"""
    if REC is not None:
        REC.mark(name, **kw)


def has_recording() -> bool:
    """这一次进程里录成过一趟没有。"""
    return HAVE


def load(path: Path | None = None) -> dict | None:
    p = Path(path) if path is not None else REC_NPZ
    if not p.exists() or not REC_JSON.exists():
        return None
    z = np.load(p)
    meta = json.loads(REC_JSON.read_text(encoding="utf-8"))
    return {"t": z["t"], "ticks": z["ticks"], "meta": meta}


def replay(arm, rec: dict, *, speed_scale: float = 1.0, on_mark=None) -> dict:
    """把 `rec` 原样走一遍。

    `on_mark(mark_dict)`：走到切点时调用。返回 False 就当场停手（用于「抬升完判没夹住，
    别往下放」）。`arm.terminated` 每步都看。
    """
    import real.arm as _arm                                        # noqa: PLC0415
    from real.joint_move import goto_q                             # noqa: PLC0415

    t = np.asarray(rec["t"], dtype=np.float64)
    ticks = np.asarray(rec["ticks"], dtype=np.int32)
    meta = dict(rec.get("meta") or {})
    if len(t) < 2:
        return {"ok": False, "why": "记录里没有点"}

    marks = sorted(meta.get("marks") or [], key=lambda m: int(m.get("i", 0)))

    prev_hook = _arm.WRITE_HOOK
    _arm.WRITE_HOOK = None          # 回放本身不再往录制里写
    try:
        # 先把臂限速挪到记录的第一点，避免从当前姿态猛跳过去。
        # **已经在第一点就别走**：`goto_path` 收到两个相同的点会抛「没有路可走」
        # （`core/trajectory.py:100`），而自动模式每一轮都是先回初始位置再回放。
        q0 = arm._ticks_to_qpos({i: int(v) for i, v in zip(IDS, ticks[0])})
        q_now = arm._ticks_to_qpos(arm.read_ticks())
        if float(np.linalg.norm(np.asarray(q_now) - np.asarray(q0))) > 1e-3:
            goto_q(arm, q0, hold=3)

        mi = 0
        due = time.perf_counter()
        prev = float(t[0])
        for k in range(len(t)):
            if getattr(arm, "terminated", False):
                return {"ok": False, "why": "停手", "at": k}
            while mi < len(marks) and int(marks[mi].get("i", 0)) <= k:
                m = marks[mi]
                mi += 1
                if on_mark is not None and on_mark(m) is False:
                    return {"ok": False, "why": f"切点「{m.get('name')}」没判过",
                            "at": k, "mark": m}
            # **按点与点之间的间隔排，不按绝对时间。** 手动那一趟会在断点上停很久，
            # 绝对时间会把那几分钟原样再等一遍；这里把单步间隔封顶 `MAX_GAP`。
            dt_ = min(max(float(t[k]) - prev, 0.0) / float(speed_scale), MAX_GAP)
            prev = float(t[k])
            due += dt_
            nap = due - time.perf_counter()
            if nap > 0:
                time.sleep(nap)
            else:
                due = time.perf_counter()      # 落后了就不追，下一步从现在算
            arm.write_ticks({i: int(v) for i, v in zip(IDS, ticks[k])})

        # 最后一个点再写几遍，等舵机把残余误差吃掉（跟 `goto_path` 的 hold 同理）。
        last = {i: int(v) for i, v in zip(IDS, ticks[-1])}
        for _ in range(4):
            time.sleep(0.05)
            arm.write_ticks(last)
        return {"ok": True, "n": len(t)}
    finally:
        _arm.WRITE_HOOK = prev_hook


def replay_latest(arm, *, on_mark=None, speed_scale: float = 1.0) -> dict:
    rec = load()
    if rec is None:
        return {"ok": False, "why": "盘上没有可回放的记录"}
    return replay(arm, rec, speed_scale=speed_scale, on_mark=on_mark)
