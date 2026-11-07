# -*- coding: utf-8 -*-
"""把「随机抽到的数」记下来，后面每一趟用同一份。

用户 2026-10-19：关节回放要求每一轮开始时方块在同一个位置，所以放置落点和腕转角度
不能再每趟重新抽 —— **第一趟抽到什么就记下来**，之后原样复用。

存 `out/joint_rec/rand_lock.json`，跟关节轨迹放在同一个目录，清的时候一起清。
"""
from __future__ import annotations

import json
from pathlib import Path

PATH = (Path(__file__).resolve().parent.parent.parent
        / "out" / "joint_rec" / "rand_lock.json")

_cache: dict | None = None


def _read() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def get(key: str, default=None):
    return _read().get(key, default)


def put(key: str, value) -> None:
    """记下一个值并立刻落盘。值必须是 JSON 能存的（数字 / 列表）。"""
    data = _read()
    data[key] = value
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                    encoding="utf-8")


def clear() -> None:
    global _cache
    _cache = {}
    try:
        PATH.unlink()
    except OSError:
        pass
