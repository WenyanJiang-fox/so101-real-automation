"""名字 -> 实现。配置里写 policy=expert，这里就查得出该用哪个类。

加一个新的只要两步：在实现文件上挂个牌子，配置里写上这个名字。主循环不用改。
"""

from __future__ import annotations

from typing import Any, Callable

_TABLE: dict[str, dict[str, type]] = {}


def register(kind: str, name: str) -> Callable[[type], type]:
    """挂个牌子。

    kind 是 arm / camera / policy / judge / skill 之一，name 是配置里写的名字。
    """

    def deco(cls: type) -> type:
        _TABLE.setdefault(kind, {})
        if name in _TABLE[kind]:
            raise ValueError(f"{kind} 里已经有一个叫 {name} 的了，换个名字")
        _TABLE[kind][name] = cls
        return cls

    return deco


def make(kind: str, name: str, **kw: Any):
    """按名字造一个出来。名字不认识就报错，并把现有的都列出来。"""
    table = _TABLE.get(kind, {})
    if name not in table:
        have = "、".join(sorted(table)) or "（一个都还没有）"
        raise KeyError(f"{kind} 里没有 {name}。现在有的是：{have}")
    return table[name](**kw)


def available(kind: str) -> list[str]:
    """这一类现在有哪些名字可用。"""
    return sorted(_TABLE.get(kind, {}))
