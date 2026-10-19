"""把面板上那句人话，翻成 `thresholds.json` 里那个颜色键。

面板那一栏让人直接写「红色的方块」「把那块绿色的拿起来」，这一层只做一件事：
从这句话里认出颜色，交回 `red` / `green` / … 这种键名。**认不出就返回 None**，
不猜、不给默认值 —— 猜错颜色等于让感知去认一个不存在的东西。

**认出来 ≠ 能认。** 颜色有没有标定过，去看 `thresholds.json` 的 `color.hsv.<键>`
（`calibrated()`），没标定的那个键用 `real.perception.load_cal` 会当场抛，面板把
那句话照原样显示出来。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TH_PATH = _ROOT / "thresholds.json"

# 每个键认哪些词。**长的排前面**（"深红" 要压过 "红"），见 `parse_color`。
WORDS: dict[str, tuple[str, ...]] = {
    "red":    ("红色", "红", "red"),
    "green":  ("绿色", "绿", "green"),
    "blue":   ("蓝色", "蓝", "blue"),
    "brown":  ("棕色", "褐色", "棕", "褐", "卡其", "brown"),
    "yellow": ("黄色", "黄", "yellow"),
    "orange": ("橙色", "橘色", "橙", "橘", "orange"),
    "purple": ("紫色", "紫", "purple"),
    "white":  ("白色", "白", "white"),
    "black":  ("黑色", "黑", "black"),
}

# 全部词按长度倒序，长的先匹。同长按键名排，保证每次结果一样。
_ORDER = sorted(
    ((w, k) for k, ws in WORDS.items() for w in ws),
    key=lambda kv: (-len(kv[0]), kv[1]))


def parse_color(text: str) -> str | None:
    """从一句话里认出颜色键；一个词都没有就返回 None。"""
    s = str(text or "").strip().lower()
    if not s:
        return None
    for word, key in _ORDER:
        if word in s:
            return key
    return None


def calibrated(path: Path = TH_PATH) -> list[str]:
    """`thresholds.json` 里真正标过的颜色（`color.hsv.<键>` 非空）。"""
    try:
        cal = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    hsv = ((cal.get("color") or {}).get("hsv") or {})
    return sorted(k for k, v in hsv.items() if v)


def known_words() -> str:
    """给界面用的一句提示：认得的词有哪些。"""
    return "红 / 绿 / 蓝 / 棕 / 黄 / 橙 / 紫 / 白 / 黑"
