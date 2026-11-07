"""策略 = 一份参数覆盖表，不是一份新技能。

「微调版」和「一步到位」走的**一直是同一份 `pick_real`**，差别只有两处：
竖直下探的终点（`GRASP_Z_FINAL_REAL`）和要不要走合爪前的画面微调（`_set_clearance`）。
所以这里只做一件事：把 `configs/strategies.json` 里的一行写进技能真正读的那几个名字。

**为什么必须写三处。** `gates.py` 在 import 的那一行就把 `GRASP_Z_FINAL_REAL` 拷成了
自己的名字（`from .knobs import ... GRASP_Z_FINAL_REAL`），之后只改 `knobs` 对它无效；
`measure.py` 读的是 `knobs` 上那一份；`skills/pick_real/__init__.py` 又导出了一份给别人。
`_set_clearance` 同一个道理：`skill.py` 也是 import 时拷走的，要改 `skill` 模块上那一份
（`_oneshot_grasp.py:54,80` 一直是这么做的，这里只是把它收拢成一个函数）。
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = _ROOT / "configs" / "strategies.json"

_cfg: dict | None = None
_cur: str | None = None
_orig_set_clearance = None


# ---- 读表 ----

def load(path=None) -> dict:
    """读策略表。**只在第一次或显式调用时读盘**，之后一直用缓存。"""
    global _cfg
    p = Path(path) if path else DEFAULT_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    if not data.get("strategies") or not data.get("default"):
        raise ValueError(f"{p} 缺 `strategies` / `default`")
    _cfg = data
    return _cfg


def cfg() -> dict:
    return _cfg if _cfg is not None else load()


def names() -> list[str]:
    """阶梯顺序（面板切下一支就按这个顺序转）。"""
    c = cfg()
    return [n for n in (c.get("ladder") or list(c["strategies"]))
            if n in c["strategies"]]


def default_name() -> str:
    return str(cfg()["default"])


def spec(name: str) -> dict:
    s = cfg()["strategies"]
    if name not in s:
        raise KeyError(f"策略表里没有 {name}；现在有：{'、'.join(sorted(s))}")
    return dict(s[name])


def label(name: str) -> str:
    try:
        return str(spec(name).get("label") or name)
    except (KeyError, OSError, ValueError):
        return name


def current() -> str | None:
    return _cur


# ---- 套用 ----

def _measure_first(arm, cams, percep, center, tag, **kw):
    """合爪前先量一眼，再决定挪不挪（2026-10-20 用户定：按当次实测现算）。

    和原来的「跳过微调」只差一步：**先量一眼**。量出来已经在带里（缝在
    `[CLEAR_BAND_LO_PX, CLEAR_BAND_HI_PX]`、`|axis| <= AXIS_TOL_PX`）就不挪 ——
    这就是一步到位；不在带里就交给完整的 `_set_clearance`，让它按实测一步步挪。

    为什么不能直接信「不量也行」：10-20 那一轮第 4 趟就是没量就合爪，合爪 847 步
    停在 36.7°（空抓基准）—— 方块根本不在两爪之间，合了个空。量这一眼约 0.3 秒。
    """
    # **完美版开关**（用户 2026-10-20：「先用完美版抓取当前目标」）：`PICK_CLEARANCE_MEASURE=0`
    # 时连这一眼都不量，水平一点不挪、直接往下扎 —— 就是 `_oneshot_grasp.py` 那一版
    # （`_skip_fine`）的行为。默认不开，按当次实测现算。
    import os                                                   # noqa: PLC0415

    if os.environ.get("PICK_CLEARANCE_MEASURE", "1") == "0":
        print(f"    [完美版] 跳过微调「{tag}」：水平不动，直接往下扎", flush=True)
        return {"ok": True}
    from .pose import _wrist                                    # noqa: PLC0415
    from .rulers import _jaw_check                              # noqa: PLC0415

    look = _jaw_check(_wrist(cams), percep, arm, f"{tag}（现算：先量一眼）",
                      axis_only=bool(kw.get("axis_only")))
    if look.get("ok"):
        print(f"    [现算] 量了一眼已经在带里（缝 {look.get('gap')}、"
              f"轴偏 {look.get('axis')}）—— 不挪，直接往下扎", flush=True)
        return {**look, "tries": 0}
    print(f"    [现算] 量了一眼不在带里（{look.get('why') or '超出门槛'}）"
          f"—— 交给微调按实测挪", flush=True)
    return _orig_set_clearance(arm, cams, percep, center, tag, **kw)


def apply(name: str) -> dict:
    """把一条策略写进技能读的那几个名字。**跑技能之前、硬件线程里调。**

    返回那条策略本身（含 `label` / `note`），调用方拿去打日志。
    """
    global _cur, _orig_set_clearance
    from skills import pick_real as PR                          # noqa: PLC0415
    from skills.pick_real import gates, knobs                   # noqa: PLC0415
    from skills.pick_real import skill as skillmod              # noqa: PLC0415

    sp = spec(name)
    z = float(sp.get("grasp_z_final_mm", -1.0)) / 1000.0
    if _orig_set_clearance is None:
        _orig_set_clearance = skillmod._set_clearance
    knobs.GRASP_Z_FINAL_REAL = z
    gates.GRASP_Z_FINAL_REAL = z
    PR.GRASP_Z_FINAL_REAL = z
    skillmod._set_clearance = (_measure_first if sp.get("skip_clearance")
                               else _orig_set_clearance)
    # 末端交给 VLA 的岔口：`skill.py` 在**下探之前**读这个模块级名字。
    # 每次都显式写一遍 —— 换回 `oneshot` 时必须把它清掉，不能留着上一支的值。
    skillmod.VLA_END = str(sp.get("vla_end") or "")
    _cur = name
    return sp


# ---- 失败码 → 下一条 ----

def code(why: str) -> str:
    """把技能那句中文 `why` 归成一个失败码。**只认已经见过的几种**，其余归 `other`。

    失败码是给「换哪一支」用的，不是新判据 —— 判据仍然只有技能里那一套。
    """
    w = str(why or "")
    rules = (
        (("找不着", "没认出来"), "no_detect"),
        (("轴偏",), "off_axis"),
        (("高度沉到",), "too_low"),
        (("缝",), "no_gap"),
        (("没夹住", "掉了"), "slip"),
        (("下探没走成",), "descend_fail"),
        (("高度",), "too_high"),
    )
    for keys, c in rules:
        if any(k in w for k in keys):
            return c
    return "other"


def next_after(code_: str, tried) -> str | None:
    """下一个该试哪条。`tried` 是这一轮已经试过的名字。

    先看失败码有没有指定（`next_on_fail`），再按阶梯顺序找一条没试过的；
    都试过了返回 `None` —— 调用方据此停采集，不自己造第三条。
    """
    c = cfg()
    if code_ in set(c.get("stop_on") or ()):
        return None
    tried = set(tried or ())
    for n in (c.get("next_on_fail") or {}).get(code_, []) or []:
        if n not in tried and n in c["strategies"]:
            return n
    for n in names():
        if n not in tried:
            return n
    return None
