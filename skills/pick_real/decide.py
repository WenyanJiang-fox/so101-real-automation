"""按这一趟量到的事实现算参数 —— 策略表降级成「还没量到事实时的默认值」。

**为什么要这一层**（2026-10-20 用户定）：以前是「开跑前从 `strategies.json` 选一行，
跑起来不许改」。台面上的情况一变（方块位置、夹爪实际的落点、上一趟扎过头多少），
那一行就不知道了 —— 只有量出来的数知道。

现在这里只回答一件事：**这一趟扎到哪儿**。

    下探终点 = 策略表给的默认值 − 上一趟同形状实测的「扎过头多少」

「扎过头」是量出来的：`_descend_to_grasp` 走完读回 TCP，和它该到的那个 z 比。
负的（比指令深）下一趟就把终点往上抬回来。账本按形状分开记 —— 不同形状的重力负载
和落点高度不一样，混在一起记就不是实测了。

**夹爪那一侧要不要挪不在这里**：那件事只能在 `_set_clearance` 量完第一眼之后才
有数（缝多少像素、轴偏多少），所以由 `strategy._measure_first` 当场量、当场定。
"""
from __future__ import annotations

#: 账本：{形状名: {"overshoot_mm": 上一趟扎过头多少（负 = 比指令深）}}
_ledger: dict[str, dict] = {}

#: 修正量允许的范围（毫米）。**两头都夹住**：往下加是补重力下沉，往上加最多补 10mm；
#: 反过来最多减 2mm（等于终点比默认的还低一点，只在实测「一直扎不到位」时才出现）。
_CORR_LO_MM = -2.0
_CORR_HI_MM = 10.0

#: 外部给的额外偏置（毫米），正 = 终点抬高（扎得浅）、负 = 再往下扎。
#: 现在只有 VLM 重试那一层写它（`ppo/capture.py`）；默认 0 = 与「完美版」逐值等价。
_bias_mm = 0.0
_BIAS_LIMIT_MM = 10.0


def set_bias(mm: float) -> None:
    """写下一趟的额外偏置。**夹在 ±10mm**：再大就不是「微调下探终点」了。"""
    global _bias_mm
    _bias_mm = max(-_BIAS_LIMIT_MM, min(_BIAS_LIMIT_MM, float(mm)))


def bias_mm() -> float:
    return _bias_mm


def remember(shape: str, *, overshoot_mm: float) -> None:
    """记下这一趟的实测偏差（米 → 毫米在调用方换好）。"""
    _ledger[str(shape)] = {"overshoot_mm": float(overshoot_mm)}


def plan(shape: str) -> dict:
    """算这一趟的下探终点。返回 `{"grasp_z_final_mm", "overshoot_mm", "why"}`。"""
    # **每次都现读 `knobs` 上那个名字**：`strategy.apply()` 是改赋值，不是在导入时绑定的
    # 那一份上改；这里要是 `from .knobs import ...`，拿到的永远是 -1mm 的旧值。
    from . import knobs                                          # noqa: PLC0415

    base = float(knobs.GRASP_Z_FINAL_REAL) * 1000.0
    prev = _ledger.get(str(shape)) or {}
    corr = 0.0
    why = "第一趟这个形状，按默认值"
    if "overshoot_mm" in prev:
        corr = -float(prev["overshoot_mm"])          # 扎深 6mm → 往上抬 6mm
        corr = max(_CORR_LO_MM, min(_CORR_HI_MM, corr))
        why = (f"上一趟实测 {float(prev['overshoot_mm']):+.1f}mm（负=比指令深），"
               f"这一趟补 {corr:+.1f}mm")
    if _bias_mm:
        why += f"；外部偏置 {_bias_mm:+.1f}mm"
    return {"grasp_z_final_mm": base + corr + _bias_mm,
            "overshoot_mm": prev.get("overshoot_mm"),
            "why": why}
