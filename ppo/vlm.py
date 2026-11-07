"""VLM 看现场：把一趟失败的实测数 + 标注图交给 deepseek-flash，拿回结构化判断。

这一层只做三件事：

    build_packet(out, ...)   把技能的返回收拾成「发给模型的」和「留在本地的」两半
    diagnose(packet)         调 API，拿回 {cause, evidence, action, params, closeness, ...}
    check(decision)          枚举和范围校验；不过就整条作废，退回规则表

**动作不是自由文本。** `action` 只能是 `ppo.policy.ACTIONS` 里那三个名字，参数只能落在
下面 `PARAM_RANGES` 的范围里。越界 = 这一条作废 —— 大模型只在「选哪一档、参数调多少」
这一层出力，不直接控关节；锚点守卫和安全闸照旧在外面兜底。

**`truth` 不发出去。** 技能自己那句 `why` 和 `strategy.code()` 归出来的失败码留在本地，
只用来核对模型的判断准不准（发出去等于把答案告诉它）。
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from pathlib import Path

_API = "https://api.deepseek.com/chat/completions"


def _secret(name: str) -> str:
    """环境变量优先；否则读仓库根目录的 secrets.env（该文件不提交）。"""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    p = Path(__file__).resolve().parents[1] / "secrets.env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, val = line.partition("=")
                if k.strip() == name:
                    return val.strip()
    return ""


_KEY = _secret("DEEPSEEK_API_KEY")
MODEL = os.environ.get("PPO_VLM_MODEL") or "deepseek-flash"

#: 本地微调过的 Qwen3-VL（`swift deploy` 的 OpenAI 兼容口）。**只换 `judge` 这一问**：
#: `diagnose` 继续走 DeepSeek —— 两个模型两种口径，不混在同一条链上。
#: 设 `VLM_JUDGE_BASE_URL=http://127.0.0.1:8100/v1/chat/completions` 即启用。
_JUDGE_API = os.environ.get("VLM_JUDGE_BASE_URL", "").strip()
_JUDGE_MODEL = os.environ.get("VLM_JUDGE_MODEL") or MODEL
_JUDGE_KEY = os.environ.get("VLM_JUDGE_API_KEY") or _KEY

FAIL_CODES = ("no_detect", "off_axis", "no_gap", "too_low", "too_high",
              "slip", "descend_fail", "safety", "other")
ACTIONS = ("baseline", "no_measure", "full")
PARAM_RANGES = {
    "descend_offset_mm": (-10.0, 10.0),
    "approach_height_mm": (45.0, 90.0),
    #: 1 号关节单步最多转多少度。默认 45 = 和安全闸上限一样，等于没加额外限制。
    #: 安全闸报「一步要动 X°（1 号关节，上限 45°）」时，模型把它调小，策略会把
    #: 一大步拆成几步走（`skills/pick_real/motion.py::_goto` 按这个值钳制）。
    "joint1_step_deg": (5.0, 45.0),
    #: **所有关节的单步上限**（夹爪不算）。默认 45 = 不额外限制（安全闸本身也是 45）。
    #: 2026-10-22：那一轮 62 次安全闸里 56 次是 5 号关节一次要转 45.2~51.2°，
    #: VLM 只会给 `joint1_step_deg`，管不到 5 号，等于什么都没改。
    "step_deg": (10.0, 45.0),
    # **目标点平移**（2026-10-21 用户定）：下一趟的定位目标整体挪这么多毫米。
    # 正 dx = 让爪子往基座 +x 挪；正 dy = 往基座 +y 挪。**绝对值，不是增量** ——
    # 下一趟按它重新算偏移，模型不给就还原成标定值。
    "nudge_dx_mm": (-30.0, 30.0),
    "nudge_dy_mm": (-30.0, 30.0),
}

#: **升级时只能从这些动作里选一个**（2026-10-22 用户定）。
#:
#: 上一次给的是连续数值（`nudge_dx_mm=-5` 这种），实测方向来回翻、置信度卡在 0.55 ——
#: 大模型做「从几个选项里挑一个」比做「回归出一个数」稳。每个动作只改一个旋钮，
#: 值 = **在当前值上加的增量**（`resolve_move` 负责换算成绝对值并夹进 `PARAM_RANGES`）。
MOVES: dict[str, tuple[str, dict[str, float]]] = {
    "none":       ("什么都不改，换一档再试", {}),
    "x_plus":     ("定位目标往 +x 挪 10mm（臂往前伸）", {"nudge_dx_mm": 10.0}),
    "x_minus":    ("定位目标往 −x 挪 10mm（往回缩）", {"nudge_dx_mm": -10.0}),
    "y_plus":     ("定位目标往 +y 挪 10mm（俯视往左）", {"nudge_dy_mm": 10.0}),
    "y_minus":    ("定位目标往 −y 挪 10mm（俯视往右）", {"nudge_dy_mm": -10.0}),
    # **2026-10-23 用户定**：VLM 只提少量变量，绝大部分仍由三档策略自己决定，
    # 且变量不能影响之后的策略。因此悬停高度 / 下探深度 / 档位全部从枚举里拿掉，
    # 只剩「感知偏移」和「安全限幅」两类。
    # **增量是 −10，不是 20**：`resolve_move` 加在默认值 30 上。写 20 会得 50、
    # 被 `PARAM_RANGES` 夹回 45 —— 等于什么都没改（2026-10-22 实测 43 次）。
    "joint1_20":  ("1 号关节单步上限压到 20°", {"joint1_step_deg": -10.0}),
    "step_minus10": ("所有关节单步上限再降 10°（最低 10°）", {"step_deg": -10.0}),
    "step_minus20": ("所有关节单步上限再降 20°（最低 10°）", {"step_deg": -20.0}),
}

#: 改定位目标的动作。**模型不能自己发明方向**，只能从程序算出的候选里挑
#: （见 `nudge_candidates`）。2026-10-22 实测：它凭历史猜的三次里两次方向相反。
NUDGE_MOVES = ("x_plus", "x_minus", "y_plus", "y_minus")

#: 只能让动作更小 / 更慢的动作。这类改动不会把臂往错的地方带。
LIMIT_MOVES = ("joint1_20", "step_minus10", "step_minus20")

#: 各旋钮「没给过」时的起点，和 `apply_action` / 技能的默认值一致。
_PARAM_DEFAULT = {"nudge_dx_mm": 0.0, "nudge_dy_mm": 0.0,
                  "descend_offset_mm": 0.0, "approach_height_mm": 80.0,
                  # 1 号的默认是 30（`skills/pick_real/knobs.JOINT1_STEP_DEG`），
                  # 不是 45。写成 45 会让 `joint1_20` 永远夹回 45。
                  "joint1_step_deg": 30.0, "step_deg": 45.0}


def resolve_move(move: str | None, current: dict | None) -> dict:
    """把枚举动作换成**绝对**参数值（当前值 + 增量，夹进 `PARAM_RANGES`）。

    未知动作 / 空动作返回 `{}`，表示这一趟不碰任何旋钮。
    """
    if move not in MOVES:
        return {}
    cur = dict(current or {})
    out: dict = {}
    for k, delta in MOVES[move][1].items():
        lo, hi = PARAM_RANGES[k]
        base = float(cur.get(k, _PARAM_DEFAULT[k]))
        out[k] = round(min(hi, max(lo, base + float(delta))), 2)
    return out


def residual_mm(state: dict | None) -> tuple[float, float] | None:
    """下一趟该把定位目标往基座系挪多少毫米（带符号）。

    优先用「下探前后的方块中心之差」—— 那是这一趟真的把方块推走多少。
    没量到两个中心时（合爪前的检查就没过），退回下探前那一眼的「离轴」残差
    `p_mm × (nx, ny)`：方块实测中心离合爪射线还有多远，就挪多少。
    两条都拿不到返回 None。
    """
    s = state or {}
    a, b = s.get("center_before_mm"), s.get("box_center_after_mm")
    if (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
            and len(a) >= 2 and len(b) >= 2):
        try:
            return (float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        except (TypeError, ValueError):
            pass
    fp = (s.get("clear") or {}).get("footprint") or {}
    try:
        p = float(fp["p_mm"])
        nx, ny = float(fp["nx"]), float(fp["ny"])
    except (KeyError, TypeError, ValueError):
        return None
    return (p * nx, p * ny)


def measured_nudge(state: dict | None, move: str) -> dict:
    """这个方向上**实测**该挪多少毫米（绝对值，带符号）。

    2026-10-22 实测：nudge 固定 10mm，而方块每趟被推 15~30mm —— 修正量比偏移量小，
    跟不过去。改成按实测残差取值（见 `residual_mm`）。
    量不出来返回 `{}`。
    """
    d = residual_mm(state)
    if d is None:
        return {}
    i = 0 if move in ("x_plus", "x_minus") else 1
    lo, hi = PARAM_RANGES["nudge_dx_mm"]
    return {"nudge_dx_mm" if i == 0 else "nudge_dy_mm":
            round(min(hi, max(lo, d[i])), 2)}


def nudge_candidates(state: dict | None) -> list[str]:
    """程序从这一趟的实测残差里算出方向，只给出朝那个方向的候选。

    残差见 `residual_mm`。**两个来源都拿不到就返回空表**，这时位移类动作
    一个都不能选 —— 宁可不动，也不能让模型凭一段历史猜方向。
    """
    d = residual_mm(state)
    if d is None:
        return []
    out: list[str] = []
    if abs(d[0]) >= 2.0:
        out.append("x_plus" if d[0] > 0 else "x_minus")
    if abs(d[1]) >= 2.0:
        out.append("y_plus" if d[1] > 0 else "y_minus")
    return out


MOVES_DOC = """move 只能从下面选**一个**。一次只改一个旋钮，且**只作用于下一集**：
  none        什么都不改，交给三档策略自己选档
  x_plus      定位目标往 +x 挪（臂往前伸），幅度由实测位移决定
  x_minus     定位目标往 −x 挪（往回缩），幅度由实测位移决定
  y_plus      定位目标往 +y 挪（俯视看往左），幅度由实测位移决定
  y_minus     定位目标往 −y 挪（俯视看往右），幅度由实测位移决定
  joint1_20   1 号关节单步上限压到 20°（安全闸报「一步太大」时用）
  step_minus10  所有关节单步上限降 10°（安全闸报的是别的关节时用）
  step_minus20  所有关节单步上限降 20°

**档位、悬停高度、下探深度都不归你管** —— 三档策略自己怎么走还怎么走。
`action` 字段照填，但程序不使用它。

**位移类（x_plus / x_minus / y_plus / y_minus）只有出现在 `state.nudge_candidates`
里时才能选** —— 那几项是程序按这一趟实测的方块位移算出来的，不在表里的方向一律拒收。
**你只选方向，挪多少由程序按 `box_center_after_mm − center_before_mm` 的实测分量算**，
上下限 ±30mm。
表是空的就说明这一趟没量到下探前后的方块中心，**这时别选任何位移类**，选 `none`
或者只动限幅类。

限幅类（joint1_20 / step_minus10 / step_minus20）和保守类（hover_up / shallower）
只准往「更小、更慢、更浅」的方向调；给一个比现在更大的值会被当场拒掉。"""

FACTS_DOC = """`state` 里这几项是**算好的事实，直接用，不要自己换算**：
  failure_code       我们程序给这一集的判定（off_axis / slip / no_detect ...），**以它为准**
  center_before_mm   下探之前量到的方块中心（基座系毫米）
  box_center_after_mm 下探之后的方块中心 —— 和上面一比就是方块被推走多少
  nudge_mm           当前生效的目标点平移（毫米，基座系）；[0,0] = 用的标定值
  workspace          臂能到哪（毫米）：
                       reach_ring_mm    可达环的半径 [内, 外]
                       ring_center_mm   可达环的环心 —— 半径是绕它量的
                       start_xy_mm      这一趟进场的起搜中心（可能是上一趟的落点）
                       calib_pick_xy_mm 配置里写死的标定位，不是这一趟用的位置
                       table_z_mm / percep_offset_mm / place_radius_mm
  has_jaw_image      true 才有【失败瞬间】那张标注图；false 时下面 `contour_ok` 给 null
  nudge_candidates   这一趟**允许**选的位移动作（程序按实测残差算的）：
                     优先用「方块被推走的位移」，量不到就用「离轴」残差 p_mm。
                     空表 = 两条都没量到，任何位移类都不许选
  vlm_params_now     当前生效的旋钮值（上一趟选的动作已经算进去了）"""

#: **提示词和校验共用的变量清单。** 改范围只改 `PARAM_RANGES`，改说明只改这里。
#: 这一段要明确告诉模型「它不指挥机械臂」，只动这几个旋钮。
VARS_DOC = """params 只能给下面这些字段。**只给需要改的，其余省略 = 保持我们现在的值**。
悬停高度、下探深度不归你管，给了也不收：
  nudge_dx_mm: -30 ~ 30。下一趟的定位目标往基座 +x 方向整体挪多少毫米
               （+x = 臂往前伸的方向；正 = 往前挪，负 = 往回挪）
  nudge_dy_mm: -30 ~ 30。同上，往基座 +y 方向（俯视看是往左）
    这两个 nudge 是**相对标定值的绝对量，不是增量**：下一趟按新值重新定位，不是在上一次
    基础上再叠。看不出往哪边偏就给 0（或干脆不写这两个字段）。
  joint1_step_deg: 5 ~ 45。**只管 1 号关节**，默认就是 30（已经生效，再给 30 = 没改）。
    安全闸报「一步要动 X°（1 号关节）」时才用它；报的是别的关节时给这个没用。
  step_deg: 10 ~ 45。**所有关节的单步上限**，默认 45（= 不额外限制）。
    安全闸报「一步要动 X°（N 号关节）」且 N 不是 1 时，用它：给 35 或 30，
    我们自己的策略会把这一大步拆成几步走，不需要你给中间点。
    默认关着 —— 只有在真的撞上安全闸时才建议动它。"""


_PROMPT = """你是 SO-101 机械臂的抓取失败诊断器。图是腕相机拍的，\
每张图前面都有文字说明它拍的是哪一刻（失败瞬间 / 下探前）。\
**如果**里面有一张【失败瞬间】的标注图，它上面画好了两爪尖的合爪轴、方块轮廓、\
以及轴偏/缝的像素读数；**没有这张图**（`state.has_jaw_image` 为 false）时，\
不要说「图上画了什么」——你看到的只有结束时的画面。
下面的 JSON 是这一趟的实测数（**全部是毫米 / 像素，基座坐标系**）。

做四件事：
1. 判断这一趟为什么没夹住（cause）。
2. 只改哪一个旋钮（move）。档位（action）由我们自己的策略决定，照填即可、不作数。
3. 给这一趟打 closeness：0 = 完全错过，1 = 差一点就夹住。
4. **检查画出来的方块轮廓对不对**（`contour_ok` + `contour_why`）：只看**失败瞬间**那张图，
   图上的红线是程序按颜色掩膜框出来的「方块」，青/绿/橙三条线是三个爪位姿，品红直线是
   合爪轴。红线跟画面里真实的方块对不上就算 false —— 具体指：把阴影/反光/别的东西框进来了、
   只框到方块的一部分、或者整条框在方块外面。**这是唯一一件只有你能判、程序自己没有数的事**：
   `state.clear` 里那些轴偏/缝读数都是基于这条红线算的，红线错了它们全都跟着错。
   **没有【失败瞬间】标注图时（`state.has_jaw_image` 为 false）`contour_ok` 给 null**，
   不要瞎猜。

`history` 是这一轮之前每一趟的结果。**已经试过、而且没有改善的档不要再选** ——
下一趟要换一个真正不同的变量。

只输出一个 JSON 对象，不要解释，不要 markdown 代码块。

cause 只能是：no_detect, off_axis, no_gap, too_low, too_high, slip, descend_fail, safety, other
action 只能是：baseline（先量一眼再微调）, no_measure（不量直接下扎）, full（完整微调）

**你只调旋钮，不指挥机械臂。** 不要给目标点、关节角、轨迹或速度 —— 那些由我们自己的
策略算。你选的档位和动作会直接写进策略，然后三档策略照旧继续跑。

""" + MOVES_DOC + """

""" + FACTS_DOC + """

`state.trigger` 说明为什么叫你：
  all_tiers_failed = 同一个目标的三档策略都试过、都没成，现在需要换一个变量；
  safety          = 安全闸拦下了一次指令（`state.safety_why` 是原话）。

contour_ok 只能是 true / false；contour_why 一句话说红线错在哪（对得上就写"对得上"）。

输出结构（只输出这一个 JSON 对象）：
{"cause":"...","evidence":"...","action":"...","move":"none","closeness":0.0,"confidence":0.0,"contour_ok":true,"contour_why":"...","next_hint":"..."}
"""


def _jsonable(v):
    """把 numpy / 元组 / 路径收拾成能 dump 的。"""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "tolist"):
        return _jsonable(v.tolist())
    return str(v)


def _mm3(v):
    """米三元组 → 毫米三元组（一位小数）；拿不出三元组就原样还回去。"""
    if v is None:
        return None
    try:
        return [round(float(x) * 1000.0, 1) for x in tuple(v)[:3]]
    except (TypeError, ValueError):
        return v


def _history_lines(history: list | None) -> list[str]:
    """把账本压成一行一趟。**这是模型知道「上一档试过没有」的唯一来源。**"""
    from .policy import ACTIONS                                    # noqa: PLC0415

    label = {a.name: a.label for a in ACTIONS}
    lines: list[str] = []
    for r in history or []:
        s = (f"第{r.get('n')}趟：动作={label.get(r.get('action'), r.get('action'))}，"
             f"{'成' if r.get('ok') else '没成'}")
        if r.get("grip_deg") is not None:
            s += f"，6 号={float(r['grip_deg']):.1f}°"
        if r.get("box_shift_mm") is not None:
            s += f"，方块被推={float(r['box_shift_mm']):.1f}mm"
        c = r.get("center")
        if c:
            s += f"，当时中心=({float(c[0]) * 1000:.0f},{float(c[1]) * 1000:.0f})mm"
        nd = r.get("nudge")
        if nd:
            s += f"，当时 nudge=({float(nd[0]):.0f},{float(nd[1]):.0f})mm"
        if r.get("failure_code"):
            s += f"，程序判定={r['failure_code']}"
        if r.get("cause"):
            s += f"，你上一次的判断={r['cause']}"
        if r.get("contour_ok") is not None:
            s += "，你上一次判轮廓" + ("对得上" if r["contour_ok"] else "对不上")
        if r.get("params"):
            s += f"，当时参数={json.dumps(r['params'], ensure_ascii=False)}"
        lines.append(s)
    return lines or ["（这是这一轮的第一趟，之前没有记录）"]


def build_packet(out: dict, *, attempt: int, strategy: str,
                 history: list | None = None, images: list[str] | None = None,
                 truth_code: str | None = None,
                 workspace: dict | None = None) -> dict:
    """把技能交回来的 `out` 收拾成一份包。

    **2026-10-22 起失败码下发**：`truth_code` 同时写进 `state.failure_code` 发给模型。
    理由是它作为升级控制器必须知道这一集是 `slip` 还是 `off_axis`（以前靠看图猜，
    8 次里有 3 次和程序判定不一致）；技能自己那句 `why` 仍然只留在本地。
    **长度一律是毫米**：以前中心是米、其余是毫米，模型得自己换算。
    """
    out = dict(out or {})
    # **`clear.why` 要摘掉**：那是技能自己下的结论（「高度沉到桌面附近（-19.1mm）」），
    # 发出去等于把答案递给模型 —— 第一轮它原样抄进 `evidence` 了。`clear` 里其余
    # 读数（缝、轴偏、沿轴、像素坐标）是原始测量，留着。
    clear = {k: v for k, v in (out.get("clear") or {}).items() if k != "why"}
    close = out.get("close") or {}
    state = {
        "attempt": int(attempt),
        "strategy": str(strategy),
        "ok": bool(out.get("ok")),
        "verdict": out.get("verdict"),
        "grip_deg": out.get("grip_deg"),
        "empty_deg": out.get("empty_deg"),
        "box_shift_mm": out.get("box_shift_mm"),
        "center_before_mm": _mm3(out.get("center_before")),
        "box_center_after_mm": _mm3(out.get("box_center_after")),
        # 这一趟实际给目标点加了多少平移（毫米，基座系 +x/+y）；0,0 = 用的标定值
        "nudge_mm": out.get("nudge_mm"),
        # 真合过爪才有这几项：合了多少步、开口从多少弧度合到多少
        "close": ({"steps": close.get("steps"),
                   "open_rad": None if close.get("q0") is None else round(float(close["q0"]), 3),
                   "end_rad": None if close.get("q") is None else round(float(close["q"]), 3)}
                  if close else None),
        # 合爪前那一眼的读数（轴偏 / 缝 / 3D 离轴、沿轴）都在这份里
        "clear": _jsonable(clear),
        # 程序给的失败判定：升级控制器需要它，**不是**让它抄答案（`why` 仍不下发）
        "failure_code": truth_code,
        "workspace": _jsonable(workspace) if workspace else None,
    }
    return {
        "state": _jsonable(state),
        "history": _history_lines(history),
        "images": [str(p) for p in (images or [])],
        "truth": {"code": truth_code, "why": str(out.get("why") or "")},
    }


def _image_block(path: str) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    data = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + data}}


def _caption(path: str) -> str:
    """每张图**前面**必须有一段文字说明它是哪一刻。

    原来三张图直接摞在一起、提示词又说「图是失败瞬间」，而排第一的是下探前的画面 ——
    模型会混着读（2026-10-20 第一次真机跑就是这样）。
    """
    n = Path(path).name
    if "jaw_check" in n:
        return "这一张是【失败瞬间】：合爪前最后一眼，图上画了合爪轴、方块轮廓和像素读数。"
    if "jaw_raw" in n:
        return "这一张是【失败瞬间的原始帧】：同一刻，没有标注。"
    if "look_see" in n:
        return "这一张是【下探前】的画面：还没下扎，只供参考，不是失败那一刻。"
    if "grasp_end" in n:
        return "这一张是【这一集结束时的腕相机画面】：合爪之后、抬升之前。"
    if "side_view" in n:
        return "这一张是【侧视相机画面】：固定三脚架、未标定，只看机械臂姿态。"
    return "腕相机画面（未标明时刻）。"


#: 上一次调用的原始响应（`finish_reason` / `content` / `reasoning_content` / `usage`）。
#: 解析失败时调用方把它落盘 —— **没有这份就看不出是推理吃光 token 还是图被拒**。
LAST: dict = {}


def diagnose(packet: dict, *, timeout: float = 240.0,
             max_tokens: int = 3000) -> dict | None:
    """调一次 API。**失败返回 None**（调用方退回规则表），不抛。"""
    content: list[dict] = [{"type": "text", "text": _PROMPT}]
    for p in packet.get("images") or []:
        blk = _image_block(p)
        if blk is not None:
            content.append({"type": "text", "text": _caption(p)})
            content.append(blk)
    payload = {"state": packet.get("state"), "history": packet.get("history")}
    content.append({"type": "text",
                    "text": "实测数：\n" + json.dumps(payload, ensure_ascii=False)})
    # **`thinking` 必须关掉。** 开着的时候这一套提示会把它带进无底洞：实测
    # `finish_reason=length`、3000 / 8000 个 completion token **全进 `reasoning_tokens`**、
    # `content` 一个字符都没有（reasoning 写到 29538 字还没收）。关掉之后 1~2 秒返回、
    # 160 个 token 出完 JSON。同一张图、同一段提示，差别只在这一个字段。
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(max_tokens),
            "thinking": {"type": "disabled"}}
    req = urllib.request.Request(
        _API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + _KEY})
    LAST.clear()
    LAST.update({"model": MODEL, "images": len(packet.get("images") or []),
                 "max_tokens": int(max_tokens)})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        msg = d["choices"][0]["message"]
        LAST.update({"finish_reason": d["choices"][0].get("finish_reason"),
                     "usage": d.get("usage"),
                     "content": msg.get("content"),
                     "reasoning_content": msg.get("reasoning_content")})
        text = (msg.get("content") or "").strip()
    except Exception as exc:                                      # noqa: BLE001
        LAST["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [vlm] 调用失败：{type(exc).__name__} {exc}")
        return None
    return parse(text)


#: **合爪前那一问**（2026-10-23 用户定）。和 `_PROMPT` 是两件事：
#: 那边问「下一步怎么调」，这边只问「此刻方块在不在两爪之间」。
#:
#: **不给它任何程序算出来的数**（离轴 / 缝 / 轴偏一个都不发）：这一问的价值就在于
#: 它是独立的一票，把程序的读数发过去等于让它复述。图也只发**原始帧**，
#: 不发 `jaw_check.jpg` 那张画了读数、写了 PASS/FAIL 的。
_JUDGE_PROMPT = """你看的是一台机械臂合爪之前的腕相机画面。

给你两张图：
  ·【此刻】下探之后、爪子张开、还没合拢的那一刻；
  ·【参考】同一趟下探之前的位置。

只回答一件事：**方块此刻是不是落在两个爪子之间、合拢时能夹到它。**

- `yes`：方块在两爪之间，合拢会夹到它；
- `no`：方块在两爪之外、已经被推出爪口，或者顶在爪面上，合拢会夹空；
- `unsure`：画面里看不清（被爪子挡住、方块不在视野里、光太暗）。

不要猜、不要给建议、不要评价策略。只输出一行 JSON，不要别的字：
{"between": "yes|no|unsure", "why": "不超过30个字，说你在图上看到的位置关系", "confidence": 0.0到1.0}
"""

_JUDGE_OK = {"yes", "no", "unsure"}
_JUDGE_ALIAS = {"是": "yes", "在": "yes", "在之间": "yes", "true": "yes",
                "否": "no", "不在": "no", "false": "no",
                "看不清": "unsure", "不确定": "unsure", "unknown": "unsure",
                "unclear": "unsure"}


def judge(images: list[str], captions: list[str] | None = None, *,
          timeout: float | None = None, samples: int = 2,
          temperature: float = 0.3) -> dict | None:
    """问一次「方块在两爪之间吗」。**失败返回 None**，调用方按「看不清」办。

    同一组图问 `samples` 次：两次一致才算数，不一致直接记 `unsure` ——
    用户要的是**稳定**的数据，宁可不表态也不拿抖出来的答案当判定。
    """
    if timeout is None:
        timeout = float(os.environ.get("VLM_JUDGE_TIMEOUT", "20.0"))
    content: list[dict] = [{"type": "text", "text": _JUDGE_PROMPT}]
    for i, p in enumerate(images or []):
        blk = _image_block(p)
        if blk is None:
            continue
        cap = (captions[i] if captions and i < len(captions) else _caption(p))
        content.append({"type": "text", "text": cap})
        content.append(blk)
    if len(content) <= 1:
        return None
    answers: list[dict] = []
    t0 = time.time()
    for _ in range(max(1, int(samples))):
        body = {"model": _JUDGE_MODEL,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 200,
                "temperature": float(temperature)}
        # `thinking` 是 DeepSeek 的字段；本地 swift 服务收到会报错，只对远端加。
        if not _JUDGE_API:
            body["thinking"] = {"type": "disabled"}
        req = urllib.request.Request(
            _JUDGE_API or _API, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + _JUDGE_KEY})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            text = (d["choices"][0]["message"].get("content") or "").strip()
        except Exception as exc:                                  # noqa: BLE001
            print(f"  [vlm·judge] 调用失败：{type(exc).__name__} {exc}")
            continue
        raw = parse(text)
        if raw is None:
            continue
        val = str(raw.get("between") or "").strip().lower()
        val = _JUDGE_ALIAS.get(val, val)
        if val not in _JUDGE_OK:
            continue
        answers.append({"between": val,
                        "why": str(raw.get("why") or "")[:120],
                        "confidence": raw.get("confidence")})
    secs = round(time.time() - t0, 1)
    if not answers:
        return None
    kinds = {a["between"] for a in answers}
    if len(kinds) != 1:
        return {"between": "unsure", "why": "两次回答不一致：" +
                " / ".join(a["between"] for a in answers),
                "confidence": min(float(a["confidence"] or 0.0) for a in answers),
                "secs": secs, "answers": answers}
    a = answers[0]
    return {"between": a["between"], "why": a["why"],
            "confidence": a["confidence"], "secs": secs, "answers": answers}


def parse(text: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象。抠不出来返回 None。"""
    s = str(text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.split("\n", 1)[-1] if "\n" in s else s
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(s[i:j + 1])
    except Exception:                                             # noqa: BLE001
        return None
    return d if isinstance(d, dict) else None


def filter_params(params: dict | None) -> dict:
    """只保留认识且落在范围内的字段；其余丢掉，**不整条作废**。

    安全闸那条路用它：模型的诊断文字值得留，某个参数越界只丢那一个。
    """
    out: dict = {}
    if not isinstance(params, dict):
        return out
    for k, (lo, hi) in PARAM_RANGES.items():
        if k not in params:
            continue
        try:
            v = float(params[k])
        except (TypeError, ValueError):
            continue
        if lo <= v <= hi:
            out[k] = round(v, 2)
    return out


def check(decision: dict | None,
          state: dict | None = None) -> tuple[dict | None, str]:
    """校验模型的输出。返回 `(可执行的决策, 问题说明)`；不过就返回 `(None, 原因)`。

    `state` 传进来就多查一条一致性：**技能压根没合过爪时，`cause` 不可能是 `slip`**
    —— 那一趟是合爪前的检查没过，`grip_deg` 为空。
    """
    if not isinstance(decision, dict):
        return None, "没有解析出 JSON"
    cause = str(decision.get("cause") or "")
    if cause not in FAIL_CODES:
        return None, f"cause 不在枚举里：{cause!r}"
    action = str(decision.get("action") or "")
    if action not in ACTIONS:
        return None, f"action 不在枚举里：{action!r}"
    # **`move` 是升级路径的唯一输出形式**（2026-10-22）：给了就必须是枚举里的一个。
    move = decision.get("move")
    if move is not None and str(move) not in MOVES:
        return None, f"move 不在枚举里：{move!r}"
    if move is not None:
        params = resolve_move(str(move), (state or {}).get("vlm_params_now"))
    else:
        params = decision.get("params") or {}      # 安全闸那条老路径仍给 params
    if not isinstance(params, dict):
        return None, "params 不是对象"
    # **幅度按实测位移取**（2026-10-23）：固定 10mm 追不上 15~30mm 的推走量。
    # 模型只负责选方向，挪多少由 `measured_nudge` 从下探前后的方块中心算。
    if move is not None and str(move) in NUDGE_MOVES:
        meas = measured_nudge(state, str(move))
        if meas:
            params.update(meas)
    # **位移只能朝实测的方块位移方向**（2026-10-22 用户定：作用可以小，但不能是负的）。
    # 状态里带了候选表就按它卡；不带（安全闸那条老路径）就不动这一条。
    cand = (state or {}).get("nudge_candidates")
    if move is not None and str(move) in NUDGE_MOVES and cand is not None:
        if str(move) not in (cand or []):
            return None, (f"move={move} 不在这次的候选里（{cand or '空'}）："
                          "位移只能朝实测的方块位移方向")
    clean: dict = {}
    for k, (lo, hi) in PARAM_RANGES.items():
        if k not in params:
            continue
        try:
            v = float(params[k])
        except (TypeError, ValueError):
            return None, f"params.{k} 不是数：{params[k]!r}"
        if not (lo <= v <= hi):
            return None, f"params.{k}={v} 越界（{lo}~{hi}）"
        clean[k] = int(v) if k == "side" else round(v, 2)
    # **限幅类只准往小调**：给一个比现在更大的上限，等于把保护撤掉。
    now = (state or {}).get("vlm_params_now") or {}
    for k, dflt in (("joint1_step_deg", 30.0), ("step_deg", 45.0)):
        if k in clean:
            base = float(now.get(k, dflt))
            if clean[k] > base + 1e-6:
                return None, f"{k}={clean[k]} 比现在（{base:g}）还大 —— 只准往小调"
    for k in ("closeness", "confidence"):
        try:
            v = float(decision.get(k))
        except (TypeError, ValueError):
            return None, f"{k} 不是数：{decision.get(k)!r}"
        if not (0.0 <= v <= 1.0):
            return None, f"{k}={v} 不在 0~1"
    # **轮廓判得对不对只记不用**：缺这个字段不当错误（老提示词下模型不会给），
    # 给了就必须是布尔 —— 字符串 "true"/"false" 也收，其余整条作废。
    ck = decision.get("contour_ok")
    if isinstance(ck, str) and ck.strip().lower() in ("true", "false"):
        ck = ck.strip().lower() == "true"
    elif ck is not None and not isinstance(ck, bool):
        return None, f"contour_ok 不是布尔：{ck!r}"
    if isinstance(state, dict):
        if cause == "slip" and state.get("grip_deg") is None:
            return None, "这一趟没合过爪（grip_deg 为空），不可能是 slip"
        if cause == "no_detect" and state.get("clear"):
            return None, "这一趟认到了方块（有轮廓读数），不可能是 no_detect"
    return {"cause": cause,
            "evidence": str(decision.get("evidence") or "")[:120],
            "action": action, "move": None if move is None else str(move),
            "params": clean,
            "closeness": round(float(decision["closeness"]), 3),
            "confidence": round(float(decision["confidence"]), 3),
            "contour_ok": ck,
            "contour_why": str(decision.get("contour_why") or "")[:80],
            "next_hint": str(decision.get("next_hint") or "")[:80]}, ""
