"""把技能和面板接起来：**一趟跑到底，在技能自己的断点上停下来等人。**

面板上的按钮不发位移、不算判据，只做两件事：

    1. 把技能跑起来（`run_pick`）—— 闸口全开，跑一趟；
    2. 在断点上写 `out/clear_go.json`（`open_gate`）—— 技能照它决定下一步怎么走。

技能停在断点上、和从断点接着走，都是 `skills/pick_real.py` **本来就有**的能力（用户
2026-10-16 定）。这一层一个字都不重写那套：`_confirm_step`（微调每一步）、
`_confirm_pass`（判过了先不合）、`_descend_to_grasp`（下探那一截）、`_confirm_lift`
（横移掉下去之后抬回来）四处，每处都是「拍当前整幅画面 + 写读数 → 等 `clear_go.json`」。

**为什么不自己拼这几段。** 这里面每一段的算式都是几百行的东西（转正挑哪一面、落点按
固定爪贴面算、下探整列解完再一口气走），抄一份到面板里就是「手搓」—— 技能改一版，
面板立刻不是一回事。

**也不走「一段一段重跑」那条路。** 那个是每个按钮都从进场重来一遍（第 4 段实测 29 秒，
其中一多半在重走前面几段），用户明确否掉：「不能我按一下就从头开始」。

`WAIT_TICK` 那一根线在 `run_pick` 里接上：技能停在断点上等的时候，面板的硬件线程
就在 `_wait_cmd` 里面 —— 不接的话，那一刻关节、夹爪、画面全是死的。见
`skills/pick_real.py` 里那个变量的说明和 `panel/rig.py` 的 `pump_wait`。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.registry import make                                   # noqa: E402
from skills import pick_real as PR                               # noqa: E402

# 这一趟要开的闸口。三个环境变量，缺一个就少停一处 —— 面板上的「允许」按了也没反应。
# `PICK_CONFIRM_STEP` 一开，`PICK_CONFIRM_DESCEND` 自动跟着开（`pick_real.py:1517`）。
GATE_ENV = ("PICK_CONFIRM_STEP", "PICK_CONFIRM", "PICK_CONFIRM_DESCEND")

# 续跑入口那个环境变量（`align` / `fine` / `close`，见 `skills/pick_real/skill.py::run`）。
# 它**不是闸口**：自动模式照样会用它，所以不放进 `GATE_ENV`。
START_ENV = "PICK_START_AT"

# 断点登记表：`(读数文件, 短名, 一句话)`。**短名只给面板显示用**，权限按下面 `ALLOWED` 走。
GATES: tuple[tuple[Path, str, str], ...] = (
    (PR.CONFIRM_STEP, "微调", "爪子贴着方块，每挪一步都停一次"),
    (PR.CONFIRM_DESCEND, "下探", "微调收工了，下一步竖直下去合爪"),
    (PR.CONFIRM_LIFT, "抬高度", "横移之后舵机在重力下掉了，要竖直抬回去"),
)

# 每个断点上外面能干什么。**写错命令技能不认会当场判失败退出这一趟**（`not cmd.get("go")`
# 那一条），所以能不能给这件事必须在面板这一头拦：给技能一个它不认的字段，等于按了「放弃」。
ALLOWED: dict[str, tuple[str, ...]] = {
    "微调": ("go", "relook", "descend", "dz", "abort"),
    "判过": ("go", "abort"),
    "下探": ("go", "dz", "abort"),
    "抬高度": ("go", "abort"),
}

INSTRUCTION = "pick up the brown box"


def _read_json(p: Path) -> dict | None:
    """读一份读数文件。**解析不了就当没有**（写的那一瞬间可能只落了一半）。"""
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def gate_now() -> tuple[str, dict] | None:
    """现在停在哪一处断点上？返回 `(短名, 读数)`；一份都没写过就是 `None`。

    **只按文件新旧认**，不看「技能是不是真在等」—— 那件事 `rig` 那头有准数
    （等的时候 `WAIT_TICK` 每一轮都在调，`rig.pump_wait` 记着时间）。两下一凑才知道
    是「正停在断点上」还是「上一趟留下的旧文件」。

    微调那一处和判过那一处**写的是同一份文件**（`clear_step.json`），靠 `pass` 这个键分：
    `_confirm_pass` 写的一定带 `"pass": True`，`_confirm_step` 写的是 `tries` 那一份。

    **判过那一处不叫「合爪」**（用户 2026-10-17：「合爪这个微调和下探中间不该有的步骤」）。
    同一个断点，技能自己那句话是「判过了、**先不合**」—— 这时候爪子上一眼才对准、还没合，
    合爪是它后面那两截（竖直下探 + 再修一次轴偏）走完才发生的事。叫「合爪」会让人以为
    这一段中间夹了一次合爪。
    """
    best = None
    for path, name, _note in GATES:
        rec = _read_json(path)
        if rec is None:
            continue
        try:
            mt = path.stat().st_mtime
        except OSError:
            continue
        if best is None or mt > best[0]:
            kind = "判过" if (name == "微调" and rec.get("pass")) else name
            best = (mt, kind, rec)
    return None if best is None else (best[1], best[2])


def gate_note(kind: str) -> str:
    for _path, name, note in GATES:
        if name == kind:
            return note
    return "判过了，先不合，等你看一眼" if kind == "判过" else ""


def open_gate(what: str, *, dz_mm: float | None = None) -> str:
    """给技能一个答复。返回一句中文，写进面板的日志。

    `what` 是 `ALLOWED` 里的那几个词：
        `go`      照技能算出来的那一步走（微调那处 / 判过那处 / 下探那处都一样）；
        `relook`  原地重新量一眼、再回到这个闸口（`move` 给 0）；
        `descend` 微调到此收工，不再挪水平，直接去走竖直下探那一截；
        `dz`      只改「钉住的那一档高度」（毫米，负 = 往下）；
        `abort`   这一趟不干了，退回去。
    """
    rec = {"go": True}
    if what == "relook":
        rec = {"move": [0.0, 0.0]}
    elif what == "descend":
        rec = {"descend": True}
    elif what == "dz":
        if dz_mm is None:
            return "✗ 改高度得给个数"
        rec = {"go": True, "dz": float(dz_mm)}
    elif what == "abort":
        rec = {"abort": True}
    elif what != "go":
        return f"✗ 不认识的答复 {what!r}"

    PR.CONFIRM_GO.parent.mkdir(parents=True, exist_ok=True)
    PR.CONFIRM_GO.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    says = {"go": "允许，照算出来的走", "relook": "原地重量一眼再回到这儿",
            "descend": "微调收工，直接下探", "abort": "放弃这一趟，退回去"}
    if what == "dz":
        return f"➤ 那一档高度改成 {dz_mm:+.1f}mm"
    return "➤ " + says[what]


def run_pick(rig, instruction: str = INSTRUCTION, *, gates: bool = True,
             start_at: str | None = None, on_start=None) -> dict:
    """跑一整趟，闸口全开 —— 技能自己会在断点上停下来等。

    **必须在硬件线程里调**（`panel/rig.py` 的 `_do` 就是从那儿调的）。技能自己会读舵机、
    发轨迹，跟刷新那条路抢串口是会读花的。

    闸口是**靠环境变量**开的，所以这一趟跑完必须还原 —— 不还原的话，面板上「手动」那一
    套以后每一次动作都带着闸口，技能之外的地方也会莫名其妙停下来。
    """
    prev_env = {k: os.environ.get(k) for k in (*GATE_ENV, START_ENV)}
    prev_tick = PR.knobs.WAIT_TICK
    prev_idle = PR.knobs.WAIT_IDLE
    prev_stop = PR.knobs.STOP_AFTER
    # **续跑入口**（用户 2026-10-18）：`align` / `fine` / `close`。不给就是从头走。
    # 跟闸口不是一回事 —— 自动模式（`gates=False`）也照样用它。
    if start_at:
        os.environ[START_ENV] = str(start_at)
    else:
        os.environ.pop(START_ENV, None)
    # `gates=False` = 自动模式：一个闸口都不开，技能全程自己走（用户 2026-10-18）。
    # 必须主动**清掉**环境变量：上一次手动那趟跑完虽然还原了，但外壳里本来就带着同样的
    # 变量的话，不清就等于闸口还开着。
    for k in GATE_ENV:
        if gates:
            os.environ[k] = "1"
        else:
            os.environ.pop(k, None)
    # 停在断点上的时候，面板的按键走这根线 —— 见文件头。
    PR.knobs.WAIT_TICK = getattr(rig, "pump_wait", None)
    # 等完了（拿到答复/超时）当场告诉面板「不等了」，面板立刻关掉「允许」按钮，
    # 不留那 0.6 秒的假窗口（用户 2026-10-18：「某些步骤要点两次」）。
    PR.knobs.WAIT_IDLE = getattr(rig, "gate_idle", None)
    PR.knobs.STOP_AFTER = None                    # 分段那一路不用了：这一趟跑到底，停在断点上
    # **把 `state` 自己拿着**：技能里每换一段就写 `state.seg`，面板靠这个知道「停在哪儿」
    # （用户 2026-10-18：关掉自动之后要从当前段接着走）。以前传 `state=None`，技能自己
    # new 一个就地丢掉，面板什么也看不到。
    from core.round_state import RoundState                         # noqa: PLC0415

    state = RoundState()
    rig._skill_state = state
    # **采集要在同一个 `state` 上挂录制**（2026-10-20）：录制靠 `state.seg` 判「从哪一段
    # 开始记、到哪一段停」，自己另 new 一个就对不上技能真正在走的那一段。
    if on_start is not None:
        on_start(state)
    try:
        skill = make("skill", "pick_real")
        out = skill.run(rig.arm, rig.cams, state=state, index=0,
                        percep=rig.percep, instruction=instruction)
        return dict(out) if isinstance(out, dict) else {"ok": bool(out), "why": str(out)}
    except Exception as exc:                                        # noqa: BLE001
        # **把 traceback 打进日志**（2026-10-20）：只回一行 `IndexError: index out of
        # range` 看不出是在哪一句抛的，面板日志里连函数名都没有。
        import traceback                                            # noqa: PLC0415

        traceback.print_exc()
        return {"ok": False, "why": f"技能抛了：{type(exc).__name__}: {exc}", "exc": True}
    finally:
        PR.knobs.WAIT_TICK = prev_tick
        PR.knobs.WAIT_IDLE = prev_idle
        PR.knobs.STOP_AFTER = prev_stop
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
