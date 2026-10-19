"""真机专用：推不动就退回来一点、重新规划再走 —— 防的是「撞上了还在顶」。

**这套不是新写的，是从 `real/servo.py` 的 `Mover` 搬过来的。** 那个文件是个手动点动
工具（`--home` / `--go` / `--upto`），里面 `STALL_DEG` 那一组常量和 `Mover._stream` /
`Mover.to` 已经把这件事写全了：每步比「指令让它动了多少」和「实测动了多少」，判到
推不动就**当场断**，再**沿原路往回退一点**（来路刚才走通过，是已知安全的），返回
`"blocked"`。抓取这条路一次都没调用过 `servo.py`，所以那套在抓取上一次都没生效过。

**为什么不放进 `core/follow.py`**：那个文件仿真也走。仿真的下探本来就有接触那一层
（`skills/pick.py` 的 `bump` / `_gone_guard`），再叠一层 TCP 判据只是多一个会误报的
东西。真机没有接触传感，只剩舵机自己，这一层才有必要。

**尺子换了一把**：`Mover` 在关节空间直接写舵机，比的是关节角（度）。抓取这条路的指令
是**末端位移增量** —— `arm.vel` 已经过斜率限幅、就是真正发出去的那一维 —— 所以这里
比毫米。判据同形，单位不同。

**`Mover` 的硬顶那条（`STALL_DEG`）没搬过来**：它要拿「指令的关节角」，而那个值在
`RealArm.drive()` 内部算完就用掉了，没留下来；在这里复算一遍 `_ik` 等于把 `drive` 重写
一遍，两个数迟早对不上。搬过来的是 `Mover` 自己注释里称作「**即时的那条**」的慢顶判据
（`real/servo.py:221-223`）—— 也是实际会先触发的那条。
"""

from __future__ import annotations

import numpy as np

from core import follow
from core.params import POS_SCALE

# **2026-10-14 先关掉。** 判据比的是「这一步打算走多远」（`arm.vel`，见 `_cmd_mm`）和
# 「实测走了多远」，**不是**真发下去的关节指令。真机上新加的 `last_action` 记下来一看：
# 要它走 1.9mm/步，关节指令只转了零点几度，实测自然到不了 0.5mm 那条地板线 —— 于是
# 每一段都从起步就被判「推不动」。三轮真机跑，进场一次都没走出去过，全是它掐的。
# 先关掉，看臂按自己的节奏能不能走到位；逆解那一件事单独查。开回来就改成 True。
ENABLED = False

# ---- 判据（毫米 / 控制步）---------------------------------------------------
# 来源是 `real/servo.py:227-233` 那一组 tick 常数，换算成这条路的单位：
#   STALL_MIN_CMD_TICKS 6.0 tick ≈ 0.53° 关节角；STALL_DEAD_TICKS 2.0 tick ≈ 0.18°。
# 真机巡航时一步的指令位移实测在 1.8~2.5mm（`out/run_real_5.log` 的 `指令=` 那一列），
# 所以 1.0mm 这条门槛在巡航档够得着；末段慢爬时指令本身就小，够不着 —— 那是想要的，
# 收尾那十几毫米本来就走得慢，不该拿它判推不动。
JAM_MIN_CMD_MM = 1.0        # 指令这一步没走过这么多，不判（起步、末段慢爬都落这儿）
JAM_DEAD_MM = 0.5           # 实测挪不到这么多就算「没动」（噪声地板）
JAM_IDLE_FRAC = 0.25        # 实测行程不到指令行程这么多倍，也算「没动」
# **2026-10-14：改成 10 试过一轮，没用，改回 3。** 猜的是「起步还没跟上就被掐」——
# `out/run_real_8.log` 里给它 10 步（≈1.5 秒），4 次「推不动」照样全是「连着 10 步、
# 实测 0.0mm」。同一轮第 9~11 行还显示：进场段前 40 步实实在在走了约 9mm 才停 ——
# 所以那些 0.0mm 不是「还没起步」，是停住了。加长窗口解决不了。
JAM_HITS = 3                # 连着几步都这样才判。一步约 0.15 秒，三步不到半秒

# ---- 退回来多少 -------------------------------------------------------------
# `Mover` 用的是「退回本段行程的一半」（`STALL_BACKOFF = 0.5`）。这里改成**定长**：
# 目的是松开顶住的那个接触、让重新规划的轨迹从一个停住的姿态起步，不是退到安全位，
# 所以「稍微退回来一点」就够。另外再怎么退也不超过本段行程的一半 —— 本段还没走出
# 5mm 的时候按实际行程退一半，不会退到起点后面去。
BACKOFF_MM = 5.0
BACKOFF_MAX_FRAC = 0.5

# 退回来之后重算再走，最多再来这么多次。**是个真障碍物的话，第二次会卡在同一个地方**，
# 再来更多次只是白顶。卡满就这么认了，把失败交回上层（下探没到底就退回上方）。
JAM_TRIES = 1


def _mm(v) -> float:
    """一个位移向量有多少毫米。"""
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64).ravel()[:3])) * 1000.0


class JamGuard:
    """`core.follow.follow` 的 `guard` 回调：推不动就返回真，让它当场断。

    `follow` 每步调一次 `guard()`，**不带参数**（`core/follow.py:139`），所以状态全在
    对象里：上一次采样到的 TCP，和这一段是从哪儿起步的。

    怎么判「推不动」：这一步**指令让它走多远**（`arm.vel` × `POS_SCALE`）对
    **实测走了多远**（TCP 位移）。
        * 指令没走够 `JAM_MIN_CMD_MM` → 不看。
        * 实测小于 `max(JAM_DEAD_MM, JAM_IDLE_FRAC × 指令)` → 这一步记一笔「没跟上」。
        * 连着 `JAM_HITS` 步都这样 → 判推不动，返回真。

    `arm.tcp()` 在 `follow` 这一步里本来就要读一次（`core/follow.py:146`）：
    `RealArm.read_ticks` 有 `_dirty` 缓存，这里读的是写完之后的第一读，外面那次直接
    吃缓存，所以**净增零次串口读**。
    """

    def __init__(self, arm, *, min_cmd: float = JAM_MIN_CMD_MM,
                 dead: float = JAM_DEAD_MM, idle_frac: float = JAM_IDLE_FRAC,
                 hits: int = JAM_HITS):
        self.arm = arm
        self.min_cmd = float(min_cmd)
        self.dead = float(dead)
        self.idle_frac = float(idle_frac)
        self.hits = int(hits)
        self.start: np.ndarray | None = None    # 这一段从哪儿起步
        self.prev: np.ndarray | None = None     # 上一次采样到的 TCP
        self.run = 0                            # 连着几步「指令动了、实测没动」
        self.jam_tcp: np.ndarray | None = None  # 判掉的那一刻的 TCP
        self.where: dict = {}

    def _cmd_mm(self) -> float:
        """这一步指令让它走多少毫米。`arm.vel` 是动作单位/步，乘 `POS_SCALE` 才是米。"""
        v = getattr(self.arm, "vel", None)
        return 0.0 if v is None else _mm(np.asarray(v, dtype=np.float64) * POS_SCALE)

    def __call__(self) -> bool:
        tcp = np.asarray(self.arm.tcp(), dtype=np.float64).ravel()[:3]
        if self.start is None:
            self.start = tcp.copy()
        if self.prev is None:
            self.prev = tcp.copy()
            return False

        cmd = self._cmd_mm()
        act = _mm(tcp - self.prev)
        self.prev = tcp.copy()

        if cmd < self.min_cmd or act >= max(self.dead, self.idle_frac * cmd):
            self.run = 0
            return False
        self.run += 1
        if self.run < self.hits:
            return False

        self.jam_tcp = tcp.copy()
        d = tcp - self.start
        self.where = {"cmd_mm": cmd, "act_mm": act, "run": self.run,
                      "moved_mm": _mm(d),
                      "dir": np.round(np.asarray(self.arm.vel, dtype=np.float64)
                                      .ravel()[:3], 2).tolist()}
        return True

    def backoff_point(self) -> np.ndarray | None:
        """退回一点点，返回那个点。沿本段来路往回退 —— 来路走过、已知安全。

        退多少是 `BACKOFF_MM`，但不退超过本段行程的一半。本段还没走出去（起点就是
        卡住的地方）就返回 None：没有「来路」可退。
        """
        if self.jam_tcp is None or self.start is None:
            return None
        d = self.jam_tcp - self.start
        span = float(np.linalg.norm(d))
        if span < 1e-9:
            return None
        back = min(BACKOFF_MM / 1000.0, BACKOFF_MAX_FRAC * span)
        return None if back < 1e-9 else self.jam_tcp - d / span * back

    def describe(self) -> str:
        w = self.where
        return (f"指令 {w.get('cmd_mm', 0.0):.1f}mm/步、实测只走了 "
                f"{w.get('act_mm', 0.0):.1f}mm，连着 {w.get('run', 0)} 步；"
                f"本段已走 {w.get('moved_mm', 0.0):.1f}mm，"
                f"顶住的方向（动作单位）{w.get('dir')}")


def guarded_follow(arm, waypoints, *, tries: int = JAM_TRIES, **kw):
    """`follow.follow` 外面包一层防撞：推不动就退回来一点、重算，最多再来 `tries` 次。

    返回 `(停因, 现场)`。停因沿用 `follow` 那一套（`reached` / `stall` / `term` /
    `bad_path` / `unreachable`），另外多一个 `jam`：退了、也重试过了，还是推不动。
    现场是 `{"jams": [...]}`，每条一句话，给日志用。

    **「重算路径」就是再调一次 `follow.follow`** —— 它每次都从 `arm.tcp()` 现铺 toppra
    轨迹（`core/follow.py:86`），退回来之后的新位置天然就是新起点，不用另写重规划。

    `kw` 原样转给 `follow.follow`。`guard` 由这里占着，传了也会被丢掉。
    """
    kw.pop("guard", None)
    if not ENABLED:
        return follow.follow(arm, waypoints, **kw), {"jams": []}
    n = max(int(tries), 0)
    jams: list[str] = []

    for k in range(n + 1):
        g = JamGuard(arm)
        out = follow.follow(arm, waypoints, guard=g, **kw)
        if out != "guard":
            return out, {"jams": jams}

        jams.append(g.describe())
        print(f"    ⚠ 推不动：{g.describe()}", flush=True)

        back = g.backoff_point()
        if back is None:
            print("    ✗ 这一步还没走出去，没有来路可退，不重试", flush=True)
            return "jam", {"jams": jams}
        if k >= n:
            print(f"    ✗ 退了 {n} 次还是推不动，这一段不走了", flush=True)
            return "jam", {"jams": jams}

        print(f"    ↺ 退回 {np.round(back, 4)}（来路上，走过、已知安全），重新规划再走一次",
              flush=True)
        # **退这一趟不带 guard**：正在离开障碍物，这段的判据没有意义（运动方向跟
        # 「往目标压」相反，本来也判不出东西）。退完再看 arm 是不是收了工。
        follow.follow(arm, [back], **kw)
        if getattr(arm, "terminated", False):
            return "term", {"jams": jams}

    return "jam", {"jams": jams}
