"""硬件那一层：一套真机装备（串口 + 腕相机 + 感知）跑在自己的线程里。

**为什么必须单独一层、还必须分线程。** 界面要一直 `cv2.waitKey` 泵消息，而真机上
任何一个动作都要几百毫秒到几秒（读六个舵机 66ms、写一次 21ms、走一段轨迹几秒）。
动作直接跑在界面线程里，画面就冻在那儿，`stop` 键也按不进去 —— 面板就没用了。
所以：**硬件全部归这一个线程**，界面通过命令队列下单、通过快照读结果。

装配**照抄 `run_sim.py`**（同一份 `configs/real.yaml`、同一段现场常数、同一串
`make()`），在 `panel/rig_assembly.py` 里。

**这一层不写任何策略。** 它只做三件事：把命令变成对 `RealArm` / `RealWristCamera` /
`BrownRectPerceptor` 的调用、把当前状态抄成一份快照、把日志交出去。判据在
`panel/measure.py`，画在 `panel/view.py`，按在 `panel/ui.py`。

这一层按职责分成了四个 mixin，合起来还是同一个 `Rig`（方法名、属性名一个没变）：

    panel/rig_assembly.py   造设备：yaml → arm / cam / percep / cal
    panel/rig_motion.py     动作原语：走轨迹、点动、回初始位置、TCP 平移、量爪行程
    panel/rig_feeds.py      喂数据：画面线程、感知线程、读数、快照
    panel/rig_gate.py       断点：判断、答复、等待期间接键

**这一层占着串口和相机。** `run_sim.py` / `view_cam.py` / `mcp_server.py` 开着的时候
打不开 —— 会当场抛，不会半死。
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from real.arm import SERVO_IDS                                  # noqa: E402

from panel.rig_assembly import AssemblyMixin                    # noqa: E402
from panel.rig_common import (DEFAULT_CONFIG, FRAME_S,          # noqa: E402
                              GATE_FRESH_S, PERCEP_S, TICK_S,
                              _prof, load_cfg)
from panel.rig_feeds import FeedsMixin                          # noqa: E402
from panel.rig_gate import GateMixin                            # noqa: E402
from panel.rig_motion import MotionMixin                        # noqa: E402

__all__ = ["Rig", "DEFAULT_CONFIG", "TICK_S", "PERCEP_S", "GATE_FRESH_S",
           "FRAME_S", "load_cfg"]

#: 一集少于这么多帧就不存 —— 和 `collect_real_dataset.py` 同一条规矩
#: （正常一集 600~900 帧；20 帧都不到的必然是中途异常）。
MIN_FRAMES = 20
#: **同一支连败这么多趟就换下一支**（用户 2026-10-20：「抓不成功就可以换」）。
#: 每趟仍然只合一次爪 —— 换的是下一趟的参数，不是这一趟重试。
TRIES_PER_STRAT = 1
#: 一轮采集最多跑这么多趟（成功的也算）。防止面板没人看着的时候一直跑下去。
MAX_ATTEMPTS = 8


class Rig(AssemblyMixin, MotionMixin, FeedsMixin, GateMixin):
    """一套真机装备 + 一个跑它的线程。界面只调 `submit()` / `snap()`。"""

    def __init__(self, cfg_path=DEFAULT_CONFIG, port=None, cam_index=None,
                 log=None):
        self._log_fn = log
        self._lock = threading.Lock()
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._quit = threading.Event()
        # **急停**。按下时只置这个事件（界面线程，零 I/O）；真正的「开爪 + 回初始位置」
        # 由硬件线程在手里这一件活一松手就做（`_loop` / `pump_wait` 两处都会看）。
        # 为什么不能直接在这儿写舵机：串口只有一个写手，技能还在写的时候插进去，
        # 两边抢总线，`out/panel_20261018_target.log` 19:48 那次就是这么乱的。
        self._estop = threading.Event()
        self._busy = False
        self._moving = False
        # **自动模式**（用户 2026-10-18）：闸口全关、跑完回家、接下一趟。
        # 开关只由面板那几个按钮/键改；`stop`、断点上「放弃」都会把它关掉。
        self._auto = False
        self._round = 0
        # **自动采集数据**（2026-10-20 用户：自动采集，但要有个面板看着）。
        # 开着的时候 = 自动连跑 + 每一趟录成 LeRobot 的一集；`held` 存盘、其余丢掉。
        # `_rec` 是 `real/recorder.py::RealRecorder`，只有采集期间不为 None。
        self._collect = False
        self._rec = None
        self._n_ok = 0                  # 这一轮采集存下的集数
        self._n_bad = 0                 # 这一轮采集丢掉的集数
        self._consec = 0                # 同一支策略连败几趟了
        # **策略阶梯**（用户 2026-10-20）：一次不行换下一支，不是换参数重跑。
        # 名字和参数都在 `configs/strategies.json`；这里只记「这一轮试过哪些」。
        self._strat = ""
        self._strat_label = ""
        self._strat_tried: set[str] = set()
        # **重试策略**（用户 2026-10-20）：采集期间由 `ppo/policy.py` 接下一趟换什么 ——
        # 不限次数，每次只换一个变量，按实测精进。手动跑（不是采集）时它是 None，
        # 面板上的策略按钮还是原来那套。
        self._ppo = None
        self._status = "starting"
        # 最近一次技能的结果（`_run_pick` 写、界面读）。没跑过就是 None。
        self._last_skill: dict | None = None
        # 这一趟技能的 `RoundState`（`skill_driver.run_pick` 挂上来的）。**跑完不清**：
        # 面板要靠 `state.seg` 决定「从哪一段接着走」（用户 2026-10-18），`state.held`
        # 决定能不能续（手里抓着东西就不续）。
        self._skill_state = None
        # 技能最近一次停在断点上等点头的时刻（`pump_wait` 每轮刷新）。0 = 没在等。
        self._gate_t = 0.0
        self._snap: dict = {}
        self._t_frame = 0.0
        self._t_percep = 0.0
        #: 侧视那一帧抄到第几号（`SideCamera.seq()`）。没变就不重复转 BGR。
        self._side_seq = -1
        # 下一眼感知的起搜中心（米，基座系）。**就是上一眼认出来的那个真位置** ——
        # 不用加也不用扣 `percep_offset_mm`，`locate` 交出来的数已经扣过了。
        # 还没认出来过就是 None，那时 `locate` 用配置里的 `pick_xy_mm` 起搜。
        self._seed = None

        self._assemble(cfg_path, port=port, cam_index=cam_index)
        # 策略表读不出来不拦启动：照技能里写死的默认值跑，日志里说清楚。
        try:
            from skills.pick_real import strategy as _strategy    # noqa: PLC0415
            self._strat = _strategy.default_name()
            self._strat_label = _strategy.label(self._strat)
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"✗ 策略表读不出来：{type(exc).__name__}: {exc}"
                      " —— 照技能里的默认值跑")

        self._thr = threading.Thread(target=self._loop, name="rig", daemon=True)
        # 感知那条线程（`_percep_loop`）。**它自己占 CPU**，硬件线程不等它。
        self._p_thr = threading.Thread(target=self._percep_loop, name="percep",
                                       daemon=True)
        # 出画面那条线程（`_pic_loop`）。**跟硬件线程完全无关** —— 见那个方法的说明。
        self._pic_thr = threading.Thread(target=self._pic_loop, name="pic",
                                         daemon=True)

    # ---- 日志 ----

    def set_log(self, fn) -> None:
        """把日志接到界面那块「日志」上（`panel/ui.py` 起窗口之后调）。

        **装配那几行还是直接打终端** —— 窗口还没起来，接不进来。装上之后两处都有：
        界面收一份，终端也照打一份（终端能往回翻，界面那块是有上限的）。
        """
        self._log_fn = fn

    def _log(self, s: str) -> None:
        """一行中文日志。终端一定有；界面接过手就两边都有。

        **这一行不许抛。** 它挂在硬件线程的最外层（`_loop` 的成功分支和 except 分支都在
        调它），抛出去那条线程就整个没了 —— 界面还活着、按什么都没反应，因为队列再也
        没人收。2026-10-17 实测：stdout 重定向到文件时按本地编码写（cp936），
        `▶`(U+25B6) / `✗`(U+2717) 编不出来，`print` 抛 `UnicodeEncodeError`。第一次抛在
        「技能开跑」那行，except 分支里又拨了一次同一行，线程直接结束。

        所以这里兜一层：编不出来就把编不了的字换成 `?` 再写一次；还不行就整行丢掉。
        日志少一行，好过整条线程没了。
        """
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            try:
                enc = sys.stdout.encoding or "utf-8"
                sys.stdout.write(line.encode(enc, "replace").decode(enc, "replace") + "\n")
                sys.stdout.flush()
            except Exception:                                       # noqa: BLE001
                pass
        except Exception:                                           # noqa: BLE001
            pass
        if self._log_fn is not None:
            self._log_fn(line)

    # ---- 命令 ----

    def submit(self, **cmd) -> None:
        # **停手要在界面线程当场关自动。** 只排队的话，自动模式在这一趟返回时会先把
        # `home` + 下一趟 `pick_run` 排进队列（`_loop` 里那一段），之后才轮到这条 stop
        # —— 人看到的就是「按了停手它自己又跑起来」。这里零 I/O，只翻一个开关。
        if cmd.get("kind") == "stop":
            self._auto = False
            # **队列里排着的也扔掉**（用户 2026-10-18）：自动模式早先排进来的
            # `home` / `pick_run` 还堵在前面，只关开关的话这一趟收工之后它们照样跑。
            self._drain_queue()
        cmd["_t_in"] = time.perf_counter()
        self._q.put(cmd)

    def _drain_queue(self) -> None:
        """扔掉自动模式排进去的 `home` / `pick_run`，**其它命令照旧留着**。

        只扔这两类是有意的：急着停手的时候人可能刚好又排了「停手」或「急停」，
        把那条一起扔掉就等于把人的叫停吃了。队列是线程安全的，在界面线程调也安全。
        """
        keep: list[dict] = []
        while True:
            try:
                cmd = self._q.get_nowait()
            except queue.Empty:
                break
            if cmd.get("kind") in ("home", "pick_run"):
                continue
            keep.append(cmd)
        for cmd in keep:
            self._q.put(cmd)

    def request_estop(self) -> None:
        """**急停：在界面线程上直接调，立刻生效。** 只置标志、清零几个开关，一次 I/O
        都不做 —— 写舵机只能由硬件线程一个人干（串口只有一个写手）。

        它做的三件事：
          1. 立 `_estop` 事件，硬件线程一回到手里就「开爪 + 回初始位置」；
          2. 立 `arm.terminated`，正在走的轨迹下一步就停（`real/joint_move.py:74`），
             技能也会在自己的检查点上往外退；
          3. 关掉自动连跑，免得急停完它自己又接一趟。

        整条链在位（`_loop` 顶部、`_do` 之后、`pump_wait` 顶部）都会看这个事件。
        """
        self._estop.set()
        self.arm.terminated = True
        self._auto = False
        # **急停要把排队的一起清掉**（用户 2026-10-18）：不清的话，急停做完之后
        # 队列里那条早先排进来的 `pick_run` 会立刻接着跑 —— 看见的就是「按了急停
        # 它还在跑」。队列操作是线程安全的，这里只是把条目取出来丢弃，不碰串口。
        self._drain_queue()
        self._log("🛑 急停已按下 —— 松爪、回初始位置（在当前这一小步做完之后立刻执行）")

    def start(self) -> None:
        self._thr.start()
        self._p_thr.start()
        self._pic_thr.start()

    def stop(self, join: float = 3.0) -> bool:
        """叫停硬件线程。**返回它是不是真的停了。**

        停了才轮到主线程接着动臂（`__main__` 的 `go_home`）。技能一趟可能还有十几秒
        的活（`out/panel_20261018_target.log`：关窗时技能还在走微调第 1~3 次），
        这里只等 `join` 秒；等不到就如实返回 False —— **调用方别再往舵机写东西**，
        否则两边抢同一条总线，`go_home` 还会把 `arm.terminated` 清回 False，
        技能连「叫停」都不认了。
        """
        self._quit.set()
        self.arm.terminated = True          # 走位中途也能当场停
        self._q.put({"kind": "quit"})
        if self._thr.is_alive():
            self._thr.join(timeout=join)
        self._stop_percep()
        return not self._thr.is_alive()

    def close(self) -> None:
        """关串口、关相机。**不松力矩** —— 松了这条臂会塌下来。"""
        for obj in (self.cam, self.arm):
            try:
                obj.close()
            except Exception as exc:                                # noqa: BLE001
                self._log(f"关闭 {type(obj).__name__} 出错：{exc}")

    # ---- 硬件线程 ----

    def _loop(self) -> None:
        while not self._quit.is_set():
            # **急停优先。** 空闲时（不在跑技能）按下的那一下，在这一拍就做掉，
            # 不用排队 —— `request_estop()` 只置事件，零 I/O。
            if self._estop.is_set():
                self._busy = True
                try:
                    self._estop_run()
                finally:
                    self._busy = False
                    self._moving = False
            # **收命令的等待要短。** 原来等 `TICK_S`（100ms），按一下按钮最多先站
            # 100ms 才轮到它；现在 20ms，刷新节拍照样在下面按 `TICK_S` 判。
            try:
                cmd = self._q.get(timeout=0.02)
            except queue.Empty:
                cmd = None
            if cmd is not None:
                if cmd.get("kind") == "quit":
                    break
                self._busy = True
                try:
                    self._do(cmd)
                except Exception as exc:                            # noqa: BLE001
                    self._log(f"✗ 命令 {cmd.get('kind')} 出错："
                              f"{type(exc).__name__}: {exc}")
                finally:
                    self._busy = False
                    self._moving = False
                # **技能被打断之后，急停接着做。** 跑技能的时候按的那一下，事件早就
                # 立起来了，但技能占着硬件线程；它一松手就轮到这儿 —— 开爪、回家。
                if self._estop.is_set():
                    self._busy = True
                    try:
                        self._estop_run()
                    finally:
                        self._busy = False
                        self._moving = False
                # **自动模式接下一趟。** 放在这里（`finally` 之后、循环里）：这一趟
                # 不管成没成都接着来；先回家再跑 —— 失败那趟可能停在半路。
                if (cmd.get("kind") == "pick_run" and self._auto
                        and not self._quit.is_set()):
                    self._round += 1
                    self._log(f"↻ 自动模式：第 {self._round} 轮收工 → 回初始位置 → 下一趟")
                    self._q.put({"kind": "home"})
                    self._q.put({"kind": "pick_run"})
            now = time.time()
            if now - self._t_frame >= TICK_S:
                try:
                    self._tick(now)
                except Exception as exc:                            # noqa: BLE001
                    self._log(f"✗ 刷新出错：{type(exc).__name__}: {exc}")
                    self._t_frame = now
        self._status = "stopped"

    # ---- 命令的实现。**每一个都是「动作 + 一句话交代结果」。** ----

    def _do(self, cmd: dict) -> None:
        kind = cmd.get("kind")
        t_in = cmd.get("_t_in")
        if t_in is not None:
            _prof("按键", f"{kind} 从按下到开始执行 "
                          f"{(time.perf_counter() - t_in) * 1000:.0f}ms")
        if kind == "stop":
            self.arm.terminated = True
            self._auto = False
            self._log("⏹ 停手 —— 轨迹走到当前这一点就停，不松力矩；自动模式已关")
            # **采集跟着停**：数据集要 `finalize()` 才读得出来，不关的话这一轮白录。
            self._collect_off("停手")
            self._status = "stopped"
            return
        if kind == "collect":
            self._collect_cmd(bool(cmd.get("on")))
            return
        if kind == "strategy":
            self._strategy_cmd(cmd.get("name"))
            return
        if kind == "auto":
            # 不动作、不碰串口 —— 只翻开关。开的时候顺手排一趟；已经在跑的话，
            # 这一条目会在当前这一趟收工之后才轮到。
            self._auto = bool(cmd.get("on"))
            self._round = 0
            self._log("自动模式：开 —— 闸口全关，跑完回初始位置接下一趟"
                      if self._auto else "自动模式：关")
            if self._auto:
                self._q.put({"kind": "pick_run"})
            else:
                # **关掉的时候把还排着的扔掉**（用户 2026-10-18）：FIFO 里那几条
                # `home` / `pick_run` 不扔的话，会紧接着把这一趟又跑起来。
                self._drain_queue()
                # 采集是靠自动连跑驱动的，连跑关了它就没有下一趟了 —— 一起收掉，
                # 免得数据集停在一个没 `finalize` 的半成品状态。
                self._collect_off("自动连跑关掉")
            return
        if kind == "estop":
            # 从 `pump_wait` 进来时（技能停在断点上），这里要当场做完，不能只是置个
            # 事件就返回 —— 返回之后又回到「等点头」，没人会再来看它。
            self._estop.set()
            self.arm.terminated = True
            self._auto = False
            self._estop_run()
            return
        if kind == "color":
            # **不动作、不碰串口**：只换感知的颜色判据，跟 `see` 一样放在
            # `terminated = False` 之前。
            self._set_target(str(cmd.get("text") or ""))
            return
        if kind == "see":
            # 「重量一眼」也**把形状缓存清掉**（`Perceptor.reset_shape`）：这一下就是
            # 人要求的「重新估一遍」，换了一块东西之后按它。
            reset = getattr(self.percep, "reset_shape", None)
            if reset is not None:
                reset()
            self._t_percep = 0.0
            self._tick(time.time(), force=True)
            return
        if kind == "torque":
            self.arm.ctrl.sync_enable_torque(list(SERVO_IDS))
            self._log(f"力矩：已使能（舵机 {list(SERVO_IDS)}）")
            return
        if kind == "grip_cal":
            self._measure_grip_range()
            return
        if kind == "gate":
            # **放在这一组里**（跟 `see` / `torque` 一样，在 `terminated = False` 之前）：
            # 写文件不动臂，不该顺手把「停手」那个标志清掉。
            self._gate_cmd(str(cmd.get("what") or "go"), cmd.get("dz"))
            return

        self.arm.terminated = False
        if kind == "jog":
            self._jog(int(cmd["joint"]), float(cmd["delta_deg"]))
        elif kind == "goto_deg":
            self._goto_deg([float(v) for v in cmd["angles"]])
        elif kind == "grip":
            self._grip(cmd.get("to"), cmd.get("deg"))
        elif kind == "home":
            self._home()
        elif kind == "nudge":
            self._nudge(float(cmd.get("dx", 0.0)), float(cmd.get("dy", 0.0)),
                        float(cmd.get("dz", 0.0)))
        elif kind == "pick_run":
            self._run_pick(str(cmd.get("instruction") or ""), cmd.get("start_at"))
        else:
            self._log(f"？不认识的命令 {kind!r}")

    def _set_target(self, text: str) -> None:
        """面板上那句目标描述 → 换感知认的颜色。

        **先认词、再验标定。** 认不出词就不猜（`targets.parse_color` 返回 None）；
        认出来但那个颜色没标定过，`set_color` 会抛，抛出来的那句本身就写着怎么标，
        原样打到日志里给用户看。
        """
        from panel import targets                                   # noqa: PLC0415

        name = targets.parse_color(text)
        if name is None:
            self._log(f"？没从「{text}」里认出颜色（认得 {targets.known_words()}）")
            return
        try:
            got = self.percep.set_color(name)
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"✗ 目标换成 {name} 不行：{exc}")
            return
        self.color = got
        # 起搜中心是**上一块东西**量出来的，换目标就作废；形状缓存 `set_color` 已经清掉。
        self._seed = None
        self._t_percep = 0.0            # 感知线程下一轮立刻用新颜色看一眼
        self._log(f"🎯 目标 → {got}（色相区间 {self.percep.ranges}）")

    # ---- 断点续跑：停在哪儿 → 从哪个入口接着走 ----

    # 这三张表就是 `skills/pick_real/skill.py` 里 `state.seg` 的原文。**改那边段名要同步
    # 改这儿**，对不上只会导致「续跑按钮不亮」，不会走错段。
    _RESUME_ALIGN = frozenset({
        "行程两端 → 合一次空爪、再张到底", "初始位置先看一眼",
        "进场 → 盒子大致位置上方", "看一眼 → 量盒子的世界坐标",
        "找目标：前后左右 → 抬头拉远", "挪到盒子正上方", "腕子对上方块侧面",
        "下探到盒子",
    })
    _RESUME_FINE = frozenset({
        "合爪前微调", "合爪前竖直下探", "下探之后（只修轴偏）",
    })
    _RESUME_CLOSE = frozenset({"微调判过（还没合爪）"})

    def resume_stage(self) -> str | None:
        """上一趟停在哪一段 → `align` / `fine` / `close`；不能续就是 `None`。

        **手里抓着东西一律不续**：合爪之后到摆件之间停下来的那一趟，方块还在爪里，
        重走抓取那一段会把它扔掉或者撞桌子。
        """
        st = getattr(self, "_skill_state", None)
        if st is None or bool(getattr(st, "held", False)):
            return None
        seg = str(getattr(st, "seg", "") or "")
        # 判过 / 已在往下扎那一支先认：这两段停下来的下一步都是「下探一截并合爪」。
        if seg in self._RESUME_CLOSE or seg == "从断点接上 → 下探一截并合爪":
            return "close"
        if seg in self._RESUME_FINE or seg.startswith("从断点接上"):
            return "fine"
        if seg in self._RESUME_ALIGN:
            return "align"
        return None

    def resume_blocked(self) -> str:
        """不能续跑时给人看的那句话；能续就是空串。

        **手里抓着东西**（合爪之后到摆件之间）一律不续 —— 方块还在爪里，重走抓取
        那一段会把它扔掉或者撞桌子。
        """
        st = getattr(self, "_skill_state", None)
        if st is not None and bool(getattr(st, "held", False)):
            return "手里抓着东西，不能续跑"
        return ""

    # ---- 自动采集：自动连跑 + 每一趟录一集 ----

    def _strategy_cmd(self, name: str | None = None) -> None:
        """换一支策略。**不碰舵机** —— 真正的套用发生在下一趟开跑之前（`_run_pick`）。

        不给名字就按阶梯顺序转下一个。手动换过之后把「这一轮试过哪些」清空，
        这样自动采集还能从头走一遍阶梯。
        """
        from skills.pick_real import strategy as _strategy          # noqa: PLC0415

        ns = _strategy.names()
        if not ns:
            self._log("✗ 策略表里一条也没有（`configs/strategies.json`）")
            return
        if name:
            nxt = str(name)
        else:
            cur = self._strat if self._strat in ns else ns[0]
            nxt = ns[(ns.index(cur) + 1) % len(ns)]
        self._strat = nxt
        self._strat_label = _strategy.label(nxt)
        self._strat_tried = set()
        self._consec = 0
        self._log(f"[策略] 下一趟用「{self._strat_label}」（{nxt}）")

    def _collect_cmd(self, on: bool) -> None:
        """采集的开关。**不碰舵机**，只开/关数据集和自动连跑。"""
        if on:
            self._collect_on()
        else:
            self._collect_off("手动关掉")

    def _collect_on(self) -> None:
        """开始采集：建数据集 → 打开自动连跑 → 排「回初始位置 + 跑一趟」。

        **侧相机打不开就不开始**（2026-10-20）：两路相机是写进 `meta/info.json` 的
        feature，少一路录出来是另一份格式，混在一个目录里读不出来。宁可当场说清楚。
        """
        if self._rec is not None:
            return
        if self.side is None:
            self._log("✗ 采集开始不了：侧相机没打开（看装配那几行的报错），"
                      "这一版只能录腕相机 —— 不开始，免得录成半份格式")
            return
        cfg = dict(self.cfg.get("collect") or {})
        root = cfg.get("root")
        repo = cfg.get("repo_id")
        if not root or not repo:
            self._log("✗ 采集开始不了：`configs/real.yaml` 的 `collect:` 缺 `root` / `repo_id`")
            return
        from real.recorder import RealRecorder                      # noqa: PLC0415

        task = f"pick up the {self.color} box"
        try:
            self._rec = RealRecorder(root, repo, task,
                                     fps=int(cfg.get("fps", 4)),
                                     size=cfg.get("size", "960x540"),
                                     side=self.side)
        except Exception as exc:                                    # noqa: BLE001
            self._rec = None
            self._log(f"✗ 数据集打不开：{type(exc).__name__}: {exc}")
            return
        self._collect = True
        self._n_ok = 0
        self._n_bad = 0
        self._consec = 0
        self._strat_tried = set()
        self._auto = True
        self._round = 0
        # **重试交给 `ppo/`**（用户 2026-10-20）：不限次数，每次只换一个变量。
        # 上限从 `configs/strategies.json` 的 `ppo.max_attempts` 读；没写 = 不限。
        try:
            from skills.pick_real import strategy as _strategy      # noqa: PLC0415

            cap = (_strategy.cfg().get("ppo") or {}).get("max_attempts")
        except Exception:                                           # noqa: BLE001
            _strategy, cap = None, None
        from ppo.policy import RetryPolicy                          # noqa: PLC0415

        self._ppo = RetryPolicy(max_attempts=cap)
        from skills import pick_real as _pr                         # noqa: PLC0415

        _pr.knobs.ANCHOR_XY = None
        self._log(f"● 采集开始：任务「{task}」→ {root}")
        self._log("   每一趟：闸口全关、抓成存盘、没抓成丢掉；"
                  f"重试由 ppo/ 接管 —— {'不限次数' if cap is None else f'最多 {cap} 次'}，"
                  "每次只换一个变量")
        try:
            from skills.pick_real import strategy as _strategy      # noqa: PLC0415
            self._log("   策略阶梯：" + " → ".join(
                _strategy.label(n) for n in _strategy.names())
                + f"（现在从「{self._strat_label or self._strat}」起）")
        except Exception:                                           # noqa: BLE001
            pass
        # 和自动连跑同一条路：先回家，再跑。之后每一趟收工由 `_loop` 接着排。
        self._q.put({"kind": "home"})
        self._q.put({"kind": "pick_run"})

    def _collect_off(self, why: str = "") -> None:
        """停采集并**把数据集收尾**（`finalize()`）。不调这一步整份数据读不出来。

        只碰磁盘，不碰串口 —— 所以主线程在 `stop()` 之后也能安全地调（`__main__` 用）。
        """
        self._collect = False
        self._auto = False
        # **重试账本先交账**：这一轮试了几次、每档各成几次、开口最大多少。
        if self._ppo is not None:
            self._log("[重试] 收工：" + self._ppo.summary())
            self._ppo = None
        try:
            from skills import pick_real as _pr                     # noqa: PLC0415

            _pr.knobs.ANCHOR_XY = None
        except Exception:                                           # noqa: BLE001
            pass
        rec, self._rec = self._rec, None
        if rec is None:
            return
        root = rec.root
        try:
            rec.close()
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"✗ 数据集收尾出错：{type(exc).__name__}: {exc}")
        self._log(f"■ 采集结束（{why or '收工'}）：成功 {self._n_ok} 集 / "
                  f"失败 {self._n_bad} 集 → {root}")

    def collect_close(self, why: str = "退出面板") -> None:
        """退出时调一次。`finalize()` 只做这一次，别让数据集停在半成品。"""
        self._collect_off(why)

    def _run_pick(self, instruction: str, start_at: str | None = None) -> None:
        """跑一整趟。**这是整条链里唯一会走完整策略的地方。**

        跑在硬件线程里（`_do` 就是在 `_loop` 里调的），所以跑的时候不会有人跟它抢串口。
        画面不会冻着：出画面那条线程（`_pic_loop`）跟硬件线程完全无关。

        **闸口全开**（`skill_driver.run_pick` 里设的环境变量）：技能会在「微调每一步 /
        下探那一截之前 / 判过了先不合 / 横移掉下去要抬回来」这四处自己停下来，等外面写
        `out/clear_go.json`。**面板不用管它在哪儿停** —— 等的时候 `WAIT_TICK` 每一轮都在
        调 `pump_wait`，`_gate_t` 就是「现在正等着」的凭据。
        """
        from panel import skill_driver                              # noqa: PLC0415

        name = instruction or skill_driver.INSTRUCTION
        self._status = "skill: pick"
        if start_at:
            says = {"align": "对准", "fine": "微调", "close": "下探一截并合爪"}
            self._log(f"↻ 从断点接着走：入口「{says.get(start_at, start_at)}」"
                      f"（上一趟停在「{getattr(self._skill_state, 'seg', '?')}」）")
        elif self._auto:
            self._log(f"▶ 自动模式第 {self._round + 1} 轮开跑（{name}）—— 全程不停")
        else:
            self._log(f"▶ 技能开跑（{name}）—— 会在断点上停下来等你点头")
        with self._lock:
            # 跑的时候别拿上一拍的读数糊弄人：`step` 清掉，界面就会显示「在动」。
            self._snap["step"] = None
            self._snap["moving"] = True
        # **这一趟要不要录。** `_rec` 只在采集期间存在；`on_start` 在技能自己的
        # `RoundState` 建好之后立刻回调，录制要靠 `state.seg` 判「从哪一段开始录」。
        rec = self._rec if self._collect else None

        def _on_start(state) -> None:
            if rec is not None:
                rec.start(self.arm, self.cam, state)

        # **重试策略接管这一趟的参数**（`ppo/`，用户 2026-10-20）：第几次、换哪个变量、
        # 定位锚点。套在「套用策略」之前 —— 它改的就是 `self._strat`。
        if self._collect and self._ppo is not None:
            act, change = self._ppo.choose()
            self._strat = act.strategy
            os.environ["PICK_CLEARANCE_MEASURE"] = "1" if act.measure else "0"
            from skills import pick_real as _pr                     # noqa: PLC0415

            _pr.knobs.ANCHOR_XY = self._ppo.anchor
            where = ("" if self._ppo.anchor is None else
                     f"；锚点 ({self._ppo.anchor[0] * 1000:.0f}, "
                     f"{self._ppo.anchor[1] * 1000:.0f})mm")
            self._log(f"[重试] 第 {self._ppo.n + 1} 次：{act.label}（{change}）{where}")

        # **套用策略**（用户 2026-10-20）：只在这一刻写技能里那几个名字 —— 跑起来之后
        # 不许再换，免得同一趟前后两半用两套参数。换策略只影响下一趟。
        if self._strat:
            try:
                from skills.pick_real import strategy as _strategy   # noqa: PLC0415
                sp = _strategy.apply(self._strat)
                self._strat_label = str(sp.get("label") or self._strat)
                self._log(f"[策略] {self._strat_label}：下探终点 "
                          f"{float(sp.get('grasp_z_final_mm', -1.0)):+.1f}mm，"
                          f"{'跳过' if sp.get('skip_clearance') else '走'}合爪前微调")
            except Exception as exc:                                # noqa: BLE001
                self._log(f"✗ 策略套用失败（{self._strat}）："
                          f"{type(exc).__name__}: {exc} —— 照技能里的默认值跑")

        t0 = time.time()
        out = skill_driver.run_pick(self, name, gates=not self._auto,
                                    start_at=start_at, on_start=_on_start)
        # **先收这一趟的实测**（`ppo/policy.py`）：6 号停几度、方块被推多少、
        # 下探之后中心在哪 → 下一趟的锚点。**必须在 `begin_target()` 之前** ——
        # 先清账本再记账，这一趟就会记成「第 1 次、起始档」。
        if self._collect and self._ppo is not None:
            self._log("[重试] " + self._ppo.observe(out))
        # **这一趟放下的位置 = 下一趟的起搜中心**（用户 2026-10-20：「目标是由上一趟
        # 放下的」）。不写成文件、不跨进程 —— 自动连跑全在同一个进程里。
        place = out.get("place") or {}
        if place.get("ok") and place.get("at"):
            from skills import pick_real as _pr                     # noqa: PLC0415

            _pr.knobs.GUESS_XY = (float(place["at"][0]), float(place["at"][1]))
            self._log(f"[下一趟] 起搜中心 = 这一趟放下的位置 "
                      f"({float(place['at'][0]) * 1000:.0f}, "
                      f"{float(place['at'][1]) * 1000:.0f})mm")
            # **换目标了**：次数、试过哪几档、锚点全清 —— 下一只方块从头来。
            if self._ppo is not None:
                self._ppo.begin_target()
        if rec is not None:
            self._collect_episode(rec, out)
        ok = bool(out.get("ok"))
        why = str(out.get("why") or "")
        self._last_skill = {"ok": ok, "why": why,
                            "give_up": bool(out.get("give_up")),
                            "seconds": time.time() - t0}
        self._log(("✓ " if ok else "✗ ") + f"技能收工：{why}"
                  f"   用时 {time.time() - t0:.1f}s")
        self._status = "skill done" if ok else "skill stopped"
        self._gate_t = 0.0              # 收工了：断点状态立刻作废，别让面板还挂着按钮

    def _collect_episode(self, rec, out: dict) -> None:
        """一趟收工：够长且真抓起来了就存，否则丢掉。**判据和采集脚本逐字一样**
        （`collect_real_dataset.py`）：`ok` + `verdict == "held"` + 没掉 + 帧数够。
        """
        n = rec.stop()
        ok = (bool(out.get("ok")) and out.get("verdict") == "held"
              and not out.get("dropped") and n >= MIN_FRAMES)
        if ok:
            rec.save()
            self._n_ok += 1
            self._consec = 0
            self._strat_tried.clear()   # 这一支能成：下一趟还从它起，阶梯重新算
        else:
            why = str(out.get("why") or ("没抓住" if not ok else f"只录到 {n} 帧"))
            rec.discard(why)
            self._n_bad += 1
            self._consec += 1
        self._log(f"采集进度：成功 {self._n_ok} / 失败 {self._n_bad}"
                  f"（这一趟 {n} 帧，策略「{self._strat_label or self._strat}」）")
        if ok:
            return
        # **重试策略在场时，停不停由它说了算**（`ppo/policy.py::should_stop`）：
        # 只有人喊停、目标找不到、自己设了次数上限这三种才停。
        if self._ppo is not None:
            stop, stop_why = self._ppo.should_stop(out)
            if stop:
                self._log(f"[重试] 停：{stop_why}")
                self._collect_off(stop_why)
            return
        if self._n_ok + self._n_bad >= MAX_ATTEMPTS:
            self._log(f"这一轮跑满 {MAX_ATTEMPTS} 趟 —— 停采集")
            self._collect_off("跑满上限")
            return
        if self._consec >= TRIES_PER_STRAT:
            self._advance_strategy(out, why)

    def _advance_strategy(self, out: dict, why: str) -> None:
        """这一支不行了：按失败码换下一支；阶梯走完就停采集。

        **技能半路收工（`give_up`）不换策略** —— 那是停手、装配、臂不在姿态上这类
        问题，换参数不会有任何变化，接着跑只会白跑。
        """
        if bool(out.get("give_up")):
            self._log("这一趟是收工/装配问题，不是抓取策略的事 —— 不换策略")
            self._collect_off("技能中途收工")
            return
        try:
            from skills.pick_real import strategy as _strategy      # noqa: PLC0415
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"✗ 策略表读不出来：{type(exc).__name__}: {exc} —— 停采集")
            self._collect_off("策略表读不出来")
            return
        c = _strategy.code(why)
        # **空抓不换策略**（用户 2026-10-20）：合了个空是「这一趟没对上位置」，
        # 换一支参数只会换一套同样没对上位置的动作 —— `decide.py` 会按实测把终点补回来，
        # 就在同一支上再扎一趟。换策略留给「轴偏来回摆 / 缝量不了」这类结构性失败。
        if c in set(_strategy.cfg().get("retry_same") or ()):
            self._consec = 0
            self._log(f"失败码 {c}（空抓）：不换策略，同一支「{self._strat_label or self._strat}」"
                      f"再来一趟")
            return
        self._strat_tried.add(self._strat)
        nxt = _strategy.next_after(c, self._strat_tried)
        if nxt is None:
            self._log(f"失败码 {c}，策略阶梯走完了 —— 停采集，不自己造第三条")
            self._collect_off("策略都用过了")
            return
        self._strat = nxt
        self._strat_label = _strategy.label(nxt)
        self._consec = 0
        self._log(f"失败码 {c}：下一趟换「{self._strat_label}」（{nxt}）")
