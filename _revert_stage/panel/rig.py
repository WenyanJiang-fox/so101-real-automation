"""硬件那一层：一套真机装备（串口 + 腕相机 + 感知）跑在自己的线程里。

**为什么必须单独一层、还必须分线程。** 界面要一直 `cv2.waitKey` 泵消息，而真机上
任何一个动作都要几百毫秒到几秒（读六个舵机 66ms、写一次 21ms、走一段轨迹几秒）。
动作直接跑在界面线程里，画面就冻在那儿，`stop` 键也按不进去 —— 面板就没用了。
所以：**硬件全部归这一个线程**，界面通过命令队列下单、通过快照读结果。

装配这件事**照抄 `run_sim.py`**（同一份 `configs/real.yaml`、同一段现场常数、同一串
`make()`），不另起一套：

    geom.REAL_TABLE_Z / REAL_PICK_XY / REAL_PERCEP_OFFSET / OBJECTS   ← 必须在 make 之前设
    make("arm", "real", port=, speed=)
    make("camera", "real_wrist", source=arm, index=)
    make("perception", "brown_rect", cams=, ref=None, objects=)

**这一层不写任何策略。** 它只做三件事：把命令变成对 `RealArm` / `RealWristCamera` /
`BrownRectPerceptor` 的调用、把当前状态抄成一份快照、把日志交出去。判据在
`panel/measure.py`，画在 `panel/view.py`，按在 `panel/ui.py`。

**这一层占着串口和相机。** `run_sim.py` / `view_cam.py` / `mcp_server.py` 开着的时候
打不开 —— 会当场抛，不会半死。
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import geom                                           # noqa: E402
from core.registry import available, make                      # noqa: E402
from real.arm import SERVO_IDS                                  # noqa: E402
from real.joint_move import goto_path, home_q                   # noqa: E402
from servo_bridge import deg_from_position                      # noqa: E402
from skills import pick_real as PR                              # noqa: E402

from panel import measure                                       # noqa: E402

DEFAULT_CONFIG = _ROOT / "configs" / "real.yaml"

# 快照刷新周期。**这个数就是「实时」的那一格。** 抓一帧约 30ms、走一次批量读舵机
# 约 66ms，加起来 100ms 就是一档；再快也只是把总线占满，读数反而更飘。
TICK_S = 0.10
# 重量一眼的间隔。感知那条链（估形状 + 投剪影）比抓帧贵得多，臂不动的时候读数也不变，
# 所以它按自己的节拍走。
PERCEP_S = 0.70
# 走位过程里抓帧的间隔（`on_step` 里）。走位是 30Hz，每步都抓会把轨迹拖慢。
MOVING_FRAME_S = 0.40
# 「技能正停在断点上等着」这个判断的新鲜度（秒）。技能等点头的时候每约 0.1 秒调一次
# `WAIT_TICK`（就是 `pump_wait`），它记下时刻；超过这个数没再被调过，就说明技能已经
# 从断点上走开了 —— 盘上那几份读数文件还在，但人不该再看到「允许」按钮。
GATE_FRESH_S = 0.6


def load_cfg(path=DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class Rig:
    """一套真机装备 + 一个跑它的线程。界面只调 `submit()` / `snap()`。"""

    def __init__(self, cfg_path=DEFAULT_CONFIG, port=None, cam_index=None,
                 log=None):
        self._log_fn = log
        self._lock = threading.Lock()
        self._q: "queue.Queue[dict]" = queue.Queue()
        self._quit = threading.Event()
        self._busy = False
        self._moving = False
        self._status = "starting"
        # 最近一次技能的结果（`_run_pick` 写、界面读）。没跑过就是 None。
        self._last_skill: dict | None = None
        # 技能最近一次停在断点上等点头的时刻（`pump_wait` 每轮刷新）。0 = 没在等。
        self._gate_t = 0.0
        self._snap: dict = {}
        self._t_frame = 0.0
        self._t_percep = 0.0
        # 下一眼感知的起搜中心（米，基座系）。**就是上一眼认出来的那个真位置** ——
        # 不用加也不用扣 `percep_offset_mm`，`locate` 交出来的数已经扣过了。
        # 还没认出来过就是 None，那时 `locate` 用配置里的 `pick_xy_mm` 起搜。
        self._seed = None

        cfg = load_cfg(cfg_path)
        rig = cfg.get("real") or {}
        if rig.get("table_z_mm") is None or rig.get("pick_xy_mm") is None:
            raise SystemExit(
                f"{cfg_path} 的 `real:` 那一段缺数：要 `table_z_mm` 和 `pick_xy_mm`。"
                "这两个数不给就把画面反推不成世界坐标 —— 拿猜的去驱动机械臂比不跑更糟。")
        # **必须在 make 之前设**：感知起来时就读它们，读不到会当场抛。
        geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
        geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                                     dtype=np.float64) / 1000.0
        geom.REAL_PERCEP_OFFSET = np.array(
            [float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
            dtype=np.float64) / 1000.0
        geom.OBJECTS = list(cfg.get("objects") or [])

        # 挂过牌子的实现才查得到。**必须在 make 之前 import。**
        import impls.judge_held                                     # noqa: F401,PLC0415
        import impls.policy_real                                    # noqa: F401,PLC0415
        import real.arm                                             # noqa: F401,PLC0415
        import real.cameras                                         # noqa: F401,PLC0415
        import real.perception                                      # noqa: F401,PLC0415
        import skills.home_real                                     # noqa: F401,PLC0415
        import skills.pick_real                                     # noqa: F401,PLC0415

        self.cfg = cfg
        self.port = str(port or rig.get("port", "COM10"))
        self.cam_index = int(cam_index if cam_index is not None
                             else rig.get("cam_index", 0))
        self._log(f"机械臂 {available('arm')} / 相机 {available('camera')} / "
                  f"感知 {available('perception')}")

        self.arm = make("arm", "real", port=self.port,
                        speed=int(rig.get("speed", 0)))
        self.arm.on_step = self._on_step
        self.arm.terminated = False
        self.cams = {"real_wrist": make("camera", "real_wrist", source=self.arm,
                                        index=self.cam_index)}
        self.cam = self.cams["real_wrist"]
        # `cam_name` 只在配置写了 `perception_cam` 时给（跟 `run_sim.py:426` 同一口径）；
        # 没写就用这一路自己的默认（`real/perception.py:57` 的 `CAM_NAME = "real_wrist"`）。
        percep_kw = ({"cam_name": str(cfg["perception_cam"])}
                     if cfg.get("perception_cam") else {})
        self.percep = make("perception", "brown_rect", cams=self.cams, ref=None,
                           objects=geom.object_specs(), **percep_kw)
        self.color = str(self.percep.color)
        # 爪子轮廓是**一次描好、一直有效**的常数（相机和两爪同在一个刚体上）。
        # 没描过、或者画幅对不上，`_jaw_shape` 会当场抛 —— 这里只记下来，不当场死，
        # 因为「没轮廓」时面板照样能手动走关节。
        try:
            self.jaw = PR._jaw_shape(self.cam)
            self.jaw_why = ""
        except RuntimeError as exc:
            self.jaw, self.jaw_why = None, str(exc)
            self._log(f"！爪轮廓读不出来：{exc}")
        self._log(f"串口 {self.port}   腕相机 index {self.cam_index} "
                  f"({self.cam.size()[0]}x{self.cam.size()[1]})   认的颜色 {self.color}")

        self._snap = self._fresh_snapshot()
        self._phase = None
        try:
            from grasp_check import load_thresholds                  # noqa: PLC0415

            self.cal = load_thresholds()
        except Exception as exc:                                    # noqa: BLE001
            self.cal = {}
            self._log(f"！判据阈值读不出来（{type(exc).__name__}: {exc}），"
                      f"夹爪那一档先不判")
        self._thr = threading.Thread(target=self._loop, name="rig", daemon=True)

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

    def snap(self) -> dict:
        """最近一次快照。**返回的是副本**，界面拿着它画，不会跟硬件线程抢。"""
        with self._lock:
            return dict(self._snap)

    # ---- 命令 ----

    def submit(self, **cmd) -> None:
        self._q.put(cmd)

    def start(self) -> None:
        self._thr.start()

    def stop(self, join: float = 3.0) -> None:
        self._quit.set()
        self.arm.terminated = True          # 走位中途也能当场停
        self._q.put({"kind": "quit"})
        if self._thr.is_alive():
            self._thr.join(timeout=join)

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
            try:
                cmd = self._q.get(timeout=TICK_S)
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
        if kind == "stop":
            self.arm.terminated = True
            self._log("⏹ 停手 —— 轨迹走到当前这一点就停，不松力矩")
            self._status = "stopped"
            return
        if kind == "see":
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
            self._run_pick(str(cmd.get("instruction") or ""))
        else:
            self._log(f"？不认识的命令 {kind!r}")

    def _run_pick(self, instruction: str) -> None:
        """跑一整趟。**这是整条链里唯一会走完整策略的地方。**

        跑在硬件线程里（`_do` 就是在 `_loop` 里调的），所以跑的时候不会有人跟它抢串口。
        画面不会冻着：技能所有动作都走 `goto_path`，那里面每个控制点都调
        `arm.on_step`（就是本类的 `_on_step`），帧照抓。

        **闸口全开**（`skill_driver.run_pick` 里设的环境变量）：技能会在「微调每一步 /
        下探那一截之前 / 判过了先不合 / 横移掉下去要抬回来」这四处自己停下来，等外面写
        `out/clear_go.json`。**面板不用管它在哪儿停** —— 等的时候 `WAIT_TICK` 每一轮都在
        调 `pump_wait`，`_gate_t` 就是「现在正等着」的凭据。
        """
        from panel import skill_driver                              # noqa: PLC0415

        name = instruction or skill_driver.INSTRUCTION
        self._status = "skill: pick"
        self._log(f"▶ 技能开跑（{name}）—— 会在断点上停下来等你点头")
        with self._lock:
            # 跑的时候别拿上一拍的读数糊弄人：`step` 清掉，界面就会显示「在动」。
            self._snap["step"] = None
            self._snap["moving"] = True
        t0 = time.time()
        out = skill_driver.run_pick(self, name)
        ok = bool(out.get("ok"))
        why = str(out.get("why") or "")
        self._last_skill = {"ok": ok, "why": why,
                            "give_up": bool(out.get("give_up")),
                            "seconds": time.time() - t0}
        self._log(("✓ " if ok else "✗ ") + f"技能收工：{why}"
                  f"   用时 {time.time() - t0:.1f}s")
        self._status = "skill done" if ok else "skill stopped"
        self._gate_t = 0.0              # 收工了：断点状态立刻作废，别让面板还挂着按钮

    def _gate_cmd(self, what: str, dz) -> None:
        """在断点上给技能一个答复：把 `out/clear_go.json` 写出去。

        **技能没停在断点上的时候不写。** 那个文件是「一次性批示」，写了没人读就一直躺在
        盘上 —— 技能走到下一个断点时先 `unlink` 它才等，所以不会串味；但人看到的是
        「按了允许却什么都没发生」，不如当场说清。
        """
        from panel import skill_driver                              # noqa: PLC0415

        if self._gate_state() is None:
            self._log("！技能没停在断点上 —— 这条批示没写（先按「跑一趟」）")
            return
        self._log(skill_driver.open_gate(what, dz_mm=None if dz is None else float(dz)))

    def pump_wait(self) -> None:
        """技能停在断点上等点头的时候，**把这期间面板按下的键做掉**。

        挂在 `skills/pick_real.py` 的 `WAIT_TICK` 上，技能每等一轮（约 0.1 秒）调一次。
        它跑的就是硬件线程 —— 串口没有第二个人抢，跟平时按按钮走的是同一条路。

        **不许在这儿再起一趟技能。** 技能正停着，再进去一层就是两趟叠在一起：串口、
        姿态、闸口文件全乱。那种命令直接丢掉并说明。

        **「停手」在这儿的意思是「这一趟不干了」**，不是「把这一小段运动掐停」——
        技能正停着没在动，掐不到什么；真按下去照原样立 `terminated`，等点头等到一半的
        技能接着走第一步就会被它顶停，人看到的是「按了允许但不动」。所以在这儿翻成
        `{"abort": true}`，让技能自己从等点头里干净地退出来。
        """
        self._gate_t = time.time()
        from panel import skill_driver                              # noqa: PLC0415

        while True:
            try:
                cmd = self._q.get_nowait()
            except queue.Empty:
                break
            kind = cmd.get("kind")
            if kind in ("pick_run", "stage"):
                self._log("！技能正停在断点上 —— 这条先不接"
                          "（先给个「允许」或者「放弃这一趟」）")
                continue
            if kind == "quit":
                self._quit.set()
                self.arm.terminated = True
                continue
            if kind == "stop":
                self._log("⏹ 停手 —— 技能正停在断点上，这一趟按「放弃」收工")
                self._log(skill_driver.open_gate("abort"))
                continue
            try:
                self._do(cmd)
            except Exception as exc:                                # noqa: BLE001
                self._log(f"✗ 命令 {kind} 出错：{type(exc).__name__}: {exc}")
            finally:
                # 跟 `_loop` 里那两行一个道理：走完一趟就把「在动」放下来，
                # 不然画面上会一直挂着 `MOVING`（`view.draw` 见它就把读数冻住）。
                self._moving = False
        # 别让画面冻在断点那一帧。**节拍跟 `_loop` 一样**：`_tick` 自己不限速，
        # 抓一帧约 30ms、批量读舵机约 66ms，每一轮都刷的话等待那条循环会被拖慢一倍。
        now = time.time()
        if now - self._t_frame >= TICK_S:
            try:
                self._tick(now)
            except Exception as exc:                                # noqa: BLE001
                self._log(f"✗ 刷新出错：{type(exc).__name__}: {exc}")


    def _goto_path(self, q_list, note: str, en: str) -> np.ndarray:
        """走一段关节轨迹。

        `note` 是给终端日志的中文，`en` 是给控制台窗口的英文。**必须分开** ——
        `self._status` 会画在控制台窗口上，而 `cv2.putText` 画中文出来全是问号
        （见 `panel/ui.py` 开头）。一句话要两种写法，就这么写。
        """
        t0 = time.time()
        self._moving = True
        self._status = en
        q_end = goto_path(self.arm, q_list)
        self._moving = False
        dt = time.time() - t0
        self._status = f"{en} - done {dt:.1f}s"
        self._log(f"{note}：走完 {dt:.1f}s")
        return q_end

    def _jog(self, joint: int, delta_deg: float) -> None:
        """单个关节走 `delta_deg` 度。**绝对关节角、梯形速度、到点补写** ——
        走的是 `real/joint_move.py` 那条，和技能里扫视/回初始位置同一条。
        """
        q0 = self.arm._ticks_to_qpos(self.arm.read_ticks())
        q1 = np.array(q0, dtype=np.float64, copy=True)
        q1[joint] = float(np.clip(q0[joint] + np.radians(delta_deg),
                                  self.arm.q_lo[joint], self.arm.q_hi[joint]))
        moved = float(np.degrees(q1[joint] - q0[joint]))
        if abs(moved) < 1e-6:
            self._log(f"关节 {joint + 1} 已经贴在限位上，这一步不动")
            return
        self._goto_path([q0, q1],
                        f"关节 {joint + 1} 走 {moved:+.2f}°（到 "
                        f"{np.degrees(q1[joint]):.2f}°）",
                        f"J{joint + 1} {moved:+.2f} deg -> {np.degrees(q1[joint]):.2f}")

    def _goto_deg(self, angles) -> None:
        """六个关节直接给绝对角度（度）。"""
        q0 = self.arm._ticks_to_qpos(self.arm.read_ticks())
        q1 = np.radians(np.asarray(angles, dtype=np.float64))
        q1 = np.clip(q1, self.arm.q_lo, self.arm.q_hi)
        self._goto_path([q0, q1], f"走到 关节(度) {np.round(np.degrees(q1), 1)}",
                        f"goto {np.round(np.degrees(q1), 1)}")

    def _grip(self, to, deg) -> None:
        """开爪 / 合爪。**走关节空间**，只动 6 号那一维，别的关节一个字节不写。

        合到底 = `arm.grip_lo`，张到最大 = `arm.grip_hi` —— 这两个数是**从 MJCF 的
        关节 range 读的**（`real/arm.py:161 joint_limits`），不是这儿写死的。
        """
        q0 = self.arm._ticks_to_qpos(self.arm.read_ticks())
        q1 = np.array(q0, dtype=np.float64, copy=True)
        if to == "close":
            q1[5], what, en = float(self.arm.grip_lo), "合到底", "grip CLOSE"
        elif to == "open":
            q1[5], what, en = float(self.arm.grip_hi), "张到最大", "grip OPEN"
        elif to == "deg" and deg is not None:
            q1[5] = float(np.clip(np.radians(float(deg)), self.arm.grip_lo,
                                  self.arm.grip_hi))
            what, en = f"走到 {float(deg):.1f}°", f"grip {float(deg):.1f} deg"
        else:
            self._log(f"？grip 命令缺 `to`：{to!r}")
            return
        self._goto_path([q0, q1],
                        f"6 号（爪）{what}：{np.degrees(q0[5]):.1f}° → "
                        f"{np.degrees(q1[5]):.1f}°",
                        f"{en} {np.degrees(q0[5]):.1f} -> {np.degrees(q1[5]):.1f}")

    def _home(self) -> None:
        """回初始位置（`real_home.RAW_HOME`）。**不是仿真那个 home** —— 这一套是
        用户 2026-10-14 定的六个原始位置，跟 `skills/home_real.py` 走的是同一条。
        """
        q0 = self.arm._ticks_to_qpos(self.arm.read_ticks())
        self._goto_path([q0, home_q(self.arm)], "回初始位置", "HOME")

    def go_home(self) -> None:
        """**退出时**走一趟「回初始位置」——走完再关串口（用户 2026-10-17 要的）。

        跟按钮那一下（`_home`）是同一条路，只多一件事：`stop()` 会把
        `arm.terminated` 立起来（那是给「走位中途叫停」用的），不把它放回 `False` 的话
        `goto_path` 一看就当场收工、一步都不走。

        **在主线程上调，而且要在 `stop()` 之后、`close()` 之前。** 这一刻硬件线程已经
        停了，臂上没有别人；串口还开着，走得动。
        """
        self.arm.terminated = False
        try:
            self._home()
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"✗ 回初始位置出错：{type(exc).__name__}: {exc}")

    def _nudge(self, dx_mm: float, dy_mm: float, dz_mm: float) -> None:
        """TCP 在**底座系**里横竖挪 `(dx, dy, dz)` 毫米，姿态保持不动。

        「保持姿态」的口径跟 `real/move_to.py:82` 一样：目标姿态就用**此刻**的姿态解，
        不让逆解顺手把腕子转过去 —— 转腕会把 TCP 拽偏（`move_to.py` 开头记着实测：
        目标 +z 抬起来，实际 z 从 86mm 掉到 43mm）。

        够不着的点**当场拒绝**，不硬走。判据用正解回代量残差，和 `reach_residual` 同法
        （但不拿它的入参口径 —— 那个函数收的是**官方 site 系**的点，`tcp()` 报的是仿真
        那套口径，两者差 6.4mm，混用会静默错位）。
        """
        q0 = self.arm._ticks_to_qpos(self.arm.read_ticks())
        pos_s, quat_s = self.arm._fk(q0)
        tcp_sim, quat_sim = self.arm._sim_pose(pos_s, quat_s)
        d = np.array([dx_mm, dy_mm, dz_mm], dtype=np.float64) / 1000.0
        goal_sim = tcp_sim + d
        pos_goal, quat_goal = self.arm._site_pose(goal_sim, quat_sim)
        q1 = self.arm._ik(pos_goal, quat_goal, q0)
        q1[5] = q0[5]
        got, _ = self.arm._fk(q1)
        res = float(np.linalg.norm(got - pos_goal))
        if res > 0.005:
            self._log(f"✗ 这个点够不着（解完正解还差 {res * 1000:.1f}mm > 5mm），"
                      f"一步不走。要挪就分几次小步走。")
            return
        self._goto_path([q0, q1],
                        f"TCP 走 ({dx_mm:+.1f}, {dy_mm:+.1f}, {dz_mm:+.1f})mm"
                        f"（够得着，逆解残差 {res * 1000:.1f}mm）",
                        f"TCP {dx_mm:+.1f},{dy_mm:+.1f},{dz_mm:+.1f}mm")

    def _measure_grip_range(self) -> None:
        """量这一趟夹爪行程的两端（空嘴合到底 / 张到最大）。

        **这就是判「夹住没夹住」的自适应基准。** 判据不看「这只盒子该是多少度」，看
        「比这一趟实测的空抓基准高多少」—— 换盒子、换爪面、舵机老化都不用重标。
        量完爪子停在**张着**的位置（跟技能同一口径，见 `pick_real.py:590`）。
        量出来的两个数记在 `arm.empty_deg` / `arm.open_deg` 上 —— **技能读的就是这两个
        名字**（`pick_real.py:2978`），所以下次跑技能时它会接着用。
        """
        from grasp_check import (empty_margin, load_thresholds,   # noqa: PLC0415
                                 measure_empty, measure_open)

        ctrl = self.arm.ctrl
        self.cal = cal = load_thresholds()      # 顺手把阈值文件重读一遍（可能刚改过）
        self._status = "measuring grip travel"
        self._log("量夹爪行程两端：先合到底，再张到最大（爪子会动）")
        empty = measure_empty(ctrl)
        opened = measure_open(ctrl)
        if empty is None or opened is None:
            self._log("✗ 行程两端没量出来（舵机没回话），判据沿用上一趟的")
            return
        self.arm.empty_deg, self.arm.open_deg = float(empty), float(opened)
        m = empty_margin(cal)
        self._log(f"行程两端：合到底 {empty:.1f}° / 张到最大 {opened:.1f}°；"
                  f"判「夹住」的区间是 {empty + m:.1f}° ~ {opened - m:.1f}°")

    # ---- 刷新：抓帧 + 读硬件 + （到点了）重量一眼 ----

    def _on_step(self) -> None:
        """走位途中由 `goto_path` 每一步调一次。**这里必须便宜** —— 每次真读舵机是
        66ms，30Hz 的轨迹会被拖成 15Hz。所以：

        · 关节角用 `last_action`（`RealArm.drive` 留的那份**发出去的**指令角，已经是
          现成的数，一次 I/O 都不要）；
        · 画面按 `MOVING_FRAME_S` 的节拍抓，不是每步抓；
        · **感知在这一段不跑** —— 臂在动，量出来的方块位置跟画面本来就对不上，
          标上去是骗人。界面上这一段显示「MOVING」并停掉标注。
        """
        now = time.time()
        with self._lock:
            act = getattr(self.arm, "last_action", None)
            if act is not None:
                self._snap["angles_deg"] = np.asarray(act, dtype=np.float64)
                self._snap["angles_src"] = "commanded"
            self._snap["moving"] = True
            self._snap["m"] = None
            self._snap["step"] = None
        if now - self._t_frame >= MOVING_FRAME_S:
            self._t_frame = now
            try:
                bgr = self._grab()
                with self._lock:
                    self._snap["bgr"] = bgr
            except Exception:                                       # noqa: BLE001
                pass

    def _grab(self) -> np.ndarray:
        """抓一帧 BGR。`shot()` 交出来的是 RGB（相机那一层的约定），这里转一次。"""
        return cv2.cvtColor(np.asarray(self.cam.shot()), cv2.COLOR_RGB2BGR)

    def _tick(self, now: float, force: bool = False) -> None:
        bgr = self._grab()
        self._t_frame = now
        do_percep = force or (now - self._t_percep >= PERCEP_S)
        if do_percep:
            self._t_percep = now
        self._refresh(bgr, do_percep)

    def _refresh(self, bgr: np.ndarray, do_percep: bool) -> None:
        """读一次硬件、按需要重量一眼，然后整份快照换新。"""
        arm = self.arm
        arm._dirty = True          # **强制真读**：不写舵机的时候缓存会一直有效，
                                   # 界面上的关节角就冻住了（`read_ticks` 的口径是
                                   # 「自上次写之后读过了吗」，见 real/arm.py:217）
        ticks = arm.read_ticks()
        q = arm._ticks_to_qpos(ticks)
        angles = np.degrees(q)
        tcp = arm.tcp()
        f, mv = arm.tips(q)
        # 6 号（爪）的角度**就从上面那一次批量读里取**，不再单读一遍。
        # `pick_real._gripper_deg` 那条会去单读一次舵机（5 次取中位数）—— 判「夹住没夹住」
        # 用它是对的（单读丢了还有退路）；**面板上不该用**：那要多花约 100ms，而且它读的
        # 是另一个瞬间，会跟上面五个关节角（同一个批量读）对不上号。换算同一条式子
        # （`servo_bridge.deg_from_position`），尺子还是同一把。
        grip_deg = (float(deg_from_position(ticks[6])) if 6 in ticks else None)
        verdict = ""
        if grip_deg is not None and getattr(arm, "empty_deg", None) is not None:
            verdict = measure.classify_grip(grip_deg, arm.empty_deg, self.cal,
                                            getattr(arm, "open_deg", None))

        snap = dict(self._snap)
        snap.update({"bgr": bgr, "angles_deg": angles, "angles_src": "measured",
                     "tcp": np.asarray(tcp, dtype=np.float64),
                     "tips": (np.asarray(f, dtype=np.float64),
                              np.asarray(mv, dtype=np.float64)),
                     "joint_q": q, "ticks": ticks, "grip_deg": grip_deg,
                     "verdict": verdict, "busy": self._busy,
                     "moving": self._moving, "status": self._status,
                     "last_skill": self._last_skill,
                     "gate": self._gate_state(),
                     "t": time.time()})
        if do_percep:
            snap.update(self._percep_pass(bgr, snap))
        with self._lock:
            self._snap = snap

    def _gate_state(self) -> dict | None:
        """现在是不是**正停在断点上等着**？等着就给一份状态，没等就是 `None`。

        两件事凑一起才算数：技能那一头 `WAIT_TICK` 刚被调过（`_gate_t` 新鲜），盘上
        又有读数文件（`skill_driver.gate_now`）。只看文件不行 —— 上一趟留下的那几份
        一直在盘上，面板会一直挂着「允许」按钮。
        """
        if time.time() - self._gate_t > GATE_FRESH_S:
            return None
        from panel import skill_driver                              # noqa: PLC0415

        got = skill_driver.gate_now()
        if got is None:
            return {"kind": "?", "rec": {}, "note": "技能停在闸口上，读数文件没读到",
                    "allow": ("go", "abort")}
        kind, rec = got
        return {"kind": kind, "rec": rec, "note": skill_driver.gate_note(kind),
                "allow": skill_driver.ALLOWED.get(kind, ("abort",))}

    def _percep_pass(self, bgr: np.ndarray, snap_now: dict) -> dict:
        """重量一眼：认出方块 + 量缝/轴偏 + 算下一步。

        `snap_now` 是**这一拍刚读到的**那份（关节角 / TCP / 爪角），用来算「下一步」；
        不是 `self._snap`（那是上一拍的，差 100ms，臂在动的时候不是一回事）。

        **方块那一份画面就是感知消费的那一帧。** `locate` 内部自己调
        `cam.shot()`，而 `RealWristCamera.shot()` 会把那一帧留在 `cam._last` 上 ——
        这里取它。不取的话量出来的读数和画出来的框差一个瞬间，臂一动就是两回事。
        """
        out: dict = {"m": None, "box": None, "facts": None, "step": None}
        try:
            center, facts = self.percep.locate(self.color, guess=self._seed)
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"！感知出错：{type(exc).__name__}: {exc}")
            out["box"] = {"center": None, "why": f"感知出错：{exc}"}
            return out
        rgb = getattr(self.cam, "_last", None)
        if rgb is not None:
            bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            out["bgr"] = bgr
        out["facts"] = facts
        if center is not None:
            # 下一眼的起搜中心就拿这一眼的结果 —— **`locate` 交出来的已经是扣过
            # `percep_offset_mm` 的真位置**（`real/perception.py:356`），所以这里不能再
            # 扣一次、也不用加回去。加一次的话起搜中心整个偏 36mm。
            self._seed = np.asarray(center, dtype=np.float64)
            out["box"] = {"center": np.asarray(center, dtype=np.float64),
                          "yaw_deg": facts.get("yaw_deg"), "iou": facts.get("iou"),
                          "size": facts.get("size"), "why": ""}
            # **这一眼 3D 那条尺子比的那个点**（`_jaw_mm` 拿它算 `p`/`ds`）在画面里的位置。
            # 它跟画面上那条尺子比的「顶面中心」不是同一个像素 —— 标出来才看得出差多少
            # （用户 2026-10-17：「你比什么就标记什么」）。投不出来就是 `None`，不画。
            out["box"]["uv3d"] = PR._project_px(self.cam, out["box"]["center"])
        else:
            out["box"] = {"center": None, "why": facts.get("no_world_why")
                          or facts.get("why") or "没认出来"}
        if self.jaw is None:
            out["m"] = {"ok": False, "why": self.jaw_why or "没有爪轮廓"}
        else:
            out["m"] = measure.jaw_metrics(bgr, self.jaw, self.percep.ranges,
                                           self.percep.fill_min,
                                           self.percep.min_area)
        out["step"] = measure.next_step(
            self.cam, out["m"], snap_now.get("angles_deg"), snap_now.get("tcp"),
            snap_now.get("grip_deg"), out["box"])
        # 中文那句话只在**相位变了**的时候打一行。每 0.7 秒打一遍的话，终端会被刷得
        # 看不清真正要紧的那几行（这正是这一包要治的病）。
        phase = out["step"]["phase"]
        if phase != self._phase:
            self._phase = phase
            self._log(f"下一步：{out['step']['why']}")
        return out

    def _fresh_snapshot(self) -> dict:
        return {"bgr": None, "ok": False, "angles_deg": None, "angles_src": "",
                "tcp": None, "tips": None, "joint_q": None, "ticks": None,
                "grip_deg": None, "grip_src": "", "busy": False, "moving": False,
                "status": "starting", "m": None, "box": None, "facts": None,
                "step": None, "last_skill": None, "gate": None, "t": 0.0}
