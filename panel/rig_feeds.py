"""三条喂数据的线程：画面、感知、读数。**各走各的节拍，互不阻塞。**

· 画面 `_pic_loop`   —— 只调相机，不碰串口；
· 感知 `_percep_loop` —— 认方块 + 量缝/轴偏 + 算下一步，算完并进快照；
· 读数 `_tick/_refresh` —— 只在硬件线程里，读一次六个舵机。

`_on_step` 是走位途中被 `goto_path` 回调的那一下，必须便宜：只用 `last_action`，
一次 I/O 都不做。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from servo_bridge import deg_from_position                      # noqa: E402
from skills import pick_real as PR                              # noqa: E402

from panel import measure                                       # noqa: E402
from panel.rig_common import FRAME_S, PERCEP_S, TICK_S, _prof   # noqa: E402


class FeedsMixin:
    """快照的写入方：画面、感知、读数。"""

    def snap(self) -> dict:
        """最近一次快照。**返回的是副本**，界面拿着它画，不会跟硬件线程抢。"""
        with self._lock:
            return dict(self._snap)

    # ---- 走位途中的那一下回调 ----

    def _on_step(self) -> None:
        """走位途中由 `goto_path` 每一步调一次。**这里必须便宜** —— 每次真读舵机是
        66ms，30Hz 的轨迹会被拖成 15Hz。所以：

        · 关节角用 `last_action`（`RealArm.drive` 留的那份**发出去的**指令角，已经是
          现成的数，一次 I/O 都不要）；
        · **不抓帧**：画面由 `_pic_loop` 那条线程独立出（2026-10-18）。原来在这儿按
          0.1 秒抓一次，等于把「出画面」挂在轨迹循环上，而且抓一次 15~20ms 要从
          33ms 的一步里扣；
        · **感知在这一段不跑** —— 臂在动，量出来的方块位置跟画面本来就对不上，
          标上去是骗人。界面上这一段显示「MOVING」并停掉标注。
        """
        with self._lock:
            act = getattr(self.arm, "last_action", None)
            if act is not None:
                self._snap["angles_deg"] = np.asarray(act, dtype=np.float64)
                self._snap["angles_src"] = "commanded"
            self._snap["moving"] = True
            self._snap["m"] = None
            self._snap["step"] = None
        # **采集时才多做这一件事**（2026-10-20）：`RealRecorder.tick()` 是「这一步要不要
        # 变成数据里的一帧」，自己按 0.2 秒的闸口节流，没到点就直接返回。它有一次抓帧
        # （腕 + 侧）和一次写盘，比上面那三行贵得多 —— 所以只在按了「采集」之后才接上。
        rec = getattr(self, "_rec", None)
        if rec is not None:
            rec.tick()

    # ---- 画面 ----

    def _grab(self) -> np.ndarray:
        """抓一帧 BGR。`shot()` 交出来的是 RGB（相机那一层的约定），这里转一次。"""
        return cv2.cvtColor(np.asarray(self.cam.shot()), cv2.COLOR_RGB2BGR)

    def _pic_loop(self) -> None:
        """**出画面单独一条线程，跟硬件线程完全无关**（2026-10-18 用户：画面不能卡死，
        不管臂在不在动）。

        原来出画面的活儿挂在硬件线程上：一是 `_loop` 的那条分支，二是走位时
        `_on_step` 里按 0.1 秒抓一次。可硬件线程一段轨迹要跑几秒、技能里一次 `locate`
        要 1~3 秒，这期间它一个循环都轮不回来 —— 臂一动画面就冻在起步那一帧。
        相机那边的 `cap.read()` 已经由 `real/cameras.py::_grab_loop` 常驻读，这里只做
        「去畸变 + 转 BGR」然后换快照里的 `bgr` / `t`，**不碰串口**。

        `t` 必须跟着换：界面判「这一帧换没换」看的就是它（`panel/ui.py::_loop`）。
        """
        while not self._quit.is_set():
            time.sleep(FRAME_S)
            try:
                bgr = self._grab()
                side = getattr(self, "side", None)
                # **侧视那一帧跟着这条路走，不再另起一条线程**（2026-10-20）：相机
                # 自己已经有读帧线程（`real/side_cam.py::_grab_loop`），这里只是把
                # 换了的那一帧抄进快照。`seq()` 没变就不转不拷 —— 一次 `cvtColor`
                # 在 860×900 上约 1~2ms，没必要每 80ms 白做一遍。
                new_side = (side is not None and side.seq() != self._side_seq)
                side_bgr = None
                if new_side:
                    self._side_seq = side.seq()
                    side_bgr = cv2.cvtColor(np.asarray(side.shot()),
                                            cv2.COLOR_RGB2BGR)
                with self._lock:
                    snap = dict(self._snap)
                    snap["bgr"] = bgr
                    snap["t"] = time.time()
                    if side_bgr is not None:
                        snap["side_bgr"] = side_bgr
                        snap["side_t"] = time.time()
                    self._snap = snap
            except Exception as exc:                                # noqa: BLE001
                self._log(f"✗ 抓帧出错：{type(exc).__name__}: {exc}")
                time.sleep(0.20)

    # ---- 读数 ----

    def _tick(self, now: float, force: bool = False) -> None:
        """**只读舵机。** 画面由 `_pic_loop` 那条线程出；感知由 `_percep_loop` 出。"""
        t0 = time.perf_counter()
        self._t_frame = now
        if force:
            self._t_percep = 0.0            # 「重量一眼」：让感知线程下一轮就跑
        self._refresh()
        _prof("刷新", f"读数 {(time.perf_counter() - t0) * 1000:.0f}ms")

    def _refresh(self) -> None:
        """读一次硬件，然后整份快照换新。**画面和感知都不在这儿**（各自一条线程）。"""
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
        # **不写 `bgr`**：画面由 `_pic_loop` 换，这一拍只换读数，别把更新的那一帧盖回去。
        snap.update({"angles_deg": angles, "angles_src": "measured",
                     "target": self.color,
                     "tcp": np.asarray(tcp, dtype=np.float64),
                     "tips": (np.asarray(f, dtype=np.float64),
                              np.asarray(mv, dtype=np.float64)),
                     "joint_q": q, "ticks": ticks, "grip_deg": grip_deg,
                     "verdict": verdict, "busy": self._busy,
                     "moving": self._moving, "status": self._status,
                     "last_skill": self._last_skill,
                     "auto": self._auto, "round": self._round,
                     "collect": self._collect, "n_ok": self._n_ok,
                     "n_bad": self._n_bad, "consec": self._consec,
                     "strategy": self._strat,
                     "strategy_label": self._strat_label,
                     "gate": self._gate_state(),
                     # 技能当前段（`RoundState.seg`）。跑完不清 —— 续跑按钮靠它。
                     "stage": str(getattr(self._skill_state, "seg", "") or ""),
                     "t": time.time()})
        with self._lock:
            self._snap = snap

    # ---- 感知 ----

    def _percep_loop(self) -> None:
        """**感知单独一个线程**（2026-10-18 时延）。

        以前它挂在硬件线程的刷新里：`locate` 一眼 0.7~3 秒，那一整段串口队列没人收，
        按什么按钮都得排队，画面也停着。现在它自己按 `PERCEP_S` 的节拍跑，算完把结果
        并进快照；硬件线程只抓帧 + 读舵机（约 0.1 秒一拍），按键随时能进来。

        **技能在跑（`_busy`）或臂在动（`_moving`）的时候不做**：那两段时间里方块本来
        就看不清，而且技能自己也在调 `percep.locate`，两个线程一起进去只会抢 CPU。
        """
        while not self._quit.is_set():
            time.sleep(0.05)
            if self._busy or self._moving:
                continue
            now = time.time()
            if now - self._t_percep < PERCEP_S:
                continue
            self._t_percep = now
            try:
                snap_now = self.snap()
                out = self._percep_pass(None, snap_now)
                with self._lock:
                    snap = dict(self._snap)
                    snap.update(out)
                    snap["t"] = time.time()
                    self._snap = snap
            except Exception as exc:                                # noqa: BLE001
                self._log(f"✗ 感知那一拍出错：{type(exc).__name__}: {exc}")

    def _stop_percep(self, join: float = 2.0) -> None:
        p = getattr(self, "_p_thr", None)
        if p is not None and p.is_alive():
            p.join(timeout=join)

    def _percep_pass(self, bgr: np.ndarray | None, snap_now: dict) -> dict:
        """重量一眼：认出方块 + 量缝/轴偏 + 算下一步。

        `snap_now` 是**这一拍刚读到的**那份（关节角 / TCP / 爪角），用来算「下一步」；
        不是 `self._snap`（那是上一拍的，差 100ms，臂在动的时候不是一回事）。

        **方块那一份画面就是感知消费的那一帧。** `locate` 内部自己调
        `cam.shot()`，而 `RealWristCamera.shot()` 会把那一帧留在 `cam._last` 上 ——
        这里取它。不取的话量出来的读数和画出来的框差一个瞬间，臂一动就是两回事。
        """
        out: dict = {"m": None, "box": None, "facts": None, "step": None}
        _p0 = time.perf_counter()
        try:
            center, facts = self.percep.locate(self.color, guess=self._seed)
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"！感知出错：{type(exc).__name__}: {exc}")
            out["box"] = {"center": None, "why": f"感知出错：{exc}"}
            return out
        _p1 = time.perf_counter()
        rgb = getattr(self.cam, "_last", None)
        if rgb is not None:
            bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            out["bgr"] = bgr
        if bgr is None:                 # 感知线程调过来时没有现成的帧，自己抓一张
            bgr = self._grab()
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
        _p2 = time.perf_counter()
        out["step"] = measure.next_step(
            self.cam, out["m"], snap_now.get("angles_deg"), snap_now.get("tcp"),
            snap_now.get("grip_deg"), out["box"])
        _prof("感知(拆)", f"locate {(_p1 - _p0) * 1000:.0f}ms + 转图/爪轮廓 "
                          f"{(_p2 - _p1) * 1000:.0f}ms + next_step "
                          f"{(time.perf_counter() - _p2) * 1000:.0f}ms")
        # 中文那句话只在**相位变了**的时候打一行。每 0.7 秒打一遍的话，终端会被刷得
        # 看不清真正要紧的那几行（这正是这一包要治的病）。
        phase = out["step"]["phase"]
        if phase != self._phase:
            self._phase = phase
            self._log(f"下一步：{out['step']['why']}")
        return out

    @staticmethod
    def _fresh_snapshot() -> dict:
        return {"bgr": None, "side_bgr": None, "side_t": 0.0,
                "ok": False, "angles_deg": None, "angles_src": "",
                "target": "",
                "tcp": None, "tips": None, "joint_q": None, "ticks": None,
                "grip_deg": None, "grip_src": "", "busy": False, "moving": False,
                "status": "starting", "m": None, "box": None, "facts": None,
                "step": None, "last_skill": None, "auto": False, "round": 0,
                "collect": False, "n_ok": 0, "n_bad": 0, "consec": 0,
                "strategy": "", "strategy_label": "",
                "gate": None, "stage": "", "t": 0.0}
