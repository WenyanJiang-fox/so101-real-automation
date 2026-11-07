"""动作原语：走一段关节轨迹、单关节点动、回初始位置、TCP 平移、量夹爪行程。

**每一个都是「动作 + 一句话交代结果」**，全部在硬件线程里执行（`Rig._do` 调它们），
所以没有人跟它们抢串口。这一层不判对错、不写策略。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from real.joint_move import goto_path, home_q                    # noqa: E402


class MotionMixin:
    """关节空间的几个动作。"""

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

        **顺序：先竖直抬高、再合爪、最后收臂**（用户 2026-10-18：「回初始位置的时候一定
        要先抬高，再合爪」）。抬到 `knobs.HOME_RAISE_Z_M`（搬运高度）；已经在它上面就
        不抬。张着爪从桌面收臂会一路蹭桌沿、蹭方块，所以合爪放在抬高之后。
        """
        from skills.pick_real import knobs                          # noqa: PLC0415

        tcp = np.asarray(self.arm.tcp(), dtype=np.float64)
        target_z = float(knobs.HOME_RAISE_Z_M)
        dz_mm = (target_z - float(tcp[2])) * 1000.0
        if dz_mm > 1.0:
            self._log(f"回初始位置：先竖直抬高 {dz_mm:.0f}mm（到 "
                      f"{target_z * 1000:.0f}mm）")
            # **分 30mm 一小步抬。** 一口气抬 106mm 逆解残差 5.5mm > 5mm，`_nudge`
            # 当场判「够不着」、一步不走（`out/panel_20261018f.log` 15:57:20），接着就
            # 直接合爪收臂了 —— 用户看到的就是「没抬就合爪」。小步每走一次都从**读回来
            # 的**姿态重新解，残差落在容里；哪一步读回没动就停，把实际高度报出来。
            #
            # **这一步被拒就减半再试**（`out/panel_20261018h.log` 16:04:45：抬 27mm 到
            # 80mm，逆解残差 6.7mm > 5mm —— 不是方向不行，是这一步太远）。30→15→7.5→3.8，
            # 都过不了容才算这一截抬不动。
            for _ in range(6):
                tcp = np.asarray(self.arm.tcp(), dtype=np.float64)
                dz_mm = (target_z - float(tcp[2])) * 1000.0
                if dz_mm <= 1.0:
                    break
                step_mm = min(30.0, dz_mm)
                moved = False
                for _try in range(5):
                    self._nudge(0.0, 0.0, step_mm)
                    after = np.asarray(self.arm.tcp(), dtype=np.float64)
                    if (float(after[2]) - float(tcp[2])) * 1000.0 >= 1.0:
                        moved = True
                        break
                    step_mm *= 0.5
                    if step_mm < 1.0:
                        break
                if not moved:
                    self._log(f"！抬不动了：停在 {float(tcp[2]) * 1000:+.0f}mm"
                              f"（要到 {target_z * 1000:+.0f}mm）—— 不再硬抬")
                    break
        else:
            self._log(f"回初始位置：已经在 {tcp[2] * 1000:+.0f}mm，不用再抬")
        self._grip("close", None)
        q0 = self.arm._ticks_to_qpos(self.arm.read_ticks())
        self._goto_path([q0, home_q(self.arm)], "回初始位置", "HOME")

    def _estop_run(self) -> None:
        """急停那两下：**先开爪，再回初始位置。** 只在硬件线程里跑。

        门在 `_estop` 事件上 —— 没按就什么都不做，按了就先把事件清掉（跑的过程中
        不会再被自己叫一次）。开爪放最前面：手里要是夹着东西，先把东西放下再说，
        带着东西收臂会一路扫过桌面。

        跑的时候要 `terminated=False`，否则 `goto_path` 第一步就收工（那条是给
        「走位中途叫停」用的）；跑完再把它立回去，让还在跑的技能从自己的检查点上退出去。
        """
        self._estop.clear()
        self.arm.terminated = False
        self._log("🛑 急停：先开爪")
        try:
            self._grip("open", None)
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"✗ 急停开爪出错：{type(exc).__name__}: {exc}")
        self._log("🛑 急停：回初始位置")
        try:
            q0 = self.arm._ticks_to_qpos(self.arm.read_ticks())
            self._goto_path([q0, home_q(self.arm)], "急停回初始位置",
                            "ESTOP HOME")
            self._log("🛑 急停：已回初始位置")
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"✗ 急停回位出错：{type(exc).__name__}: {exc}")
        finally:
            self.arm.terminated = True

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

    def _align_check(self) -> None:
        """指尖对账：**不动臂**，量「两指尖中点」和「侧相机看到的方块中心」差多少。

        用法：人用面板把臂摆到「指尖正对方块」，按「指尖对账」。

        侧相机的外参是常数，所以这个差就是**绝对偏差** —— 它跟臂摆在哪无关。
        腕相机那条链同口径的数要人在读数里看（那一份每 0.7 秒刷一次，这里不动它，
        免得跟感知线程抢同一个 `_est` 缓存）。

        `percep_side.locate` 交出来的已经扣过 `percep_offset_mm`，但那个数是在**腕相机
        链**上量的，对侧相机不适用。这里加回去，报**原始读数** —— 对账要比没扣过的。
        """
        import cv2                                                # noqa: PLC0415
        import time                                               # noqa: PLC0415
        from core import geom                                     # noqa: PLC0415
        from real.perception import draw                          # noqa: PLC0415

        if getattr(self, "percep_side", None) is None:
            self._log("！指尖对账：侧相机没建起来（缺 real_side*.json 或没插）")
            return
        tips = [np.asarray(t, dtype=np.float64) for t in self.arm.tips()]
        mid = np.asarray(self.arm.jaw_gap_center(), dtype=np.float64)
        grip_deg = float(np.degrees(self.arm.grip_q()))
        off = np.asarray(geom.real_percep_offset(), dtype=np.float64)
        center, facts = self.percep_side.locate(self.color)
        self._log("===== 指尖对账（侧相机，原始读数）=====")
        self._log(f"  两指尖中点 ({mid[0] * 1000:+.1f}, {mid[1] * 1000:+.1f}) mm"
                  f"   两尖相距 {np.linalg.norm(tips[1] - tips[0]) * 1000:.1f}mm"
                  f"   爪角 {grip_deg:+.1f}°")
        # 腕相机那一份直接从**最近一拍的快照**取（不再调一次 `locate`：感知线程
        # 正在用同一份缓存，而且这一下要 1.5 秒）。臂没动，一两百毫秒前的读数就是
        # 现在这个姿态的读数。
        wc = ((getattr(self, "_snap", None) or {}).get("box") or {}).get("center")
        if wc is not None:
            wr = np.asarray(wc, dtype=np.float64).copy()
            wr[:2] += off
            self._log(f"  腕相机读方块 ({wr[0] * 1000:+.1f}, {wr[1] * 1000:+.1f}) mm"
                      f"（读数里那一份，原始）")
            dw = (wr[:2] - mid[:2]) * 1000.0
            self._log(f"  差（腕 − 指尖） dx={dw[0]:+.1f} dy={dw[1]:+.1f} mm")
        if center is None:
            self._log(f"  侧相机没量出来：{facts.get('no_world_why') or facts.get('why')}")
            self._status = "align check: side sees nothing"
            return
        raw = np.asarray(center, dtype=np.float64).copy()
        raw[:2] += off                     # 加回被扣掉的 percep_offset
        d = (raw[:2] - mid[:2]) * 1000.0
        self._log(f"  侧相机读方块 ({raw[0] * 1000:+.1f}, {raw[1] * 1000:+.1f}) mm"
                  f"（已加回 {np.round(off * 1000, 1)}）")
        self._log(f"  差（侧 − 指尖） dx={d[0]:+.1f} dy={d[1]:+.1f} mm")
        self._status = f"align: dx={d[0]:+.1f} dy={d[1]:+.1f} mm"
        try:
            bgr = cv2.cvtColor(
                np.asarray(self.cams["real_side"].shot()), cv2.COLOR_RGB2BGR)
            out = (_ROOT / "out" / f"align_check_{int(time.time())}.png")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), draw(bgr, self.percep_side.last or {}))
            self._log(f"  那一帧：{out}")
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"！对账那一帧没存下来：{type(exc).__name__}: {exc}")
