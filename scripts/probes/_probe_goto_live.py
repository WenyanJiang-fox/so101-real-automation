"""**直接调线上那段 `_goto`**（不是重写一遍），喂假的舵机交付比，看它收不收敛。

不碰串口、不动臂：`RealArm(ctrl=StubCtrl())` + 把 `read_ticks`/`_ticks_to_qpos`
换成读一个可变的 `q`；`real.joint_move.goto_q` 换成「按交付比走一部分」。

起点/目标抄 2026-10-18 01:54 那趟日志（多关节大步，2 号被放大到 2.7 倍那一次）。
跑四组交付比：日志反推的那组、全 0.6、全 0.4、最差关节 0.3。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import real.joint_move as jm                                      # noqa: E402
from real.arm import RealArm                                      # noqa: E402
from skills.pick_real import _goto                                # noqa: E402

GOAL = np.array([215.3, 40.2, 9.4]) / 1000.0
Q0 = np.radians(np.array([-11.5, 2.2, 26.1, 67.1, -141.2, 44.6]))

CASES = {
    "日志反推": np.array([1.0, 1.1, 0.57, 1.03, 0.7, 1.0]),
    "全 0.6": np.array([0.6, 0.6, 0.6, 0.6, 0.6, 1.0]),
    "全 0.4": np.array([0.4, 0.4, 0.4, 0.4, 0.4, 1.0]),
    "最差 0.3": np.array([0.3, 0.3, 0.3, 0.3, 0.3, 1.0]),
}


class StubCtrl:
    def sync_read_positions(self, ids):
        return {int(i): 2048 for i in ids}

    def sync_set_positions(self, mapping, speed=0):
        pass


def make_arm(deliver):
    arm = RealArm(ctrl=StubCtrl())
    state = {"q": np.array(Q0, dtype=np.float64)}

    arm.read_ticks = lambda: {}
    arm._ticks_to_qpos = lambda ticks: np.array(state["q"], dtype=np.float64)

    def fake_goto_q(a, q_goal, *, q_from=None, **kw):
        qf = (np.asarray(q_from, dtype=np.float64) if q_from is not None
              else np.array(state["q"], dtype=np.float64))
        # 夹爪那一维按指令到位；其余按交付比
        got = (np.asarray(q_goal, dtype=np.float64) - qf) * deliver
        got[5] = np.asarray(q_goal, dtype=np.float64)[5] - qf[5]
        state["q"] = qf + got
        return np.array(state["q"], dtype=np.float64)

    jm.goto_q = fake_goto_q
    return arm, state


def tcp_err(arm):
    q = arm._ticks_to_qpos(arm.read_ticks())
    p = np.asarray(arm._sim_pose(*arm._fk(q))[0], dtype=np.float64)
    return float(np.linalg.norm(p - GOAL)) * 1000.0


def main() -> int:
    orig = jm.goto_q
    for name, deliver in CASES.items():
        arm, state = make_arm(deliver)   # 这一组把 jm.goto_q 换成自己的
        print(f"\n===== {name} =====")
        print(f"起点离目标 {tcp_err(arm):.1f}mm")
        out = _goto(arm, GOAL, vmax=0.05, grip=1.0, free=(0, 1, 2, 3, 4))
        print(f"→ {out}  最终 {tcp_err(arm):.1f}mm")
    jm.goto_q = orig
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
