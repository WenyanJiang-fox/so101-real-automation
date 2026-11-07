"""离线重放 `_goto` 的三趟：**放大 vs 不放大，谁收敛**（不碰串口、不动臂）。

起点和放大倍数全抄 2026-10-18 01:54 那趟日志的第 2 趟：

    q_now  [ -11.5   2.2  26.1  67.1 -141.2  44.6]   离目标 8.4mm
    放大   1号×1.1 2号×2.7 4号×1.1 5号×1.2            （第 1 趟实测给的）
    实际交付（这一趟读完）：2号 +3.7°、3号 -2.5°、4号 +3.3°，3 号只跟了 57%

舵机的交付比例是**从那一趟日志反推的**（不是模型算的），拿来只比两种规则谁稳：
  · old = 现在线上那套「按实测放大」（gmax=3）；
  · new = 放开五关节那条路放大关掉（gmax=1），每趟从走到的姿态重解。
逆解、正解、姿态口径全部走 `pick_real._ik_free` / `RealArm`，同一套数。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from real.arm import RealArm                                       # noqa: E402
from skills.pick_real import _ik_free, GAIN_MIN_ASK_DEG, GAIN_MAX  # noqa: E402

GOAL = np.array([215.3, 40.2, 9.4]) / 1000.0
Q0 = np.radians(np.array([-11.5, 2.2, 26.1, 67.1, -141.2, 44.6]))
GAIN0 = np.array([1.1, 2.7, 1.0, 1.1, 1.2, 1.0])    # 日志里第 1 趟之后印的
DELIVER = np.array([1.0, 1.1, 0.57, 1.03, 0.7, 1.0])  # 日志反推的交付比


class StubCtrl:
    def sync_read_positions(self, ids):
        return {int(i): 2048 for i in ids}

    def sync_set_positions(self, mapping, speed=0):
        pass


def run(mode: str) -> None:
    arm = RealArm(ctrl=StubCtrl())
    q = np.array(Q0, dtype=np.float64)
    arm.read_ticks = lambda: {}
    arm._ticks_to_qpos = lambda ticks: np.array(q, dtype=np.float64)
    gain = np.array(GAIN0, dtype=np.float64) if mode == "old" else np.ones(6)
    fixed = None
    print(f"\n== {mode} ==  起点离目标 "
          f"{np.linalg.norm(np.asarray(arm._sim_pose(*arm._fk(q))[0]) - GOAL) * 1000:.1f}mm")
    for k in range(3):
        if mode == "fixed" and fixed is not None:
            q_goal = fixed.copy()
        else:
            tgt_pos, tgt_quat = arm._site_pose(GOAL, arm.quat())
            q_goal = _ik_free(arm, tgt_pos, tgt_quat, q, (0, 1, 2, 3, 4))
            if mode == "fixed":
                fixed = q_goal.copy()
        res = np.linalg.norm(np.asarray(arm._sim_pose(*arm._fk(q_goal))[0]) - GOAL)
        want = np.asarray(q_goal, dtype=np.float64) - q
        want[5] = 0.0
        q_cmd = q + want * gain
        got = want * gain * DELIVER                       # 舵机按比例交付
        q = q + got
        err = np.linalg.norm(np.asarray(arm._sim_pose(*arm._fk(q))[0]) - GOAL)
        print(f"  第 {k + 1} 趟：指令 {np.round(np.degrees(np.abs(want * gain)), 1)}°"
              f"（模型残差 {res * 1000:.1f}mm）→ 走完离目标 {err * 1000:.1f}mm")
        # 下一趟的放大（和 `_goto` 同一套：got/want、≤0 或 <0.3° 不放大、夹到 gmax）
        ratio = np.divide(got, want * gain, out=np.ones(6),
                          where=np.abs(want * gain) > 1e-12)
        ratio[(ratio <= 0.0) | (np.abs(want * gain) <= np.radians(GAIN_MIN_ASK_DEG))] = 1.0
        ratio = np.clip(ratio, 1.0 / GAIN_MAX, 1.0)
        ratio[5] = 1.0
        gain = np.ones(6) if mode == "new" else 1.0 / ratio


def main() -> int:
    run("old")
    run("new")
    run("fixed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
