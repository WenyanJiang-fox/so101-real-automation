"""离线扫描：**新规则下「挑一个关节」挑得对不对**（不动臂、不碰串口）。

挑关节只用臂的运动学模型（`RealArm._fk` / `_sim_pose` / `_joint_dir_xy`），跟仿真
（`impls/arm_sim.py`）和真机是同一套数。所以不需要另造一个 MuJoCo 场景：这里用
`StubCtrl` + 覆盖 `_ticks_to_qpos`，在多个姿态、多个方向上直接把 `_free_for` 和
`_ik_free` 跑一遍。

三个问题：
  1. `_free_for` 是不是每次都只返回**一个**关节；
  2. 返回的那个关节，在模型里是不是真的沿这一步方向有力（门槛同代码：mm/度、cos）；
  3. 只放这一个关节、走 5mm 和 20mm（小步 / 大步），实际位移沿指令方向占多少、
     横漂和 Δz 有多大 —— **挑对了 = 投影为正且占大部分**。

姿态在关节限位内均匀抽；方向在水平面内均匀抽。随机种子固定，可复现。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from real.arm import RealArm                                       # noqa: E402
from skills import pick_real as PR                                 # noqa: E402

N_POSE = 60          # 抽多少个姿态
N_DIR = 6            # 每个姿态抽多少个方向
STEP_LENS = (0.005, 0.020)   # 小步 / 大步（米）
SEED = 20261018


class StubCtrl:
    """串口替身：读回中位、写下去丢掉。和 `_probe_free_for.py` 同一份。"""

    def sync_read_positions(self, ids):
        return {int(i): 2048 for i in ids}

    def sync_set_positions(self, mapping, speed=0):
        pass


def main() -> int:
    rng = np.random.default_rng(SEED)
    arm = RealArm(ctrl=StubCtrl())
    base = PR._gap_free()          # 缝那一段的首选（写死的兜底）

    n_cases = 0
    n_single = 0                   # 只返回一个关节的次数
    n_strong = 0                   # 返回的关节过了两门槛
    n_none_strong = 0              # 这个姿态/方向下谁都没力臂（退兜底）
    step_stat = {L: [] for L in STEP_LENS}   # (投影比, 横漂比, dz比)

    q_lo = np.asarray(arm.q_lo, dtype=np.float64)[:5]
    q_hi = np.asarray(arm.q_hi, dtype=np.float64)[:5]

    for _pose in range(N_POSE):
        q = rng.uniform(q_lo, q_hi)
        q = np.concatenate([q, [0.0]])          # 夹爪那一维不用（_free_for 不看它）
        q_now = q.copy()
        arm.read_ticks = lambda: {}
        arm._ticks_to_qpos = lambda ticks: np.array(q_now, dtype=np.float64)
        tcp = np.asarray(arm._sim_pose(*arm._fk(q_now))[0], dtype=np.float64)
        quat = np.asarray(arm.quat(), dtype=np.float64)
        if not np.all(np.isfinite(tcp)):
            continue

        # 这一姿态下每个关节的 (mm/度, 方向余弦)，给「谁都没力臂」判据用
        dirs = {}
        for j in range(5):
            d, mm = PR._joint_dir_xy(arm, q_now, j)
            dirs[j] = (None if d is None else d, float(mm))

        for _d in range(N_DIR):
            th = rng.uniform(0.0, 2.0 * np.pi)
            need = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
            n_cases += 1

            got = PR._free_for(arm, need, base)
            if len(got) == 1:
                n_single += 1
            j = int(got[0])
            d, mm = dirs[j]
            cos = 0.0 if d is None else abs(float(d @ need))
            strong = (mm >= PR.GAP_FALLBACK_MIN_MM_DEG
                      and cos >= PR.GAP_FALLBACK_MIN_COS)
            if strong:
                n_strong += 1
            else:
                # 是不是这个姿态/方向下本来就谁都没力臂（那就该退兜底）
                any_strong = any(
                    dd is not None
                    and mmj >= PR.GAP_FALLBACK_MIN_MM_DEG
                    and abs(float(dd @ need)) >= PR.GAP_FALLBACK_MIN_COS
                    for dd, mmj in dirs.values())
                if not any_strong:
                    n_none_strong += 1

            for L in STEP_LENS:
                tgt = np.array([tcp[0] + need[0] * L,
                                tcp[1] + need[1] * L, tcp[2]], dtype=np.float64)
                qg = PR._ik_free(arm, *arm._site_pose(tgt, quat), q_now, (j,))
                p = np.asarray(arm._sim_pose(*arm._fk(qg))[0], dtype=np.float64)
                disp = p - tcp
                proj = float(disp[:2] @ need) / L
                lateral = float(np.linalg.norm(disp[:2] - need * (disp[:2] @ need))) / L
                dzr = float(disp[2]) / L
                step_stat[L].append((proj, lateral, dzr))

    def _pct(x, p):
        return float(np.percentile(np.asarray(x), p))

    print(f"姿态 {N_POSE} × 方向 {N_DIR} = {n_cases} 例（种子 {SEED}）")
    print(f"  只返回一个关节：{n_single}/{n_cases}")
    print(f"  返回的关节过了两门槛：{n_strong}/{n_cases}")
    print(f"  谁都没力臂、只能退兜底的：{n_none_strong}/{n_cases}")

    for L in STEP_LENS:
        a = np.asarray(step_stat[L], dtype=np.float64)
        proj, lat, dzr = a[:, 0], a[:, 1], a[:, 2]
        print(f"\n只放选中的那一个关节，走 {L * 1000:.0f}mm：")
        print(f"  沿指令方向的投影比  中位 {_pct(proj, 50):+.2f}  "
              f"10%分位 {_pct(proj, 10):+.2f}  最小 {proj.min():+.2f}")
        print(f"  投影 ≥0.5 的比例：{float((proj >= 0.5).mean()) * 100:.0f}%   "
              f"投影 ≤0（走反）：{float((proj <= 0.0).mean()) * 100:.0f}%")
        print(f"  横漂 / 步长        中位 {_pct(lat, 50):.2f}  "
              f"90%分位 {_pct(lat, 90):.2f}")
        print(f"  |Δz| / 步长        中位 {_pct(np.abs(dzr), 50):.2f}  "
              f"90%分位 {_pct(np.abs(dzr), 90):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
