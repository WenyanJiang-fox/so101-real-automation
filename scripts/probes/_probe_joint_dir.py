"""离线探针：把日志里**每一趟走位**的带符号关节指令还原出来，跟实际走的方向对一遍。

**为什么能还原。** `_goto` 的指令是
    q_goal = arm._ik(*arm._site_pose(goal, arm.quat()), q_now)
逆解是确定性的，`goal`（那行 `→ [x y z]mm`）和 `q_now`（上一趟 `走到` 那行的关节角）
日志里都有；`arm.quat()` 就是那组关节角的正解姿态，也能算出来。所以指令正负可以重算。

**为什么要正负。** `_goto` 里那句 `要动` 是 `np.degrees(np.abs(q_goal - q_now))` ——
取了绝对值，日志上看不出方向。run 69 抬高度那两趟就是这么被盖住的。

    G:/anaconda/envs/sim2real/python.exe -u _probe_joint_dir.py out/run_real_69.log [out/...]

只看指令幅度超过下面 `MIN_CMD_DEG` 的关节：日志的关节角印到 0.1°，0.1° 以下的差
是四舍五入噪声，不算数。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from real.arm import RealArm                                       # noqa: E402

MIN_CMD_DEG = 0.3

RE_GO = re.compile(r"走位（关节空间）第\s*(\d+)\s*趟：\[([^\]]+)\]mm\s*→\s*\[([^\]]+)\]mm"
                   r"\s*要动\s*\[([^\]]+)\]")
RE_END = re.compile(r"走到 \[([^\]]+)\]mm\s*差\s*([\d.]+)mm\s*关节\s*\[([^\]]+)\]")


class FakeCtrl:
    def __init__(self) -> None:
        self.ticks: dict[int, int] = {}

    def sync_read_positions(self, ids):
        return {int(i): int(self.ticks.get(int(i), 2048)) for i in ids}

    def sync_set_positions(self, goal, speed=0):
        raise AssertionError("探针脚本不许写舵机")


def nums(s: str) -> np.ndarray:
    return np.array([float(x) for x in
                     re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", s)],
                    dtype=np.float64)


def main() -> None:
    arm = RealArm(ctrl=FakeCtrl())
    rows = []
    for path in sys.argv[1:]:
        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        cur_q = None                       # 上一趟走完读回的关节角（度）
        pend = None                        # 刚读到、还没配到 `走到` 的那条 `走位`
        for ln in lines:
            mgo = RE_GO.search(ln)
            if mgo:
                pend = (nums(mgo.group(2)), nums(mgo.group(3)), nums(mgo.group(4)))
                continue
            mend = RE_END.search(ln)
            if mend and pend is not None:
                start_deg, end_deg = cur_q, nums(mend.group(3))
                cur_q = end_deg                    # **不管这一趟算不算，起点都要往前走**
                if start_deg is not None and start_deg.size == 6:
                    q0 = np.radians(start_deg)
                    goal = pend[1] / 1000.0
                    _pos, quat_c = arm._sim_pose(*arm._fk(q0))
                    q_goal = arm._ik(*arm._site_pose(goal, quat_c), q0)
                    rows.append({
                        "file": p.name,
                        "cmd": np.degrees(np.abs(q_goal - q0)),
                        "sign": np.sign(np.degrees(q_goal - q0)),
                        "got": np.degrees(np.radians(end_deg) - q0),
                        "err_mm": float(mend.group(2)),
                        "said": (pend[2] if len(pend) > 2 else None),
                        "q0": start_deg,
                        "start_z_mm": float(pend[0][2]),
                        "goal_z_mm": float(goal[2] * 1000.0),
                    })
                pend = None
                continue
            if mgo is None and mend is None:
                continue
        _ = lines

    if not rows:
        print("没解析到任何一趟走位")
        return

    names = ["1号", "2号", "3号", "4号", "5号", "6号"]
    print(f"解析到 {len(rows)} 趟走位（{MIN_CMD_DEG}° 以下的指令不算）\n")

    print("== 0. 先验还原对不对：重算的「要动」跟日志那句「要动」比 ==")
    diffs = []
    for r in rows:
        if r["said"] is None or r["said"].size != 6:
            continue
        diffs.append(np.abs(r["cmd"] - r["said"]))
    if diffs:
        d = np.vstack(diffs)
        print(f"   {len(d)} 趟能对上账，逐关节差值：中位 "
              f"{np.median(d):.3f}°  95 分位 {np.percentile(d, 95):.3f}°  "
              f"最大 {d.max():.3f}°")
        print("   分关节（差值 > 0.15° 的格子占比 / 中位差 / 最大差）：")
        for j in range(6):
            print(f"     {names[j]}  {100 * (d[:, j] > 0.15).mean():5.1f}%"
                  f"   {np.median(d[:, j]):.3f}°   {d[:, j].max():.3f}°")
        print("   （日志把关节角印到 0.1°，0.1° 以下是四舍五入；哪一号对不上，"
              "哪一号的结论就不能信）\n")


    print("== 每号关节：指令超过门槛的趟里，实际方向反了几趟 ==")
    tot = np.zeros(6, dtype=int)
    bad = np.zeros(6, dtype=int)
    for r in rows:
        for j in range(6):
            if r["cmd"][j] < MIN_CMD_DEG:
                continue
            tot[j] += 1
            if np.sign(r["got"][j]) != r["sign"][j] and abs(r["got"][j]) > 0.15:
                bad[j] += 1
    down_up = np.zeros(6, dtype=int)       # 指令要往下（负）、实际往上（正）
    up_down = np.zeros(6, dtype=int)       # 指令要往上（正）、实际往下（负）
    mags = {j: [] for j in range(6)}
    for r in rows:
        for j in range(6):
            if r["cmd"][j] < MIN_CMD_DEG:
                continue
            if np.sign(r["got"][j]) != r["sign"][j] and abs(r["got"][j]) > 0.15:
                if r["sign"][j] < 0:
                    down_up[j] += 1
                else:
                    up_down[j] += 1
                mags[j].append(abs(r["got"][j]))
    for j in range(6):
        pct = (100.0 * bad[j] / tot[j]) if tot[j] else float("nan")
        mag = (f"实际反向幅度中位 {np.median(mags[j]):.2f}°"
               f"、最大 {max(mags[j]):.2f}°" if mags[j] else "")
        print(f"  {names[j]}  指令够大的 {tot[j]:3d} 趟，走反 {bad[j]:3d} 趟"
              f"（{pct:5.1f}%）  其中「指令往下→实际往上」{down_up[j]:3d} 趟"
              f"、「指令往上→实际往下」{up_down[j]:3d} 趟   {mag}")

    print("\n== 每号关节「角度 +1°」时 TCP 往哪走（取反向那几趟的姿态附近算）==")
    poses = [np.radians(r["q0"]) for r in rows if any(
        r["cmd"][j] >= MIN_CMD_DEG and np.sign(r["got"][j]) != r["sign"][j]
        and abs(r["got"][j]) > 0.15 for j in range(6))]
    if poses:
        base = np.mean(np.vstack(poses), axis=0)
        p0, _ = arm._sim_pose(*arm._fk(base))
        print(f"   参照姿态 {np.round(np.degrees(base), 1)}°　TCP "
              f"{np.round(p0 * 1000, 1)}mm")
        for j in range(6):
            q = base.copy()
            q[j] += np.radians(1.0)
            p, _ = arm._sim_pose(*arm._fk(q))
            d = (p - p0) * 1000.0
            print(f"     {names[j]} +1° → TCP 挪 [{d[0]:+6.2f} {d[1]:+6.2f} "
                  f"{d[2]:+6.2f}]mm   z {'升' if d[2] > 0.02 else '降' if d[2] < -0.02 else '几乎不动'}")

    print("\n== 反向那几趟，这一步本来是要把 TCP 往哪走 ==")
    for j in range(6):
        ups = downs = 0
        for r in rows:
            if r["cmd"][j] < MIN_CMD_DEG:
                continue
            if not (np.sign(r["got"][j]) != r["sign"][j] and abs(r["got"][j]) > 0.15):
                continue
            if r["goal_z_mm"] > r["start_z_mm"] + 0.05:
                ups += 1
            elif r["goal_z_mm"] < r["start_z_mm"] - 0.05:
                downs += 1
        if ups or downs:
            print(f"  {names[j]}：反向的趟里，目标是抬高的 {ups} 趟、降低的 {downs} 趟")

    allsteps = {"up": 0, "down": 0, "flat": 0}
    for r in rows:
        if r["goal_z_mm"] > r["start_z_mm"] + 0.05:
            allsteps["up"] += 1
        elif r["goal_z_mm"] < r["start_z_mm"] - 0.05:
            allsteps["down"] += 1
        else:
            allsteps["flat"] += 1
    print(f"  （全部 {len(rows)} 趟里：抬高 {allsteps['up']}、降低 {allsteps['down']}、"
          f"高度不动 {allsteps['flat']}）")

    print("\n== 反着走的那几趟（前 40 条）==")
    n = 0
    for r in rows:
        for j in range(6):
            if r["cmd"][j] < MIN_CMD_DEG:
                continue
            if np.sign(r["got"][j]) != r["sign"][j] and abs(r["got"][j]) > 0.15:
                print(f"  {r['file']} {names[j]}  指令 {r['sign'][j] * r['cmd'][j]:+.2f}°"
                      f"  实际 {r['got'][j]:+.2f}°   这一趟差 {r['err_mm']:.1f}mm")
                n += 1
                break
        if n >= 40:
            break

    print("\n== 指令方向对了、但幅度只走到一半以下的（前 20 条）==")
    n = 0
    for r in rows:
        for j in range(6):
            if r["cmd"][j] < MIN_CMD_DEG:
                continue
            if (np.sign(r["got"][j]) == r["sign"][j]
                    and 0.0 <= r["got"][j] * r["sign"][j] < 0.5 * r["cmd"][j]):
                print(f"  {r['file']} {names[j]}  指令 {r['sign'][j] * r['cmd'][j]:+.2f}°"
                      f"  实际 {r['got'][j]:+.2f}°   这一趟差 {r['err_mm']:.1f}mm")
                n += 1
                break
        if n >= 20:
            break


if __name__ == "__main__":
    main()
