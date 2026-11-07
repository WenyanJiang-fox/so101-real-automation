"""离线：`skills/pick_real.py` 里新加的「摆件」那一段。**不动臂、不碰串口。**

三问：
  1. `_place_xy_real` 抽出来的点，是不是每一个都够得着、都离原处够远、都在方位窗里；
  2. 松爪高度 `z_rel = 桌面 + 半个边长 - 手里最低那点` 算出来对不对
     （拿假 tcp/tips 喂进去，看最低那点是不是正好落在桌面 + 半个边长）；
  3. 「没落到半个边长」那道闸门（`clear > half + 5mm` 就不松爪）判得对不对。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import geom                                              # noqa: E402
from real.arm import RealArm                                       # noqa: E402
from skills import pick_real as PR                                 # noqa: E402

# 三个真机常数平时是 `panel/rig.py` / `real/perception.py` 从 yaml 里灌进 `core.geom`
# 的；这个探针不起面板，就照 `run_sim.py` 那几行自己灌一遍（只读 yaml，不碰串口）。
import yaml                                                        # noqa: E402

_cfg = yaml.safe_load((Path(__file__).resolve().parents[2]
                       / "configs" / "real.yaml").read_text(encoding="utf-8"))
_real = _cfg["real"]
geom.REAL_TABLE_Z = float(_real["table_z_mm"]) / 1000.0
geom.REAL_PICK_XY = np.array([float(v) for v in _real["pick_xy_mm"]],
                             dtype=np.float64) / 1000.0
geom.REAL_PERCEP_OFFSET = np.array(
    [float(v) for v in _real.get("percep_offset_mm", (0.0, 0.0))],
    dtype=np.float64) / 1000.0


class StubCtrl:
    def sync_read_positions(self, ids):
        return {int(i): 2048 for i in ids}

    def sync_set_positions(self, mapping, speed=0):
        pass


def main() -> int:
    base = np.asarray(geom.MARKER_CENTER, dtype=np.float64)
    pick = np.asarray(geom.real_pick_xy(), dtype=np.float64).reshape(-1)[:2]
    base_deg = float(np.degrees(np.arctan2(pick[1] - base[1], pick[0] - base[0])))
    print(f"基点 {np.round(base * 1000, 1)}mm   工位 {np.round(pick * 1000, 1)}mm"
          f"   方位 {base_deg:+.1f}°   桌面 z {geom.real_table_z() * 1000:+.1f}mm")
    print(f"可达环 {geom.REACH_MIN_R * 1000:.0f}~{geom.REACH_MAX_R * 1000:.0f}mm")

    print("\n== 1. `_place_xy_real` 抽 500 次")
    center = np.array([pick[0], pick[1], 0.0], dtype=np.float64)
    n_bad_reach = n_bad_far = n_bad_deg = 0
    r_min, r_max, d_min = 1e9, -1e9, 1e9
    deg_min, deg_max = 1e9, -1e9
    for i in range(500):
        xy, why = PR._place_xy_real(center)
        r = float(np.hypot(xy[0] - base[0], xy[1] - base[1]))
        d = float(np.hypot(xy[0] - center[0], xy[1] - center[1]))
        deg = float(np.degrees(np.arctan2(xy[1] - base[1], xy[0] - base[0])))
        r_min, r_max = min(r_min, r), max(r_max, r)
        d_min = min(d_min, d)
        deg_min, deg_max = min(deg_min, deg), max(deg_max, deg)
        n_bad_reach += bool(geom.out_of_reach(xy))
        n_bad_far += bool(d < PR.PLACE_AWAY_M)
        n_bad_deg += bool(deg > base_deg + PR.PLACE_DEG_LEFT + 1e-6
                          or deg < base_deg - PR.PLACE_DEG_RIGHT - 1e-6)
        if i == 0:
            print(f"   头一次：({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f})mm —— {why}")
    print(f"   半径实测 {r_min * 1000:.0f}~{r_max * 1000:.0f}mm"
          f"（要 {PR.PLACE_R_MIN_M * 1000:.0f}~{PR.PLACE_R_MAX_M * 1000:.0f}，"
          f"可达环内 {'✓' if not n_bad_reach else '✗'}）")
    print(f"   离原处最近 {d_min * 1000:.0f}mm（要 ≥{PR.PLACE_AWAY_M * 1000:.0f}）"
          f"   越界次数 {n_bad_far}")
    print(f"   方位实测 {deg_min:+.1f}~{deg_max:+.1f}°（窗 {base_deg:+.1f}"
          f" 左+{PR.PLACE_DEG_LEFT:.0f}/右-{PR.PLACE_DEG_RIGHT:.0f}）   越界次数 {n_bad_deg}")
    print(f"   → 够不着 {n_bad_reach} / 太近 {n_bad_far} / 跑出方位窗 {n_bad_deg}")

    print("\n== 2. 松爪高度算式")
    arm = RealArm(ctrl=StubCtrl())
    arm.read_ticks = lambda: {}
    table = float(geom.real_table_z())
    for half_mm, tip_mm, box_low_mm in ((9.0, 4.0, -12.0), (9.0, -14.0, -12.0),
                                        (20.0, 3.0, -40.0)):
        half = half_mm / 1000.0
        tcp = np.array([0.20, 0.05, 0.10], dtype=np.float64)
        arm.tcp = lambda tcp=tcp: tcp
        arm.tips = lambda tip=tip_mm, tcp=tcp: (
            np.array([tcp[0] + 0.01, tcp[1], tcp[2] + tip / 1000.0]),
            np.array([tcp[0] - 0.01, tcp[1], tcp[2] + tip / 1000.0]))
        grasp_z = float(tcp[2])
        c0 = np.array([0.20, 0.05, grasp_z + half + box_low_mm / 1000.0])
        rel_tips = float(min(arm.tips()[0][2], arm.tips()[1][2]) - tcp[2])
        rel_box = float((c0[2] - half) - grasp_z)
        rel_low = float(min(rel_tips, rel_box))
        z_rel = float(table + half - rel_low)
        # 到达那一刻：TCP 落在 `z_rel`，两个偏移量跟着走，于是现场量到的最低点就是
        # `z_rel + rel_low`（照 `_place_real` 里 `low = min(tips2..., tcp2[2]+rel_box)`）。
        low = float(z_rel + rel_low)
        want = table + half
        print(f"   半边长 {half_mm:4.1f}  爪尖 {tip_mm:+6.1f}  方块底 {box_low_mm:+6.1f}"
              f" → 低的那个 {'爪尖' if rel_tips <= rel_box else '方块底'}"
              f"  松爪 TCP {z_rel * 1000:+7.1f}mm"
              f"  那一刻最低点 {low * 1000:+7.1f}mm"
              f"  要 {want * 1000:+7.1f}mm"
              f"  差 {abs(low - want) * 1000:.3f}mm")

    print("\n== 3. 那道闸门（`clear > half + 5mm` 就不松爪）")
    half = 0.009
    for low_mm in (half * 1000, half * 1000 - 3, half * 1000 + 3,
                   half * 1000 + 6, half * 1000 + 60):
        clear = low_mm / 1000.0
        fire = bool(clear > half + PR.PLACE_Z_TOL_MM / 1000.0)
        print(f"   最低点离桌面 {low_mm:6.1f}mm   半个边长 {half * 1000:.1f}mm"
              f"   → {'✗ 不松爪' if fire else '✓ 松爪'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
