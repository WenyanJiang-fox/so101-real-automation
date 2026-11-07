"""把**某一趟真机跑出来的路径点**放回仿真里走一遍 —— 是回放，不是重算。

和 `run_sim.py --config configs/sim_replay.yaml` 那次的区别，一句话：
**那次仿真自己又量了一遍盒子在哪、自己又解了一遍路径；这次一个数都不重量。**

真机那一趟（`out/run_real_13.log`）量出来的两个数直接钉进 `skills/pick_real.py` 的
`REPLAY`（见那个文件的注释）：起搜中心 `guess`、盒子中心 `center`、朝向 `yaw_deg`。
往下**走的还是同一个 `PickRealSkill.run`** —— 进场点、纠位点、转正法向、下探的
entry/mid/aim、抬升高度全是它自己从那三个数算出来的。所以仿真走的这条，就是真机
那一趟**打算**走的那条。

**一处坐标平移**：真机桌面在基座系 z = -9mm（`configs/real.yaml` 的 `table_z_mm`），
仿真的地面在 z = 0。所以搬过来的 z 一律 **+9mm**，保持「离桌面多高」不变。下面每
一段都把这个平移打在日志里。

用法：
    G:/anaconda/envs/sim2real/python.exe replay_real_path.py
    G:/anaconda/envs/sim2real/python.exe replay_real_path.py --log out/run_real_13.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import impls.arm_sim_real          # noqa: F401,E402  注册 sim_real 机械臂
import impls.cam_sim               # noqa: F401,E402  注册两路仿真相机
import impls.judge_held            # noqa: F401,E402
import real.perception             # noqa: F401,E402  注册 brown_rect（真机那一支）
import skills.home_real            # noqa: F401,E402
import skills.pick_real            # noqa: F401,E402
from core import geom, shapes                              # noqa: E402
from core.registry import make                             # noqa: E402
from core.round_state import RoundState                    # noqa: E402
from run_sim import CONTROL_HZ, write_video                # noqa: E402

# 真机桌面在基座系的 z（`configs/real.yaml: table_z_mm`）。仿真的地面是 0。
REAL_TABLE_Z_MM = -9.0
Z_SHIFT = -REAL_TABLE_Z_MM / 1000.0        # 搬过来时 z 加这么多

# 那一趟日志里的一行：
#     量出来：x=239 y=47 z=-1mm  yaw -42.7°（起搜中心 [227.  68.   1.]mm，挪了 …）
_LINE = re.compile(
    r"量出来：x=(-?\d+) y=(-?\d+) z=(-?\d+)mm\s+yaw\s+(-?[\d.]+)°"
    r"（起搜中心 \[\s*(-?[\d.]+)\.\s+(-?[\d.]+)\.\s+(-?[\d.]+)\.\]mm")


def read_log(path: Path) -> tuple[dict, str]:
    """从日志里读**第一条**「量出来」—— 就是第一趟尝试量到的那个盒子。

    不读后面那几条：第 2、3 趟是臂走不动之后在原地乱量的（`run_real_13.log` 第 48 行
    那次量出 x=-73mm，明显是相机里已经没有盒子了）。回放要的是第一次看准的那个数。
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    for ln in text.splitlines():
        m = _LINE.search(ln)
        if not m:
            continue
        cx, cy, cz, yaw, gx, gy, gz = (float(v) for v in m.groups())
        return ({"guess": np.array([gx, gy, gz]) / 1000.0,
                 "center": np.array([cx, cy, cz]) / 1000.0,
                 "yaw_deg": yaw}, ln.strip())
    raise SystemExit(f"没在 {path} 里找到「量出来」那一行")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="out/run_real_13.log",
                    help="从哪一趟真机日志里读路径点（只读第一条「量出来」）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="视频写到哪（默认按时长自动起名）")
    args = ap.parse_args()

    log = (HERE / args.log) if not Path(args.log).is_absolute() else Path(args.log)
    spot, src_line = read_log(log)

    print("=" * 78)
    print(f"回放来源：{log}")
    print(f"  {src_line}")
    print(f"坐标平移：真机桌面 z={REAL_TABLE_Z_MM:.0f}mm → 仿真地面 z=0，"
          f"搬过来的 z 一律 +{-REAL_TABLE_Z_MM:.0f}mm")
    print("=" * 78)

    # 真机现场那两个常数。仿真这一趟不拿它们算路径（`REPLAY` 已经把起搜中心钉死了），
    # 但感知建对象时要读，摆上去。
    geom.REAL_TABLE_Z = 0.0
    geom.REAL_PICK_XY = np.array([0.227, 0.068])
    geom.OBJECTS = [{"shape": "cube", "color": "red", "side_mm": 20.0}]
    geom.EXTRA_MARKS = []

    arm = make("arm", "sim_real", objects=geom.object_specs(),
               markers=False, ring=False)
    cams = {n: make("camera", n, source=arm)
            for n in ("sim_overhead", "sim_wrist")}
    percep = make("perception", "brown_rect", cams=cams,
                  ref=arm.ref_frame(), objects=geom.object_specs(),
                  cam_name="sim_wrist")

    arm.reset(args.seed)

    # 盒子摆到那一趟量出来的地方。**这一步是「操作员把盒子放上去」**，真机上就是人放的；
    # 高度用仿真里那个自然躺着的高度（20mm 方块中心在 10mm），不硬塞真机那个 z：
    # 真机量到的 -1mm 是它那一趟自己的 2mm 误差，塞进仿真的地面下面去只会被顶回来。
    side = float(shapes.from_spec(geom.OBJECTS[0]).rest_z)
    arm.place_cube(0, tuple(spot["center"][:2]), side, float(np.radians(spot["yaw_deg"])))
    print(f"摆件：盒子摆在 x={spot['center'][0] * 1000:.0f} "
          f"y={spot['center'][1] * 1000:.0f} z={side * 1000:.0f}mm，"
          f"朝向 {spot['yaw_deg']:+.1f}°（都是那一趟量出来的）")
    print(f"  回放的下探终点 z={(spot['center'][2] + Z_SHIFT) * 1000:.0f}mm"
          f"＝盒子中心下面 {abs(spot['center'][2] + Z_SHIFT - side) * 1000:.0f}mm"
          f"（真机那个 -1mm 比它自己方块的躺高还低 2mm）\n")

    # **三个数钉死，别的全放它自己算。**
    skills.pick_real.knobs.REPLAY = {
        "guess": spot["guess"] + np.array([0.0, 0.0, Z_SHIFT]),
        "center": spot["center"] + np.array([0.0, 0.0, Z_SHIFT]),
        "yaw_deg": spot["yaw_deg"],
        "src": f"{log.name} 第 1 趟",
    }

    frames: list[dict] = []
    n = [0]

    def on_step() -> None:
        n[0] += 1
        if n[0] % args.stride:
            return
        frames.append({name: c.shot() for name, c in cams.items()})

    arm.on_step = on_step
    state = RoundState()

    print("真机那一趟走的是：home_real → pick_real → home_real\n")
    home = make("skill", "home_real")
    print("── 1. 回初始位置")
    print(f"   {home.run(arm, cams, state=state)}")

    print("\n── 2. 抓一次（路径点是真机那一趟的）")
    out = make("skill", "pick_real").run(
        arm, cams, state=state, percep=percep,
        instruction="pick up the red box", index=0)
    print(f"   {out}")

    print("\n── 3. 回初始位置")
    print(f"   {home.run(arm, cams, state=state)}")

    if not frames:
        print("没录到画面")
        return 1
    fps = args.fps or CONTROL_HZ / args.stride
    name = args.out or (f"replay_{log.stem}_overhead_wrist_{n[0]}step.mp4")
    path = HERE / "out" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(path, frames, list(cams), fps)
    print(f"\n控制步 {n[0]}，留帧 {len(frames)}，{fps:.1f}fps")
    print(f"视频：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
