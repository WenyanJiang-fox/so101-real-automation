"""采一组标定视图：臂挪二十几个姿态，每个姿态拍一张靶标、记一次关节读数。

**一次采样同时喂两件事**，因为它们要的是同一批数据：
    · 内参    同一块板子在不同位置/倾角下的多张图 → `cv2.calibrateCamera`
    · 手眼    同一批图 + 每个姿态的关节读数（FK）→ `cv2.calibrateHandEye`
所以图存下来时把关节角和舵机原始值一起记在旁边，换 cv2 之后不用重采。

**第四版：改成一维关节直驱，不走逆解。** 第三版想靠「抬高抓取点」和「掰工具轴」
拉开相机到板子的距离，实测全灭 —— 78 个点里抬高到 +60mm 以上一个都解不出，
那是运动学死区。真正管用的是 **4 号 wrist_flex 直接改相机的光轴方向**：实测桌面
落点能从 0.07m 推到 0.52m，相机到板子的距离从 0.16m 拉到 0.4m。而且**掰下去只会
把爪子抬高**（实测 −30° 时 TCP 前移 80mm、抬高 21mm），压不到平板，所以只动
1 号(pan) 和 4 号(flex) 这两维天然安全，不需要逆解把关。同一招的扫描版是
`real/sweep_look.py`（那边是粗扫找板子，这边是细采喂标定）。

**为什么非要掰。** 腕相机离桌面只有约 159mm、光轴正对下方，看到的桌面只有约
0.33m × 0.18m 一块。**不掰的视图全是「正对板子」** —— 板子摊在桌面上、光轴垂直
打下去，板平面和像平面平行，这是标定里最经典的病态姿态：焦距和距离完全耦合，
解出来的 fx 能差 ±10% 而重投影误差几乎不变。掰出 30~60° 的斜视角才把这条解开。

安全约定：
  * **默认什么都不动**，不加 `--go` 只打印打算走哪些点。
  * **TCP 永远不低于初始点位** —— 平板就摊在正下方，往下探就会压到。判据用
    `arm._fk` 的返回值，那是 **TCP（gripperframe）**，不是抓取点，两者差 58.7mm。
  * 中途 Ctrl+C 随时停，停在当时位置，**不松力矩**（松了这条臂会塌下来）。

跑法（在 G:\\sim2real\\so101-real-harness 下）：
    G:/anaconda/envs/sim2real/python.exe real/calib_views.py
    G:/anaconda/envs/sim2real/python.exe real/calib_views.py --go
出图：G:\\sim2real\\so101-real-harness\\out\\calibview_000.jpg ...
读数：G:\\sim2real\\so101-real-harness\\out\\calib_views.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.trajectory import build                        # noqa: E402
from real_home import RAW_HOME                           # noqa: E402
from servo_bridge import open_controller                 # noqa: E402

OUT = _ROOT / "out"

# 相机分辨率。**必须和标定时用的一档一致** —— 内参只对拍它的那个分辨率成立。
CAM_W, CAM_H = 1920, 1080
FOURCC = "MJPG"

# 一个姿态至少要认出几个角点才收。4 个是最低（能解一张板子的位姿），
# 但太少的姿态喂进去会把病态解带进标定，留一倍余量。
MIN_CORNERS = 8

# 关节序号（0 起）：1 号 pan -> 0，2 号 lift -> 1，3 号 elbow -> 2，
# 4 号 wrist_flex -> 3，5 号 wrist_roll -> 4。
J_PAN, J_LIFT, J_ELBOW, J_FLEX, J_ROLL = 0, 1, 2, 3, 4

# **主网格之外再补的几维**（2026-10-18：1/4 号那批图没转过 5 号，腕子一转估计飘 20mm）。
# 每一项 = (关节, 参考 pan, 参考 flex, 相对初始点位的角度列表)。**一次只动一个关节，
# 不做笛卡尔积** —— 2/3/5 号同时大改会让相机离板子太远、或者压低 TCP。
# 参考姿态用 pan 0 / flex −40：实测这一档能拍到整块板。
EXTRA_DEG = (
    (J_ROLL, 0.0, -40.0, (-60.0, -30.0, 30.0, 60.0)),
    (J_LIFT, 0.0, -40.0, (-8.0, 8.0)),
    (J_ELBOW, 0.0, -40.0, (-8.0, 8.0)),
)

# 4 号相对初始点位的转角（度）。**初始点位上 4 号已经在 88.1°，模型上限 ±95°** ——
# 正向只剩 6.9°，掰不动；负向还有 183°。所以只往负向掰。实测 −40 / −60 两档都能
# 拍到整块板（26~31 个角点），−20 那档就够不着了，所以更浅的档不用给。
FLEX_DEG = (-30.0, -40.0, -50.0, -60.0)

# 1 号相对初始点位的转角（度）。**不超过 ±60** —— 再大扫到桌外和那堆 3D 打印件上
# 去了。板子偏在一侧，实测 pan +60 那一片是空的，留着不亏：拍不到的会被跳过。
PAN_DEG = (-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0)

# 安全带：TCP 不许比初始点位更低。
SAFE_MARGIN = 0.005


def open_cam(index: int):
    """开相机并要到高分辨率。**不设这个就只有 640x480**，格点会糊掉。"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"打不开相机 index={index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    for _ in range(10):
        ok, fr = cap.read()
    if not ok or fr is None:
        cap.release()
        raise RuntimeError("相机读不出帧")
    got = (fr.shape[1], fr.shape[0])
    if got != (CAM_W, CAM_H):
        cap.release()
        raise RuntimeError(f"要 {CAM_W}x{CAM_H}，实得 {got} —— 标定分辨率对不上，不凑合")
    return cap


def main() -> int:                                             # noqa: C901
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="采一组标定视图（内参 + 手眼共用）")
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--index", type=int, default=1, help="相机序号")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--jvel", type=float, default=0.15, help="关节速度上限（弧度/秒），慢点稳")
    ap.add_argument("--jacc", type=float, default=0.6)
    ap.add_argument("--settle", type=float, default=0.6, help="走到位后等多久再拍（秒）")
    ap.add_argument("--want", type=int, default=24,
                    help=f"收够这么多张就停。计划里有 {len(PAN_DEG) * len(FLEX_DEG)} 个姿态，"
                         f"拍不到板子的会被跳过")
    ap.add_argument("--pan", default=None,
                    help=f"1 号的角度集，逗号分隔（度）。不给就用默认那七档")
    ap.add_argument("--flex", default=None,
                    help=f"4 号的角度集，逗号分隔（度）。不给就用默认那四档")
    ap.add_argument("--append", action="store_true",
                    help="接着已有的 out/calib_views.json 往下收（图号也接着编），"
                         "不覆盖。近距那轮就靠它并进来")
    ap.add_argument("--go", action="store_true", help="真的动。不给就只打印计划")
    args = ap.parse_args()

    pan_deg = PAN_DEG if args.pan is None else tuple(
        float(v) for v in args.pan.split(",") if v.strip())
    flex_deg = FLEX_DEG if args.flex is None else tuple(
        float(v) for v in args.flex.split(",") if v.strip())

    import make_charuco as mc                                    # noqa: PLC0415
    _, board = mc.build(8, 6)
    det = cv2.aruco.CharucoDetector(board)

    from real.arm import RealArm                                 # noqa: PLC0415

    ctrl = open_controller(args.port)
    arm = RealArm(ctrl=ctrl, speed=0)

    q_home = arm._ticks_to_qpos(RAW_HOME)
    # **`arm._fk` 给的是 TCP（gripperframe），不是抓取点。** 初始姿态下 TCP 是
    # 0.0593、抓取点是 0.1175，差 58.7mm。拿抓取点的值当门槛会把初始姿态自己判成不安全。
    tcp_home_z = float(arm._fk(q_home)[0][2])
    print(f"初始点位 关节(度) {np.round(np.degrees(q_home), 1)}")
    print(f"TCP 永远不低于 z = {tcp_home_z:.4f}（平板就在下面）\n")

    # 先把整条路算完再动。打印出来的计划**就是**接下来要走的那条，不是跑到一半
    # 再临场跳。够不着 / 会压低 TCP 的点在这一步就出局。
    print(f"计划 {len(pan_deg) * len(flex_deg)} 个姿态（不动臂），只列收下的：")
    solved: list[dict] = []
    for fi, flex in enumerate(flex_deg):
        # 一层里 pan 蛇形来回走，层与层之间相邻两点的 1 号不用横跨整个行程。
        pans = pan_deg if fi % 2 == 0 else tuple(reversed(pan_deg))
        for pan in pans:
            q_goal = q_home.copy()
            q_goal[J_PAN] += math.radians(pan)
            q_goal[J_FLEX] += math.radians(flex)
            tcp_z = float(arm._fk(q_goal)[0][2])
            if tcp_z < tcp_home_z - SAFE_MARGIN:
                print(f"  pan {pan:+5.0f}°  flex {flex:+5.0f}°  "
                      f"**TCP {tcp_z:.4f} 比初始低，丢**")
                continue
            solved.append({"pan": pan, "flex": flex, "q": q_goal, "tcp_z": tcp_z,
                           "extra": ""})
            print(f"  {len(solved):2d}  pan {pan:+5.0f}°  flex {flex:+5.0f}°  "
                  f"TCP z {tcp_z:.4f}")

    # ---- 补的那几维：一次只动 2/3/5 号其中一个 ----
    for joint, pan_ref, flex_ref, vals in EXTRA_DEG:
        for v in vals:
            q_goal = q_home.copy()
            q_goal[J_PAN] += math.radians(pan_ref)
            q_goal[J_FLEX] += math.radians(flex_ref)
            q_goal[joint] += math.radians(v)
            tcp_z = float(arm._fk(q_goal)[0][2])
            tag = f"J{joint + 1} {v:+5.0f}°"
            if tcp_z < tcp_home_z - SAFE_MARGIN:
                print(f"  pan {pan_ref:+5.0f}°  flex {flex_ref:+5.0f}°  {tag}  "
                      f"**TCP {tcp_z:.4f} 比初始低，丢**")
                continue
            solved.append({"pan": pan_ref, "flex": flex_ref, "q": q_goal,
                           "tcp_z": tcp_z, "extra": tag})
            print(f"  {len(solved):2d}  pan {pan_ref:+5.0f}°  flex {flex_ref:+5.0f}°  "
                  f"{tag}  TCP z {tcp_z:.4f}")

    print(f"\n收下 {len(solved)} 个姿态，收够 {args.want} 张就停。")
    if not args.go:
        print("没给 --go，一步不动。上面这份没问题再加 --go。")
        ctrl.close()
        return 0

    cap = open_cam(args.index)
    js = OUT / "calib_views.json"
    records: list[dict] = []
    if args.append and js.exists():
        records = json.loads(js.read_text(encoding="utf-8"))
        print(f"续采：已有 {len(records)} 张，图号从 {len(records):03d} 接着编。")
    n0 = len(records)                      # --want 管的是**这一轮新收**几张
    q_prev = arm._ticks_to_qpos(arm.read_ticks())
    try:
        for k, step in enumerate(solved):
            q_goal = step["q"]

            traj = build([q_prev, q_goal], vmax=args.jvel, amax=args.jacc, density=0.01)
            _ts, qs, _ = traj.sample(1.0 / args.hz)
            dt = 1.0 / args.hz
            t0 = time.time()
            for i, q in enumerate(qs):
                arm.write_ticks(arm._qpos_to_ticks(q))
                nap = (i + 1) * dt - (time.time() - t0)
                if nap > 0:
                    time.sleep(nap)
            for _ in range(10):
                arm.write_ticks(arm._qpos_to_ticks(q_goal))
                time.sleep(dt)
            q_prev = q_goal

            time.sleep(args.settle)
            for _ in range(3):
                ok, fr = cap.read()
            if not ok or fr is None:
                print(f"[{k:2d}] 读不出帧，跳过")
                continue

            ch_c, _ch_i, m_c, _m_i = det.detectBoard(fr)
            n_c = 0 if ch_c is None else len(ch_c)
            n_m = 0 if m_c is None else len(m_c)
            if n_c < MIN_CORNERS:
                print(f"[{k:2d}] 跳过：只认出 {n_c} 个角点（{n_m} 个码），不够 {MIN_CORNERS}")
                continue

            ticks = arm.read_ticks()
            q_now = arm._ticks_to_qpos(ticks)
            tcp_now = arm._fk(q_now)[0]
            p = OUT / f"calibview_{len(records):03d}.jpg"
            cv2.imwrite(str(p), fr)
            records.append({
                "file": p.name,
                "pan_deg": step["pan"],
                "flex_deg": step["flex"],
                "extra": step.get("extra", ""),
                "corners": n_c, "markers": n_m,
                "q_deg": [float(v) for v in np.degrees(q_now)],
                "raw_ticks": {str(s): int(v) for s, v in sorted(ticks.items())},
                "tcp_world": [float(v) for v in tcp_now],
            })
            print(f"[{k:2d}] 收了 {p.name}  角点 {n_c:2d}  码 {n_m:2d}  "
                  f"pan {step['pan']:+5.0f}° flex {step['flex']:+5.0f}° "
                  f"{step.get('extra', '')}  TCP {np.round(tcp_now, 4)}")
            if len(records) - n0 >= args.want:
                print(f"\n这一轮收够 {args.want} 张，后面的姿态不走了。")
                break
    except KeyboardInterrupt:
        print("\n中断。臂停在当时位置，**力矩不松**。")
    finally:
        cap.release()

    js.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n这一轮收了 {len(records) - n0} 张，总共 {len(records)} 张，读数在 {js}")
    if len(records) < 6:
        print("**少于 6 张，标定解不稳。** 看看是不是板子出画面了、或者反光太重。")
    ctrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
