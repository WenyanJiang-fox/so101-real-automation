# -*- coding: utf-8 -*-
"""同一个方块、换几个臂姿，两路相机各读一次，对账。

**为什么跑这一趟。** 腕相机反推出来的世界坐标会不会随臂姿态漂，从来没量过；
`configs/real.yaml` 里那个 `percep_offset_mm: [35.8, 0]` 也只在腕转 −80° 附近
的三个姿态上量过（文件里自己写着「还没验过换腕转/换方位后会不会跟着转」）。
侧相机的外参是常数，正好当独立参照。

**两路记的都是原始读数。** 这里把 `geom.REAL_PERCEP_OFFSET` 置成 0，感知不扣那个
35.8；不然两路都被同一个数平移，差看不出东西。

判读（不预设结论，表里自己算）：
  · 侧相机在各姿态之间的散布 = 感知自身的噪声底（方块没动，相机也没动）；
  · 腕相机在各姿态之间的散布 = 随臂姿态的漂移；
  · 每个姿态的 `腕 − 侧` 是不是同一个 [35.8, 0]。

跑：`G:/anaconda/envs/sim2real/python.exe _probe_side_offset.py`
**会动臂。** 先回初始位置，再挪到方块正上方，然后 ±15mm 平移、抬高 25mm、
腕子绕工具轴 ±30°。一次合爪都不做。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np                                               # noqa: E402

from core import geom                                            # noqa: E402
from core.params import SPEED, speed_vmax                        # noqa: E402
from core.registry import make                                   # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg            # noqa: E402

SETTLE_S = 0.6          # 走完等舵机站稳再拍
OFFSETS_MM = [("x+15", (15.0, 0.0, 0.0)), ("x-15", (-15.0, 0.0, 0.0)),
              ("y+15", (0.0, 15.0, 0.0)), ("y-15", (0.0, -15.0, 0.0)),
              ("z+25", (0.0, 0.0, 25.0))]
ROLL_DEG = 30.0         # 绕工具轴转多少度（腕相机跟着滚，方块位置不该变）
ALIGN_Z_MM = 120.0      # 挪到方块上方时的高度（远高于桌面，不碰任何东西）


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_cfg(DEFAULT_CONFIG)
    rig = cfg["real"]
    geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
    geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                                 dtype=np.float64) / 1000.0
    # **这一趟要原始读数**：不扣 35.8。两路同口径才比得出「差」。
    geom.REAL_PERCEP_OFFSET = np.zeros(2, dtype=np.float64)
    geom.OBJECTS = list(cfg.get("objects") or [])

    import cv2                                                   # noqa: E402
    import impls.judge_held                                      # noqa: E402,F401
    import impls.policy_real                                     # noqa: E402,F401
    import real.arm                                               # noqa: E402,F401
    import real.cameras                                           # noqa: E402,F401
    import real.perception                                        # noqa: E402,F401
    import skills.home_real                                       # noqa: E402,F401
    import skills.pick_real                                       # noqa: E402,F401
    from real import joint_move as jm                             # noqa: E402
    from real.perception import draw                              # noqa: E402
    from skills.pick_real.motion import _goto                     # noqa: E402

    arm = make("arm", "real", port=str(rig.get("port", "COM10")),
               speed=int(rig.get("speed", 0)))
    arm.terminated = False
    cams = {"real_wrist": make("camera", "real_wrist", source=arm,
                               index=int(rig.get("cam_index", 1)))}
    cams["real_side"] = make("camera", "real_side", source=arm)
    perc = {n: make("perception", "brown_rect", cams={n: c}, ref=None,
                    objects=geom.object_specs(), cam_name=n)
            for n, c in cams.items()}

    print(f"机械臂 {arm.port if hasattr(arm, 'port') else rig.get('port')}")
    for n, c in cams.items():
        w, h = c.size()
        print(f"  相机 {n:11s} {w}x{h}")
    print(f"感知偏移：{np.round(geom.real_percep_offset() * 1000, 1)}mm（置零，记原始读数）")

    records: list[dict] = []
    ts = int(time.time())

    def read(tag: str) -> dict:
        """当前姿态下两路各读一次。返回记录。"""
        time.sleep(SETTLE_S)
        rec = {"tag": tag,
               "tcp_mm": np.round(np.asarray(arm.tcp()) * 1000.0, 1).tolist(),
               "angles_deg": np.round(np.asarray(arm.angles()), 1).tolist(),
               "cams": {}}
        for name, c in cams.items():
            item: dict = {}
            try:
                bgr = cv2.cvtColor(np.asarray(c.shot()), cv2.COLOR_RGB2BGR)
                center, facts = perc[name].locate(
                    f"pick up the {perc[name].color} "
                    f"{perc[name].spec.get('shape', 'object')}")
                item["found"] = bool(facts.get("found"))
                item["uv"] = None if facts.get("uv") is None else [
                    round(float(v), 1) for v in facts["uv"]]
                if center is not None:
                    item["world_mm"] = [round(float(v) * 1000.0, 1) for v in center]
                    item["yaw_deg"] = facts.get("yaw_deg")
                    item["iou"] = facts.get("iou")
                    item["size"] = facts.get("size")
                else:
                    item["why"] = (facts.get("no_world_why")
                                   or facts.get("why") or "没量出来")
                out = ROOT / "out" / f"side_offset_{ts}_{tag}_{name}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), draw(bgr, facts or {}))
                item["img"] = str(out)
            except Exception as exc:                         # noqa: BLE001
                item["why"] = f"{type(exc).__name__}: {exc}"
            rec["cams"][name] = item

        w = rec["cams"]["real_wrist"].get("world_mm")
        s = rec["cams"]["real_side"].get("world_mm")
        rec["wrist_minus_side_mm"] = (None if (w is None or s is None)
                                      else [round(w[i] - s[i], 1) for i in range(3)])
        records.append(rec)
        show(rec)
        return rec

    def show(rec: dict) -> None:
        print(f"\n[{rec['tag']}]  TCP {rec['tcp_mm']} mm")
        for name in ("real_wrist", "real_side"):
            it = rec["cams"][name]
            short = "腕" if name == "real_wrist" else "侧"
            if "world_mm" in it:
                print(f"   {short}  x={it['world_mm'][0]:6.1f} y={it['world_mm'][1]:6.1f}"
                      f" z={it['world_mm'][2]:6.1f} mm   yaw {it['yaw_deg']:+6.1f}°"
                      f"   iou {it['iou']:.2f}   {it.get('size')}")
            else:
                print(f"   {short}  没量出来：{it.get('why')}")
        d = rec["wrist_minus_side_mm"]
        if d is not None:
            print(f"   差（腕−侧） dx={d[0]:+6.1f} dy={d[1]:+6.1f} dz={d[2]:+6.1f} mm")

    def move_tcp(goal) -> np.ndarray:
        got = _goto(arm, np.asarray(goal, dtype=np.float64),
                    vmax=speed_vmax(SPEED))
        tcp = np.asarray(arm.tcp(), dtype=np.float64)
        print(f"  → TCP ({tcp[0] * 1000:.1f}, {tcp[1] * 1000:.1f}, "
              f"{tcp[2] * 1000:.1f})mm  {got}", flush=True)
        return tcp

    def roll_to(q_base: np.ndarray, deg: float) -> None:
        """**相对 `q_base`** 转 `deg`（不是相对此刻）—— 连续转两次才不会叠。"""
        q = np.asarray(q_base, dtype=np.float64).copy()
        q[4] += np.radians(deg)
        jm.goto_q(arm, q)
        got = np.asarray(arm.angles())
        print(f"  → 腕转 {got[4]:+.1f}°（基准 {np.degrees(q_base[4]):+.1f}°）",
              flush=True)

    try:
        print("\n[1] 回初始位置")
        print("   ", make("skill", "home_real").run(arm, cams), flush=True)
        read("home")

        # 挪到方块正上方：优先用侧相机的 x,y（它不随臂动），退而用配置里的摆放点。
        it = records[-1]["cams"]["real_side"]
        cube = it.get("world_mm")
        if cube is not None:
            target_xy = (cube[0] / 1000.0, cube[1] / 1000.0)
            print(f"\n[2] 挪到方块上方 —— 目标来自侧相机 {cube[:2]}mm")
        else:
            target_xy = (float(geom.REAL_PICK_XY[0]), float(geom.REAL_PICK_XY[1]))
            print(f"\n[2] 挪到方块上方 —— 侧相机没量出来，退回配置摆放点 "
                  f"{np.round(geom.REAL_PICK_XY * 1000, 0)}mm")
        arm.terminated = False
        base = move_tcp([target_xy[0], target_xy[1], ALIGN_Z_MM / 1000.0])
        read("aligned")

        print("\n[3] 平移/抬高")
        for tag, (dx, dy, dz) in OFFSETS_MM:
            arm.terminated = False
            move_tcp(base + np.array([dx, dy, dz]) / 1000.0)
            read(tag)

        print("\n[4] 回到正上方，腕子绕工具轴转")
        arm.terminated = False
        move_tcp(base)
        q_base = arm._ticks_to_qpos(arm.read_ticks())
        for deg, tag in ((+ROLL_DEG, f"roll+{ROLL_DEG:.0f}"),
                         (-ROLL_DEG, f"roll-{ROLL_DEG:.0f}")):
            arm.terminated = False
            roll_to(q_base, deg)
            read(tag)
        arm.terminated = False
        jm.goto_q(arm, q_base)
        print("  → 腕子转回基准", flush=True)
    finally:
        out = ROOT / "out" / f"side_offset_{ts}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"offset_mm": [0.0, 0.0],
                                   "records": records}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\n记录写到 {out}")
        summarize(records)
        for c in cams.values():
            try:
                c.close()
            except Exception as exc:                         # noqa: BLE001
                print("关相机：", exc)
        try:
            arm.close()
        except Exception as exc:                             # noqa: BLE001
            print("关串口：", exc)
    return 0


def summarize(records: list[dict]) -> None:
    """把能算的都算出来：各自散布 + 逐姿态的差。"""
    print("\n===== 汇总 =====")
    for name, short in (("real_wrist", "腕"), ("real_side", "侧")):
        pts = [r["cams"][name]["world_mm"] for r in records
               if r["cams"][name].get("world_mm")]
        if len(pts) < 2:
            print(f"{short}：只有 {len(pts)} 个读数，算不了散布")
            continue
        a = np.asarray(pts, dtype=np.float64)
        span = a.max(axis=0) - a.min(axis=0)
        print(f"{short}：{len(pts)} 个读数   x/y/z 极差 "
              f"{span[0]:.1f}/{span[1]:.1f}/{span[2]:.1f} mm   "
              f"均值 {np.round(a.mean(axis=0), 1)}")
    print("\n逐姿态的「腕−侧」（mm）：")
    print(f"  {'姿态':<12s} {'dx':>7s} {'dy':>7s} {'dz':>7s}")
    for r in records:
        d = r.get("wrist_minus_side_mm")
        if d is None:
            print(f"  {r['tag']:<12s}   ——（有一路没量出来）")
            continue
        print(f"  {r['tag']:<12s} {d[0]:+7.1f} {d[1]:+7.1f} {d[2]:+7.1f}")
    ds = [r["wrist_minus_side_mm"] for r in records
          if r.get("wrist_minus_side_mm")]
    if len(ds) >= 2:
        a = np.asarray(ds, dtype=np.float64)
        print(f"  差值极差 {np.round(a.max(axis=0) - a.min(axis=0), 1)} mm   "
              f"均值 {np.round(a.mean(axis=0), 1)} mm")


if __name__ == "__main__":
    raise SystemExit(main())
