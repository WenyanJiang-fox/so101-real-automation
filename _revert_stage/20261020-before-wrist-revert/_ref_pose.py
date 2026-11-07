"""采一个「人手动摆出来的、真夹住了」的姿态当参照。**两步走。**

    G:/anaconda/envs/sim2real/python.exe _ref_pose.py unlock      # 松力矩，你去用手摆
    G:/anaconda/envs/sim2real/python.exe _ref_pose.py lock        # 摆好了，锁住并记录

为什么要这个：`skills/pick_real.py` 的落点算式是「固定爪尖 + side·(半边长 + 爪面探出 +
缝)·开合轴」，那个**缝**现在是照着仿真抄的 1mm，真机上没有对照物 —— 三趟跑下来爪里
都是空的（`out/run_real_25.log` / `out/run_real_26.log`），说不清是缝大了还是别的。
人手摆一个真夹住的样子，就能把这几个数当场量出来。

**`unlock` 松的是舵机的使能位，重力会把臂拽下来 —— 松手前先扶住。** 摆好之后别撒手，
跑 `lock`：它先**读当前位置、把当前位置写回去当目标**，再使能力矩 —— 顺序反了舵机会
朝上一次的目标位置猛地弹过去。

不动任何别的文件，只往 `out/` 里写两张：`ref_pose.jpg`（腕相机整幅原图）、
`ref_pose.json`（关节原始值 + 量出来的几个数）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import real.arm          # noqa: E402,F401  导入即挂上牌子
import real.cameras      # noqa: E402,F401
import real.perception   # noqa: E402,F401

from core.registry import make                    # noqa: E402
from real.arm import RealArm                      # noqa: E402
from servo_bridge import SERVO_TO_JOINT, open_controller  # noqa: E402

SERVO_IDS = sorted(SERVO_TO_JOINT)
CFG = HERE / "configs" / "real.yaml"


def _cfg() -> dict:
    with open(CFG, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _h(v) -> np.ndarray:
    """取水平分量并归一。开合轴只认水平方向（腕子竖直往下时才成立）。"""
    a = np.asarray(v, dtype=np.float64).reshape(3).copy()
    a[2] = 0.0
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _see(arm, cfg: dict, cfg_real: dict, tag: str = "ref_pose"):
    """腕相机看一眼，返回 (盒子中心 or None, facts)。顺手把整幅原图存下来。

    感知反推世界坐标要用「桌面在基座系多高」「起搜中心在哪」这两个现场常数 ——
    跟 `run_sim.py` 建组件之前干的是同一件事，一处不漏地照做。
    """
    from core import geom
    geom.OBJECTS = list(cfg.get("objects") or [])
    geom.REAL_TABLE_Z = float(cfg_real["table_z_mm"]) / 1000.0
    geom.REAL_PICK_XY = np.array([float(v) for v in cfg_real["pick_xy_mm"]]) / 1000.0
    geom.REAL_PERCEP_OFFSET = np.array(
        [float(v) for v in cfg_real.get("percep_offset_mm") or (0.0, 0.0)]) / 1000.0

    cam = make("camera", "real_wrist", source=arm,
               index=int(cfg_real.get("cam_index", 1)))
    try:
        percep = make("perception", "brown_rect", cams={"real_wrist": cam},
                      ref=None, objects=geom.object_specs())
        frame = cam.shot()
        import cv2
        out = HERE / "out"
        out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / f"{tag}.jpg"),
                    cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        center, facts = percep.locate("pick up the red box")
        return center, facts
    finally:
        cam.close()


def cmd_look(cfg: dict) -> int:
    """只看一眼、只报数，一个字节都不写舵机。"""
    cfg_real = cfg.get("real") or {}
    c = open_controller(cfg_real.get("port", "COM10"))
    arm = RealArm(ctrl=c, speed=int(cfg_real.get("speed", 0)))
    try:
        tip = np.asarray(arm.tcp(), dtype=np.float64) * 1000.0
        center, facts = _see(arm, cfg, cfg_real)
        print(f"固定爪尖现在在 {np.round(tip, 1)}mm")
        if center is None:
            print(f"盒子没量出来：{facts.get('no_world_why') or facts.get('why')}")
            return 1
        print(f"感知说盒子中心 {np.round(np.asarray(center) * 1000, 1)}mm"
              f"  yaw {facts.get('yaw_deg')}°  重合度 {facts.get('iou')}")
        print(f"  爪尖离它 {np.linalg.norm(center - arm.tcp()) * 1000:.1f}mm")
        return 0
    finally:
        c.close()


def cmd_unlock(cfg: dict) -> int:
    port = (cfg.get("real") or {}).get("port", "COM10")
    c = open_controller(port)
    try:
        c.sync_disable_torque(SERVO_IDS)
        print(f"串口 {port} 已开，6 个舵机的力矩**已松开**。")
        print("松手前先扶住 —— 重力会立刻把臂拽下去。")
        print("摆到「两爪真的夹住盒子」那个姿态，别撒手，回来跑：")
        print("  G:/anaconda/envs/sim2real/python.exe _ref_pose.py lock")
        time.sleep(0.5)
    finally:
        c.close()
    return 0


def cmd_lock(cfg: dict) -> int:
    cfg_real = cfg.get("real") or {}
    port = cfg_real.get("port", "COM10")
    c = open_controller(port)
    arm = RealArm(ctrl=c, speed=int(cfg_real.get("speed", 0)))
    cam = None
    try:
        # ---- 先把「人摆出来的姿态」读下来，再锁 ----
        arm._dirty = True
        ticks = {int(k): int(v) for k, v in arm.read_ticks().items()}
        q = arm._ticks_to_qpos(ticks)
        print("读到的原始位置：")
        for sid in SERVO_IDS:
            print(f"    {sid} {SERVO_TO_JOINT[sid]:<14} {ticks[sid]:5d}"
                  f"   关节 {np.degrees(q[sid - 1]):7.1f}°")
        print(f"    TCP(固定爪尖) {np.round(arm.tcp() * 1000, 1)}mm")

        # **顺序不能反**：先把当前位置写成目标位置，再使能力矩。反过来舵机会朝上一次
        # 留下的旧目标弹过去（那个目标还停在上一轮跑完的地方）。
        arm.write_ticks(ticks)
        c.sync_enable_torque(SERVO_IDS)
        time.sleep(0.6)
        arm._dirty = True
        ticks2 = {int(k): int(v) for k, v in arm.read_ticks().items()}
        moved = max(abs(ticks2[s] - ticks[s]) for s in SERVO_IDS)
        print(f"力矩已使能。锁住前后最大差 {moved} 个刻度"
              f"（{'没动，锁在了原处' if moved <= 8 else '！！动了，当心'}）")

        # ---- 量：固定爪尖 → 盒子中心，沿开合轴分解 ----
        xt = _h(arm.tool_x())
        f_tip = np.asarray(arm.tcp(), dtype=np.float64)
        gap = np.asarray(arm.jaw_gap_center(), dtype=np.float64)
        side = 1.0 if float(np.dot(f_tip - gap, xt)) >= 0.0 else -1.0
        f2, m2 = arm.tips(q)
        m_axis = float(np.dot(np.asarray(m2) - np.asarray(f2), xt)) * 1000.0

        center, facts = _see(arm, cfg, cfg_real, tag="ref_pose")

        rec = {"ticks": ticks, "joints_deg": [round(float(v), 2)
                                              for v in np.degrees(q)],
               "tcp_mm": [round(float(v), 2) for v in f_tip * 1000.0],
               "side": side, "moving_tip_axis_mm": round(m_axis, 2),
               "tool_x": [round(float(v), 4) for v in xt],
               "found": bool(center is not None),
               "facts_why": facts.get("no_world_why") or facts.get("why")}

        print(f"\n开合轴（水平）{np.round(xt, 3)}  固定爪在它 {'+' if side > 0 else '-'} 侧")
        print(f"动爪尖沿开合轴离固定爪尖 {m_axis:.1f}mm")
        if center is None:
            print(f"盒子的世界坐标这次没量出来：{rec['facts_why']}\n"
                  f"（画面存了 {out / 'ref_pose.jpg'}，你自己能看见爪子和盒子在哪）")
        else:
            center = np.asarray(center, dtype=np.float64)
            d = (center - f_tip) * 1000.0
            ax = float(np.dot(d, xt))
            rest = d - ax * xt
            pad = float(arm.fixed_face_reach(
                xt, side, (float(center[2]) - 0.01, float(center[2]) + 0.01)))
            rec.update({"center_mm": [round(float(v), 2) for v in center * 1000.0],
                        "along_axis_mm": round(ax, 2),
                        "rest_mm": [round(float(v), 2) for v in rest],
                        "pad_mm": round(pad * 1000.0, 2)})
            print(f"量出来：盒子中心 {np.round(center * 1000, 1)}mm"
                  f"  yaw {facts.get('yaw_deg')}°  重合度 {facts.get('iou')}")
            print(f"  固定爪尖 → 盒子中心  沿开合轴 {ax:+.1f}mm"
                  f"   侧向 {np.linalg.norm(rest[:2]):.1f}mm   竖直 {rest[2]:+.1f}mm")
            print(f"  爪面比爪尖还往盒子里探 {pad * 1000:.1f}mm")
            print(f"  【要的数】落点算式里那个「缝」应该让盒子中心离爪尖 {ax:.1f}mm ——"
                  f" 现在代码按 半边长 + {pad * 1000:.1f} + 1.0 算")

        (out / "ref_pose.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n记录：{out / 'ref_pose.json'}\n画面：{out / 'ref_pose.jpg'}")
        return 0
    finally:
        if cam is not None:
            cam.close()
        c.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="采一个手动夹住的姿态当参照")
    ap.add_argument("step", choices=["unlock", "lock", "look"])
    args = ap.parse_args()
    cfg = _cfg()
    if args.step == "unlock":
        return cmd_unlock(cfg)
    return cmd_lock(cfg) if args.step == "lock" else cmd_look(cfg)


if __name__ == "__main__":
    sys.exit(main())
