# -*- coding: utf-8 -*-
"""松力矩让人用手掰 / 掰完开回力矩并**原地固定**，顺手做指尖对账。

    G:/anaconda/envs/sim2real/python.exe _free_and_fix.py free
        先把**当前位置**写成舵机目标，再松力矩 —— 这样掰完开回力矩时，
        目标就是掰完那一刻的位置，臂不会弹回旧目标。

    G:/anaconda/envs/sim2real/python.exe _free_and_fix.py fix
        读当前位置 → 写成目标 → 开回力矩 → 复读确认没跳 → 报
        「两指尖中点」「腕相机读方块」「侧相机读方块」和两个差值。

    G:/anaconda/envs/sim2real/python.exe _free_and_fix.py live
        松力矩 + 开一个侧相机窗口，实时画「指尖中点 → 方块中心」那根线。
        摆到线几乎没有再按 ESC（或 q）。**退出时力矩仍然是松的**，
        接着跑 `fix` 才开回力矩。

**松力矩之后臂会自己往下塌（重力）。** 掰的时候扶住。
`free` 结束会关串口；力矩位是舵机里记着的，关串口不会把它开回来。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np                                               # noqa: E402

from core import geom                                            # noqa: E402
from core.registry import make                                   # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg            # noqa: E402
from real.arm import SERVO_IDS                                   # noqa: E402


def _arm():
    cfg = load_cfg(DEFAULT_CONFIG)
    rig = cfg["real"]
    geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
    geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                                 dtype=np.float64) / 1000.0
    geom.REAL_PERCEP_OFFSET = np.array(
        [float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
        dtype=np.float64) / 1000.0
    geom.OBJECTS = list(cfg.get("objects") or [])
    import real.arm as _ra                                        # noqa: F401,PLC0415

    arm = make("arm", "real", port=str(rig.get("port", "COM10")),
               speed=int(rig.get("speed", 0)))
    return cfg, rig, arm


def cmd_free() -> int:
    _cfg, _rig, arm = _arm()
    try:
        cur = arm.read_ticks()
        print(f"掰之前 关节(度) {np.round(arm.angles(), 1)}")
        print(f"   原始位置 {dict(sorted(cur.items()))}")
        # **先把当前位置写成目标**：舵机的位置控制目标是最后一次写进去的值。
        # 不先写的话，力矩一开回来它会直奔那个旧目标（回初始位置那组数）。
        arm.write_ticks(cur)
        ok = arm.ctrl.sync_disable_torque(list(SERVO_IDS))
        print(f"力矩：已松（{ok}）—— 可以掰了。掰的时候扶住，它会往下塌。")
    finally:
        arm.close()
    return 0


def _lock(arm) -> None:
    """开回力矩、**原地锁死**：先把当前位置写成目标，再开力矩。

    舵机的位置目标是最后一次写进去的值 —— 不先写就开力矩，臂会直奔旧目标。
    开回来停 1 秒再复读，把「有没有跳」的量报出来。
    """
    cur = arm.read_ticks()
    q_before = np.asarray(arm.angles(), dtype=np.float64)
    arm.write_ticks(cur)
    ok = arm.ctrl.sync_enable_torque(list(SERVO_IDS))
    time.sleep(1.0)
    q_after = np.asarray(arm.angles(), dtype=np.float64)
    jump = float(np.max(np.abs(q_after - q_before)))
    print(f"力矩：已开（{ok}）   开回后最大关节变化 {jump:.2f}°")
    print(f"关节(度) {np.round(q_after, 1)}")
    print(f"TCP {np.round(arm.tcp() * 1000, 1)} mm")


def _measure(arm, cams: dict) -> None:
    """量「两指尖中点」和每一路相机的方块读数，打印差值并存帧。

    `cams` 是已经建好的相机对象（`free`/`live` 两条路共用这一份量法）。
    """
    import cv2                                                    # noqa: PLC0415
    from real.perception import draw                              # noqa: PLC0415
    import real.cameras                                           # noqa: F401,PLC0415
    import real.perception                                        # noqa: F401,PLC0415

    tips = [np.asarray(t, dtype=np.float64) for t in arm.tips()]
    mid = np.asarray(arm.jaw_gap_center(), dtype=np.float64)
    grip_deg = float(np.degrees(arm.grip_q()))
    print(f"\n两指尖中点 ({mid[0] * 1000:+.1f}, {mid[1] * 1000:+.1f}) mm"
          f"   两尖相距 {np.linalg.norm(tips[1] - tips[0]) * 1000:.1f}mm"
          f"   爪角 {grip_deg:+.1f}°")
    print("  （侧 − 指尖）为正 = 侧相机认的方块在指尖的 +x/+y 一侧")

    off = np.asarray(geom.real_percep_offset(), dtype=np.float64)
    for name, short in (("real_wrist", "腕"), ("real_side", "侧")):
        if name not in cams:
            print(f"\n{short}相机：没建起来")
            continue
        # **这一份是诊断用的，阈値比线上松。** 指尖贴着方块时两路的剪影都只剩
        # 一条边，重合度会被压到 0.4~0.5；线上那个 0.5 是抓取时的门槛，这里要的
        # 是「差多少」，不是「敢不敢抓」。
        percep = make("perception", "brown_rect", cams={name: cams[name]},
                      ref=None, objects=geom.object_specs(), cam_name=name,
                      min_iou=float(os.environ.get("CHECK_MIN_IOU", 0.3)))
        center, facts = percep.locate(percep.color)
        if center is None:
            print(f"\n{short}相机：量不出来 —— "
                  f"{facts.get('no_world_why') or facts.get('why')}")
            try:
                bgr = cv2.cvtColor(np.asarray(cams[name].shot()),
                                   cv2.COLOR_RGB2BGR)
                out = ROOT / "out" / f"freefix_{name}_{int(time.time())}.png"
                cv2.imwrite(str(out), draw(bgr, facts or {}))
                print(f"  那一帧：{out}")
            except Exception as exc:                          # noqa: BLE001
                print(f"  ！那一帧没存下来：{type(exc).__name__}: {exc}")
            continue
        raw = np.asarray(center, dtype=np.float64).copy()
        raw[:2] += off                      # 加回被扣掉的 percep_offset
        d = (raw[:2] - mid[:2]) * 1000.0
        print(f"\n{short}相机读方块 ({raw[0] * 1000:+.1f}, {raw[1] * 1000:+.1f}) mm"
              f"   形状 {facts.get('size')}   重合度 {facts.get('iou')}")
        print(f"  差（{short} − 指尖） dx={d[0]:+.1f} dy={d[1]:+.1f} mm")
        try:
            bgr = cv2.cvtColor(np.asarray(cams[name].shot()),
                               cv2.COLOR_RGB2BGR)
            out = ROOT / "out" / f"freefix_{name}_{int(time.time())}.png"
            cv2.imwrite(str(out), draw(bgr, facts or {}))
            print(f"  那一帧：{out}")
        except Exception as exc:                              # noqa: BLE001
            print(f"  ！那一帧没存下来：{type(exc).__name__}: {exc}")


def cmd_live() -> int:
    """松力矩，开一个侧相机实时窗口，画「指尖中点 → 方块中心」那根线。

    **这根线是相机自己算的**：`arm.jaw_gap_center()` 给指尖中点（正解，世界系），
    `percep.locate()` 给方块中心（世界系），两个都投到侧相机画面上，画出来。
    线短 = 两指尖在方块正上方。

    **一退出窗口就开回力矩、原地锁死，然后当场量两路。** 不用再跑 `fix`。
    """
    import cv2                                                    # noqa: PLC0415
    from core.vision import project                               # noqa: PLC0415
    from real.perception import draw                              # noqa: PLC0415
    import real.cameras                                           # noqa: F401,PLC0415
    import real.perception                                        # noqa: F401,PLC0415

    cfg, rig, arm = _arm()
    cams: dict = {}
    try:
        # 腕相机现在就建好：退出窗口后要当场量，不再多等一次开设备。
        cams["real_wrist"] = make("camera", "real_wrist", source=arm,
                                  index=int(rig.get("cam_index", 1)))
        cams["real_side"] = make("camera", "real_side", source=arm)
        cam = cams["real_side"]
        percep = make("perception", "brown_rect", cams={"real_side": cam},
                      ref=None, objects=geom.object_specs(), cam_name="real_side",
                      min_iou=float(os.environ.get("CHECK_MIN_IOU", 0.3)))
        cur = arm.read_ticks()
        arm.write_ticks(cur)          # 先写目标，等会儿开回力矩不会弹
        ok = arm.ctrl.sync_disable_torque(list(SERVO_IDS))
        print(f"力矩：已松（{ok}）—— 可以掰了。扶住，它会往下塌。")
        print("窗里品红色 ✕ = 两指尖中点，品红线连到方块中心；线越短越对正。")
        print("摆好按 ESC 或 q 退出 —— 退出即开回力矩锁死并自动量两路。")

        off = np.asarray(geom.real_percep_offset(), dtype=np.float64)
        win = "side: tip-mid -> cube   ESC to lock & measure"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 540, 960)       # 1080x1920 原图，按 125% 缩放会撑爆屏
        n = 0
        try:
            while True:
                center, facts = percep.locate(percep.color)
                bgr = cv2.cvtColor(np.asarray(cam.shot()), cv2.COLOR_RGB2BGR)
                img = draw(bgr, facts or {})
                mid = np.asarray(arm.jaw_gap_center(), dtype=np.float64)
                mt = cam.mount()
                pm = (project(mt["pos"], mt["mat"], mt["k"], mid)
                      if mt is not None else None)
                if facts.get("found") and pm is not None:
                    u0, v0 = int(round(pm[0])), int(round(pm[1]))
                    cv2.drawMarker(img, (u0, v0), (255, 0, 255),
                                   cv2.MARKER_TILTED_CROSS, 24, 2)
                    uc, vc = facts["rect"][0]
                    uc, vc = int(round(uc)), int(round(vc))
                    cv2.line(img, (u0, v0), (uc, vc), (255, 0, 255), 2)
                    if center is not None:
                        raw = np.asarray(center, dtype=np.float64).copy()
                        raw[:2] += off          # 加回被扣掉的 percep_offset
                        d = (raw[:2] - mid[:2]) * 1000.0
                        cv2.putText(img,
                                    f"tip-mid -> cube  dx={d[0]:+.1f} "
                                    f"dy={d[1]:+.1f} mm",
                                    (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (255, 0, 255), 1, cv2.LINE_AA)
                        if n % 12 == 0:
                            print(f"  dx={d[0]:+.1f} dy={d[1]:+.1f} mm"
                                  f"   iou {facts.get('iou')}")
                        n += 1
                else:
                    cv2.putText(img, "cube not measured", (8, 64),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1,
                                cv2.LINE_AA)
                cv2.imshow(win, img)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
        # **退出窗口那一刻就把臂锁住**，然后当场量两路。
        print("窗口已关 —— 开回力矩，原地锁死")
        _lock(arm)
        _measure(arm, cams)
    finally:
        for c in cams.values():
            try:
                c.close()
            except Exception:                                     # noqa: BLE001
                pass
        arm.close()
    return 0


def cmd_fix() -> int:
    import real.cameras                                           # noqa: F401,PLC0415
    import real.perception                                        # noqa: F401,PLC0415

    cfg, rig, arm = _arm()
    cams: dict = {}
    try:
        _lock(arm)
        for name in ("real_wrist", "real_side"):
            kw = ({"index": int(rig.get("cam_index", 1))}
                  if name == "real_wrist" else {})
            cams[name] = make("camera", name, source=arm, **kw)
        _measure(arm, cams)
    finally:
        for c in cams.values():
            try:
                c.close()
            except Exception:                                 # noqa: BLE001
                pass
        arm.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    what = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if what == "free":
        raise SystemExit(cmd_free())
    if what == "fix":
        raise SystemExit(cmd_fix())
    if what == "live":
        raise SystemExit(cmd_live())
    print("用法：_free_and_fix.py free | fix | live")
    raise SystemExit(2)
