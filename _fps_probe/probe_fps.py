"""探针：真机这条路能不能撑起 30fps 采集。**只量时间，不搬臂。**

    G:/anaconda/envs/sim2real/python.exe _fps_probe/probe_fps.py
    G:/anaconda/envs/sim2real/python.exe _fps_probe/probe_fps.py --no-write

量四件事：

1. 现在这条驱动读一次六个舵机要多久（含 `_read_packet` 里每个 ID 固定的
   `time.sleep(0.01)`，见 `mcp_robot/sources/servo_control_standalone.py:412`）。
2. 把那 10ms 关掉之后读一次要多久（用驱动自己的 `read_sleep` 开关，默认 0.01，
   改成 0 就是关）。
3. 写一次位置要多久，同样量 `write_settle` 前后的差（`sync_set_positions` 里写速度
   之后那 20ms）。
4. 相机：`shot()`（去畸变 + 转 RGB）一次多久，抓帧线程真实出帧多少 Hz。

最后跑 5 秒 30Hz 的闭环：每拍「快读 + 抓帧」，量实际节拍、抖动、有没有重复帧。

**不搬臂。** 写测试只把开头读到的、校验过的当前位置写回去；任何一个 ID 的读数跟
开头的差超过 50 tick 就不写。串口打不开（面板还占着）就直接报错退出，不重试。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real.cameras import RealWristCamera                      # noqa: E402
from servo_bridge import open_controller                      # noqa: E402

SERVO_IDS = (1, 2, 3, 4, 5, 6)
OUT = Path(__file__).resolve().parent / "result.json"


def _timed(fn, n: int):
    ts, vals = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        vals.append(fn())
        ts.append(time.perf_counter() - t0)
    return ts, vals


def _stat(ts) -> dict:
    ms = [t * 1000.0 for t in ts]
    return {"n": len(ms), "mean_ms": round(statistics.fmean(ms), 2),
            "min_ms": round(min(ms), 2), "max_ms": round(max(ms), 2),
            "p50_ms": round(statistics.median(ms), 2)}


def _hash(frame: np.ndarray) -> int:
    """整帧太贵，取 1/8 抽样算 crc32。判「是不是同一帧」够用。"""
    return zlib.crc32(np.ascontiguousarray(frame[::8, ::8]).tobytes())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="真机 30fps 能力探针（不动臂）")
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=5.0, help="闭环那一段跑多久")
    ap.add_argument("--no-write", action="store_true", help="只读不写")
    ap.add_argument("--write-loop", action="store_true",
                    help="闭环那一段也带写（写当前位置，模拟自动运行时的总线占用）")
    a = ap.parse_args()

    res: dict = {"port": a.port, "cam_index": a.cam}

    print(f"打开 {a.port} …")
    try:
        ctrl = open_controller(port=a.port)
    except Exception as exc:                                  # noqa: BLE001
        print(f"✗ 串口打不开：{type(exc).__name__}: {exc}")
        print("  面板/技能还占着 COM10？先关掉再跑。")
        return 2
    print("串口打开")

    try:
        # ---- 1. 基线读：现在的驱动 ----
        print("\n[1] 现在这条驱动读一次（20 次）…")
        slow, slow_vals = _timed(lambda: ctrl.sync_read_positions(list(SERVO_IDS)), 20)
        res["read_slow"] = _stat(slow)
        base = ctrl.sync_read_positions(list(SERVO_IDS))
        bad = [k for k, v in base.items() if v is None]
        # 写之前先把「这份读数能不能当目标位置」验一遍：缺项、贴边、20 次之间抖动
        # 超过 20 tick，都说明这个数不能写回去，宁可不测。
        unsteady = []
        for k in SERVO_IDS:
            seen = [v.get(k) for v in slow_vals]
            if any(x is None for x in seen) or \
                    max(seen) - min(seen) > 20 or not (0 < base.get(k, -1) < 4095):
                unsteady.append(k)
        res["write_guard_unsteady"] = unsteady
        if bad:
            print(f"✗ 舵机 {bad} 读不到，后面的写测试全部跳过")
        if unsteady:
            print(f"✗ 舵机 {unsteady} 读数不稳/贴边，写测试全部跳过")
        print(f"   {res['read_slow']}")

        # ---- 2. 关掉固定 sleep 之后（用驱动自己的开关，不再 mock） ----
        rs0, ws0 = ctrl.read_sleep, ctrl.write_settle
        res["read_sleep_default"] = rs0
        res["write_settle_default"] = ws0
        ctrl.read_sleep = 0.0
        print("\n[2] read_sleep=0 之后（20 次）…")
        fast, vals = _timed(lambda: ctrl.sync_read_positions(list(SERVO_IDS)), 20)
        res["read_fast"] = _stat(fast)
        same = all(all(v.get(k) == base.get(k) for k in SERVO_IDS) for v in vals if v)
        res["read_fast_matches_baseline"] = bool(same)
        print(f"   {res['read_fast']}  读回值和基线一致={same}")
        if not same:
            print("   ⚠ 有几次读回跟基线不同——可能是臂被碰了，也可能是收包乱了。")

        # ---- 3. 写：只写开头读到的当前位置 ----
        if not a.no_write and not bad and not unsteady:
            cur = {k: int(v) for k, v in base.items()}
            ctrl.write_settle = ws0          # 先按驱动的默认值量
            print(f"\n[3] 写一次位置（10 次，write_settle={ws0}，写的是当前位置）…")
            slow_w, _ = _timed(lambda: ctrl.sync_set_positions(dict(cur), speed=0), 10)
            res["write_slow"] = _stat(slow_w)
            print(f"   {res['write_slow']}")
            ctrl.write_settle = 0.0
            print("   write_settle=0 之后（10 次）…")
            fast_w, _ = _timed(
                lambda: ctrl.sync_set_positions(dict(cur), speed=0), 10)
            res["write_fast"] = _stat(fast_w)
            print(f"   {res['write_fast']}")
        else:
            print("\n[3] 写测试跳过（--no-write 或读不到）")

        # ---- 4. 相机 ----
        print(f"\n[4] 打开腕相机 index={a.cam} …")
        cam = RealWristCamera(index=a.cam)
        # mount() 这里必然是 None：它要臂（`source`）来算手眼位姿，探针没接臂。
        print(f"   实际尺寸 {cam.size()}")
        shot_ts, frames = _timed(cam.shot, 20)
        res["shot"] = _stat(shot_ts)
        res["cam_size"] = list(cam.size())
        print(f"   shot()（去畸变+转RGB）：{res['shot']}")

        # 抓帧线程真实出帧率：每 2ms 看一次 `_raw` 换了没有。
        t_end = time.time() + 2.0
        last_id, n_new, t_prev = None, 0, None
        dts = []
        while time.time() < t_end:
            with cam._raw_lock:                                     # noqa: SLF001
                raw = cam._raw
            if raw is not None and id(raw) != last_id:
                last_id = id(raw)
                n_new += 1
                now = time.time()
                if t_prev is not None:
                    dts.append(now - t_prev)
                t_prev = now
            time.sleep(0.002)
        res["cam_raw_hz"] = round(n_new / 2.0, 1)
        if dts:
            res["cam_raw_dt_ms"] = {"mean": round(statistics.fmean(dts) * 1000, 2),
                                    "max": round(max(dts) * 1000, 2)}
        print(f"   抓帧线程出帧：{res['cam_raw_hz']} Hz  {res.get('cam_raw_dt_ms')}")

        # ---- 5. 30Hz 闭环：快读 + 抓帧 ----
        write_loop = a.write_loop and not a.no_write and not bad and not unsteady
        cur = {k: int(v) for k, v in base.items()} if write_loop else None
        print(f"\n[5] 30Hz 闭环 {a.seconds:.0f} 秒（快读 + 抓帧，"
              f"{'带写当前位置' if write_loop else '写关闭'}）…")
        period = 1.0 / 30.0
        t0 = time.perf_counter()
        ticks, hashes, read_ms, shot_ms, write_ms = [], [], [], [], []
        k = 0
        while True:
            # **绝对时间网格**：第 k 拍定在 t0 + k*period。按「上一拍起点 + period」
            # 排的话，Windows 的 sleep 每拍多睡约 1ms，误差被每一拍继承下去 ——
            # 实测只会跑到 29.3Hz（改之前的结果）。
            target = t0 + k * period
            now = time.perf_counter()
            if now < target:
                time.sleep(target - now)
            if time.perf_counter() - t0 >= a.seconds:
                break
            k += 1
            tk = time.perf_counter()
            r0 = time.perf_counter()
            w0 = r0
            if write_loop:
                ctrl.sync_set_positions(dict(cur), speed=0)
                w0 = time.perf_counter()
            ctrl.sync_read_positions(list(SERVO_IDS))
            r1 = time.perf_counter()
            f = cam.shot()
            r2 = time.perf_counter()
            write_ms.append((w0 - r0) * 1000)
            read_ms.append((r1 - w0) * 1000)
            shot_ms.append((r2 - r1) * 1000)
            ticks.append(tk - t0)
            hashes.append(_hash(f))
        n = len(ticks)
        dts = np.diff(ticks) * 1000 if n > 1 else np.array([0.0])
        repeat = sum(1 for i in range(1, n) if hashes[i] == hashes[i - 1])
        res["loop"] = {
            "target_hz": 30.0, "ticks": n,
            "actual_hz": round(n / (ticks[-1] - ticks[0]) if n > 1 else 0.0, 1),
            "dt_mean_ms": round(float(dts.mean()), 2),
            "dt_p95_ms": round(float(np.percentile(dts, 95)), 2),
            "dt_max_ms": round(float(dts.max()), 2),
            "write_mean_ms": round(statistics.fmean(write_ms), 2),
            "read_mean_ms": round(statistics.fmean(read_ms), 2),
            "shot_mean_ms": round(statistics.fmean(shot_ms), 2),
            "consecutive_duplicate_frames": int(repeat),
            "unique_frames": int(len(set(hashes))),
        }
        print(f"   {res['loop']}")

        cam.close()
        # 关掉再按 `fast=True` 开一次，确认 `open_controller` 那条接线是通的。
        ctrl.close()
        c2 = open_controller(port=a.port, fast=True)
        res["fast_open_knobs"] = [c2.read_sleep, c2.write_settle]
        c2.close()
        print(f"   open_controller(fast=True) -> read_sleep/write_settle "
              f"{res['fast_open_knobs']}")
    finally:
        try:
            ctrl.close()
        except Exception:                                          # noqa: BLE001
            pass

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
