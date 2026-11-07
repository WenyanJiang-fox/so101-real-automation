"""harness 主循环：一轮接一轮地跑，直到采够或触发停止条件。

    G:/anaconda/envs/sim2real/python.exe harness.py --driver expert --episodes 20

现在只验一件事：仿真环境能不能不重启、连续跑几十轮不崩不卡不涨内存。
所以这一版只有「采够条数就停」，组合停止判据（颜色平衡 / 滑窗成功率 / 硬上限）
和 LeRobot 写入都还没接 —— 先把循环本身跑稳。

每轮做的事：挑布局（颜色取欠采样的）→ 换色 → 跑一条 → 判定 → 打一行。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from backend import SimBackend              # noqa: E402
from drivers import ExpertDriver            # noqa: E402
from judge import GroundTruthJudge          # noqa: E402
from layout import COLORS, next_layout      # noqa: E402


def _rss_mb():
    """进程常驻内存。psutil 没装就返回 None，不为了一个监控值卡住主流程。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="SO-101 harness 主循环（仿真）")
    ap.add_argument("--driver", choices=("expert", "policy"), default="expert")
    ap.add_argument("--episodes", type=int, default=20, help="跑多少轮")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--jitter-xy", type=float, default=0.030)
    ap.add_argument("--jitter-yaw", type=float, default=np.pi / 6)
    ap.add_argument("--max-attempts", type=int, default=None,
                    help="兜底上限，默认 episodes*4（规划失败会重试）")
    ap.add_argument("--frames", dest="frames", action="store_true", default=True,
                    help="每帧调一次 observation()，把渲染也压进循环（默认开）")
    ap.add_argument("--no-frames", dest="frames", action="store_false")
    ap.add_argument("--viewer", action="store_true", help="开 MuJoCo 窗口看")
    args = ap.parse_args()

    if args.driver == "policy":
        raise SystemExit("策略驱动器还没写，先用 --driver expert")

    import paths  # noqa: E402  backend 已经把 SIM_DIR 放进 sys.path

    backend = SimBackend()
    driver = ExpertDriver(backend)
    judge = GroundTruthJudge()
    rng = np.random.default_rng(args.seed)

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(backend.env.m, backend.env.d)
        print("已开 viewer")

    max_attempts = args.max_attempts or args.episodes * 4
    counts = {c: 0 for c in COLORS}
    results = []
    n_frames_total = 0
    t_start = time.time()

    print(f"\n开始：driver={driver.name}  目标 {args.episodes} 轮  "
          f"最多试 {max_attempts} 次  抖动 ±{args.jitter_xy*1000:.0f}mm/±{np.degrees(args.jitter_yaw):.0f}°")
    print(f"{'#':<4}{'布局':<34}{'判定':<58}{'帧':>6}{'耗时s':>8}{'RSS MB':>9}")
    print("-" * 122)

    attempt = 0
    saved = 0
    while saved < args.episodes and attempt < max_attempts:
        attempt += 1
        layout = next_layout(counts, rng, (paths.PICK_X, paths.PICK_Y),
                             args.jitter_xy, args.jitter_yaw)
        backend.apply_layout(layout)

        trace = []
        n_steps = [0]
        n_frames = [0]

        def on_frame():
            n_steps[0] += 1
            if n_steps[0] % backend.steps_per_frame == 0:
                n_frames[0] += 1
                if args.frames:
                    backend.observation()  # 渲染但不留图：压测 renderer 复用

        t0 = time.time()
        if driver.run_episode(backend, layout, trace, on_frame=on_frame) is None:
            print(f"{attempt:<4}{str(layout):<34}规划失败，重摆")
            continue

        v = judge.evaluate(backend, layout, trace)
        dt = time.time() - t0
        n_frames_total += n_frames[0]
        counts[v.color] += 1
        results.append(v.success)
        saved += 1

        rss = _rss_mb()
        print(f"{attempt:<4}{str(layout):<34}{v.line():<58}{n_frames[0]:>6}"
              f"{dt:>8.1f}{(f'{rss:8.0f}' if rss else '       -')}")

        if viewer:
            viewer.sync()
            if not viewer.is_running():
                print("viewer 被关了，提前结束")
                break

    if viewer:
        viewer.close()

    wall = time.time() - t_start
    ok = sum(results)
    print("-" * 122)
    print(f"跑了 {saved} 轮 / 尝试 {attempt} 次，总 {wall:.1f}s"
          f"（{wall / max(saved, 1):.1f}s/轮），共 {n_frames_total} 帧")
    print(f"成功率 {ok}/{len(results)}"
          + (f"（{ok / len(results) * 100:.0f}%）" if results else ""))
    print("颜色分布：" + "  ".join(f"{c}={counts[c]}" for c in COLORS))
    rss = _rss_mb()
    if rss:
        print(f"结束时 RSS {rss:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
