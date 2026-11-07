# -*- coding: utf-8 -*-
"""把原始真机数据集抽稀：**没动的帧剪掉，慢的地方加快**，写成一个新数据集。

    先看效果（不写盘）：
    G:/anaconda/envs/sim2real/python.exe -u postprocess_real_dataset.py --dry-run
    确认后去掉 --dry-run 就会写到 `datasets\\so101_real_grasp_trim`。

**判据全部按「相邻两帧的关节角最大差」（度）**，实测过的分布（650 帧那一集）：
p50 0.53°、p75 0.88°、p90 1.67°、p95 1.81°。所以：

* `--idle-deg 0.3`：两帧之间六个关节谁都没动过 0.3° → **这一帧直接剪掉**（不做锚点）。
* `--min-delta-deg 1.0`：上次留下的帧到现在，攒够 1° 才再留一帧 —— 慢的地方自然抽稀。
* `--max-skip 3`：再怎么慢，隔 3 帧也得留一帧，防止极慢段被拉成幻灯片。

三个数都能调。默认这一组把 650 帧压到 273 帧（42%），85 秒 → 39 秒。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SRC_DEFAULT = r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_960_4hz"
DST_DEFAULT = r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_960_4hz_trim"
REPO_SRC = "so101_real/pick_grasp"
REPO_DST = "so101_real/pick_grasp_trim"

#: 只搬这三列（外加每帧的 task 字符串）。
KEEP_FEATURES = ("observation.state", "action", "observation.images.camera1")


def keep_mask(states: np.ndarray, idle_deg: float, min_delta_deg: float,
              max_skip: int) -> np.ndarray:
    """True = 这一帧留下。

    * 相邻两帧的**最大关节差** < `idle_deg` → 剪掉，不补锚点（「没动的地方」）。
    * 否则累计位移，攒够 `min_delta_deg` 再留一帧（「慢的地方加快」）。
    * 连 `max_skip` 帧都没攒够 → 也留一帧兜底。
    第一帧和最后一帧无条件留（起手和抬升结束）。
    """
    n = int(len(states))
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    keep[0] = True
    acc = 0.0
    skipped = 0
    for i in range(1, n):
        step = float(np.abs(states[i] - states[i - 1]).max())
        if step < idle_deg:
            continue
        acc += step
        skipped += 1
        if acc >= min_delta_deg or skipped >= int(max_skip):
            keep[i] = True
            acc = 0.0
            skipped = 0
    keep[-1] = True
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dst", default=DST_DEFAULT)
    ap.add_argument("--idle-deg", type=float, default=0.3)
    ap.add_argument("--min-delta-deg", type=float, default=1.0)
    ap.add_argument("--max-skip", type=int, default=3)
    ap.add_argument("--crf", type=int, default=23,
                    help="输出视频质量。源是 CRF 30，这里转一次码，给好一点")
    ap.add_argument("--dry-run", action="store_true", help="只算不写")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not (src_root / "meta" / "info.json").exists():
        print(f"✗ 源数据集不存在：{src_root}")
        return 1
    if not args.dry_run and dst_root.exists():
        print(f"✗ 目标已存在，不覆盖：{dst_root}（换一个 --dst）")
        return 1

    src = LeRobotDataset(REPO_SRC, root=str(src_root))
    n_ep = int(src.meta.total_episodes)
    fps = int(src.meta.fps)
    print(f"源：{src_root}  {n_ep} 集 / {src.meta.total_frames} 帧 / fps {fps}")
    print(f"参数：idle {args.idle_deg}°  min-delta {args.min_delta_deg}°  "
          f"max-skip {args.max_skip}" + ("   [dry-run]" if args.dry_run else ""))

    feats = {k: v for k, v in src.meta.features.items() if k in KEEP_FEATURES}
    ds = None
    if not args.dry_run:
        ds = LeRobotDataset.create(
            repo_id=REPO_DST, fps=fps, root=str(dst_root),
            robot_type=src.meta.robot_type, use_videos=True, features=feats,
            rgb_encoder=RGBEncoderConfig(crf=int(args.crf)))

    tot_in = 0
    tot_out = 0
    for ep in range(n_ep):
        ed = LeRobotDataset(REPO_SRC, root=str(src_root), episodes=[ep],
                            return_uint8=True)
        n = len(ed)
        tab = ed.select_columns(["observation.state", "action"])
        states = np.stack([np.asarray(tab[i]["observation.state"],
                                      dtype=np.float32) for i in range(n)])
        keep = keep_mask(states, args.idle_deg, args.min_delta_deg, args.max_skip)
        idx = np.flatnonzero(keep)
        tot_in += n
        tot_out += len(idx)
        print(f"  第 {ep} 集：{n} → {len(idx)} 帧（{len(idx) / n * 100:.0f}%，"
              f"{n / fps:.0f}s → {len(idx) / fps:.0f}s）")
        if ds is None:
            continue
        for i in idx:
            f = ed[i]
            ds.add_frame({
                "observation.state": np.asarray(f["observation.state"],
                                                dtype=np.float32),
                "action": np.asarray(f["action"], dtype=np.float32),
                # `return_uint8=True` 给的是 CHW，写回去要 HWC
                "observation.images.camera1": np.asarray(
                    f["observation.images.camera1"], dtype=np.uint8).transpose(1, 2, 0),
                "task": f["task"],
            })
        ds.save_episode()

    if ds is not None:
        ds.finalize()
    if tot_in:
        print(f"合计：{tot_in} → {tot_out} 帧（{tot_out / tot_in * 100:.0f}%），"
              f"{tot_in / fps:.0f}s → {tot_out / fps:.0f}s")
    if ds is not None:
        print(f"写出：{dst_root}")
    else:
        print("dry-run，没有写盘。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
