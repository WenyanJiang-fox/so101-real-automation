# -*- coding: utf-8 -*-
"""在数据集的**副本**上修 metadata，再逐条核对：

1. meta/episodes 里 episode_index 21..49 顺次前移一位（原 20 号整条丢了）；
2. info.json 的 total_episodes / total_frames 跟着改；
3. 用 lerobot 0.6.1 读回来，逐条比 `item['episode_index']` 和 `frame_index`。

只核对，不写新数据集。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

WORK = Path(r"G:\sim2real\so101-real-harness\datasets\_clean_src")


def patch_meta() -> None:
    info_p = WORK / "meta" / "info.json"
    info = json.loads(info_p.read_text(encoding="utf-8"))
    files = sorted((WORK / "meta" / "episodes").glob("chunk-*/*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    old = df["episode_index"].tolist()
    new = list(range(len(df)))
    df["episode_index"] = new
    df.to_parquet(files[0], index=False)
    for f in files[1:]:
        f.unlink()
    info["total_episodes"] = len(df)
    info["total_frames"] = int(df["length"].sum())
    info_p.write_text(json.dumps(info, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"episodes {len(old)}，重编号 {old[:3]}…{old[-3:]} -> {new[:3]}…{new[-3:]}")
    print("total_frames", info["total_frames"])


def check() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("so101_real/pick_grasp_2cam", root=str(WORK))
    print("loaded", ds.num_episodes, "episodes,", len(ds), "frames")
    bad = 0
    for ep in range(ds.num_episodes):
        fi = ds.meta.episodes["dataset_from_index"][ep]
        to = ds.meta.episodes["dataset_to_index"][ep]
        for pos in {int(fi), int(to) - 1}:
            it = ds[pos]
            got_ep = int(it["episode_index"])
            got_f = int(it["frame_index"])
            want_f = 0 if pos == fi else int(to) - 1 - int(fi)
            if got_ep != ep or got_f != want_f:
                bad += 1
                if bad <= 5:
                    print(f"  ✗ ep{ep} pos{pos}: episode_index={got_ep} frame_index={got_f}"
                          f"（应为 {ep}/{want_f}）")
    print("mismatch", bad)
    return bad


if __name__ == "__main__":
    patch_meta()
    sys.exit(1 if check() else 0)
