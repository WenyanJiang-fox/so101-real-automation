# -*- coding: utf-8 -*-
"""把采集时写坏的两条 episode 从数据里摘出去，用官方 writer 重出一份干净数据集。

坏在哪（实测）：
  - episode 14 的 data parquet 有 154 行，但视频只有 74 帧、meta length=74；
    前 80 行是`清空缓存没删掉`的失败尝试，后 74 行才是真正录到视频的那一版。
  - episode 20 在 data 里剩 87 行，但 meta 和视频里都没有它。

做法：按 episode_index 直接筛 data parquet（**不信任 meta 的行号区间**），
每帧按 (视频文件, 起始帧) 从原视频里取，写进新数据集。只保留 0..19 与 21..49。
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

SRC = Path(os.environ.get(
    "REBUILD_SRC", r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_2cam_4hz"))
OUT = Path(os.environ.get(
    "REBUILD_OUT", r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_2cam_clean_4hz"))
REPO = os.environ.get("REBUILD_REPO", "so101_real/pick_grasp_2cam_clean")
# 原始数据集后来又追加了 episode，`REBUILD_EPISODES` 给总数（默认还是当初那 50 条）。
KEEP = [e for e in range(int(os.environ.get("REBUILD_EPISODES", "50"))) if e != 20]
BAD_EP, BAD_KEEP_TAIL = 14, 74


def load_rows() -> pd.DataFrame:
    fs = sorted(glob.glob(str(SRC / "data" / "chunk-*" / "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def load_meta() -> pd.DataFrame:
    fs = sorted(glob.glob(str(SRC / "meta" / "episodes" / "chunk-*" / "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def ep_rows(data: pd.DataFrame, ep: int) -> pd.DataFrame:
    x = data[data["episode_index"] == ep]
    if ep == BAD_EP:                       # 只留和视频对得上的后 74 行
        x = x.tail(BAD_KEEP_TAIL)
    return x.reset_index(drop=True)


def video_frames(path: Path, start: int, n: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        for i, fr in enumerate(c.decode(st)):
            if i < start:
                continue
            if i >= start + n:
                break
            out.append(fr.to_ndarray(format="rgb24"))
    if len(out) != n:
        raise RuntimeError(f"{path.name} 取到 {len(out)} 帧，期望 {n}（start {start}）")
    return out


def vcol(key: str, name: str) -> str:
    return f"videos/{key}/{name}"


def main() -> int:
    data, meta = load_rows(), load_meta().set_index("episode_index")
    fps = int(json.loads((SRC / "meta" / "info.json").read_text(encoding="utf-8"))["fps"])
    cams = ["observation.images.camera1", "observation.images.camera2"]
    feats = {}
    info = json.loads((SRC / "meta" / "info.json").read_text(encoding="utf-8"))
    for k, v in info["features"].items():
        feats[k] = {kk: v[kk] for kk in ("dtype", "shape", "names") if kk in v}

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if (OUT / "meta" / "info.json").exists():
        raise SystemExit(f"{OUT} 已存在，先删掉再跑")
    ds = LeRobotDataset.create(repo_id=REPO, fps=fps, root=str(OUT), robot_type="so101",
                               use_videos=True, features=feats)
    total = 0
    for new_ep, ep in enumerate(KEEP):
        rows = ep_rows(data, ep)
        n = len(rows)
        clips = {}
        for cam in cams:
            fi = int(meta.loc[ep, vcol(cam, "file_index")])
            frm = float(meta.loc[ep, vcol(cam, "from_timestamp")])
            p = SRC / "videos" / cam / "chunk-000" / f"file-{fi:03d}.mp4"
            clips[cam] = video_frames(p, round(frm * fps), n)
        for i in range(n):
            frame = {
                "observation.state": rows.loc[i, "observation.state"].astype(np.float32),
                "action": rows.loc[i, "action"].astype(np.float32),
                "task": "pick up the green box",
            }
            for cam in cams:
                frame[cam] = clips[cam][i]
            ds.add_frame(frame)
        ds.save_episode()
        total += n
        print(f"ep {ep:02d} -> {new_ep:02d}: {n} 帧（累计 {total}）", flush=True)
    ds.finalize()
    print(f"完成：{len(KEEP)} 条，{total} 帧 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
