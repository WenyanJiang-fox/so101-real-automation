# -*- coding: utf-8 -*-
"""把数据集里断号的 `episode_index` 接上（268、269 那种空洞）。

**为什么必须接**：lerobot 0.6 的 `LeRobotDatasetMetadata.get_video_file_path()` 拿
`episode_index` 当**行号**去索引 `meta/episodes`，所以 episode_index 必须等于行号。
`pick_grasp_2cam_clean2_4hz` 有 283 条但编号是 0..284（缺 268、269），于是
`IndexError: Episode index 283 out of range. Episodes: 283`；openpi 那边接着回退到
hub 上找这个仓库，服务器连不通 huggingface.co，训练就起不来。

只改 `episode_index` 这一列：`data/chunk-*/*.parquet` 和 `meta/episodes/chunk-*/*.parquet`。
`index` `frame_index` `dataset_from_index/to_index` 和视频时间戳都不动 —— 行还是那些行。
`info.json` 的 `total_episodes` / `total_frames` 按改完的实际数重算，不再手工减。

    python -u _renumber_episodes.py <数据集目录> [--apply]

不给 `--apply` 只打印将要改什么。
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

#: 从 268 之后整体往前挪 2 格（268、269 两条不存在，270 起接上）。
GAP_FROM = 270
SHIFT = 2


def _remap(i: int) -> int:
    return i - SHIFT if i >= GAP_FROM else i


def _fix_file(path: str, apply: bool) -> list[int]:
    t = pq.read_table(path)
    if "episode_index" not in t.column_names:
        return []
    old = [int(i) for i in t.column("episode_index").to_pylist()]
    new = [_remap(i) for i in old]
    if old != new:
        col = pa.array(new, type=t.schema.field("episode_index").type)
        t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index", col)
        if apply:
            tmp = path + ".tmp"
            pq.write_table(t, tmp)
            os.replace(tmp, path)
        print(f"  {path}: {len(old)} 行")
    return new


def main() -> int:
    root = Path(sys.argv[1])
    apply = "--apply" in sys.argv
    data_files = sorted(glob.glob(str(root / "data" / "chunk-*" / "*.parquet")))
    meta_files = sorted(glob.glob(str(root / "meta" / "episodes" / "chunk-*" / "*.parquet")))

    seen: set[int] = set()
    frames = 0
    for p in data_files:
        for e in _fix_file(p, apply):
            seen.add(e)
            frames += 1
    for p in meta_files:
        _fix_file(p, apply)

    info_p = root / "meta" / "info.json"
    info = json.loads(info_p.read_text(encoding="utf-8"))
    print(f"info.json: {info['total_episodes']} 条 / {info['total_frames']} 帧 → "
          f"{len(seen)} 条 / {frames} 帧")
    if apply:
        info["total_episodes"] = len(seen)
        info["total_frames"] = frames
        info_p.write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")
    print(f"{'已改' if apply else '（预演，没写盘）'}：data {len(data_files)} 个，"
          f"meta/episodes {len(meta_files)} 个；episode 编号连续: "
          f"{sorted(seen) == list(range(len(seen)))}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
