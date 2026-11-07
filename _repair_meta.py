# -*- coding: utf-8 -*-
"""把硬停截断的分片摘掉，元数据按还能读的集数重算。

扫描 `data/` 和 `meta/episodes/` 下所有 parquet，读不出来的（没有 parquet 尾）连同
同名视频一起挪到 `_backup/20261022_truncated_dataset/`，然后重写 `info.json` 和
`stats.json`，最后用 `LeRobotDataset.resume` 真开一次确认。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.io_utils import write_stats

ROOT = Path(r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_2cam_clean2_4hz")
BAK = Path(r"G:\sim2real\so101-real-harness\_backup\20261022_truncated_dataset")


def readable(p: Path) -> bool:
    try:
        pq.read_table(p)
        return True
    except Exception:                                              # noqa: BLE001
        return False


BAK.mkdir(parents=True, exist_ok=True)
for p in sorted((ROOT / "data").rglob("*.parquet")) + sorted(
        (ROOT / "meta/episodes").rglob("*.parquet")):
    if readable(p):
        print("OK ", p.relative_to(ROOT))
        continue
    stem = p.stem
    group = [p] + list((ROOT / "videos").glob(f"*/chunk-*/{stem}.mp4"))
    for f in group:
        if f.exists():
            dst = BAK / f.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst))
            print("挪走", f.relative_to(ROOT))

rows: list[dict] = []
for f in sorted((ROOT / "meta/episodes").rglob("*.parquet")):
    if readable(f):
        rows += pq.read_table(f).to_pylist()
print("剩余集数", len(rows))

per_ep: list[dict] = []
for r in rows:
    feats: dict[str, dict] = {}
    for k, v in r.items():
        if k.startswith("stats/"):
            _, feat, stat = k.split("/", 2)
            feats.setdefault(feat, {})[stat] = np.atleast_1d(np.asarray(v, dtype=np.float64))
    per_ep.append(feats)
write_stats(aggregate_stats(per_ep), ROOT)

info = json.loads((ROOT / "meta/info.json").read_text(encoding="utf-8"))
info["total_episodes"] = len(rows)
info["total_frames"] = int(sum(int(r["length"]) for r in rows))
info["splits"] = {"train": f"0:{len(rows)}"}
(ROOT / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False),
                                    encoding="utf-8")
print("info.json：", info["total_episodes"], "集 /", info["total_frames"], "帧")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ds = LeRobotDataset.resume("so101_real/pick_grasp_2cam_clean2", root=str(ROOT))
print("resume OK：len =", len(ds), " episodes =", ds.meta.total_episodes)
