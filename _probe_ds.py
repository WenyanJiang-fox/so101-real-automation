# -*- coding: utf-8 -*-
"""只读：把远端数据集的实际内容和 info.json 对一遍，看缺哪一块。

    /root/miniconda3/envs/pi05/bin/python -u /root/autodl-tmp/_probe_ds.py
"""
from __future__ import annotations

import glob
import json
import os

import pyarrow.parquet as pq

ROOT = "/root/autodl-tmp/hf_lerobot/so101_real/pick_grasp_2cam_clean2_4hz"


def main() -> int:
    info = json.load(open(f"{ROOT}/meta/info.json", encoding="utf-8"))
    print("info:", info["total_episodes"], "episodes,", info["total_frames"],
          "frames,", info["fps"], "fps,", info["chunks_size"], "chunks_size")

    data_files = sorted(glob.glob(f"{ROOT}/data/chunk-*/*.parquet"))
    eps: set[int] = set()
    rows = 0
    for f in data_files:
        t = pq.read_table(f, columns=["episode_index"])
        e = t.column("episode_index").to_pylist()
        eps.update(e)
        rows += len(e)
    print("data:", len(data_files), "files,", len(eps), "unique episodes,", rows, "rows")
    miss = sorted(set(range(int(info["total_episodes"]))) - eps)
    print("data 里缺的 episode:", miss[:20], "(共", len(miss), ")")

    meta_rows = 0
    for f in sorted(glob.glob(f"{ROOT}/meta/episodes/chunk-*/*.parquet")):
        meta_rows += len(pq.read_table(f, columns=["episode_index"]).column("episode_index"))
    print("meta/episodes:", meta_rows, "行")

    for key in info["features"]:
        if info["features"][key]["dtype"].startswith("video"):
            vids = glob.glob(f"{ROOT}/videos/{key}/chunk-*/*.mp4")
            print(f"videos {key}: {len(vids)} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
