# -*- coding: utf-8 -*-
"""量一下 `_build_endseg.py` 的三个阶段各花多少：解码 / add_frame / save_episode。

    python -u _bench_endseg.py [集数]

只读源数据、写到临时目录，跑完删掉。
"""
from __future__ import annotations

import glob
import os
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_2cam_clean2_4hz")
TMP = Path(r"G:\sim2real\so101-real-harness\datasets\_bench_tmp")
CAMS = ["observation.images.camera1", "observation.images.camera2"]

sys.path.insert(0, r"G:\sim2real\so101-real-harness")
from _build_endseg import VideoStream, cut_at                            # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def main() -> int:
    data = pd.concat([pd.read_parquet(f) for f in
                      sorted(glob.glob(str(SRC / "data" / "chunk-*" / "*.parquet")))],
                     ignore_index=True)
    meta = pd.concat([pd.read_parquet(f) for f in
                      sorted(glob.glob(str(SRC / "meta" / "episodes" / "chunk-*" / "*.parquet")))],
                     ignore_index=True).set_index("episode_index")
    info = json.loads((SRC / "meta" / "info.json").read_text(encoding="utf-8"))
    feats = {k: {kk: v[kk] for kk in ("dtype", "shape", "names") if kk in v}
             for k, v in info["features"].items()}
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if TMP.exists():
        shutil.rmtree(TMP)
    ds = LeRobotDataset.create(repo_id="bench/x", fps=int(info["fps"]), root=str(TMP),
                               robot_type="so101", use_videos=True, encoder_threads=4,
                               streaming_encoding=bool(int(os.environ.get("BENCH_STREAM", "0"))),
                               features=feats)
    eps = [e for e in range(20, 20 + N)]
    streams = {}
    t_dec = t_add = t_save = 0.0
    for ep in eps:
        rows = data[data["episode_index"] == ep].reset_index(drop=True)
        grip = np.array([float(np.asarray(a)[5]) for a in rows["observation.state"]])
        start = cut_at(grip)
        n = len(rows) - start
        fi = int(meta.loc[ep, f"videos/{CAMS[0]}/file_index"])
        off = int(round(float(meta.loc[ep, f"videos/{CAMS[0]}/from_timestamp"]) * 4)) + start
        t0 = time.time()
        clips = {}
        for cam in CAMS:
            key = (cam, fi)
            if key not in streams:
                for old in [k for k in streams if k[0] == cam and k[1] != fi]:
                    streams.pop(old).close()
                streams[key] = VideoStream(SRC / "videos" / cam / "chunk-000"
                                           / f"file-{fi:03d}.mp4")
            clips[cam] = streams[key].take(off, n)
        t1 = time.time()
        for i in range(n):
            ds.add_frame({
                "observation.state": np.asarray(rows.loc[start + i, "observation.state"],
                                                dtype=np.float32),
                "action": np.asarray(rows.loc[start + i, "action"], dtype=np.float32),
                "task": "pick up the green box",
                **{cam: clips[cam][i] for cam in CAMS}})
        t2 = time.time()
        ds.save_episode()
        t3 = time.time()
        t_dec += t1 - t0
        t_add += t2 - t1
        t_save += t3 - t2
        print(f"  ep {ep}: {n} 帧  解码 {t1 - t0:.1f}s  add_frame {t2 - t1:.1f}s  "
              f"save {t3 - t2:.1f}s", flush=True)
    ds.finalize()
    print(f"合计 {len(eps)} 集：解码 {t_dec:.1f}s  add_frame {t_add:.1f}s  "
          f"save {t_save:.1f}s")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
