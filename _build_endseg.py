# -*- coding: utf-8 -*-
"""把 262 集抓取数据砍成「末端一段」，专门用来微调 VLA 的下探 + 合爪。

    G:/anaconda/envs/sim2real/python.exe -u _build_endseg.py

切点在 6 号舵机：`i15` = 这一集第一次读到 gripper > 15° 的那一帧（脚本在下探
一半处张爪，`skill.py::_descend(open_at=0.5)`）。从 `i15 - 2` 切到集尾。

实测（262 集）：i15 中位第 51 帧、集长中位 82 帧 → 末端每集中位 28 帧。
**动作分布和推理时完全对得上**：模型接手的那一刻，爪刚张开、还在往下走。

**丢掉 ep 0~19**（2026-10-23 实测）：那 20 集的 6 号舵机张到 **99~100°**
（`state` 和 `action` 都是），而 20 集之后张到 **19.5°** —— 两套张开值混在一个
数据集里，末段微调会把「张开」学成两个答案。现役技能的上限是 20°，留后者。

**视频只解一遍。** 一集只取 30 帧，但每集的起点在 mp4 中间；按集去开文件会把
「从头解到起点」重复算一遍（实测 40 秒/集）。这里按文件分组，两个相机各开一路
顺序解码器，一集一集往后推 —— 每个 mp4 只解一遍。
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import pandas as pd

SRC = Path(os.environ.get(
    "ENDSEG_SRC",
    r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_2cam_clean2_4hz"))
OUT = Path(os.environ.get(
    "ENDSEG_OUT",
    r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_2cam_clean2_end_4hz"))
REPO = os.environ.get("ENDSEG_REPO", "so101_real/pick_grasp_2cam_end")
LEAD = 2                 # 张爪之前留两帧（合爪前的那个状态）
#: 张开值超过这个数的集整集丢掉：那是旧标定（张到 99°），和现在 20° 的上限打架。
OPEN_CAP = 30.0
CAM_KEYS = ["observation.images.camera1", "observation.images.camera2"]


class VideoStream:
    """一个 mp4 的顺序解码器。`take(start, n)` 只会往前走，不会回头。"""

    def __init__(self, path: Path):
        self.path = path
        self._c = av.open(str(path))
        self._it = self._c.decode(self._c.streams.video[0])
        self._i = -1

    def take(self, start: int, n: int) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for fr in self._it:
            self._i += 1
            if self._i < start:
                continue
            out.append(fr.to_ndarray(format="rgb24"))
            if len(out) >= n:
                return out
        raise RuntimeError(f"{self.path.name} 到文件尾只取到 {len(out)} 帧，"
                           f"期望 {n}（start {start}）")

    def close(self) -> None:
        self._c.close()


def load_rows() -> pd.DataFrame:
    fs = sorted(glob.glob(str(SRC / "data" / "chunk-*" / "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def load_meta() -> pd.DataFrame:
    fs = sorted(glob.glob(str(SRC / "meta" / "episodes" / "chunk-*" / "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).set_index(
        "episode_index")


def cut_at(gripper: np.ndarray) -> int:
    """第一次张开（>15°）之前两帧。找不到就退到「第一帧 >10°」，再没有就 0。"""
    for thr in (15.0, 10.0, 0.0):
        hit = np.where(gripper > thr)[0]
        if len(hit):
            return max(0, int(hit[0]) - LEAD)
    return 0


def main() -> int:
    data, meta = load_rows(), load_meta()
    info = json.loads((SRC / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    feats = {k: {kk: v[kk] for kk in ("dtype", "shape", "names") if kk in v}
             for k, v in info["features"].items()}

    # ---- 先把每集的切点和文件名算出来 ----
    groups: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for ep in sorted(data["episode_index"].unique()):
        rows = data[data["episode_index"] == ep].reset_index(drop=True)
        grip = np.array([float(np.asarray(a)[5]) for a in rows["observation.state"]])
        if float(grip.max()) > OPEN_CAP:
            print(f"  [跳过] ep {ep}: 张开值 {grip.max():.1f}° 是旧标定", flush=True)
            continue
        start = cut_at(grip)
        n = len(rows) - start
        if n <= 2:
            print(f"  [跳过] ep {ep}: 切完只剩 {n} 帧", flush=True)
            continue
        fi = int(meta.loc[ep, f"videos/{CAM_KEYS[0]}/file_index"])
        off = int(round(float(meta.loc[ep, f"videos/{CAM_KEYS[0]}/from_timestamp"])
                        * fps))
        groups[fi].append((ep, off + start, n, start))
    for fi in groups:
        groups[fi].sort(key=lambda t: t[1])
    print(f"要搬 {sum(len(v) for v in groups.values())} 集，"
          f"分在 {len(groups)} 个视频文件里", flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if (OUT / "meta" / "info.json").exists():
        raise SystemExit(f"{OUT} 已存在，先删掉再跑")
    # **`streaming_encoding=True` 是这一版能跑完的关键**（2026-10-23 实测）。
    # 默认那条路是「每帧先写成 png，`save_episode` 再把 png 读回来编码」，本机实测
    # **22 秒/集**（29 帧 × 2 路），242 集要一个半小时；改成流式直喂编码器之后
    # **1.3 秒/集**。编码参数（libsvtav1 / crf 30 / preset 12）一个字没动。
    ds = LeRobotDataset.create(repo_id=REPO, fps=fps, root=str(OUT),
                               robot_type="so101", use_videos=True,
                               streaming_encoding=True,
                               encoder_threads=int(os.environ.get("ENDSEG_ENC_THREADS", "4")),
                               features=feats)

    total, kept = 0, 0
    for fi in sorted(groups):
        streams = {}
        try:
            for cam in CAM_KEYS:
                p = SRC / "videos" / cam / "chunk-000" / f"file-{fi:03d}.mp4"
                streams[cam] = VideoStream(p)
            for ep, off, n, _start in groups[fi]:
                clips = {cam: streams[cam].take(off, n) for cam in CAM_KEYS}
                rows = data[data["episode_index"] == ep].reset_index(drop=True)
                s0 = len(rows) - n
                for i in range(n):
                    frame = {
                        "observation.state": np.asarray(
                            rows.loc[s0 + i, "observation.state"], dtype=np.float32),
                        "action": np.asarray(rows.loc[s0 + i, "action"],
                                             dtype=np.float32),
                        "task": "pick up the green box",
                    }
                    for cam in CAM_KEYS:
                        frame[cam] = clips[cam][i]
                    ds.add_frame(frame)
                del clips
                ds.save_episode()
                total += n
                kept += 1
                if kept % 20 == 0:
                    print(f"  … {kept} 集 / {total} 帧（ep {ep}，本集 {n}）", flush=True)
        finally:
            for s in streams.values():
                s.close()
        print(f"  文件 file-{fi:03d} 走完（累计 {kept} 集 / {total} 帧）", flush=True)

    ds.finalize()
    print(f"完成：{kept} 集，{total} 帧 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
