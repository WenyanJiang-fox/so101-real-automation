"""把 LeRobot v3 数据集里的每一集单独切一段 mp4 出来，方便一集一集看。

**为什么要切。** LeRobot 的 `videos/.../file-XXX.mp4` 是**一个文件装很多集**（同一
`chunk` 里的集接在一条视频流上），每集的起止时间写在
`meta/episodes/chunk-*/file-*.parquet` 的 `from_timestamp` / `to_timestamp` 里。
直接双击 mp4 播，看到的是几趟连放，不是一集。

用法（只读数据集，不碰机械臂、不碰串口）：

    G:/anaconda/envs/sim2real/python.exe export_episode_videos.py
    G:/anaconda/envs/sim2real/python.exe export_episode_videos.py --cam camera2

输出到 `out/episodes/<cam>_ep{集号}_{秒数}.mp4`，原分辨率、原帧率，不缩放不裁剪。
"""
from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import av
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = _ROOT / "datasets" / "so101_real_grasp_2cam_4hz"
DEFAULT_OUT = _ROOT / "out" / "episodes"


def load_episodes(root: Path, cam: str) -> pd.DataFrame:
    """所有 chunk/file 的集元数据拼起来，只留 `cam` 这一路的时间戳。"""
    files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise SystemExit(f"{root} 里没有 meta/episodes/*.parquet")
    cols = {
        f"videos/observation.images.{cam}/file_index": "file_index",
        f"videos/observation.images.{cam}/from_timestamp": "t0",
        f"videos/observation.images.{cam}/to_timestamp": "t1",
    }
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    keep = ["episode_index", "length", *cols]
    return df[keep].rename(columns=cols).sort_values("episode_index")


def cut(src: Path, dst: Path, t0: float, t1: float) -> int:
    """把 `src` 的 [t0, t1) 秒重新编码成 `dst`。返回写了多少帧。"""
    n = 0
    with av.open(str(src)) as cin:
        vs = cin.streams.video[0]
        vs.thread_type = "AUTO"
        fps = int(round(float(vs.average_rate)))
        with av.open(str(dst), "w") as cout:
            os_ = cout.add_stream("libx264", rate=fps)
            os_.pix_fmt = "yuv420p"
            os_.width, os_.height = vs.codec_context.width, vs.codec_context.height
            cin.seek(int(t0 / vs.time_base), stream=vs)
            for frame in cin.decode(vs):
                t = float(frame.pts * frame.time_base)
                if t + 1e-6 < t0:
                    continue
                if t >= t1 - 1e-6:
                    break
                # **时间戳要自己给**：`pts=None` 时编码器按它自己的时基连号写，
                # 75 帧会被压成 0.25 秒。这里按源帧率一帧一格。
                frame.pts = n
                frame.time_base = Fraction(1, fps)
                for pkt in os_.encode(frame):
                    cout.mux(pkt)
                n += 1
            for pkt in os_.encode(None):
                cout.mux(pkt)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="按集切出 mp4")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--cam", default="camera1", help="camera1（腕）/ camera2（侧）")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root, out = Path(a.root), Path(a.out) / a.cam
    out.mkdir(parents=True, exist_ok=True)
    df = load_episodes(root, a.cam)
    print(f"{root}：{len(df)} 集，相机 {a.cam}")
    for r in df.itertuples(index=False):
        src = (root / "videos" / f"observation.images.{a.cam}" /
               f"chunk-{int(r.file_index) // 1000:03d}" /
               f"file-{int(r.file_index) % 1000:03d}.mp4")
        dst = out / f"ep{int(r.episode_index):03d}_{r.t0:.2f}-{r.t1:.2f}s.mp4"
        n = cut(src, dst, float(r.t0), float(r.t1))
        print(f"  第 {int(r.episode_index)} 集：{n} 帧 / {r.t1 - r.t0:.2f}s → {dst}")
    print(f"全部在 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
