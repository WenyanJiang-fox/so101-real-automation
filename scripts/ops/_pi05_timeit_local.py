# -*- coding: utf-8 -*-
"""在本机走 SSH 隧道掐一次推理时间，载荷和真客户一样是 JPEG q85。

    G:/anaconda/envs/sim2real/python.exe -u _pi05_timeit_local.py

和服务器本机的 `_pi05_timeit.py` 用同一组帧（`out/pi05_try3/0090_*`），
差别只有隧道这一段。目的：确认 JPEG 换上来以后慢的是不是只剩链路。
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from _pi05_link import connect

BASE = "out/pi05_try3"
WRIST = f"{BASE}/0090_wrist.jpg"
SIDE = f"{BASE}/0090_side.jpg"
STATE = np.array([-25.2, -10.0, 8.9, 94.3, -128.8, 1.9], dtype=np.float32)
HOST, PORT = "127.0.0.1", 8000


def _jpg(a: np.ndarray, q: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.asarray(a, dtype=np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def main() -> int:
    wrist = cv2.imread(WRIST)
    side = cv2.imread(SIDE)
    if wrist is None or side is None:
        print("图没找到：", WRIST, SIDE)
        return 1
    wrist = cv2.resize(wrist, (960, 540), interpolation=cv2.INTER_AREA)
    obs = {"images": {"left_wrist_0_rgb": _jpg(wrist), "base_0_rgb": _jpg(side)},
           "state": STATE,
           "prompt": "pick up the green box"}
    print(f"载荷 {len(obs['images']['left_wrist_0_rgb']) + len(obs['images']['base_0_rgb'])} B",
          flush=True)
    t0 = time.time()
    client = connect(HOST, PORT)
    print(f"连接 {time.time() - t0:.2f}s", flush=True)
    for i in range(3):
        t = time.time()
        out = client.infer(obs)
        acts = np.asarray(out["actions"])
        print(f"第 {i + 1} 次：{time.time() - t:.2f}s  动作 {acts.shape}  "
              f"{np.round(acts[0], 1)}", flush=True)
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
