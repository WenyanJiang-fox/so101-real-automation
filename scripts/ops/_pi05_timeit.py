# -*- coding: utf-8 -*-
"""在**服务器本机**掐一次推理的时间，不走 SSH 隧道。

    /root/miniconda3/envs/pi05/bin/python -u /root/autodl-tmp/_pi05_timeit.py

用 `out/pi05_try3` 里存下来的两张帧 + 最后一组关节角，连 `127.0.0.1:8000`，
连问 3 次，逐次打印墙钟。目的只有一个：把「隧道 + 传图」和「策略本身」分开。
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/root/autodl-tmp")
from _pi05_link import connect                                       # noqa: E402

BASE = "/root/autodl-tmp/pi05_probe"
WRIST = f"{BASE}/0090_wrist.jpg"
SIDE = f"{BASE}/0090_side.jpg"
STATE = np.array([-25.2, -10.0, 8.9, 94.3, -128.8, 1.9], dtype=np.float32)


def main() -> int:
    wrist = cv2.imread(WRIST)
    side = cv2.imread(SIDE)
    if wrist is None or side is None:
        print("图没找到：", WRIST, SIDE)
        return 1
    wrist = cv2.resize(wrist, (960, 540), interpolation=cv2.INTER_AREA)

    def _jpg(a):
        ok, buf = cv2.imencode(".jpg", np.asarray(a, dtype=np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return buf.tobytes()

    obs = {"images": {"left_wrist_0_rgb": _jpg(wrist),
                      "base_0_rgb": _jpg(side)},
           "state": STATE,
           "prompt": "pick up the green box"}
    print(f"载荷 {len(obs['images']['left_wrist_0_rgb']) + len(obs['images']['base_0_rgb'])} B",
          flush=True)
    t0 = time.time()
    client = connect("127.0.0.1", 8000)
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
