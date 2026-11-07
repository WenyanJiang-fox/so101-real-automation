# -*- coding: utf-8 -*-
"""把服务端 `_decode_images` 换成 cv2 版本：cv2 编/解同一条通道，数组和原来逐位相同。

PIL 解出来是 RGB，会和 cv2 编码时的 BGR 假设错开一次通道，所以不用它。

    /root/miniconda3/envs/pi05/bin/python -u /root/autodl-tmp/_patch_jpeg_server2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

F = Path("/root/autodl-tmp/openpi/src/openpi/serving/websocket_policy_server.py")

NEW = '''def _decode_images(obs):
    """带 JPEG 的图先解成数组（客户端为了省带宽发的是 bytes）。

    `images` 里每个值是 JPEG 字节流（前两字节 0xFF 0xD8）就解，否则原样留着。
    用 cv2 解：和客户端 `cv2.imencode` 是同一条通道编码，数组逐位不变。
    """
    try:
        import cv2
        import numpy as _np
    except Exception:                                               # noqa: BLE001
        return obs
    imgs = obs.get("images") if isinstance(obs, dict) else None
    if not isinstance(imgs, dict):
        return obs
    for k, v in list(imgs.items()):
        if isinstance(v, (bytes, bytearray)) and len(v) > 2 and v[:2] == b"\\xff\\xd8":
            imgs[k] = cv2.imdecode(_np.frombuffer(bytes(v), dtype=_np.uint8),
                                   cv2.IMREAD_COLOR)
    return obs


'''


def main() -> int:
    src = F.read_text(encoding="utf-8")
    start = src.find("def _decode_images(obs):")
    end = src.find("class WebsocketPolicyServer:")
    if start < 0 or end < 0 or end < start:
        print("结构不对，没找到函数或类")
        return 1
    src = src[:start] + NEW + src[end:]
    F.write_text(src, encoding="utf-8")
    print("已换成 cv2 版")
    return 0


if __name__ == "__main__":
    sys.exit(main())
