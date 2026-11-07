# -*- coding: utf-8 -*-
"""给服务器上的 `websocket_policy_server.py` 加上「JPEG 字节进、数组出」。

链路实测 0.06 MB/s：两张 960x540 原图 3.1MB 正好要 50 秒，这就是每块慢的全部原因。
客户端改成发 JPEG（约 140KB），服务端在推理前解回数组。

    /root/miniconda3/envs/pi05/bin/python -u /root/autodl-tmp/_patch_jpeg_server.py
"""
from __future__ import annotations

import sys
from pathlib import Path

F = Path("/root/autodl-tmp/openpi/src/openpi/serving/websocket_policy_server.py")

HELPER = '''
def _decode_images(obs):
    """带 JPEG 的图先解成数组（客户端为了省带宽发的是 bytes）。

    `images` 里每个值是 JPEG 字节流（前两字节 0xFF 0xD8）就解，否则原样留着。
    返回 RGB uint8，和原来发裸数组时的口径一致。
    """
    try:
        import io

        import numpy as _np
        from PIL import Image
    except Exception:                                               # noqa: BLE001
        return obs
    imgs = obs.get("images") if isinstance(obs, dict) else None
    if not isinstance(imgs, dict):
        return obs
    for k, v in list(imgs.items()):
        if isinstance(v, (bytes, bytearray)) and len(v) > 2 and v[:2] == b"\\xff\\xd8":
            imgs[k] = _np.asarray(Image.open(io.BytesIO(bytes(v))).convert("RGB"),
                                  dtype=_np.uint8)
    return obs


'''


def main() -> int:
    src = F.read_text(encoding="utf-8")
    if "_decode_images" in src:
        print("已经打过补丁，跳过")
        return 0
    anchor = "class WebsocketPolicyServer:"
    if anchor not in src:
        print("找不到 class 定义")
        return 1
    src = src.replace(anchor, HELPER.lstrip("\n") + anchor, 1)
    old = "obs = msgpack_numpy.unpackb(await websocket.recv())"
    new = "obs = _decode_images(msgpack_numpy.unpackb(await websocket.recv()))"
    if old not in src:
        print("找不到 unpack 那一行")
        return 1
    src = src.replace(old, new, 1)
    F.write_text(src, encoding="utf-8")
    print("补丁已写入", F)
    return 0


if __name__ == "__main__":
    sys.exit(main())
