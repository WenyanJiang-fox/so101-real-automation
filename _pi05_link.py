# -*- coding: utf-8 -*-
"""连策略服务的 websocket 客户端，关掉 keepalive。

`openpi_client.websocket_client_policy.WebsocketClientPolicy` 用 `websockets.sync.client
.connect` 的默认值 `ping_interval=20s`；服务端首次推理要 JIT 编译几十秒，事件循环被
占住回不了 ping，连接就被判 1011 掉线。这里把 ping 全关掉（`None`），接收靠阻塞 `recv`。

    from _pi05_link import connect
    client = connect("127.0.0.1", 8000)
    out = client.infer(obs)      # {"actions": (10, 6)}
    client.close()
"""
from __future__ import annotations

import websockets.sync.client

from openpi_client import msgpack_numpy


class PolicyLink:
    """和 `WebsocketClientPolicy` 同接口的最小子集：`infer` / `get_server_metadata` / `close`。"""

    def __init__(self, ws, metadata):
        self._ws = ws
        self._meta = metadata
        self._packer = msgpack_numpy.Packer()

    def get_server_metadata(self) -> dict:
        return self._meta

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"服务端报错：\n{resp}")
        return msgpack_numpy.unpackb(resp)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:                                          # noqa: BLE001
            pass


def connect(host: str = "127.0.0.1", port: int = 8000,
            open_timeout: float = 60.0) -> PolicyLink:
    ws = websockets.sync.client.connect(
        f"ws://{host}:{port}",
        compression=None,
        max_size=None,
        ping_interval=None,          # **不 ping**：服务端编译期间回不了 pong
        ping_timeout=None,
        close_timeout=5,
        open_timeout=open_timeout,
    )
    meta = msgpack_numpy.unpackb(ws.recv())
    return PolicyLink(ws, meta)
