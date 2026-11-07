# -*- coding: utf-8 -*-
"""把腕相机变成一路**连续**的画面，落在 `out/live.jpg`，给监督窗看。

**为什么要在抓取进程里出图。** 同一台相机开不了两次（`real/cameras.py` 文件头那句），
`ppo.capture` 已经把它攥在手里了。所以由它顺手按 10 帧/秒把最新一帧写成一个文件，
面板只负责读 —— 面板那一路永远不碰相机、不碰串口。

**画什么。** 和 `_jaw_check` 画的同一套东西，只是每帧重画一遍：

    · 爪子三圈轮廓（固定爪青、动爪张开绿、动爪合上橙）—— 相机和两爪在同一个刚体上，
      这几圈在画面里是**死的**，描一次就够；
    · 合爪轴（品红长线）—— 同理，死的；
    · 方块红区轮廓（红线）—— 这个每帧现找，掩膜口径跟 `_jaw_check` 一模一样。

**不在这上面写判据数字。** `gap` / `axis` 的判定源只有一个（`rulers._jaw_check`），
这里再算一遍就成了第二个真值。面板要数字的时候看日志里每一眼那一行。

**写文件用「先写临时文件再换名」**，面板永远读不到写了一半的图。
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LIVE_IMG = ROOT / "out" / "live.jpg"
JAW_SHAPE_JSON = ROOT / "assets" / "camera_calibration" / "real_wrist_jaw_shape.json"

#: 和 `_jaw_check` 同一个口径：固定爪青、张开绿、合上橙、合爪轴品红、方块红。
COLORS = {"fixed_jaw": (255, 255, 0), "moving_open": (0, 255, 0),
          "moving_closed": (0, 165, 255)}


class LiveView:
    """后台线程：按 `fps` 把相机最新一帧画上标记写成 `LIVE_IMG`。"""

    def __init__(self, cam, percep=None, path: Path = LIVE_IMG,
                 fps: float = 10.0) -> None:
        self.cam = cam
        self.percep = percep
        self.path = Path(path)
        self.period = 1.0 / max(float(fps), 1.0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jaw: dict | None = None
        self._jaw_why = ""
        self.frames = 0

    # ---- 生命周期 ----

    def start(self) -> "LiveView":
        self._jaw, self._jaw_why = self._load_jaw()
        self._thread = threading.Thread(target=self._loop, name="live-view",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)

    # ---- 线程 ----

    def _load_jaw(self) -> tuple[dict | None, str]:
        try:
            d = json.loads(JAW_SHAPE_JSON.read_text(encoding="utf-8"))
        except Exception as exc:                                  # noqa: BLE001
            return None, f"读不到 {JAW_SHAPE_JSON.name}：{exc}"
        try:
            w, h = (int(v) for v in self.cam.size())
            if [int(v) for v in d.get("image_size", ())] != [w, h]:
                return None, (f"{JAW_SHAPE_JSON.name} 是在 {d.get('image_size')} 上描的，"
                              f"相机出的是 {[w, h]} —— 不画爪轮廓")
        except Exception:                                         # noqa: BLE001
            pass
        return d, ""

    def _loop(self) -> None:
        import cv2                                                # noqa: PLC0415

        said_err = ""
        while not self._stop.is_set():
            t0 = time.time()
            try:
                bgr = cv2.cvtColor(self.cam.shot(), cv2.COLOR_RGB2BGR)
            except Exception as exc:                              # noqa: BLE001
                msg = f"取帧失败：{type(exc).__name__}: {exc}"
                if msg != said_err:
                    print(f"    [live] {msg}", flush=True)
                    said_err = msg
                time.sleep(0.20)
                continue
            try:
                self._draw(bgr)
                self._write(bgr)
                self.frames += 1
            except Exception as exc:                              # noqa: BLE001
                msg = f"画/写失败：{type(exc).__name__}: {exc}"
                if msg != said_err:
                    print(f"    [live] {msg}", flush=True)
                    said_err = msg
            dt = time.time() - t0
            if dt < self.period:
                time.sleep(self.period - dt)

    def _draw(self, bgr) -> None:
        """爪子轮廓 + 合爪轴 + 方块轮廓。**只画，不判。**"""
        import cv2                                                # noqa: PLC0415

        jaw = self._jaw
        if jaw is not None:
            try:
                shapes = {k: jaw[k] for k in COLORS if k in jaw}
            except Exception:                                     # noqa: BLE001
                shapes = {}
            for name, pts in shapes.items():
                poly = (np.asarray(pts, dtype=np.float32)
                        .reshape(-1, 1, 2).astype(np.int32))
                cv2.polylines(bgr, [poly], True, COLORS[name], 3, cv2.LINE_AA)
            try:
                ref = np.asarray(jaw["axis_ref_px"], dtype=np.float64)
                u = np.asarray(jaw["axis_dir_px"], dtype=np.float64)
                u = u / max(float(np.linalg.norm(u)), 1e-12)
                far = 0.8 * float(bgr.shape[1])
                cv2.line(bgr, tuple(np.round(ref - far * u).astype(int)),
                         tuple(np.round(ref + far * u).astype(int)),
                         (255, 0, 255), 2, cv2.LINE_AA)
            except Exception:                                     # noqa: BLE001
                pass
        if self.percep is not None:
            from real.perception import brown_mask, candidates     # noqa: PLC0415
            try:
                mask = brown_mask(bgr, self.percep.ranges)
                cands = [d for d in candidates(mask, self.percep.min_area)
                         if d["rect_fill"] >= self.percep.fill_min]
                if cands:
                    cv2.drawContours(bgr, [cands[0]["contour"]], -1,
                                     (0, 0, 255), 2)
            except Exception:                                     # noqa: BLE001
                pass
        # 左上角一行 ASCII：这一张是什么时候的（cv2 写不了中文）。
        cv2.putText(bgr, "LIVE " + time.strftime("%H:%M:%S"),
                    (16, bgr.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(bgr, "LIVE " + time.strftime("%H:%M:%S"),
                    (16, bgr.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 2, cv2.LINE_AA)

    def _write(self, bgr) -> None:
        """先写 `live.tmp`，再三遍机会换名。面板正在读的那一下换不过去就丢一帧。"""
        import cv2                                                # noqa: PLC0415

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.jpg")
        ok, buf = cv2.imencode(".jpg", bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return
        buf.tofile(str(tmp))
        for _ in range(3):
            try:
                os.replace(str(tmp), str(self.path))
                return
            except PermissionError:
                time.sleep(0.005)
