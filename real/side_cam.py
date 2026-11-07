"""侧视 USB 相机：只录数据，不参与抓取。

**为什么单独一个类、不塞进 `RealWristCamera`。** 那一路是腕上的，要跟着手走、要手眼
标定、要去畸变、`mount()` 得算机位；这一路固定在三脚架上，**没有标定、也没有机位**，
谁问 `mount()` 都只能回答「不知道」。两件事只有「开相机、拿一帧」这一句是共用的。

**取景是裁出来再转过来的**（2026-10-20 用户定的两组数，和 `view_cam.py` 里看的同一个）：

    crop   = 600,180,1460,1080   只留白纸、方块、机械臂，剪掉左边的地面和右边的柱子
    rotate = 180                 相机是倒着架的（不转的话桌沿在上、臂从下往上长）

`shot()` 交的是 **RGB**，和 `RealWristCamera.shot()` 同一个口径（`real/cameras.py`
文件头那条约定）。

**不设分辨率就出 640×480**，裁剪框对的是 1920×1080 的坐标，所以这里两个都得给，
而且**必须走 DSHOW**（2026-10-20 实测：不给后端时 index 0 落到 MSMF 上会卡死，同一个
相机用 DSHOW 三秒内出帧）。`open_camera` 两样都不给时会退回 `VideoCapture(index)`，
所以这里写死 1920×1080 + MJPG。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from color_check import grab_frame, open_camera          # noqa: E402

#: 侧相机在 1920×1080 下的取景框 `(x0, y0, x1, y1)`，和 `view_cam.py --crop` 同一组。
DEFAULT_CROP = (600, 180, 1460, 1080)
#: 相机倒着架的：纸在画面下方、臂从上方垂下来才对。
DEFAULT_ROTATE = 180.0


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    """绕画面中心转 `deg` 度，画布放大到装得下整张图。和 `view_cam.py::_rotate` 同一套。"""
    if not deg:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(deg), 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw = int(round(h * sin + w * cos))
    nh = int(round(h * cos + w * sin))
    m[0, 2] += nw / 2.0 - w / 2.0
    m[1, 2] += nh / 2.0 - h / 2.0
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


class SideCamera:
    """侧视相机。接口只到「拿一帧」—— 没有 `mount()`，因为它没有标定。"""

    name = "real_side"

    def __init__(self, index: int = 0, crop=DEFAULT_CROP,
                 rotate: float = DEFAULT_ROTATE, size=(1920, 1080),
                 fourcc: str = "MJPG", warmup: int = 8):
        self.index = int(index)
        self.crop = tuple(int(v) for v in crop) if crop else None
        self.rotate = float(rotate)
        self._cap = open_camera(self.index, size=(int(size[0]), int(size[1])),
                                fourcc=fourcc)
        first = grab_frame(self._cap, warmup)
        self._src_size = (int(first.shape[1]), int(first.shape[0]))
        if self.crop is not None:
            x0, y0, x1, y1 = self.crop
            if not (0 <= x0 < x1 <= self._src_size[0]
                    and 0 <= y0 < y1 <= self._src_size[1]):
                self._cap.release()
                raise RuntimeError(
                    f"侧相机裁剪框 {self.crop} 超出了实际画幅 {self._src_size}；"
                    f"相机这一档没谈下来，先跑 view_cam.py --probe 看看")
        out = self._process(first)
        self._size = (int(out.shape[1]), int(out.shape[0]))
        self._last = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        # **自己一条读帧线程**（2026-10-20）。这一档 1920×1080 MJPG 读一次要 ~210ms；
        # 原先 `shot()` 是当场读，谁调谁等 —— 采集时它挂在硬件线程的 `_on_step` 上，
        # 每录一帧就把走位拖慢 210ms；面板再加一路实时画面就要两个线程抢同一台相机。
        # 现在只有一个线程读 `cap`，别人读缓存（`real/cameras.py::_grab_loop` 同一个做法）。
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._seq = 0
        self._thr = threading.Thread(target=self._grab_loop, name="side_cam",
                                     daemon=True)
        self._thr.start()

    def _process(self, frame: np.ndarray) -> np.ndarray:
        if self.crop is not None:
            x0, y0, x1, y1 = self.crop
            frame = frame[y0:y1, x0:x1]
        return _rotate(frame, self.rotate)

    def _grab_loop(self) -> None:
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except Exception:                                       # noqa: BLE001
                ok, frame = False, None
            if ok and frame is not None:
                rgb = cv2.cvtColor(self._process(frame), cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._last = rgb
                    self._seq += 1
            else:
                time.sleep(0.05)

    def shot(self) -> np.ndarray:
        """最新那一帧（RGB）。**不读相机** —— 读帧线程在跑，这里只交缓存。

        相机断流时交的还是上一帧，不抛 —— 录数据时断一帧好过整趟死。
        """
        with self._lock:
            return self._last

    def seq(self) -> int:
        """换了第几帧（面板靠它判「这一帧要不要重画」）。"""
        return self._seq

    def size(self) -> tuple[int, int]:
        """(宽, 高)，转完、裁完之后的。"""
        return self._size

    def close(self) -> None:
        self._stop.set()
        thr = getattr(self, "_thr", None)
        if thr is not None:
            thr.join(timeout=1.0)
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
            self._cap = None


if __name__ == "__main__":
    # 直接跑 = 拍一张存下来看看。不动机械臂。
    import time

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cam = SideCamera(index=idx)
    print(f"侧相机 {idx}：原生 {cam._src_size} → 出图 {cam.size()}  "
          f"裁剪 {cam.crop}  转 {cam.rotate:g}°")
    rgb = cam.shot()
    out = _ROOT / "out" / f"side_cam_{int(time.time())}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"存了 {out}")
    cam.close()
