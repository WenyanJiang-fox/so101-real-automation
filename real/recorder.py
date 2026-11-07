"""把真机控制循环的每一步录成 LeRobot v3.0 的一帧。

**fps 是量出来的，不是拍脑袋写的。** 一个控制步的实测中位间隔 125ms（8.0Hz）。
**2026-10-20 改定频 0.2s**：采样闸口 `MIN_INTERVAL_S` 从 0.09 提到 0.2，于是落成
「隔一步录一帧」≈ 250ms ≈ **4Hz**，`fps` 跟着写 4，不再后期剪帧。
`stop()` 会把这一集实测的帧间隔中位数打出来，换机器/换相机之后按那个数改。

**接在哪。** `real/joint_move.py::_tick` 每个轨迹点都调 `arm.on_step`，`RealArm.drive`
也调。所以这里只提供一个无参的 `tick()`，由外面挂到 `arm.on_step` 上 —— 不另起线程、
不轮询，走位停下来的那一刻录制也停下。

**录什么。** 只录抓取这一段（用户 2026-10-20）：`observation.state` / `action` 两列
六关节角（度，0~180 舵机口径，跟 `RealArm.angles()` 同一个数），一路腕相机
`observation.images.camera1`（**960×540**，2026-10-20 用户定）。摆件那一段照跑但
不录 —— 它只负责把方块换到一个新位置。

**开场不录**（2026-10-20 用户定，减少停顿）：技能还没进任何段之前、以及夹爪标定段
（`CALIB_SEG`）都不进数据。实测这两块占 30 帧左右，全是原地不动。

**为什么是 960×540 而不是正方形。** 相机原生 1920×1080（16:9）。存成正方的 512×512
是左右压扁 1.78 倍；存成 960×540 只做等比缩放，取景和几何关系都不变。模型那头反正
也要自己缩放 —— SmolVLA 走 `resize_with_pad(..., 512, 512)`（保持长宽比、补黑边），
pi0/pi0.5 走 224×224，存什么分辨率都躲不掉这一步。实测 PNG 写入：1920×1080 190.3ms、
960×540 39.1ms、512² 22.0ms，所以全尺寸不值。

**fps 只是元数据。** 采样节拍由 `MIN_INTERVAL_S` 控制，两者别混。
"""
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

# **编码子进程的 BLAS 线程数钉成 1**（2026-10-21）：lerobot 存盘时按相机数开
# `ProcessPoolExecutor`（`dataset_writer.py:344`），每个 worker 重新 import numpy，
# OpenBLAS 默认按核数（本机 8）开线程池。内存一紧就在 worker 里报
# `OpenBLAS error: Memory allocation still failed after 10 retries`，整个进程池被杀，
# 这一集存不进去。子进程是 spawn 出来的、继承这里的环境变量 —— 只影响 BLAS 线程数，
# 不碰几何和控制。**必须在任何子进程创建之前设**，所以放在模块最上面。
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

#: 和 `real/arm.py:JOINT_ORDER` 一致，也是 `so101-smolvla-sim/paths.py:JOINT_NAMES` 那一串。
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

#: 抬升那一段的段名（`skills/pick_real/skill.py::_close_and_lift`）。走到它之后
#: 只要段名变了（摆件的「摆件：搬过去」），录制就停。
LIFT_SEG = "抬升到搬运高度"

#: 夹爪行程两端的标定段（`skill.py:304`）。**这一段不录**：它是「合一次空爪、张一次到底」
#: 测这一趟的空抓基准，和抓取动作无关，实测占 15~30 帧、全是原地不动。用户 2026-10-20 定：
#: 停顿在录制层去掉。段名之前（`seg == ""`）也不录 —— 那是技能还没开始动。
CALIB_SEG = "行程两端 → 合一次空爪、再张到底"

#: 两帧之间至少要隔这么久（秒）。**定频采集**：不在后期剪帧，录进来就是均匀的。
#:
#: 2026-10-20 用户定：`0.09 → 0.20`。实测一个控制步的中位间隔是 **125ms（8.0Hz）**，
#: 所以 0.2 的闸口落成「隔一步录一帧」≈ 250ms ≈ **4Hz**，`fps` 也写 4。
#: 想严格 5Hz 只能给控制循环加 sleep 把步长压到 200ms，采集会更慢，不值得。
MIN_INTERVAL_S = 0.2

#: 临时计时（`PANEL_PROFILE=1` 才打）：把 `tick()` 里每一段花多少毫秒拆开。
_PROF = bool(os.environ.get("PANEL_PROFILE"))

#: 一集的墙钟上限（秒）。**只防卡死，不改抓取判据。** 参考：整趟抓取实测 95.5 秒，
#: 现在每步多 ~80ms 录帧，按两倍算 200 秒；摆件再留 100 秒。超过就立 `arm.terminated`
#: —— `goto_path` / `_close_until_stall` / `skill._stop_here` 都会看它。
EPISODE_MAX_S = 420.0


#: 图像存多大：`(宽, 高)`。相机原生 1920×1080，取一半。
DEFAULT_SIZE = (960, 540)


def norm_size(size) -> tuple[int, int]:
    """`512` → `(512, 512)`；`(960, 540)` / `"960x540"` → `(960, 540)`。给的是 `(宽, 高)`。"""
    if isinstance(size, str):
        t = size.lower().replace("×", "x")
        w, h = t.split("x", 1) if "x" in t else (t, t)
        return (int(w), int(h))
    if isinstance(size, int):
        return (int(size), int(size))
    w, h = size
    return (int(w), int(h))


def features(size=DEFAULT_SIZE, side_size=None) -> dict:
    """三列 feature（给了侧相机就是四列）。键名和 `so101-smolvla-sim/collect_demos.py:89-100`
    一致；第二路叫 `camera2`，排在腕相机后面。

    `shape` 的顺序是 `(高, 宽, 3)`（`names` 就写着），和 `cv2.resize` 的 `dsize`
    顺序**相反**，别把宽高填反了。

    **加不加这一路是数据集级别的决定**：`meta/info.json` 里的 feature 定死之后不能改，
    所以腕相机和侧相机必须一开始就说清楚，中途不能反悔。
    """
    w, h = norm_size(size)
    feats = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "observation.images.camera1": {
            "dtype": "video", "shape": (h, w, 3),
            "names": ["height", "width", "channels"]},
    }
    if side_size is not None:
        sw, sh = norm_size(side_size)
        feats["observation.images.camera2"] = {
            "dtype": "video", "shape": (sh, sw, 3),
            "names": ["height", "width", "channels"]}
    return feats


class RealRecorder:
    """一个数据集 + 一个「这一步要不要记」的开关。"""

    def __init__(self, root, repo_id: str, task: str, *, fps: int = 4,
                 size=DEFAULT_SIZE, side=None,
                 max_episode_s: float = EPISODE_MAX_S):
        self.root = Path(root)
        self.repo_id = str(repo_id)
        self.task = str(task)
        self.fps = int(fps)
        self.size = norm_size(size)
        # **侧相机是可选的一路**：给就给整段录进去，不给就只录腕相机。给了之后
        # `side.size()` 决定 `camera2` 的 shape —— 侧相机那边已经裁过、转过了，
        # 这里不再缩放，画面原样进数据。
        self.side = side
        self.side_size = norm_size(side.size()) if side is not None else None
        self.max_episode_s = float(max_episode_s)
        self.ds = self._open()
        self.arm = None
        self.cam = None
        self.state = None
        self.active = False
        self.n_frames = 0
        self._t_ep = 0.0
        self._t_last = 0.0
        self._dts: list[float] = []
        self._warned = False
        self._started = False
        # **写盘单独一条线程**（2026-10-20）。`add_frame` 实测 76~109ms（两路视频编码），
        # 它原来挂在控制步上：走位线程每录一帧就要等它，轨迹从 30Hz 掉到 10Hz 左右。
        # 现在 `tick()` 只把组好的帧塞进队列就返回；顺序还是先进先出，攒够再存。
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._err: BaseException | None = None
        self._wr = threading.Thread(target=self._write_loop, name="rec-writer",
                                    daemon=True)
        self._wr.start()

    # ---- 数据集 ----

    def _open(self) -> LeRobotDataset:
        """有 `meta/info.json` 就接着写，没有就新建（照 `collect_demos.py:101-103`）。

        **`encoder_threads=1`**（2026-10-21）：本机 16GB、存盘时只剩 ~1.6GB，SVT-AV1
        开编码器直接 `MemoryError: avcodec_open2(libsvtav1)`。这个值对 libsvtav1 会被
        翻译成 `svtav1-params lp=<n>`（`lerobot/configs/video.py:197-208`），线程池小了
        峰值内存跟着降。两路相机各开一个进程、每个都开 SVT，线程数再按核数（8）铺就是
        内存爆的直接原因。
        """
        if (self.root / "meta" / "info.json").exists():
            print(f"[录] 续写已有数据集：{self.root}", flush=True)
            ds = LeRobotDataset.resume(self.repo_id, root=str(self.root),
                                       encoder_threads=1)
            self._check_features(ds)
            return ds
        # **`create` 要求目录不存在**（`dataset_metadata.py:832` 用 `exist_ok=False`）。
        # 目录在、里面却没有数据集，说明是别的东西，不替用户删。
        if self.root.exists():
            raise RuntimeError(f"{self.root} 已存在但没有 meta/info.json，"
                               f"不覆盖；换一个 --root，或者自己确认后清空")
        print(f"[录] 新建数据集：{self.root}", flush=True)
        return LeRobotDataset.create(
            repo_id=self.repo_id, fps=self.fps, root=str(self.root),
            robot_type="so101", use_videos=True, encoder_threads=1,
            features=features(self.size, self.side_size))

    def _check_features(self, ds) -> None:
        """**续写前先核对 feature**（2026-10-22 踩过）。

        `meta/info.json` 里的相机列是建数据集时定死的。往只有 `camera1` 的数据集
        里写带 `camera2` 的帧，`add_frame` 会抛
        `ValueError: Feature mismatch ... Extra features: {'observation.images.camera2'}`，
        而且是在**第一集抓成、`save()` 的时候**才炸 —— 那一集白录，才知道参数错了。
        """
        want = features(self.size, self.side_size)
        have = dict(getattr(ds.meta, "features", {}) or {})
        # **只查「这次要写的」，不看数据集多出来的列。** lerobot 自己会往
        # `meta/info.json` 里补 `timestamp` / `episode_index` / `frame_index` /
        # `index` / `task_index`，那些不是我们写的，拿它们报错会挡住每一次续写。
        bad: list[str] = []
        for k in sorted(want):
            if k not in have:
                bad.append(f"{k}（数据集里没有，这次要写）")
                continue
            if k.startswith("observation.images"):
                ws = tuple(want[k].get("shape") or ())
                hs = tuple(have[k].get("shape") or ())
                if ws != hs:
                    bad.append(f"{k}（数据集 {hs}，这次 {ws}）")
        if bad:
            raise RuntimeError(
                f"{self.root} 的 feature 和这次录的不一致，一条都没写：\n  "
                + "\n  ".join(bad)
                + "\n换一个 --root / --repo-id（两路相机的数据集见 "
                  "`configs/real.yaml` 的 `collect.root`），"
                  "或者把 `collect.side_cam` 整段去掉再跑。")

    # ---- 一集 ----

    def start(self, arm, cam, state) -> None:
        """开始录这一集。`state` 是技能自己的 `RoundState`（要读它的 `held` / `seg`）。"""
        self.arm = arm
        self.cam = cam
        self.state = state
        self.active = True
        self.n_frames = 0
        self._dts = []
        self._warned = False
        self._started = False
        now = time.time()
        self._t_ep = now
        self._t_last = 0.0
        print(f"[录] 开始第 {self.ds.meta.total_episodes} 集（fps {self.fps}，"
              f"{self.size[0]}×{self.size[1]}）", flush=True)

    def stop(self) -> int:
        """停止采样。返回这一集录了多少帧。"""
        if self.active:
            self.active = False
            span = time.time() - self._t_ep
            extra = ""
            if self._dts:
                med = float(np.median(self._dts))
                extra = (f"  实测帧间隔中位 {med * 1000:.0f}ms"
                         f"（{1.0 / med:.1f} Hz）")
            print(f"[录] 停止采样：{self.n_frames} 帧 {span:.1f}s "
                  f"（{self.n_frames / span if span > 0 else 0:.1f} Hz）{extra}",
                  flush=True)
        return self.n_frames

    def tick(self) -> None:
        """挂到 `arm.on_step` 上的回调。没有参数，自己去看 `state.seg`。"""
        if not self.active:
            return
        now = time.time()
        # ---- 看门狗：只防卡死 ----
        if now - self._t_ep > self.max_episode_s:
            if not self._warned:
                self._warned = True
                print(f"[录] ⏱ 这一集超过 {self.max_episode_s:.0f}s，立 terminated 收工",
                      flush=True)
            self.arm.terminated = True
            return
        # ---- 开场不录：技能还没开始动，或者正在量夹爪行程两端 ----
        seg = getattr(self.state, "seg", "") or ""
        if not self._started:
            if not seg or seg == CALIB_SEG:
                return
            self._started = True
            # 第一帧从这个时刻算起，别把开场那几十秒算进帧间隔
            self._t_last = 0.0
            print(f"[录] 开录：技能进入「{seg}」", flush=True)
        # ---- 抓成了、已经走完抬升：摆件不录 ----
        if getattr(self.state, "held", None) is not None and seg != LIFT_SEG:
            self.stop()
            return
        # ---- 节拍 ----
        if now - self._t_last < MIN_INTERVAL_S:
            return
        if self._t_last > 0.0:
            self._dts.append(now - self._t_last)
        self._t_last = now
        _t0 = time.perf_counter()
        img = self.cam.shot()
        _t1 = time.perf_counter()
        # `dsize` 是 `(宽, 高)`：1920×1080 → 960×540 只做等比缩放，不压扁。
        img = cv2.resize(np.asarray(img), self.size, interpolation=cv2.INTER_AREA)
        _t2 = time.perf_counter()
        st = np.asarray(self.arm.angles(), dtype=np.float32)
        _t3 = time.perf_counter()
        act = getattr(self.arm, "last_action", None)
        act = st.copy() if act is None else np.asarray(act, dtype=np.float32)
        frame = {
            "observation.state": st,
            "action": act,
            "observation.images.camera1": img,
            "task": self.task,
        }
        _t4 = time.perf_counter()
        if self.side is not None:
            # 侧相机自己已经裁好、转好了（`real/side_cam.py`），这里原样进数据。
            frame["observation.images.camera2"] = np.asarray(self.side.shot(),
                                                             dtype=np.uint8)
        _t5 = time.perf_counter()
        # **只入队，不等写盘**（2026-10-20）。队列满了才阻塞 —— 那是写盘真的跟不上，
        # 此时宁可把控制步拖慢，也不丢帧（丢帧等于把数据的时间轴剪了个洞）。
        self._q.put(frame)
        _t6 = time.perf_counter()
        self.n_frames += 1
        if _PROF and self.n_frames % 20 == 0:
            print(f"[prof 录] 腕图 {(_t1 - _t0) * 1000:.0f}ms + 缩放 "
                  f"{(_t2 - _t1) * 1000:.0f}ms + 读舵机 {(_t3 - _t2) * 1000:.0f}ms"
                  f" + 组帧 {(_t4 - _t3) * 1000:.0f}ms + 侧图 "
                  f"{(_t5 - _t4) * 1000:.0f}ms + 入队 "
                  f"{(_t6 - _t5) * 1000:.0f}ms = {(_t6 - _t0) * 1000:.0f}ms",
                  flush=True)

    # ---- 写盘那条线程 ----

    def _write_loop(self) -> None:
        """队列消费方。**只有这一条线程碰 `ds.add_frame`**，顺序就是入队顺序。"""
        while True:
            frame = self._q.get()
            try:
                if frame is None:
                    return
                self.ds.add_frame(frame)
            except BaseException as exc:                            # noqa: BLE001
                self._err = exc
                print(f"[录] ✗ 写帧失败：{type(exc).__name__}: {exc}", flush=True)
            finally:
                self._q.task_done()

    def _drain(self) -> None:
        """等队列里的帧全部落进缓冲。存/丢之前必须调 —— 不然最后几帧会串到下一集。"""
        self._q.join()
        if self._err is not None:
            exc, self._err = self._err, None
            raise RuntimeError(f"录数据时写帧失败：{exc}") from exc

    # ---- 存/丢 ----

    def save(self) -> None:
        """存下这一集。

        **这里不 `finalize()`。** `DatasetWriter.finalize` 会把 parquet 写手关掉、
        置 `_finalized`，后面几集就写不进去了；它只在采集全部结束时调一次
        （`close()`）—— 不调那一次，整个数据集读不出来（`merge_datasets.py:106-107`）。
        """
        self.active = False
        self._drain()
        try:
            self.ds.save_episode()
        except BaseException:
            # **保存崩了必须重建 buffer**（2026-10-21）：lerobot 的 `save_episode` 第一件
            # 事就是 `episode_buffer.pop("size")`（`dataset_writer.py:282`），编码那一步
            # 一抛异常，buffer 就永远缺 `size` —— 之后每一帧 `add_frame` 都
            # `KeyError: 'size'`，整份采集再也录不进去。这一集按失败处理（异常照抛给
            # 调用方），但下一集要能接着录。
            try:
                self.ds.clear_episode_buffer(delete_images=True)
                print("[录] 保存失败 —— buffer 已重建，下一集接着录", flush=True)
            except Exception as exc:                               # noqa: BLE001
                print(f"[录] ✗ 重建 buffer 也失败：{type(exc).__name__}: {exc}",
                      flush=True)
            raise
        print(f"[录] 已存第 {self.ds.meta.total_episodes - 1} 集，"
              f"共 {self.ds.meta.total_frames} 帧 / "
              f"{self.ds.meta.total_episodes} 集", flush=True)

    def discard(self, why: str = "") -> None:
        """丢掉这一集（连同临时图片）。"""
        self.active = False
        n = self.n_frames
        self._drain()
        self.ds.clear_episode_buffer(delete_images=True)
        print(f"[录] 丢掉这一集（{n} 帧）{('：' + why) if why else ''}", flush=True)

    def close(self) -> None:
        self.active = False
        try:
            self._drain()
            if self.ds.has_pending_frames():
                self.ds.clear_episode_buffer(delete_images=True)
        except Exception as exc:                                   # noqa: BLE001
            print(f"[录] 收尾时清缓冲失败：{exc}", flush=True)
        self._q.put(None)                  # 写盘线程收工之后再 finalize
        self._wr.join(timeout=10.0)
        self.ds.finalize()
        print(f"[录] 数据集收尾：{self.ds.meta.total_episodes} 集 / "
              f"{self.ds.meta.total_frames} 帧 → {self.root}", flush=True)
