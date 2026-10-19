"""真机相机：从 USB 摄像头上抓一帧，接口和仿真的 `impls/cam_sim.py` 逐字一样。

`Camera` 只要求三件事（`core/interfaces.py:73`）：

    shot()   给一帧
    size()   画面多大
    mount()  自己在世界里的位姿 + 内参

**`mount()` 是现算的，不是查表。** 腕部相机跟着手走，臂一动手位姿就变；仿真那边
（`cam_sim.mount()`）也是现读 `cam_xpos/cam_xmat`。这里现算的原料有两份：

    · 这一刻的关节读数 → 模型正解（`RealArm._fk`）→ gripperframe 在基座系里的位姿
    · 手眼标定解出来的那个固定变换（相机相对 gripperframe）

两份一乘就是相机在基座系里的位姿。这就是「相机标定 + 手眼标定」的全部用处：
把像素换算成基座系里的坐标。**没标定过就返回 None**，接口本来就允许
（`core/interfaces.py:89`）—— 不要给一个凑出来的位姿，感知会拿它去反推世界坐标，
算出来的点看着有值、其实是错的。

**畸变在出图这一步抹掉，不进核心。** harness 的整条视觉链都是纯针孔，仿真那台也是，
所以约定就是「相机交给核心的必须是一张针孔图」。真机这颗镜头不是针孔的（1920x1080
下 k2 = −0.040），`shot()` 先用标定的畸变系数把整幅拉正再交出去。这样 `mount()`
不用多带一个畸变槽，`core/vision.py` 一个字不用改，下游自己算内参的那条支路
（`skills/pick.py::_look`）也自动跟着对。

**坐标系要换一次：cv2 的相机系 ≠ MuJoCo 的相机系。** 手眼解出来的那份是 cv2 的
约定（x 右、y **下**、z 朝前），而 `core/vision.backproject` 用的是 MuJoCo 那一套
（x 右、y 上、视线是 −z）。差一个 `diag(1,−1,−1)`。少换这一次，反推出来的点会
上下、前后双双翻掉，而且看着还挺像个正常的坐标。

**`shot()` 必须返回 RGB。** cv2 给的是 BGR，仿真那台给的是 RGB（`cam_sim.shot()`
取的是 MuJoCo 观测）。两条路的出口必须是同一种东西，少转这一次，所有颜色判据全反。

屏幕上的 `camera 1` 就是这一路（`G:\\sim2real\\so101-real-harness\\view_cam.py` 看的同一个）。
**同一个相机开不了两次** —— 那边窗口开着的时候这边会打不开。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# 上级目录（= 项目根）要在 sys.path 上，color_check 才导得进来 —— 直接跑本文件时
# 也要成立，不能只依赖调用方先把路径铺好。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from color_check import grab_frame, open_camera          # noqa: E402
from core.interfaces import Camera                        # noqa: E402
from core.registry import register                        # noqa: E402

CALIB_DIR = _ROOT / "assets" / "camera_calibration"
INTRINSICS_JSON = CALIB_DIR / "real_wrist.json"          # 内参（fx fy cx cy + 畸变）
HAND_EYE_JSON = CALIB_DIR / "real_wrist_mount.json"      # 外参（相机相对 gripperframe）

# cv2 相机系 → MuJoCo 相机系。见文件头那段「坐标系要换一次」。
CV2_TO_MJ = np.diag([1.0, -1.0, -1.0])

# 要相机出 MJPG。不设这个，1920x1080 拿不到（同一套写法见 `real/calib_views.py`）。
FOURCC = "MJPG"


def _T(rot, tvec) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = np.asarray(rot, dtype=np.float64)
    m[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return m


def pose_from_fk(cal: dict, pos_g, quat_g) -> dict:
    """抓手的位姿 + 手眼 → 相机在基座系里的位姿（返回 `mount()` 那个 dict）。

    **抽出来是为了能离线复算。** `mount()` 拿实时的关节读数喂它；`real/mount_check.py
    的 `--views` 拿存下来的 `raw_ticks` 喂同一个函数，这样验的就是**线上真跑的那套
    算法**，不是另写一份看着一样的。少了这一层，离线复算验的是复算自己。
    """
    from scipy.spatial.transform import Rotation                # noqa: PLC0415

    # `_fk` 给的四元数是 [w,x,y,z]，scipy 要 [x,y,z,w]。
    T_b_g = _T(Rotation.from_quat([quat_g[1], quat_g[2], quat_g[3], quat_g[0]])
               .as_matrix(), pos_g)
    # 相机在基座系里 = 抓手在基座系里 × 手眼那个固定变换，再换成 MuJoCo 那套坐标轴。
    T = T_b_g @ cal["T_g_cv"] @ np.block(
        [[CV2_TO_MJ, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
    return {"pos": T[:3, 3].copy(), "mat": T[:3, :3].copy(),
            "fovy": cal["fovy"], "k": cal["k"].copy()}


def load_mount_cal() -> dict | None:
    """把两份标定读出来。**缺一份就返回 None**，不拿半个凑。"""
    if not (INTRINSICS_JSON.exists() and HAND_EYE_JSON.exists()):
        return None
    intr = json.loads(INTRINSICS_JSON.read_text(encoding="utf-8"))
    he = json.loads(HAND_EYE_JSON.read_text(encoding="utf-8"))
    k3 = np.asarray(intr["k"], dtype=np.float64)
    size = tuple(int(v) for v in intr["image_size"])
    # `mount["k"]` 在 harness 里的形状是 (fx, fy, cx, cy) 四个数，**不是 3x3** ——
    # `core/vision.backproject` 就是那么解的（`fx, fy, cx, cy = k`）。
    k4 = np.array([k3[0, 0], k3[1, 1], k3[0, 2], k3[1, 2]], dtype=np.float64)
    # fovy 是从 fy 反推的，这样「按视场角算」和「按标定算」两条路给的是同一个数。
    fovy = float(np.degrees(2.0 * np.arctan(size[1] / (2.0 * k4[1]))))
    return {"T_g_cv": _T(he["R_cam2gripper"], he["t_cam2gripper"]),
            "k": k4, "fovy": fovy, "image_size": size,
            # 3x3 那套 K 和畸变系数单独留着，只给 `_build_rectify_map` 用。**别把它塞进
            # 给 harness 的 `mount()`** —— 那边整条链是纯针孔，畸变得在出图这一步就抹掉。
            "k3": k3, "dist": np.asarray(intr.get("dist", []), dtype=np.float64),
            "source": f"{INTRINSICS_JSON.name} + {HAND_EYE_JSON.name}"}


@register("camera", "real_wrist")
class RealWristCamera(Camera):
    """真机上那一路。暂时只有一路，后面补第二路时再加一个同形状的类。"""

    name = "real_wrist"

    def __init__(self, source=None, index: int = 1, warmup: int = 5, **kw):
        # `source` 是主循环统一塞进来的「后端」（`run_sim.py:247` 那句 make 对每一路
        # 相机都传 `source=arm`）。仿真那台靠它拿画面，真机不需要 —— 画面从摄像头来。
        # **收下不用，是为了让主循环那句 make 一个字都不用改。**
        self.source = source
        self.index = int(index)
        self.warmup = int(warmup)
        self._cal = load_mount_cal()      # 内参 + 手眼，缺一份就是 None
        self._size_warned = False
        # **按标定那一档分辨率开。** 不给这两样，这台相机默认只出 640x480，那份
        # 1920x1080 的内参就作废了（`open_camera` 的注释里说了为什么两样都得给）。
        want = tuple(self._cal["image_size"]) if self._cal else None
        self._cap = open_camera(self.index, size=want,
                                fourcc=FOURCC if want else None)
        self._last = None
        self._rectify = None
        try:
            first = grab_frame(self._cap, self.warmup)
        except Exception:
            self._cap.release()
            self._cap = None
            raise
        # 尺寸要按**真拿到的那一帧**定，再拿它去建重映射表 —— 表是按像素算的，
        # 跟标定那一档差一个像素都对不上（`size()` 的注释里说了 `cap.set` 会骗人）。
        self._size = (int(first.shape[1]), int(first.shape[0]))
        self._rectify = self._build_rectify_map()
        self._last = cv2.cvtColor(self._rectify_frame(first), cv2.COLOR_BGR2RGB)
        # **一个抓帧线程常驻**（2026-10-18 时延）：`cap.read()` 只许这一个线程调。
        # 以前每次 `shot()` 都现读两帧（丢一帧再取一帧），一次 30~70ms，而且读到的
        # 是驱动缓存里最旧的那一帧；面板每 0.1 秒抓一次、感知那条链自己还要再抓一次，
        # 串口那一路就被抓帧挤住了。现在它自己按相机节拍把最新一帧放在 `_raw` 上，
        # `shot()` 只做「去畸变 + 转 RGB」，不再碰总线。
        self._raw: np.ndarray | None = None
        self._raw_lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._stop = threading.Event()
        with self._raw_lock:
            self._raw = first.copy()
        self._grabber = threading.Thread(target=self._grab_loop, name="cam-grab",
                                         daemon=True)
        self._grabber.start()

    def _grab_loop(self) -> None:
        """按相机自己的节拍把帧读出来、只留最新一帧。**唯一的读取者。**"""
        while not self._stop.is_set():
            with self._cap_lock:
                cap = self._cap
            if cap is None:
                break
            ok, frame = cap.read()
            if ok and frame is not None:
                with self._raw_lock:
                    self._raw = frame
            else:
                time.sleep(0.005)

    def _build_rectify_map(self):
        """预先把「去畸变」的像素重映射表算出来。不适用就返回 None。

        **为什么在出图这一步做、不在反推那一步做。** harness 的整条视觉链
        （`core/vision.backproject` / `project` / `estimate_pose`）都是**纯针孔**，
        仿真那台相机也是纯针孔，所以「相机交给核心的必须是一张针孔图」是这个接口的
        隐含约定。真机这颗镜头不是针孔的，那就由它自己把图拉正 —— 这样核心不用知道
        畸变这回事，`mount()` 也不用多带一个槽，下游（包括 `skills/pick.py::_look`
        那条自己算内参的支路）一个字都不用改。

        **不拉正的代价实测过。** 拿 23 张远距姿态复算，同一块没动过的板子反推出来
        的点散到 19.86mm（最差 34.37mm），量出来的格子间距只有 23.68mm（该是 28.0），
        而这个偏差跟「角点离主点的半径」相关 +0.88、跟相机倾角只 +0.30 —— 就是畸变。
        拉正之后散度掉到 3.53mm、间距回到 27.6~28.1mm。

        **输出仍按原来那套 K 出图**（`P=K`，不换 `getOptimalNewCameraMatrix`），
        所以 `mount()["k"]` 跟拉正后的图是严格配套的。代价是画面四角会出现采样不到
        的黑边 —— 那边的视场是畸变挤出来的，本来就不该有内容。
        """
        cal = self._cal
        if cal is None or cal["dist"].size == 0:
            return None
        if tuple(self._size) != tuple(cal["image_size"]):
            return None                     # 分辨率跟标定对不上，这张表对不上像素
        m1, m2 = cv2.initUndistortRectifyMap(cal["k3"], cal["dist"], None,
                                             cal["k3"], tuple(self._size),
                                             cv2.CV_16SC2)
        return m1, m2

    def _rectify_frame(self, frame: np.ndarray) -> np.ndarray:
        """出图前把畸变抹掉。没标定畸变（或者分辨率对不上）就原样返回。"""
        if self._rectify is None:
            return frame
        return cv2.remap(frame, self._rectify[0], self._rectify[1], cv2.INTER_LINEAR)

    def shot(self) -> np.ndarray:
        """要**最新**那一帧。抓帧在 `_grab_loop` 那个线程里，这里只做去畸变 + 转 RGB。

        **不再自己 `cap.read()`**（2026-10-18）：多一个人读同一条 VideoCapture 会打架，
        而且每次读两帧、拿到的还是驱动缓存里最旧的那一帧。现在这里拿到的是抓帧线程
        刚读出来的那一帧（最多一个相机周期旧）。
        """
        with self._raw_lock:
            raw = self._raw
        if raw is None:
            frame = self._rectify_frame(grab_frame(self._cap, 2))
        else:
            frame = self._rectify_frame(raw)
        self._last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._last

    def size(self) -> tuple[int, int]:
        """(宽, 高)。按实际拿到的那一帧报，不按 `cap.set` 要的那个值 ——
        DSHOW 上 set 经常返回 True 但实际没生效（这台相机 2026-10-13 就踩过）。"""
        return self._size

    def mount(self) -> dict | None:
        """相机这一刻在**模型基座系**里的位姿 + 内参。标定没做全就返回 None。

        约定和 `core/interfaces.py:89` / `core/vision.backproject` 一致：
        `pos` 是相机原点，`mat` 是 cam→base 的 3x3（MuJoCo 那一套：x 右、y 上、
        视线 −z），`fovy` 是垂直视场角（度），`k` 是多给的一个 (fx,fy,cx,cy)。

        **要臂。** `source` 得是一个 `RealArm`（主循环对每一路相机都传 `source=arm`，
        仿真那台也是这么拿模型的）。没有臂就当标定不全，返回 None。

        **帧和位姿不是同一瞬间的。** 这里读的是「问的这一下臂在哪」，`shot()` 是
        下一次抓帧。臂在动的时候这两个会差开；静止的时候无所谓。
        """
        cal = self._cal
        if cal is None:
            return None
        w, h = self._size
        if (w, h) != tuple(cal["image_size"]):
            # **分辨率对不上，内参就作废** —— 标定是按 1920x1080 做的，换一档像素数
            # 要整套重标。宁可说没有机位，不要拿错标度的内参去反推。
            if not self._size_warned:
                print(f"！相机出的是 {w}x{h}，标定是 {cal['image_size'][0]}x"
                      f"{cal['image_size'][1]}，内参对不上 —— `mount()` 返回 None")
                self._size_warned = True
            return None
        arm = self.source
        if arm is None or not hasattr(arm, "_fk"):
            return None
        try:
            q = arm._ticks_to_qpos(arm.read_ticks())
            pos_g, quat_g = arm._fk(q)
        except Exception:                                        # noqa: BLE001
            # 总线读挂了就如实说没机位，不要把上一次的位姿拿出来充数。
            return None
        return pose_from_fk(cal, pos_g, quat_g)

    def close(self) -> None:
        # **先让抓帧线程停下，再关相机**：反过来的话它正在 `read()` 里，
        # `release()` 之后那一读会报错。
        stop = getattr(self, "_stop", None)
        if stop is not None:
            stop.set()
            g = getattr(self, "_grabber", None)
            if g is not None and g.is_alive():
                g.join(timeout=1.0)
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
            self._cap = None


if __name__ == "__main__":
    # 直接跑 = 只用相机拍一张存下来看看。不动机械臂。
    import time

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cam = RealWristCamera(index=idx)
    w, h = cam.size()
    print(f"相机 {idx}：{w}x{h}   mount()={cam.mount()}")
    rgb = cam.shot()
    out = _ROOT / "out" / f"real_cam{idx}_{int(time.time())}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"存了 {out}")
    cam.close()
