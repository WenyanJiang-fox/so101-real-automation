"""装配：一份 yaml → 一套能跑的真机（机械臂 + 腕相机 + 感知 + 判据阈值）。

**照抄 `run_sim.py` 的同一份 `configs/real.yaml`、同一串 `make()`**，不另起一套。
这个 mixin 只做「造东西」，不参与任何一次动作，也不碰命令队列。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import geom                                           # noqa: E402
from core.registry import available, make                      # noqa: E402
from skills import pick_real as PR                              # noqa: E402

from panel.rig_common import DEFAULT_CONFIG, load_cfg           # noqa: E402


class AssemblyMixin:
    """`Rig.__init__` 里「把设备造出来」的那一半。"""

    def _assemble(self, cfg_path=DEFAULT_CONFIG, port=None, cam_index=None) -> None:
        cfg = load_cfg(cfg_path)
        rig = cfg.get("real") or {}
        if rig.get("table_z_mm") is None or rig.get("pick_xy_mm") is None:
            raise SystemExit(
                f"{cfg_path} 的 `real:` 那一段缺数：要 `table_z_mm` 和 `pick_xy_mm`。"
                "这两个数不给就把画面反推不成世界坐标 —— 拿猜的去驱动机械臂比不跑更糟。")
        # **必须在 make 之前设**：感知起来时就读它们，读不到会当场抛。
        geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
        geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                                     dtype=np.float64) / 1000.0
        geom.REAL_PERCEP_OFFSET = np.array(
            [float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
            dtype=np.float64) / 1000.0
        geom.OBJECTS = list(cfg.get("objects") or [])

        # 挂过牌子的实现才查得到。**必须在 make 之前 import。**
        import impls.judge_held                                     # noqa: F401,PLC0415
        import impls.policy_real                                    # noqa: F401,PLC0415
        import real.arm                                             # noqa: F401,PLC0415
        import real.cameras                                         # noqa: F401,PLC0415
        import real.perception                                      # noqa: F401,PLC0415
        import skills.home_real                                     # noqa: F401,PLC0415
        import skills.pick_real                                     # noqa: F401,PLC0415

        self.cfg = cfg
        self.port = str(port or rig.get("port", "COM10"))
        self.cam_index = int(cam_index if cam_index is not None
                             else rig.get("cam_index", 0))
        self._log(f"机械臂 {available('arm')} / 相机 {available('camera')} / "
                  f"感知 {available('perception')}")

        self.arm = make("arm", "real", port=self.port,
                        speed=int(rig.get("speed", 0)))
        self.arm.on_step = self._on_step
        self.arm.terminated = False
        self.cams = {"real_wrist": make("camera", "real_wrist", source=self.arm,
                                        index=self.cam_index)}
        self.cam = self.cams["real_wrist"]
        # 侧相机：内参/外参都标好了才建。建不出来不致命 —— 腕相机那条路照常走。
        self.percep_side = None
        try:
            self.cams["real_side"] = make("camera", "real_side", source=self.arm)
            self._log(f"侧相机 index {self.cams['real_side'].index} "
                      f"({self.cams['real_side'].size()[0]}x"
                      f"{self.cams['real_side'].size()[1]})")
            # **单独一份感知，专供「指尖对账」那一下**。它只读侧相机，不掺进
            # `self.percep`（那份跟着 `perception_cam` 走，抓取还在用）。
            self.percep_side = make("perception", "brown_rect",
                                    cams={"real_side": self.cams["real_side"]},
                                    ref=None, objects=geom.object_specs(),
                                    cam_name="real_side")
        except Exception as exc:                                    # noqa: BLE001
            self._log(f"！侧相机没建起来（{type(exc).__name__}: {exc}），只用腕相机")
        # `cam_name` 只在配置写了 `perception_cam` 时给（跟 `run_sim.py:426` 同一口径）；
        # 没写就用这一路自己的默认（`real/perception.py:57` 的 `CAM_NAME = "real_wrist"`）。
        percep_kw = ({"cam_name": str(cfg["perception_cam"])}
                     if cfg.get("perception_cam") else {})
        self.percep = make("perception", "brown_rect", cams=self.cams, ref=None,
                           objects=geom.object_specs(), **percep_kw)
        self.color = str(self.percep.color)
        # 爪子轮廓是**一次描好、一直有效**的常数（相机和两爪同在一个刚体上）。
        # 没描过、或者画幅对不上，`_jaw_shape` 会当场抛 —— 这里只记下来，不当场死，
        # 因为「没轮廓」时面板照样能手动走关节。
        try:
            self.jaw = PR._jaw_shape(self.cam)
            self.jaw_why = ""
        except RuntimeError as exc:
            self.jaw, self.jaw_why = None, str(exc)
            self._log(f"！爪轮廓读不出来：{exc}")
        self._log(f"串口 {self.port}   腕相机 index {self.cam_index} "
                  f"({self.cam.size()[0]}x{self.cam.size()[1]})   认的颜色 {self.color}")

        self._snap = self._fresh_snapshot()
        self._phase = None
        try:
            from grasp_check import load_thresholds                  # noqa: PLC0415

            self.cal = load_thresholds()
        except Exception as exc:                                    # noqa: BLE001
            self.cal = {}
            self._log(f"！判据阈值读不出来（{type(exc).__name__}: {exc}），"
                      f"夹爪那一档先不判")
