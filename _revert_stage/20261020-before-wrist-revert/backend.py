"""后端：harness 主循环唯一认识的「机械臂 + 相机」接口。

仿真这一版是 so101-smolvla-sim/sim_env.py 的薄封装，不重写、不改那边任何文件。
真机版以后用 servo_bridge + 两个相机在 RealBackend 里补，主循环不用动。

**这一层是硬边界**：主循环不许直接摸 MuJoCo 或串口。真机接上来时只换后端。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from layout import CUBE_RGBA, Layout

SIM_DIR = Path(r"G:\sim2real\so101-smolvla-sim")
if str(SIM_DIR) not in sys.path:
    # paths.py 必须在别的 import 之前进来（它设 HF_ENDPOINT 等环境变量）
    sys.path.insert(0, str(SIM_DIR))


class SimBackend:
    """MuJoCo 仿真后端。"""

    def __init__(self, render: bool = True):
        import sim_env  # noqa: E402  延迟导入：装环境变量、也只在这一层依赖仿真

        self._sim_env = sim_env
        self.env = sim_env.SO101SimEnv(render=render)
        # 离屏 renderer 只有在同一线程里创建 + 使用才正常（Windows WGL）。
        # 这里先渲染一次把 renderer 建起来，主循环全程单线程，不会跨线程。
        self.env.observation()

        # 方块是单个 geom、没挂材质（matid=-1），所以改 geom_rgba 就是换色。
        # 按 body 找而不是记死 id，XML 改了也不会错位。
        import mujoco  # noqa: E402

        bid = self.env.cube_b
        hits = [g for g in range(self.env.m.ngeom)
                if self.env.m.geom_bodyid[g] == bid]
        if len(hits) != 1:
            raise RuntimeError(f"方块 body 下有 {len(hits)} 个 geom，换色逻辑要重写")
        self._cube_geom = hits[0]
        # 30fps 一帧 = 多少个物理步。on_frame 是按物理步回调的，采样要按这个降频
        self.steps_per_frame = int(sim_env.STEPS_PER_FRAME)

    # ---- 一轮的生命周期 ----

    def apply_layout(self, layout: Layout) -> None:
        """把布局写进场景。现在只有颜色是仿真独有的副作用，位置/偏航由
        driver 在 run_episode 里摆（expert 会自己 reset_scene）。"""
        self.env.m.geom_rgba[self._cube_geom] = CUBE_RGBA[layout.color]

    # ---- 状态 ----

    def gripper_deg(self) -> float:
        """6 号舵机角度，0-180，和真机 get_servo_angle 同口径。"""
        return float(self.env.state_deg()[5])

    def servo_from_raw_joint_deg(self, raw_deg: float) -> float:
        """expert 的 trace 存的是关节原始角（度），不是舵机 0-180。

        爪关节 range 是 [-10°, 100°]，直接拿去比阈值会差一大截。
        这里按 deg_to_qpos 的同一段线性映射倒回来，模块 A 才有可比性。
        """
        lo, hi = self.env.grip_range
        return float((np.radians(raw_deg) - lo) / (hi - lo) * 180.0)

    def state_deg(self) -> np.ndarray:
        return self.env.state_deg()

    def observation(self):
        return self.env.observation()

    def is_success(self) -> bool:
        return bool(self.env.is_success())

    def n_contacts(self) -> int:
        return int(self.env.n_cube_contacts())

    @property
    def raw(self):
        """给驱动器用的逃生舱（expert 需要 env.c 这个 MuJoCo 控制器）。"""
        return self.env


class RealBackend:
    """真机后端。现在只有骨架 —— 等硬件接上再填，接口和 SimBackend 必须一致。"""

    def __init__(self, port: str = "COM12"):
        raise NotImplementedError(
            "真机后端还没写。要接的部件：servo_bridge.open_controller（6 号读角度）"
            "+ 腕部/全局两路相机取帧。填完之前 harness 只能跑 --backend sim。"
        )
