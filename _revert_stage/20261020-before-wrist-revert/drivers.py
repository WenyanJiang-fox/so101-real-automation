"""驱动器：一轮里的动作是谁产生的。

两个实现对外一个接口：`run_episode(backend, layout, trace, on_frame) -> trace | None`，
返回 None 表示这一轮规划失败（方块够不着之类），不是失败，直接跳过重摆。
`on_frame` 每物理步回调一次，采集时靠它按 30fps 降频存帧。
"""

from __future__ import annotations

import sys

from backend import SIM_DIR


class ExpertDriver:
    """脚本专家。用它跑出来的都是成功示范，用来验证循环本身。"""

    name = "expert"

    def __init__(self, backend):
        if str(SIM_DIR) not in sys.path:
            sys.path.insert(0, str(SIM_DIR))
        from expert import Expert  # noqa: E402

        # 和 SO101SimEnv 共用同一份仿真状态 —— 不共用的话 expert 操作的是另一个
        # 场景，主循环看到的方块位置永远不动（collect_demos.py 也是这么接的）。
        self.expert = Expert(controller=backend.raw.c)

    def run_episode(self, backend, layout, trace, on_frame=None):
        return self.expert.run_episode(cube_xy=layout.xy, cube_yaw=layout.yaw,
                                       on_frame=on_frame, trace=trace)


class PolicyDriver:
    """SmolVLA 策略驱动器。加载 checkpoint，一步一个动作。

    还没写 —— 第一版先只验 expert 的连续跑。写的时候照抄 sim_rollout.py 的
    to_batch/load_policy/select_action：动作块要 policy.reset() 清队列，
    n_frames 的单位是 30fps 帧不是物理步。
    """

    name = "policy"

    def __init__(self, backend, ckpt=None):
        raise NotImplementedError("策略驱动器还没写，先用 --driver expert 验循环")
