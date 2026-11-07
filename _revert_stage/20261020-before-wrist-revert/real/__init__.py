"""真机那一半。

**和仿真那条线完全隔开。** 上级目录里的 `run_sim.py`、`skills/`、`impls/arm_sim.py`、
`impls/cam_sim.py` 一个字都不动 —— 仿真照跑。本目录只往上借 `core/`（六个接口 +
注册表 + 参数），别的什么都不碰。

为什么要分开、而不是在 `run_sim.py` 里加开关：`run_sim.py` 和 `skills/` 里有十几处
**没加守卫**的仿真专属调用（`arm.cube_pos`、`arm.n_cubes`、`arm.finger_cube_contacts`、
`arm.terminated`、`arm.last_action` …）。想让同一条主循环也跑真机，就得给那十几处
全加上 `hasattr` 守卫 —— 那是在改一条**已经在跑的仿真流水线**，风险不对等。

所以真机走自己的入口：相机 → 感知 → 机械臂，这条链自己拼。
`core/registry.py` 是共享的，一边注册 `sim_wrist`、一边注册 `real_wrist`，互不知道。
"""
