"""仿真机械臂 + 真机那几样读数：真机技能跑在仿真上时用它。

**为什么要有这个文件。** `skills/pick_real.py` 是按真机写的，它调的东西
`impls/arm_sim.py` 的 `SimArm` 上没有：

    arm.ctrl                6 号舵机的读写（`servo_bridge` / `grasp_check` 要的）
    arm.read_ticks()        六个关节此刻的原始位置
    arm._ticks_to_qpos()    原始位置 → 关节角
    arm._qpos_to_ticks()    关节角 → 原始位置
    arm.write_ticks()       把目标关节角写下去（= 发位置指令）
    arm.model / .data / ._fk()   `real/sweep_look.py` 算抬头姿态要的

**仿真里这几样都有对应物，只是没人接上 —— 这不是伪造，是换单位报同一件事：**

    6 号舵机角度   ⟷ 夹爪关节角（`data.qpos[gripper]`）。都是「此刻爪子开多大」。
    舵机原始值     ⟷ 用真机那份标定换算出来的关节位置（下面那对函数）。
    位置指令       ⟷ 模型的六个 `<position>` 执行器。把目标角写进 `data.ctrl`
                     再推进物理，就是「舵机收到目标位置、自己转过去」。
                     它们 `forcerange` 是 ±3.35 N·m，**夹在东西上照样转不到底** ——
                     真机靠这个差值判「夹住了没」，这里一模一样。

**换算是真机那一对函数**（`so101_nexus.lerobot_adapter.normalization`），标定也是真机
那份（`assets/lerobot_calibration/little_little_pig.json`）。所以 `real_home.RAW_HOME`
那串真机舵机原始值在仿真里解出来的姿态，跟真机上走到的是同一个姿态。

**一个字没改 `impls/arm_sim.py`。** 这里全部是加出来的。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from so101_nexus.config import SO101_JOINT_NAMES, SO101_TCP_SITE_NAME  # noqa: E402
from so101_nexus.kinematics import (                                    # noqa: E402
    EE_IK_DAMPING, EE_IK_ITERATIONS, ee_ik_delta_q)
from so101_nexus.lerobot_adapter.normalization import (                # noqa: E402
    motor_ticks_to_sim_rad, sim_rad_to_motor_ticks)

from core.registry import register                                     # noqa: E402
from impls.arm_sim import SimArm                                       # noqa: E402
from real.arm import (CAL_JSON, GRIP_OPEN_DEG, OW, SERVO_TO_JOINT_REV,  # noqa: E402
                      load_calibration)
from servo_bridge import SERVO_TO_JOINT                                # noqa: E402

JOINT_ORDER = tuple(SO101_JOINT_NAMES)
TCP_SITE = SO101_TCP_SITE_NAME
GRIPPER_ID = 6
# 逆解迭代几次。跟 `real/arm.py:IK_ITERS` 同一个来源（nexus 的那个常数），不另立一份。
IK_ITERS = EE_IK_ITERATIONS

# 舵机角 0-180° 那根刻度是怎么来的：`servo_control_standalone.set_servo_angle` 里
# `position = 1024 + angle * 2048 / 180`，读回来 `angle = (position - 1024) * 180 / 2048`。
# 这两个数是**驱动里的**，不在这儿另立一份口径。
TICK_AT_0DEG = 1024
TICK_AT_180DEG = 3072

# 合爪 / 张爪要推进多少步才算「走到底」。真机那边是 `grasp_check.measure_empty`
# 里的 `time.sleep(0.8)` —— 仿真没有墙钟可等，只能让步进把时间推出来。
# 50Hz × 80 步 = 1.6 秒，比真机那 0.8 秒宽一倍；这个上限够不着的时候由 `tol` 收。
GRIP_SETTLE_STEPS = 80
GRIP_SETTLE_TOL = 0.01          # 弧度


def _deg_to_tick(deg: float) -> int:
    """舵机角（0-180°）→ 原始值。跟驱动里那条线性式逐字一样，含两端夹紧。"""
    pos = int(TICK_AT_0DEG + float(deg) * (TICK_AT_180DEG - TICK_AT_0DEG) / 180.0)
    return max(TICK_AT_0DEG, min(TICK_AT_180DEG, pos))


def _tick_to_deg(tick: float) -> float:
    """原始值 → 舵机角（0-180°）。"""
    return (float(tick) - TICK_AT_0DEG) * 180.0 / (TICK_AT_180DEG - TICK_AT_0DEG)


class _SimGripperCtrl:
    """`servo_control_standalone` 的一点替身：**只认 6 号**。

    `servo_bridge.close_gripper` / `open_gripper` / `read_gripper_deg` 只用到
    `set_servo_angle(id, deg, speed)` 和 `get_servo_angle(id)` 这两个方法，所以这里
    就只接这两个。接多了会让人以为仿真里真有串口。
    """

    def __init__(self, arm: "SimRealArm"):
        self.arm = arm

    def set_servo_angle(self, servo_id, angle_deg, speed=0) -> float:
        self._only_gripper(servo_id)
        return self.arm.goto_grip_deg(float(angle_deg))

    def get_servo_angle(self, servo_id) -> float:
        self._only_gripper(servo_id)
        return self.arm.read_grip_deg()

    def close(self) -> None:
        pass

    @staticmethod
    def _only_gripper(servo_id) -> None:
        if int(servo_id) != GRIPPER_ID:
            raise RuntimeError(
                f"仿真这一支只接了 {GRIPPER_ID} 号（夹爪）—— 抓取判据只读它一个。"
                f"收到的是 {servo_id} 号，别的舵机在这儿没有对应物。")


@register("arm", "sim_real")
class SimRealArm(SimArm):
    """`SimArm` 加一层「真机读数」。接口跟 `RealArm` 对齐，技能看不出区别。"""

    name = "sim_real"

    def __init__(self, **kw):
        super().__init__(**kw)
        mujoco = self._mujoco
        u = self.u

        self.model = u.model
        # `_fk` 用的草稿。**不能算在 `u.data` 上**：那会把正在跑的这一局挪走。
        self.data = mujoco.MjData(u.model)
        self.site = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        if self.site < 0:
            raise RuntimeError(f"模型里没有 site {TCP_SITE!r} —— 换模型了？")

        self._qadr = {}
        self._jid = {}
        for name in JOINT_ORDER:
            jid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"模型里没有关节 {name!r}")
            self._qadr[name] = int(u.model.jnt_qposadr[jid])
            self._jid[name] = int(jid)

        # 夹爪的关节行程。`ctrl` 那一头要以它夹紧 —— 舵机角 0° 落在行程外面
        # （见下面 `grip_q_from_deg` 的注释），不夹的话目标角会跑到关节限位外面。
        # `jnt_range` 是按**关节 id** 索引的，`_qadr` 是 qpos 地址，两个不是一回事。
        grip_jid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        self.grip_lo, self.grip_hi = (float(v) for v in u.model.jnt_range[grip_jid])
        # 逆解要把关节夹进行程里（`real/arm.py::_ik` 同一步）：六个关节一起收。
        self.q_lo = np.array([float(u.model.jnt_range[self._jid[n]][0])
                              for n in JOINT_ORDER], dtype=np.float64)
        self.q_hi = np.array([float(u.model.jnt_range[self._jid[n]][1])
                              for n in JOINT_ORDER], dtype=np.float64)
        # 逆解里姿态误差的权重。跟 `RealArm.ow` 同一个默认值（`real/arm.py` 的 `OW`），
        # `skills/pick_real.py` 进场那一段会临时改它。
        self.ow = OW

        self.cal = load_calibration(CAL_JSON)
        # **20° 张开上限对应的刻数**（和 `RealArm.grip_open_tick` 同名同口径）。
        # 2026-10-21 用户定：不管什么时候张开都不许超过 20°，仿真这一支跟着真机走。
        _q_open = np.zeros(6, dtype=np.float64)
        _q_open[5] = float(np.clip(np.radians(GRIP_OPEN_DEG),
                                   self.grip_lo, self.grip_hi))
        self._grip_open_tick = int(self._qpos_to_ticks(_q_open)[GRIPPER_ID])
        # 缓存口径跟 `RealArm.read_ticks` 一样：写下去就作废。区别是仿真里写完那一步
        # 就已经走到位了（`_write_ctrl` 当场推进物理），所以读是白拿的。
        self._dirty = True
        self.last_ticks: dict[int, int] = {}
        self.n_steps = 0
        # 技能自己填的两个数（`pick_real` 第 0 段量完挂在这儿，判官再取）
        self.empty_deg = None
        self.open_deg = None
        self.ctrl = _SimGripperCtrl(self)

    # ---- 关节原始值：`real/joint_move.py` 要的四样 ----

    def _qpos_now(self) -> np.ndarray:
        d = self.u.data
        return np.array([d.qpos[self._qadr[n]] for n in JOINT_ORDER], dtype=np.float64)

    def read_ticks(self) -> dict[int, int]:
        """六个关节此刻的原始位置（`{1..6: 0-4095}`）。"""
        if not self._dirty and self.last_ticks:
            return self.last_ticks
        self.last_ticks = self._qpos_to_ticks(self._qpos_now())
        self._dirty = False
        return self.last_ticks

    def _ticks_to_qpos(self, ticks) -> np.ndarray:
        by_name = {SERVO_TO_JOINT[int(k)]: int(v) for k, v in ticks.items()}
        return motor_ticks_to_sim_rad(by_name, calibration=self.cal,
                                      gripper_limits_rad=(self.grip_lo, self.grip_hi))

    def _qpos_to_ticks(self, qpos) -> dict[int, int]:
        by_name = sim_rad_to_motor_ticks(np.asarray(qpos, dtype=np.float64),
                                         calibration=self.cal,
                                         gripper_limits_rad=(self.grip_lo, self.grip_hi))
        return {SERVO_TO_JOINT_REV[name]: int(v) for name, v in by_name.items()}

    def write_ticks(self, ticks) -> None:
        """把目标关节角写下去 —— 就是「发位置指令」。

        真机那边是写六个舵机的目标位置、舵机自己转过去。仿真这边模型的六个
        `<position>` 执行器就是那六个伺服，写 `data.ctrl` 再推进物理是同一件事。

        **6 号在这里硬夹到 20° 上限**（跟 `RealArm.write_ticks` 同一条规矩）。
        """
        ticks = {int(k): int(v) for k, v in ticks.items()}
        t6 = ticks.get(GRIPPER_ID)
        if t6 is not None and t6 > self._grip_open_tick:
            ticks[GRIPPER_ID] = self._grip_open_tick
        self._write_ctrl(self._ticks_to_qpos(ticks))
        self._dirty = True
        self.n_steps += 1

    def _write_ctrl(self, q6) -> None:
        """把六个关节的目标角写进执行器、推进一个控制间隔、再新出一帧观测。

        **走环境自己的两条缝**（`_actuator_ids` / `_advance_physics`），不直接
        `mj_step` —— 直接步进会把环境的计数器和新旧帧甩开，`_observe` 出来的
        TCP 就是陈的。
        """
        u = self.u
        u.data.ctrl[u._actuator_ids] = np.asarray(q6, dtype=np.float64)
        u._advance_physics()
        self.obs, self.info = u._observe()

    # ---- 6 号舵机：`arm.ctrl` 那一头 ----

    def grip_q_from_deg(self, deg: float) -> float:
        """舵机角（0-180°）→ 夹爪关节角。

        **走真机那份标定**：先按驱动那条线性式换成原始值，再用
        `motor_ticks_to_sim_rad` 的夹爪那一支换关节角。结果是 0° 落在关节行程**外面**
        （真机也一样：舵机 0° 对应的位置比夹爪能合到的还紧，所以是顶死在那儿）。
        所以调用方要把它夹进 `[grip_lo, grip_hi]`。
        """
        ticks = dict(self.read_ticks())
        ticks[GRIPPER_ID] = _deg_to_tick(deg)
        return float(self._ticks_to_qpos(ticks)[5])

    def read_grip_deg(self) -> float:
        """此刻的 6 号舵机角（度）。**读的是关节实际位置**，不是发出去的指令 ——
        夹着东西合不拢时这里就是一个比 `empty_deg` 大的数，判据全靠这个差。"""
        return _tick_to_deg(self._qpos_to_ticks(self._qpos_now())[GRIPPER_ID])

    @property
    def grip_open(self) -> float:
        """张开指令角（弧度）。和 `RealArm.grip_open` 同一个上限：20°。"""
        return float(np.clip(np.radians(GRIP_OPEN_DEG),
                             self.grip_lo, self.grip_hi))

    @property
    def grip_open_tick(self) -> int:
        """20° 上限对应的 6 号刻数（`measure_open` 要它）。"""
        return int(self._grip_open_tick)

    def goto_grip_deg(self, deg: float, max_steps: int = GRIP_SETTLE_STEPS,
                      tol: float = GRIP_SETTLE_TOL) -> float:
        """把爪子开到 `deg`，等它停稳再报角度。其余五个关节原地不动。

        真机是「发一条目标位置 + `time.sleep(0.8)`」；仿真没有别的线程在推进物理，
        所以这里自己推进到停稳（或者撞上步数上限）。返回停稳后读到的角度。
        """
        q_t = float(np.clip(self.grip_q_from_deg(deg), self.grip_lo, self.grip_hi))
        q = self._qpos_now()
        q[5] = q_t
        for _ in range(int(max_steps)):
            self._write_ctrl(q)
            if abs(float(self.u.data.qpos[self._qadr["gripper"]]) - q_t) <= tol:
                break
        self._dirty = True
        return self.read_grip_deg()

    # ---- 正解：`real/sweep_look.poses` 要的 ----

    def _fk(self, qpos) -> tuple[np.ndarray, np.ndarray]:
        """关节角 → (TCP 位置, 姿态四元数 [w,x,y,z])。算在草稿 `data` 上。

        **不用补 `RealArm` 那个 `TCP_DELTA_BODY` 换算。** 那个差是「官方 MJCF 的
        `gripperframe` vs nexus 的 `gripperframe`」—— 仿真本来就在 nexus 这边，
        读到的就是那一套。
        """
        mujoco = self._mujoco
        for i, name in enumerate(JOINT_ORDER):
            self.data.qpos[self._qadr[name]] = float(qpos[i])
        mujoco.mj_kinematics(self.model, self.data)
        pos = np.array(self.data.site_xpos[self.site], dtype=np.float64)
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site])
        return pos, quat

    # ---- TCP 口径 ↔ 正解口径：仿真里是恒等 ----

    def _sim_pose(self, pos, quat):
        """(正解口径) → (对外报的 TCP 口径)。

        `RealArm` 这一步要补 `TCP_DELTA_BODY`（官方 MJCF 的 `gripperframe` 和 nexus 的
        `gripperframe` 差 19.9mm + 90°）。仿真本来就在 nexus 这边读的，两个口径是同一
        个东西，所以这里是恒等 —— 跟 `_fk` 那句注释一个道理。
        """
        return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)

    _site_pose = _sim_pose

    def _jac(self, qpos) -> np.ndarray:
        """TCP 的 6×6 雅可比（三行平移、三行旋转），列按 `JOINT_ORDER`。

        **`mj_kinematics` 之后必须再调一次 `mj_comPos`。** `mj_jacSite` 要用子树质心
        数据，只调正解那一块是陈的，整块雅可比会是零 —— 逆解看着收敛了，其实报的是
        原样。`real/arm.py::_jac` 有同一句。算在草稿 `data` 上，真身不动。
        """
        mujoco = self._mujoco
        self._fk(qpos)
        mujoco.mj_comPos(self.model, self.data)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
        cols = [self.model.jnt_dofadr[self._jid[n]] for n in JOINT_ORDER]
        return np.vstack([jacp[:, cols], jacr[:, cols]])

    def _ik(self, target_pos, target_quat, qpos0, iters: int = IK_ITERS,
            ow: float | None = None) -> np.ndarray:
        """阻尼最小二乘迭代到那个位姿。算法逐字照抄 `real/arm.py::_ik`。

        参数含义、为什么只迭代几次、为什么要把关节夹进行程，全在那边写着 —— 这一份
        不另立解释，免得两边说法打架。**唯一的差别是解的模型**：那边解官方 MJCF，
        这边直接解 nexus 的模型（本来就跑在它上面）。
        """
        # 夹取上下限和 `real/arm.py` 同一条：硬件行程再叠安全下限（2026-10-25）。
        from real import safety                                      # noqa: PLC0415
        lo, hi = safety.clip_bounds(self.q_lo, self.q_hi)
        q = np.array(qpos0, dtype=np.float64, copy=True)
        for _ in range(int(iters)):
            pos, quat = self._fk(q)
            jac = self._jac(q)
            dq = ee_ik_delta_q(jac, pos, quat, np.asarray(target_pos, dtype=np.float64),
                               np.asarray(target_quat, dtype=np.float64),
                               orientation_weight=(self.ow if ow is None else float(ow)),
                               damping=EE_IK_DAMPING)
            q = np.clip(q + dq, lo, hi)
        return q

    # ---- 夹爪的两样：`pick_real` 的落点算式要的 ----

    def fixed_face_reach(self, axis, side: float = 1.0, z_band=None, q=None) -> float:
        """`SimArm.fixed_face_reach(axis, side)` —— 接口对齐版，只多了两个参数。

        真机那一份要 `z_band`（整根固定指是上宽下窄的楔子，不过滤高度带会量到最上面
        那一截）。仿真没有这个问题：它的指垫球就钉在夹持高度上，所以这两个参数在这儿
        没有用武之地，收下不动。`q` 同理（真机那份可以在草稿姿态上算）。
        """
        return float(super().fixed_face_reach(axis, side))

    def jaw_gap_center(self, q=None) -> np.ndarray:
        """两组指垫中点的中点（世界系）—— 合爪时物体该待的那一点。

        `tcp()` 报的是固定爪尖，两者差着半个开口。`SimArm` 只给了固定爪那一个
        （`fixed_tip_pos`），活动爪那一半在这儿补上。`q` 参数是跟 `RealArm` 对齐用
        的，调用方（`skills/pick_real.py`）不传。
        """
        d = self.u.data
        fixed = np.asarray(d.geom_xpos[self.fix_tip], dtype=np.float64).mean(axis=0)
        moving = np.asarray(d.geom_xpos[self.mov_tip], dtype=np.float64).mean(axis=0)
        return (fixed + moving) / 2.0
