"""真机机械臂。对外接口和 `impls/arm_sim.py` 的 `SimArm` 逐字对齐。

主循环（`core/follow.py`）和技能调的是这几个：`tcp()` / `quat()` / `tool_x()` /
`drive(绝对坐标, closing=, grip=, pos_slew=, rot_slew=, brake=)` / `grip_q()` /
`reach_residual()` / `terminated` / `on_step`。仿真那边怎么用，这边就怎么接，
**技能和主循环一个字都不用改**。

**正逆解去哪来的**：本目录不算运动学。正解是把关节角塞进官方 MJCF
（`SO-ARM100/Simulation/SO101/so101_new_calib.xml`）跑 `mj_kinematics`，读里面那个
叫 `gripperframe` 的 site。**但对外报的 TCP 要补一个换算**：仿真用的不是这个文件，
是 nexus 的 menagerie 模型，同一张 `gripperframe` 牌子下的点在两个文件里差 19.9mm
和 90°。`tcp()` / `quat()` / `tool_x()` / `drive()` 一律报**仿真那一套**，底下的
`_fk` / `_jac` / `_ik` 仍是官方 site 原样（手眼标定和 `real/servo.py` 都用那一套）。
来处见下面 `TCP_DELTA_BODY` 那一段。逆解调 `so101_nexus.kinematics.ee_ik_delta_q`
（阻尼最小二乘），不自己推雅可比。

**舵机原始值 ↔ 关节角**：`so101_nexus.lerobot_adapter.normalization` 里的
`sim_rad_to_motor_ticks` / `motor_ticks_to_sim_rad`，标定来自
`assets/lerobot_calibration/little_little_pig.json`。**这是仿真自己在用的那一对函数**，
不是照着重写的 —— 两边差一点点，仿真的高度参数（`LIFT_Z` 那些）在真机上就全错位。
换算已验：六个关节中位 raw 进去，出来是 0 rad，往返零误差。

**为什么中位是 0**：`so101_new_calib.xml` 的关节零点是行程中点，而 `mid` 取的是
标定表里 `range_min/range_max` 的中点 —— 两边是同一个口径，所以对得上。
（另一份 `so101_old_calib.xml` 零点是水平伸展，**不能换**。）

**没写的东西**：仿真那些靠真值的查询（`cube_pos` / `n_cubes` /
`finger_cube_contacts` / `bump_scan` / `dry_run`）真机上一概没有，也不假装有 ——
技能里凡是调它们的，在这一路上都得换写法。`contacts()` 按接口约定返回 0。
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco                                                     # noqa: E402
from lerobot.motors import MotorCalibration                       # noqa: E402
from so101_nexus.kinematics import (EE_DELTA_ACTION_SCALE,        # noqa: E402
                                    EE_IK_DAMPING, EE_IK_ITERATIONS,
                                    ee_ik_delta_q, rotvec_to_quat,
                                    quat_multiply)
from so101_nexus.lerobot_adapter.normalization import (           # noqa: E402
    motor_ticks_to_sim_rad, sim_rad_to_motor_ticks)

from core.geom import align_action                                 # noqa: E402
from core.interfaces import Arm                                   # noqa: E402
from core.registry import register                                # noqa: E402
from servo_bridge import (GRIPPER_ID, GRIPPER_MAX_OPEN_DEG,      # noqa: E402
                          SERVO_TO_JOINT, open_controller)

MJCF = Path(r"G:\sim2real\SO-ARM100\Simulation\SO101\so101_new_calib.xml")
CAL_JSON = _ROOT / "assets" / "lerobot_calibration" / "little_little_pig.json"
TCP_SITE = "gripperframe"

# ---- 对外报的 TCP 口径：照仿真那一套 ----
# 仿真跑的不是上面这个文件，是 nexus 的 menagerie 模型。`gripper` body 在两边逐字节
# 相同（pos/quat 一样），但挂在它下面的 `gripperframe` site 被 nexus 改过：
#   官方 so101_new_calib.xml:113  pos="-0.0079 -0.000218 -0.0981274" quat="0.707107 0 0.707107 0"
#   menagerie so101.xml:216       pos=" 0.012  -0.000218 -0.098127"  quat="0 0 1 0"
# 位置差 19.9mm（沿 body x，就是「固定爪面 → 两爪正中」那一段），姿态差一次绕 y 的 90°。
# **仿真整套代码都是照 menagerie 那套写的**：`core/params.py` 说「竖直向下的工具轴」、
# `core/geom.face_normal` 说工具 x 是开合方向、`align_action` 的 `z_des=[0,0,-1]` 是
# 手指朝下。官方文件里这些轴正好转了 90°——工具 x 是手指方向、工具 z 才是开合轴。
# 所以 `tcp()/quat()/tool_x()/drive()` 对外一律补上这个差，仿真的技能和跟随循环
# 才能一个字不改地跑。**底下的 `_fk`/`_jac`/`_ik` 仍是官方 site 原样** ——
# `real/cameras.py` 的手眼标定、`calibrate_handeye.py`、`real/servo.py` 都是照
# 官方 site 标/写的，那几处一个字节都不能动。
#
# 这两个数是**两个文件的差值**，不是量出来的、也不是调出来的。
#
# ---- 2026-10-14：位置那一半换成「固定爪尖」----
# 用户定：下探/进场/纠位的「对准」基准点用固定爪尖。原来那个点比爪尖多伸出去 13.7mm
# （初始位置、下探到底、纠位三个姿态都是 13.7mm，结构定值）。所以「把 TCP 对准盒子中心
# 下探」实际是让爪尖停在盒子中心上方 13.7mm —— run_21 三次合爪全空。
# 新值由 `_probe_tipdelta.py` 现算（`d = R_site^T @ (爪尖 − 官方 site)`，三个姿态结果
# 一致，长度 6.4mm）。爪尖体坐标（`gripper` body）[-0.0089, -0.0002, -0.1044]。
# **姿态那一半（Y90）不动**：爪尖没有朝向，仿真要的「工具 z 朝下、工具 x 是开合方向」
# 还得照 menagerie 那套补。
TCP_DELTA_BODY = np.array([6.273e-03, 1.8e-05, -1.0e-03])
TCP_QUAT_Y90 = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])       # wxyz
TCP_QUAT_Y90_INV = np.array([np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0])  # wxyz

# ---- 两个爪尖：体坐标定值（米），**在就是上面这个模型上量准的** ----
# 见 memory `so101-fingertip-points`。固定爪挂在 `gripper` 体上（不随 6 号关节动），
# 动爪就是 `moving_jaw_so101_v1` 这个体本身。这个定值在这一版模型上复现得出
# 「合到底两尖相距 6.5mm、张到最大 141.1mm」。**别拿 `geom_size` 去缩放推** ——
# 对 mesh geom 那是包围盒尺寸，推出来是看着差不多的假数。
TIP_FIXED_BODY = "gripper"
TIP_MOVING_BODY = "moving_jaw_so101_v1"
TIP_FIXED = np.array([-0.0089, -0.0002, -0.1044])
TIP_MOVING = np.array([-0.0087, -0.0820, 0.0189])

#: **夹爪张开上限（qpos 度）。2026-10-21 用户定：任何时候张开都不许超过 20°。**
#: 这是硬上限，不是「建议值」：`RealArm.write_ticks` 会把 6 号的刻数夹到
#: `grip_open_tick`，所有绕过 `write_ticks` 的开爪（`measure_open`）也必须用它。
#: `grip_hi`（MJCF 行程上限 100.0°）只留给读数换算和夹取范围的钳位。
GRIP_OPEN_DEG = float(GRIPPER_MAX_OPEN_DEG)

# 动作单位 → 物理量。**这三个数不自己定，跟 nexus 的 `EE_DELTA_ACTION_SCALE` 取齐**：
# 位置 0.02 米/单位（= core.params.POS_SCALE）、姿态 0.1 弧度/单位、夹爪 0.2 弧度/单位。
# 取成一样的，仿真里调好的那些速度档（SPEED/SLOW/CARRY/GENTLE…）在真机上才是同一个意思。
POS_SCALE = float(EE_DELTA_ACTION_SCALE[0])
ROT_SCALE = float(EE_DELTA_ACTION_SCALE[3])
GRIP_SCALE = float(EE_DELTA_ACTION_SCALE[6])

# 和 `impls/arm_sim.py` 的两个档取一样
SPEED_POS = 0.08
SPEED_ROT = 0.15
# 姿态误差在逆解里的权重。**nexus 默认 0.01 太轻，工具轴压不下去**（arm_sim.py:33
# 的 OW 就是为这个设的），真机沿用同一档，免得两边走出不同姿态。
OW = 0.05

# 逆解迭代几次。**和 nexus 的 `_solve_ee_ik` 取同一个数**（3 次），不自己定：
# 阻尼最小二乘本身就是一个牛顿步，够得着的目标两三下就到位；真正让手臂「一步一步
# 挪过去」的是外层的控制循环，不是这里。**不要加「单次增量上限」**：加了之后
# 3 次走不完，`reach_residual` 会把够得着的点报成够不着，`follow` 就地不动。
IK_ITERS = EE_IK_ITERATIONS

JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper")
SERVO_IDS = (1, 2, 3, 4, 5, 6)

# **总线上的每一次读写都要串起来**（2026-10-18）。面板现在有两个线程会碰到这条臂：
# 硬件线程（刷读数、走轨迹）和感知线程（`cam.mount()` 要读一次舵机才知道相机在哪）。
# SCS 协议是一问一答，两份读同时进去就互相把包吃掉 —— 实测现象是每两眼有一眼
# `mount()` 返回 None（「没做相机标定」）。读和写都进这把锁，事务本身只有
# 21~66ms，串起来不影响原来的节拍。
_BUS = threading.RLock()


def load_calibration(path: Path = CAL_JSON) -> dict:
    """读 lerobot 标定文件，转成 nexus 要的 `MotorCalibration`。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = set(JOINT_ORDER) - set(raw)
    if missing:
        raise RuntimeError(f"标定文件缺关节：{sorted(missing)}（文件 {path}）")
    return {name: MotorCalibration(id=int(raw[name]["id"]),
                                   drive_mode=int(raw[name]["drive_mode"]),
                                   homing_offset=int(raw[name]["homing_offset"]),
                                   range_min=int(raw[name]["range_min"]),
                                   range_max=int(raw[name]["range_max"]))
            for name in JOINT_ORDER}


def load_model(path: Path = MJCF):
    """建正逆解用的模型。**只当运动学模型用** —— 不 `mj_step`、不读接触、不渲染。

    就算这样也要把整份 MJCF 装进来：`gripperframe` 这个 site 的位姿是相对 `gripper`
    body 定的，脱离模型没法算。
    """
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    if site < 0:
        raise RuntimeError(f"{path} 里没有 site {TCP_SITE!r} —— 换文件了？")
    # 关节按名字找地址，不写死 0..5：模型里还挂着别的关节（backlash 那类）时，
    # 写死下标会静默错位。
    adr = {}
    for name in JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"{path} 里没有关节 {name!r}")
        adr[name] = int(model.jnt_qposadr[jid])
    return model, data, site, adr


def joint_limits(model) -> dict[str, tuple[float, float]]:
    """六个关节各自的行程（弧度）。**从 MJCF 自己的关节 range 读**，不另抄一份数。

    逆解必须照着它夹：不夹的话逆解会给出「数学上最接近、但机械上转不过去」的角，
    再换算成舵机原始值就落到 [0,4095] 外面（实测 `wrist_roll` 跑到 -176.9°，
    超出它自己的 -157.2°~162.8°）。
    """
    out = {}
    for name in JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if not bool(model.jnt_limited[jid]):
            raise RuntimeError(f"{name} 在 MJCF 里没有限位（jnt_limited=0），"
                               "逆解就没得夹了，先去模型里把 range 补上")
        out[name] = (float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1]))
    return out


@register("arm", "real")
class RealArm(Arm):
    """真机从臂。`ctrl` 传进来就用它（测试时可以塞假的），不传就自己开串口。"""

    name = "real"

    def __init__(self, port: str = "COM10", ctrl=None, speed: int = 0,
                 mjcf: Path = MJCF, cal_json: Path = CAL_JSON, debug: bool = False,
                 fast: bool = False):
        self.cal = load_calibration(cal_json)
        self.model, self.data, self.site, self.adr = load_model(mjcf)
        lim = joint_limits(self.model)
        self.q_lo = np.array([lim[n][0] for n in JOINT_ORDER], dtype=np.float64)
        self.q_hi = np.array([lim[n][1] for n in JOINT_ORDER], dtype=np.float64)
        self.grip_lo, self.grip_hi = lim["gripper"]

        self.port = port
        self.debug = bool(debug)
        # **速度档。默认 0 = 舵机自己最快。这个数没在真机上标过**，第一次上机
        # 要按实走的结果重定：太大跟不上、会甩，太小每步的位移追不上指令。
        self.speed = int(speed)
        # `fast=True` 只给 30fps 采集用（去掉驱动两处固定等待，读 65.8→1.3ms、
        # 写 20.9→0.48ms）。默认 False，面板和技能走的东西一个数都不变。
        self.fast = bool(fast)
        self.ctrl = (ctrl if ctrl is not None
                     else open_controller(port, fast=self.fast))
        # 读缓存。True = 手上的读数不作数，下一次读得真去总线问。见 `read_ticks`。
        self._dirty = True
        # **必须在下面那次 `grip_q()` 之前建好**（2026-10-21）：`grip_q` → `read_ticks`，
        # 某个舵机第一次就读不到时会走 `self.last_ticks.get(sid)` 这个兜底分支。原来这一行
        # 写在 `_grip_cmd` 后面，于是兜底分支自己抛 `AttributeError: ... no attribute
        # 'last_ticks'`，把真正的原因（某个舵机读不到）盖掉了。
        self.last_ticks: dict[int, int] = {}

        # **20° 上限对应的刻数**（按这份标定算一次，`write_ticks` 拿它当硬夹）。
        # 2026-10-21 用户定：任何时候张开都不许超过 20°，所以这一层必须拦在写之前。
        _q_open = np.zeros(6, dtype=np.float64)
        _q_open[5] = self.grip_open
        self._grip_open_tick = int(self._qpos_to_ticks(_q_open)[GRIPPER_ID])
        self._cap_hits = 0          # 6 号被夹回上限的次数（只在第一次打日志）

        # 和 SimArm 一样的两个速度状态
        self.vel = np.zeros(3, dtype=np.float64)
        self.rvel = np.zeros(3, dtype=np.float64)
        # 夹爪那一维是**增量**（跟 nexus 的 pd_ee_delta_pose 一致），所以自己攒一个
        # 「当前命令要去到的角」。起点按真机此刻的读数，不是 0。
        self._grip_cmd = float(self.grip_q())
        self.terminated = False
        self.n_steps = 0
        # 逆解里姿态误差的权重。默认就是模块级 `OW`（和 `arm_sim.py` 同一档）。
        # **进场那一段要临时调小**，见 `skills/pick_real.py` 的 `REAL_IK_OW`。
        self.ow = OW

    # ---- 串口那一层 ----

    def read_ticks(self) -> dict[int, int]:
        """读回六个舵机的原始位置。**有缓存**，见 `_dirty`。读不到的沿用上一次的值。

        **为什么要缓存**：这个驱动读一次要 66ms（`_read_packet` 里每个舵机先
        `time.sleep(0.01)`，见 `mcp_robot/sources/servo_control_standalone.py:412`），
        而 `core/follow.py` 每个控制步要问三次「手臂在哪」（循环开头 `arm.tcp()`、
        `drive` 里再读一次、收尾再 `tcp()`）。全走真读就是 3×66+21 = 219ms/步，
        实测只有 4.5 步/秒，走起来一顿一顿。

        缓存的口径是**「自上次写舵机之后读过了吗」**，不是按时间猜：
          * 没写过 -> 手上的读数就是手臂现在的位置，直接给（写目标位置不改变
            「手臂此刻在哪」这件事）。
          * 写过之后的头一次读 -> 真读。这一次要的就是「刚发的指令动没动」。

        所以每个控制步真读一次，`follow` 那三次读会合成一次。代价是反馈量比指令
        晚一个控制步 —— 但那本来就是观测的固有延迟，不是这里引入的。
        """
        with _BUS:
            if not self._dirty and self.last_ticks:
                return self.last_ticks
            got = self.ctrl.sync_read_positions(list(SERVO_IDS))
            out = {}
            for sid in SERVO_IDS:
                v = got.get(sid)
                if v is None:
                    v = self.last_ticks.get(sid)
                    if v is None:
                        raise RuntimeError(f"舵机 {sid} 读不到，而且没有上一次的值可沿用")
                out[sid] = int(v)
            self.last_ticks = out
            self._dirty = False
            return out

    def write_ticks(self, ticks: dict[int, int]) -> None:
        """写六个舵机的目标刻数。**6 号在这里硬夹到 20° 上限**（`grip_open_tick`）。

        2026-10-21 用户定「不管什么时候张开爪子，都不能大于 20°」。技能、轨迹、
        `follow`、面板全走这一条，所以夹在这里就是全局的；真被夹了会打一行字，
        不去悄悄改掉——看见这行说明上面有地方写了超限的张开目标。
        """
        ticks = {int(k): int(v) for k, v in ticks.items()}
        t6 = ticks.get(GRIPPER_ID)
        if t6 is not None and t6 > self._grip_open_tick:
            # **只报第一次。** 回初始位置那条轨迹里 6 号是一帧一帧往 1972 刻爬的，
            # 每帧都要夹一次；2026-10-21 实测刷了 15 行一样的字，看不出别的东西。
            self._cap_hits += 1
            if self._cap_hits == 1:
                print(f"！6 号张开超过 {GRIP_OPEN_DEG:.0f}° 上限：{t6} 刻 → 夹到 "
                      f"{self._grip_open_tick} 刻（这一趟后面再超限不再重复报）",
                      flush=True)
            ticks[GRIPPER_ID] = self._grip_open_tick
        with _BUS:
            self.ctrl.sync_set_positions(ticks, speed=self.speed)
            # 写了就作废缓存：下一次读得真去总线问，才看得见这一步有没有动。
            self._dirty = True

    def _ticks_to_qpos(self, ticks: dict[int, int]) -> np.ndarray:
        by_name = {SERVO_TO_JOINT[sid]: int(v) for sid, v in ticks.items()}
        return motor_ticks_to_sim_rad(by_name, calibration=self.cal,
                                      gripper_limits_rad=(self.grip_lo, self.grip_hi))

    def _qpos_to_ticks(self, qpos: np.ndarray) -> dict[int, int]:
        by_name = sim_rad_to_motor_ticks(np.asarray(qpos, dtype=np.float64),
                                         calibration=self.cal,
                                         gripper_limits_rad=(self.grip_lo, self.grip_hi))
        return {SERVO_TO_JOINT_REV[name]: int(v) for name, v in by_name.items()}

    # ---- 正解 ----

    def _fk(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """关节角 → (TCP 位置, 姿态四元数 [w,x,y,z])。"""
        for name in JOINT_ORDER:
            self.data.qpos[self.adr[name]] = float(qpos[JOINT_ORDER.index(name)])
        mujoco.mj_kinematics(self.model, self.data)
        pos = np.array(self.data.site_xpos[self.site], dtype=np.float64)
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site])
        return pos, quat

    def _jac(self, qpos: np.ndarray) -> np.ndarray:
        """TCP 的 6×6 雅可比（三行平移、三行旋转），列按 JOINT_ORDER。

        **`mj_kinematics` 之后必须再调一次 `mj_comPos`。** `mj_jacSite` 要用 subtree
        的质心数据，只调正解的话那一块是陈的 —— 实测漏掉这一步，整块雅可比全是零，
        逆解一步都不动（`_ik` 看着像收敛了，其实返回的是原样）。nexus 的
        `base_env.py:549` 有同一句注释，是照它抄的。
        """
        self._fk(qpos)
        mujoco.mj_comPos(self.model, self.data)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
        cols = [self.model.jnt_dofadr[self.adr[n]] for n in JOINT_ORDER]
        return np.vstack([jacp[:, cols], jacr[:, cols]])

    def _ik(self, target_pos, target_quat, qpos0, iters: int = IK_ITERS,
            ow: float | None = None) -> np.ndarray:
        """阻尼最小二乘迭代到那个位姿。够不着就给「最接近的」——和 nexus 一个脾气。

        **只迭代 `iters` 次（默认 3），且不夹单次增量**：阻尼最小二乘算出来的就是
        一个牛顿步，够得着的话两三下到位。夹单次增量等于把一步拆成几十步，`_ik`
        反而在 `IK_STEPS` 次里走不完，够得着的点也会被 `reach_residual` 报成够不着。
        唯一的夹是夹进关节行程（nexus 的 `_solve_ee_ik` 同一步）：不夹的话关节会
        被推到转不过去的角上，换算成舵机原始值就出 0..4095。

        `ow`：姿态误差的权重。不给就用 `self.ow`（默认 `OW`）。**姿态要求一硬，就会
        跟位置要求抢关节**：实测从某个姿态解进场点，`ow=0.05` 时 4 号被顶到 95° 上限、
        差 19.6mm；`ow=0` 时位置优先，4 号退回 93.4°、差 4.9mm，而工具 Z 轴的朝下
        分量还是 −0.994（`ow=0.05` 时是 −0.990）—— **爪子照样是竖直的**。
        姿态不是不管，是交给 `drive` 一步步转过去，别让它在逆解里跟位置平起平坐。
        """
        # **解的时候就带上安全下限**（2026-10-25 用户：「直接带上限制」）：2 号肩抬
        # −45°、3 号肘两臂夹角 ≥75°。先解到硬件行程上再让闸判，结果是解得出、但一出
        # 手就撞闸停手；带着限制解，逆解会在允许范围里换一组关节分配。
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

    # ---- Arm 接口 ----

    def reset(self, seed: int = 0) -> dict:
        """真机没有「开一局」这回事：不复位场景、不摆东西。清掉速度状态就返回。

        **返回空字典是如实**：接口说「真机里没人知道真值、由人来给」。叫它去摆方块
        或者报「这一局抓哪个」都是编的。
        """
        self.vel[:] = 0.0
        self.rvel[:] = 0.0
        self.terminated = False
        self.n_steps = 0
        self._dirty = True          # 重来一次就以真读数为准，别拿上一次的缓存
        self._grip_cmd = float(self.grip_q())
        return {}

    def tip(self) -> tuple[np.ndarray, np.ndarray]:
        q = self._ticks_to_qpos(self.read_ticks())
        return self._fk(q)

    @staticmethod
    def _rot(quat) -> np.ndarray:
        """四元数 [w,x,y,z] → 3x3 旋转矩阵（列是工具系三轴在基座系里的方向）。"""
        R = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=np.float64))
        return R.reshape(3, 3)

    @staticmethod
    def _sim_pose(pos_s, quat_s) -> tuple[np.ndarray, np.ndarray]:
        """官方 `gripperframe` 的位姿 → 仿真那套 TCP 口径的位姿。

        位置沿**官方 site 的 z 轴**挪 19.9mm（那正是 body x 在官方工具系里的方向），
        姿态再绕工具 y 转 90°。见文件头那两个常数的来处。
        """
        R = RealArm._rot(quat_s)
        out_q = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(out_q, np.asarray(quat_s, dtype=np.float64), TCP_QUAT_Y90)
        return np.asarray(pos_s, dtype=np.float64) + R @ TCP_DELTA_BODY, out_q

    @staticmethod
    def _site_pose(pos_c, quat_c) -> tuple[np.ndarray, np.ndarray]:
        """仿真那套 TCP 口径的位姿 → 官方 `gripperframe` 的位姿（`_sim_pose` 的反向）。

        逆解要的是官方 site 的位姿。**退那 19.9mm 要用官方工具系的方向**（先把自己
        的姿态转回去再退），不能拿仿真口径的姿态直接退 —— 两者的 x/z 差 90°，
        退出来会落到另一个轴上。
        """
        q_s = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(q_s, np.asarray(quat_c, dtype=np.float64), TCP_QUAT_Y90_INV)
        return (np.asarray(pos_c, dtype=np.float64) - RealArm._rot(q_s) @ TCP_DELTA_BODY,
                q_s)

    def tcp(self) -> np.ndarray:
        q = self._ticks_to_qpos(self.read_ticks())
        return self._sim_pose(*self._fk(q))[0]

    def quat(self) -> np.ndarray:
        q = self._ticks_to_qpos(self.read_ticks())
        return self._sim_pose(*self._fk(q))[1]

    def tool_x(self) -> np.ndarray:
        """工具系 x 轴 —— 爪的开合方向（和 `SimArm.tool_x` 同口径）。"""
        from scipy.spatial.transform import Rotation

        q = self.quat()
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()[:, 0]

    def tips(self, q=None) -> tuple[np.ndarray, np.ndarray]:
        """两个爪尖的世界坐标 (固定, 动)。定值和来处见文件头那两个常数。

        `q` 不给就按**这一刻读回来的**关节角算。`_fk` 顺手把 `data` 的 xpos/xmat 填上，
        所以后面直接读；它只动 qpos、别的什么都不碰。
        """
        qs = (self._ticks_to_qpos(self.read_ticks()) if q is None
              else np.asarray(q, dtype=np.float64))
        self._fk(qs)
        out = []
        for name, local in ((TIP_FIXED_BODY, TIP_FIXED),
                            (TIP_MOVING_BODY, TIP_MOVING)):
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            out.append(np.asarray(self.data.xpos[b], dtype=np.float64)
                       + np.asarray(self.data.xmat[b]).reshape(3, 3) @ local)
        return out[0], out[1]

    def jaw_gap_center(self, q=None) -> np.ndarray:
        """两组爪尖的中点（世界系）—— 合爪时物体该待的那一点。

        `tcp()` 报的是**固定爪尖**，两者差着半个开口：合到底差 3.3mm、张到最大差
        70.6mm。落点该按哪个算，见 `skills/pick_real.py` 下探那一段。
        """
        f, m = self.tips(q)
        return (f + m) / 2.0

    def fixed_face_reach(self, axis, side: float = 1.0, z_band=None, q=None) -> float:
        """固定爪**越过爪尖、朝盒子那一侧**还伸出多少米。

        这是仿真那边 `impls/arm_sim.py:657` 的同名量，那边问的是指垫球的表面；
        这个模型里没有指垫球，所以直接拿**固定爪那几个碰撞 geom 的网格顶点**算：
        把顶点投到开合轴上，取「比爪尖更靠盒子那一侧」的最大超出量。

        **为什么非要它**：落点算式里的「缝」如果从爪尖算，缝就只是爪尖到盒子的距离，
        而真正会刮到盒子的是爪面。`_probe_jawbox.py` 实测（`run_real_24.log` 第 2 趟
        下探到底那一列的真实关节角 + 量出来的盒子）：盒子近面离爪尖 2.5mm 时，
        固定爪的料和盒子已经**重叠 1.35mm** —— 也就是说爪面比爪尖还往里探了约
        3.9mm，落点写的「5mm 缝」实际只剩 1mm 出头。下探时固定爪是刮着盒子近面
        下去的，盒子就是这么被推歪的。

        `side` 跟落点那个 `side` 是同一个（+1 = 固定爪在开合轴的正侧）。
        朝物体那一侧是 `-side`，所以取 `side * (爪尖 − 顶点)` 投影的最大值。

        **`z_band` 不能省。** 整根固定指是一块上宽下窄的楔子，越往上越往开口里探；
        不过滤的话量到的是**最上面那一截**的横偏（这个姿态下 37mm），跟夹持高度上
        的真面差着十万八千里。仿真那边没有这个问题，因为它的指垫球就钉在夹持高度上。
        给 `(z_lo, z_hi)` 就只数落在这个高度带里的顶点 —— 那才是会碰到盒子的料。
        """
        qs = (self._ticks_to_qpos(self.read_ticks()) if q is None
              else np.asarray(q, dtype=np.float64))
        x = np.asarray(axis, dtype=np.float64)
        nx = float(np.linalg.norm(x))
        if nx < 1e-9:
            return 0.0
        x = x / nx
        base = float(np.dot(self.tips(qs)[0], x))
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TIP_FIXED_BODY)
        out = 0.0
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != bid:
                continue
            # **只看碰撞 geom。** 视觉那几块是同一形状的复制品，算进去只会把
            # 「哪个才是真会撞上的面」这件事搅浑。
            if (self.model.geom_contype[g] == 0
                    and self.model.geom_conaffinity[g] == 0):
                continue
            mid = int(self.model.geom_dataid[g])
            if mid < 0:                       # 不是 mesh，没有顶点可用
                continue
            a = int(self.model.mesh_vertadr[mid])
            n = int(self.model.mesh_vertnum[mid])
            v = np.asarray(self.model.mesh_vert[a:a + n], dtype=np.float64)
            R = np.asarray(self.data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
            world = v @ R.T + np.asarray(self.data.geom_xpos[g], dtype=np.float64)
            if z_band is not None:
                lo, hi = float(z_band[0]), float(z_band[1])
                keep = (world[:, 2] >= lo) & (world[:, 2] <= hi)
                if not keep.any():
                    continue
                world = world[keep]
            out = max(out, float(side) * float(np.max(base - world @ x)))
        return float(out)

    def angles(self) -> np.ndarray:
        """六个关节各多少度（和 `SimArm.angles` 同口径）。"""
        return np.degrees(self._ticks_to_qpos(self.read_ticks()))

    def grip_q(self) -> float:
        """夹爪关节现在的角度（弧度）。合爪 = 把它压小。"""
        return float(self._ticks_to_qpos(self.read_ticks())[5])

    @property
    def grip_open(self) -> float:
        """**张开指令角（弧度）。** 所有「张爪」写这个，不写 `grip_hi`。

        用户 2026-10-21 定：**任何时候张开都不许超过 20°**（`GRIP_OPEN_DEG`）。
        `grip_hi` 是 MJCF 的行程上限（100°），写下去只会让 6 号顶在止位上。
        仍然夹到 `[grip_lo, grip_hi]` 里，防止配置改小。
        """
        return float(np.clip(np.radians(GRIP_OPEN_DEG), self.grip_lo, self.grip_hi))

    @property
    def grip_open_tick(self) -> int:
        """20° 上限对应的 6 号刻数。**绕过 `write_ticks` 的开爪必须用它当目标。**"""
        return int(self._grip_open_tick)

    def contacts(self) -> int:
        """真机上没有接触传感器，按接口约定返回 0。**不是「没碰到」的意思。**"""
        return 0

    def drive(self, to_position, to_direction=None, closing=None,
              grip=None, pos_slew=0.012, rot_slew=0.03, brake=False) -> None:
        """朝目标走一小步。**算法和 `SimArm.drive` 一模一样**，只是那一步不是交给
        仿真环境去做，是自己解逆解再把关节角度写进舵机。

        参数的说明见 `impls/arm_sim.py:481` 那一份，这里不重复。两步之间的差别只有
        一个：仿真的 PD 一步到位，真机是舵机自己转过去、会滞后，所以外层循环
        （`core/follow.py`）看的反馈是「读回来的关节角」，比仿真多一个串口往返的延迟。
        """
        q = self._ticks_to_qpos(self.read_ticks())
        # **整段都在「仿真那套 TCP 口径」里算**（见文件头）。`_ik` 要的是官方 site
        # 的位姿，所以末尾再换回去 —— 换回去只是同一个刚体上的另一点，逆解照样收敛。
        tcp, quat = self._sim_pose(*self._fk(q))

        d = np.asarray(to_position, dtype=np.float64) - tcp
        want = np.clip(d / POS_SCALE, -1.0, 1.0)
        if brake:
            room = float(np.linalg.norm(d)) / POS_SCALE
            cap = float(np.sqrt(2.0 * float(pos_slew) * max(room, 0.0)))
            mag = float(np.linalg.norm(want))
            if mag > cap > 0.0:
                want = want * (cap / mag)
        self.vel = self.vel + np.clip(want - self.vel, -pos_slew, pos_slew)

        z_des = np.array([0.0, 0.0, -1.0]) if to_direction is None else np.asarray(to_direction, float)
        rwant = align_action(quat, z_des, gain=ROT_SCALE, x_des=closing) * SPEED_ROT
        self.rvel = self.rvel + np.clip(rwant - self.rvel, -rot_slew, rot_slew)

        # 一步的位移 / 转角：动作单位 × 物理尺度。和 nexus 的 `_action_to_ctrl` 同一个式子。
        target_pos = tcp + self.vel * POS_SCALE
        target_quat = quat_multiply(rotvec_to_quat(self.rvel * ROT_SCALE), quat)

        g = grip() if callable(grip) else (0.0 if grip is None else float(grip))
        self._grip_cmd = float(np.clip(self._grip_cmd + g * GRIP_SCALE,
                                       self.grip_lo, self.grip_hi))

        # 换回官方 site 口径再解（`_site_pose` 是上面那个换算的反向）。
        q_next = self._ik(*self._site_pose(target_pos, target_quat), q)
        q_next[5] = self._grip_cmd          # 夹爪那一维直接就是算好的目标角，不进逆解
        self._write(q_next)
        # 留证：**发出去的**那六个关节角。`q_next` 算完就进 `_write` 用掉了，本来什么
        # 都不留 —— `run_sim.py:487` 那个 `getattr(arm, "last_action", None)` 一直在找它，
        # 找不到就退回记实测角，于是轮次记录里 `arm` 和 `action` 变成同一份数据，
        # 出问题时分不出「舵机不肯转」还是「根本没让它转」。单位取度，跟 `angles()` 对齐。
        self.last_action = np.degrees(q_next)

        self.n_steps += 1
        if self.on_step is not None:
            self.on_step()

    def _write(self, qpos: np.ndarray) -> None:
        ticks = self._qpos_to_ticks(qpos)
        for sid, v in ticks.items():
            if not (0 <= v <= 4095):
                raise RuntimeError(f"关节角算出来落到舵机行程外：舵机 {sid} 要 {v}")
        self.write_ticks(ticks)

    def reach_residual(self, position) -> float:
        """这个点够不够得着：返回「够过去以后离目标还差多少米」，0 就是够得着。

        **和 `SimArm.reach_residual` 同一个做法**：不自己写可达性判断，就让它解一次
        逆解，再把结果正推回去量差多少 —— 逆解对够不着的点不报错，只给最接近的。
        **全程是纯计算**：从此刻的关节角出发解，算完就扔，真机一步没动。
        """
        q = self._ticks_to_qpos(self.read_ticks())
        _, quat = self._fk(q)
        q_sol = self._ik(np.asarray(position, dtype=np.float64), quat, q)
        got, _ = self._fk(q_sol)
        return float(np.linalg.norm(got - np.asarray(position, dtype=np.float64)))

    def close(self) -> None:
        if getattr(self, "ctrl", None) is not None:
            self.ctrl.close()
            self.ctrl = None


SERVO_TO_JOINT_REV = {v: k for k, v in SERVO_TO_JOINT.items()}
