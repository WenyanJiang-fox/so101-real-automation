"""把 mcp_robot 里的真机舵机驱动接进本目录。

驱动本体在 G:\\sim2real\\mcp_robot\\sources\\servo_control_standalone.py。
本目录不改它、不复制它，只用 sys.path 引进来 —— 那边一改，这边跟着变。

驱动里跟本目录相关的接口（已核实，行号对着 servo_control_standalone.py）：
  ServoControllerStandalone(port_name="COM4", baudrate=1000000, timeout=0.1, debug=False,
                            read_sleep=0.01, write_settle=0.02)   L289
  set_servo_angle(servo_id, angle, speed=0) -> bool        角度 0-180，L456
  get_servo_angle(servo_id) -> float | None                角度 0-180，L499
  read_position(servo_id) -> int | None                    原始值 0-4095，L557
  enable_torque / disable_torque / go_home(speed=0)

本期用不到、但驱动里缺的东西（要加的话在这里补）：
  只读了 ADDR_PRESENT_POSITION = 56（L285）。负载 / 电流寄存器没读，
  所以「舵机使多大劲」这个物理量现在拿不到。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

DRIVER_DIR = Path(r"G:\sim2real\mcp_robot\sources")
if str(DRIVER_DIR) not in sys.path:
    sys.path.insert(0, str(DRIVER_DIR))

from servo_control_standalone import (  # noqa: E402
    COMM_SUCCESS,
    INST_READ,
    ServoControllerStandalone,
)

# 舵机 -> 关节 的规范映射（全项目一致）
SERVO_TO_JOINT = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}
GRIPPER_ID = 6

# 夹爪行程（用户实测，见 memory so101-gripper-jaw-travel）：
#   0°   = 合到底，爪面间距约 30.9 mm
#   180° = 全开，  爪面间距约 55.9 mm
# 方向如果和真机对不上，第一次用之前先确认一遍。
#
# **`GRIPPER_OPEN_DEG = 180` 已经删掉。** 2026-10-21 用户定：任何时候张开爪子都不许
# 超过 20°（qpos 口径）。180° 换算成原始位置是 3072，比标定上限 2944 还远 —— 原来
# `open_gripper` 就是拿它开的，这是全项目唯一一处绕过 `RealArm.grip_open` 的开爪。
# 现在开爪必须显式给 `tick`（`RealArm.grip_open_tick`），没有默认值。
GRIPPER_CLOSED_DEG = 0.0
#: **张开上限（qpos 度）。2026-10-21 用户定：不管什么时候都不能超过这个数。**
#: 真机（`RealArm`）和仿真（`SimRealArm`）都从这一条取，别再在别处写死角度。
GRIPPER_MAX_OPEN_DEG = 20.0

DEFAULT_PORT = "COM12"
DEFAULT_BAUD = 1_000_000


def deg_from_position(position) -> float:
    """原始位置值 → 舵机角度（0-180）。**跟驱动里 `get_servo_angle` 同一条式子**
    （`G:\\sim2real\\mcp_robot\\sources\\servo_control_standalone.py:516`）。

    为什么在这儿重写一遍：`arm.read_ticks()` 走的是**批量同步读**（`sync_read_positions`，
    回的就是原始位置值），而 `read_gripper_deg` 走的是**单舵机读**。两条读同一个寄存器，
    换算必须一模一样，判据（`classify_ref` 的 `empty_deg` / `open_deg`）才在同一把尺子上。
    """
    return (float(position) - 1024.0) * 180.0 / 2048.0


def open_controller(port: str = DEFAULT_PORT, baudrate: int = DEFAULT_BAUD,
                    debug: bool = False,
                    fast: bool = False) -> ServoControllerStandalone:
    """打开串口。开不了直接抛，不返回半死的 controller。

    `fast=False`（默认）用驱动原来的两处等待：读一次约 65.8ms、写一次约 20.9ms。
    `fast=True` 把这两处去掉：读约 1.3ms、写约 0.48ms，读回值逐次相同（实测见
    `G:\\sim2real\\so101-real-harness\\_fps_probe\\result.json`）。**只给 30fps 采集用**；
    面板和技能走默认，行为一个数都不变。代价见驱动 `__init__` 的 `read_sleep` 注释：
    某个舵机不回应时是忙等。
    """
    kw = {"read_sleep": 0.0, "write_settle": 0.0} if fast else {}
    c = ServoControllerStandalone(port_name=port, baudrate=baudrate, debug=debug, **kw)
    if not c.is_open:
        raise RuntimeError(f"串口打不开：{port}（被别的程序占用了？mcp_server.py 在跑？）")
    return c


def read_gripper_deg(controller, samples: int = 5, delay: float = 0.02):
    """读 6 号舵机角度，连读 samples 次取中位数。

    单次 SCS 读有丢包概率，中位数比平均值抗离群值。全读失败返回 None。
    """
    vals = []
    for _ in range(max(1, samples)):
        a = controller.get_servo_angle(GRIPPER_ID)
        if a is not None:
            vals.append(a)
        if delay:
            time.sleep(delay)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def read_all_deg(controller, samples: int = 3, delay: float = 0.02) -> dict:
    """读全部 6 个舵机的角度，{servo_id: 角度 or None}。"""
    out = {}
    for sid in range(1, 7):
        vals = []
        for _ in range(max(1, samples)):
            a = controller.get_servo_angle(sid)
            if a is not None:
                vals.append(a)
            if delay:
                time.sleep(delay)
        if vals:
            vals.sort()
            out[sid] = vals[len(vals) // 2]
        else:
            out[sid] = None
    return out


def close_gripper(controller, speed: int = 0) -> bool:
    """合爪。speed=0 是舵机最高速，会撞得比较狠，标定时可以用小一点的值。"""
    return controller.set_servo_angle(GRIPPER_ID, GRIPPER_CLOSED_DEG, speed)


def open_gripper(controller, speed: int = 0, tick: int | None = None) -> bool:
    """开爪。**`tick` 必给**，它是 20° 上限对应的刻数（`RealArm.grip_open_tick`）。

    原来这里写死 `GRIPPER_OPEN_DEG = 180`（= 3072 刻），比标定上限 2944 还远；
    用户 2026-10-21 定死「不管什么时候张开爪子都不能大于 20°」，所以这条路不再有
    默认值 —— 没给 `tick` 就直接报错，免得又悄悄开到最大。

    刻数按驱动那条线性式换成舵机角再写（`set_servo_angle` 就在本文件引用的驱动里），
    仿真那一支的 `_SimGripperCtrl.set_servo_angle` 收的也是这个角度，两边同一条路。
    """
    if tick is None:
        raise ValueError("open_gripper 必须给 tick（用 RealArm.grip_open_tick）")
    deg = (int(tick) - 1024) * 180.0 / 2048.0
    return controller.set_servo_angle(GRIPPER_ID, deg, speed)


# ---- 读任意寄存器（驱动缺的那一块，按本文件开头说的「缺的在这里补」）----
# 驱动只把 ADDR_PRESENT_POSITION 读死了（servo_control_standalone.py L570），可认领
# 端口要比的是标定寄存器。**这里不重写协议**：直接用它自己的 `_send_packet` 和
# `_read_packet`，只是把地址和长度换成要读的那个。
ADDR_MIN_POSITION_LIMIT = 9      # 行程下限，2 字节
ADDR_MAX_POSITION_LIMIT = 11     # 行程上限，2 字节
ADDR_HOMING_OFFSET = 31          # 零点偏移，2 字节，**有符号**
HOMING_SIGN_BIT = 11             # 它的符号位在第 11 位（sign-magnitude 编码）

# 三个地址对着 lerobot 的 STS3215 表核实过：CONTROL_TABLE 里 9/11/31 各 2 字节，
# STS_SMS_SERIES_ENCODINGS_TABLE 里 "Homing_Offset": 11。
# 见 G:\anaconda\envs\sim2real\Lib\site-packages\lerobot\motors\feetech\tables.py


def read_register(controller, servo_id: int, address: int, size: int = 2):
    """读一个寄存器，返回原始整数；读不到返回 None。**只发读指令，不写。**"""
    if controller._send_packet(servo_id, INST_READ, [address, size]) != COMM_SUCCESS:
        return None
    result, error, data = controller._read_packet(servo_id)
    if result != COMM_SUCCESS or error != 0 or len(data) < size:
        return None
    return sum(int(b) << (8 * i) for i, b in enumerate(data[:size]))   # 小端


def decode_sign_magnitude(raw: int, sign_bit: int = HOMING_SIGN_BIT) -> int:
    """sign-magnitude 解回有符号整数 —— 零点偏移就是这么存的（-1936 那种）。

    和 lerobot 的 `encoding_utils.decode_sign_magnitude` 同一个算法，照抄过来是为了
    不把 lerobot 引进本目录（那个包在这个环境里还缺 feetech 的 SDK）。
    """
    magnitude = raw & ((1 << sign_bit) - 1)
    return -magnitude if (raw >> sign_bit) & 1 else magnitude


def read_homing_offset(controller, servo_id: int):
    """读一个舵机的零点偏移，已解符号。读不到返回 None。"""
    raw = read_register(controller, servo_id, ADDR_HOMING_OFFSET)
    return None if raw is None else decode_sign_magnitude(raw)


def read_calibration_registers(controller, samples: int = 3, delay: float = 0.02) -> dict:
    """读回六个舵机的标定三件套：{servo_id: {homing_offset, range_min, range_max}}。

    每个舵机的每个值连读 samples 次取中位数 —— 单次 SCS 读有丢包概率，中位数比
    平均值抗离群值（跟 `read_all_deg` 一个道理）。全读失败记 None。

    **只读。** 认领端口靠它，认错了的代价是开错端口，不是动错臂。
    """
    out = {}
    for sid in range(1, 7):
        vals = {}
        for key, addr in (("homing_offset", ADDR_HOMING_OFFSET),
                          ("range_min", ADDR_MIN_POSITION_LIMIT),
                          ("range_max", ADDR_MAX_POSITION_LIMIT)):
            got = []
            for _ in range(max(1, samples)):
                raw = read_register(controller, sid, addr)
                if raw is not None:
                    got.append(raw)
                if delay:
                    time.sleep(delay)
            if not got:
                vals[key] = None
                continue
            got.sort()
            raw = got[len(got) // 2]
            vals[key] = (decode_sign_magnitude(raw) if key == "homing_offset" else raw)
        out[sid] = vals
    return out
