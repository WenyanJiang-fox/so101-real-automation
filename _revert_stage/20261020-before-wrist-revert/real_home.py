"""真机从臂的「初始点位」= **手指朝下、腕相机正对桌面**。

2026-10-14 重定。怎么来的：在 `real/servo.py` 里解「抓取点（`grasp_point`，
不是 `gripperframe`）停在原地、手指（+工具x）指向世界 -z」，解出来带臂走到位，
再把读数记下来。实测走完手指离正下 **0.77°**，抓取点在 (0.1528, -0.0029, 0.1204)，
画面里两个爪子从上方垂下、指尖相碰，桌面和盒子都在视野内。

**改过两次，别搞混：**

* 第一次（2026-10-14 上午）：换掉「松扭矩自由落稳」那个原始姿势。那个姿势的工具轴
  是**横的**（离正下 42°），是重力塌出来的，跟「拿爪子去桌上夹东西」没关系 —— 从它
  出发，任何下探都得先把腕子翻过来，而腕一转，腕相机的画面也跟着翻天。
* 第二次（2026-10-14 下午）：松扭矩让臂自己塌下去，用户上手把它摆成上面这组数。
  跟第一次那版比，**只有 5 号（wrist_roll）变了**：263 -> 1018，绕工具轴自转 66.4°。
  抓取点只挪了 1.5mm，手指仍然离正下 0.5°。变的纯粹是**腕相机的画面滚转**。

**这是通电抱住的姿态，不是自由落稳的姿态。** 通电时它可重复（走完读回 1.1mm 内
一致）；断电松手会塌到别处去，不是回到这里。

数值是**舵机原始位置**（0-4095）。记原始值是因为它不截断：
`servo_control_standalone.set_servo_angle` 把角度硬夹在 [1024, 3072]（L456），
而 5 号 1018 就贴着那个下限，走角度那条路会先被拽到 1024。

**扭矩不是本模块开的，harness 也不开。** 舵机里的使能位是断电才清；手动
`disable_torque` 之后必须手动 `sync_enable_torque` 开回来（`real/move_to.py` 的
`--torque` 就是这个），否则舵机收到目标位置也不动。

端口：从臂 `little_little_pig` 在 COM10。主臂那只已经没插。
"""

from __future__ import annotations

PORT = "COM10"

# 舵机 -> 关节 的映射取自 servo_bridge，本目录只有那一份，不复制。
from servo_bridge import SERVO_TO_JOINT  # noqa: E402

# 下面括号里是**关节角**（模型 FK 算的，`RealArm._ticks_to_qpos`）。
# 注意它跟本模块 `raw_to_deg()` 给出的**舵机角**不是一回事：6 号舵机 2803 原始值
# 关节角是 89.8°，`raw_to_deg` 给的是 156.4°。两个都用得着，别混。
RAW_HOME = {
    1: 2229,   # shoulder_pan   关节   1.1°
    2: 1481,   # shoulder_lift  关节 -40.0°
    3: 2409,   # elbow_flex     关节  42.5°
    4: 3107,   # wrist_flex     关节  88.1°
    5: 1018,   # wrist_roll     关节 -90.5°
    6: 1450,   # gripper        关节 -9.9°（关节行程 [-10°,100°]，合到底）
}

# 姿态实测（模型 FK）：抓取点 (0.1523, -0.0024, 0.1192)，手指方向 [-0.01, 0, -1]，
# 离正下 0.5°。
# **5 号这一转把腕相机的画面滚了 66.4°**（画面里那只箱子从右边缘跑到了下边缘），
# 但**抓取点像素没动**：老姿态跑 `jaw_cal` 给 (334,175)，新姿态给 (333,174)。
# 相机和爪子装在同一截腕上、绕同一根轴转，爪子相对相机没变，它自己的投影就不动。

# 6 号舵机 2026-10-14 改过两次：
#   上午改成张开的 2803（行程 90.5%），理由是「下探得张着嘴下」。
#   下午改回合到底的 1450 —— 那个理由不成立了：`skills/pick_real.py` 自己会在
#   下探过半时才张嘴、进场前还专门合一次，home 开爪只剩副作用。2803 的实际后果是
#   整轮开头 20% 的画面里爪子都张着，**收尾那次「回初始位置」还会把已经夹住的方块
#   当场放掉**（2026-10-14 的 `out\round_01.json`：抬升完 TCP z=68mm 方块还在爪里，
#   回原位段开到 89.8° 时脱手）。
# 1450 只是「相对舵机 0-180° 那个刻度」的 37.4°，不是相对爪子自己的行程。
# 按标定文件 `assets/lerobot_calibration/little_little_pig.json`（range_min=1425、
# range_max=2944）算，1450 落在行程的 1.6% 处，**几乎全闭**。爪子的真行程是
# 原始值 1425（闭）到 2944（开）。

# 角度换算，和 servo_control_standalone 里那套线性映射同一份：
#   position = 1024 + angle * 2048 / 180
#   angle    = (position - 1024) * 180 / 2048
RAW_MIN = 1024.0
RAW_PER_DEG = 2048.0 / 180.0


def raw_to_deg(raw: float) -> float:
    """原始位置 -> 舵机角度 0-180。"""
    return (raw - RAW_MIN) / RAW_PER_DEG


def deg_to_raw(deg: float, clamp: bool = True) -> int:
    """舵机角度 -> 原始位置。

    clamp 夹的是舵机自己的可写范围 [0, 4095]，**不是** [1024, 3072] ——
    后者是 `set_servo_angle` 那一个函数自己的限制，不是舵机的限制。
    """
    raw = int(round(RAW_MIN + deg * RAW_PER_DEG))
    return max(0, min(4095, raw)) if clamp else raw


HOME_DEG = {sid: round(raw_to_deg(raw), 2) for sid, raw in RAW_HOME.items()}


def describe() -> str:
    """一行一个舵机的对照表，给人看。"""
    return "\n".join(
        f"{sid}  {SERVO_TO_JOINT[sid]:<14} 原始 {RAW_HOME[sid]:>4}  = 舵机角 {HOME_DEG[sid]:>7.2f}°"
        for sid in sorted(RAW_HOME)
    )


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"端口 {PORT}（从臂 little_little_pig）")
    print(describe())
