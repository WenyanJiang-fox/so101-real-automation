"""机械臂动作的安全闸：自动走位之前、走完之后各问一遍。

**阈值全部按实测取**（`out/` 里已经跑过的日志，不是拍的）：

    单次走位里单个关节的位移（`要动 [...] 度`）
        正常：`oneshot_011017` / `collect_gain_034151` / `collect_2eps_033229` /
              `collect_4hz_032736` / `panel_20261020_1425` 五份里，中位数 6~9°，
              最大 34.4°（都在「初始位置 → 悬停点」那一段进场）。
        失控：`panel_20261020_1442` 第 2 趟，要动 81.1 / 41.9 / 100.0 / 93.0 度。
        → `MAX_STEP_DEG = 45`

    抓取点（TCP，基座系，毫米）
        正常：水平半径 172~280、z −18~95（初始点位 z = 119）。
        失控：水平半径 358.6、z 444.7（关节 [85.2 −99.8 −56.2 2.0 −66.9 99.3]）。
        → 半径 ≤ 300、−30 ≤ z ≤ 200

**越界不是重试，是停手。** 立 `arm.terminated` 之后，技能每一处走位回来都用
`_stopped` 看到它、交 `give_up`，面板据此停采集。人看一眼再决定，不自动接着跑。

**手工动作不过这道闸。** 面板上的 jog / home / 急停回位是人在看着按的，而且从
一个歪掉的姿态回初始位置本来就可能要动一百多度 —— 那道坎只拦自动走位。
"""
from __future__ import annotations

import numpy as np

#: 单次走位、单个关节最多动多少度（夹爪那一维不算，它一次合到底就是 100°）。
MAX_STEP_DEG = 45.0

#: 抓取点允许的工作包络（毫米，基座系）。
TCP_R_MAX_MM = 300.0
TCP_Z_MIN_MM = -30.0
TCP_Z_MAX_MM = 200.0

#: **2 号肩抬的绝对下限（度）**（2026-10-24 用户定）：「后仰角度不应该比初始位置
#: 还要激进」。**方向：这个关节的数值越负 = 越往后仰** —— 初始点位 −40.0°
#: （`real_home.RAW_HOME[2]`），正常抓取在 −20~−30°，失控那一趟读到 −99.6~−99.9°。
#: MJCF 自己的限位是 ±100°，所以单步 45° 闸和 TCP 包络都拦不住它（每一步都只动
#: 零点几度、TCP 也还在包络里）。这一条补的就是「绝对值」这一维。
#: 2026-10-24 用户定：**−45.0°**（初始位置是 −40.0°，往后再放 5°）。
JOINT2_MIN_DEG = -45.0


def joint2_home_deg(arm) -> float:
    """2 号肩抬的绝对下限（度，越负越后仰）。**固定 −45.0°**（用户定的）。

    初始点位实测 −40.0°（`RAW_HOME[2]`，同一份标定换算），放 5° 余量到这里。
    不再从 `RAW_HOME` 现算 —— 用户要的是一个能直接对表看的数。
    """
    return JOINT2_MIN_DEG


#: **3 号肘（elbow_flex）的两臂夹角下限**（2026-10-25 用户定）：「3 号舵机控制的两个
#: 臂不能小于 75°」。**换算实测**：`so101_new_calib.xml` 做前向运动学，取上臂、下臂、
#: 腕三个铰接点的连线夹角 —— `q3 = 0` 时两臂夹角 **73.8°**，之后每 1° 关节角正好
#: 对应 1° 夹角（+42.5° → 116.3°，−42.5° → 31.3°）。所以：
#:     两臂夹角 = 73.8° + q3         →  q3 下限 = 75.0 − 73.8 = **+1.2°**
#: 方向：q3 越负 = 折得越狠（两臂夹角越小）。初始点位 q3 = +42.5°（夹角 116°）。
ELBOW_MIN_INTERIOR_DEG = 75.0
ELBOW_INTERIOR_AT_ZERO_DEG = 73.8
ELBOW3_MIN_DEG = ELBOW_MIN_INTERIOR_DEG - ELBOW_INTERIOR_AT_ZERO_DEG


def check_elbow_abs(arm, q) -> str | None:
    """这个姿态的 3 号肘折过没有：两臂夹角不许小于 75°。"""
    q3 = float(np.degrees(np.asarray(q, dtype=np.float64).reshape(-1)[2]))
    if q3 < ELBOW3_MIN_DEG - 1e-6:
        return _trip(arm, f"3 号肘要到 {q3:.1f}°，两臂夹角只剩 "
                          f"{ELBOW_INTERIOR_AT_ZERO_DEG + q3:.1f}°"
                          f"（下限 {ELBOW_MIN_INTERIOR_DEG:.0f}°）")
    return None


def check_joint2_abs(arm, q) -> str | None:
    """这个姿态的 2 号肩抬有没有越过绝对下限。"""
    q2 = float(np.degrees(np.asarray(q, dtype=np.float64).reshape(-1)[1]))
    lim = joint2_home_deg(arm)
    if q2 < lim - 1e-6:
        return _trip(arm, f"2 号肩抬要到 {q2:.1f}°，比初始位置的 "
                          f"{lim:.1f}° 还要后仰")
    return None


def check_joint_abs(arm, q) -> str | None:
    """这个姿态的绝对下限：2 号肩抬、3 号肘，两个都看。"""
    return check_joint2_abs(arm, q) or check_elbow_abs(arm, q)


def clip_bounds(q_lo, q_hi) -> tuple[np.ndarray, np.ndarray]:
    """把硬件的关节行程再收紧到两条绝对下限上，交给逆解当夹取上下限。

    **逆解在解的时候就该守这两条**（2026-10-25 用户：「直接带上限制」）：先解后拦的话，
    解出来的构型必然先撞闸、整趟停手；带着限制解，它会在允许的范围里重新分配关节，
    解不到就由「够不着」那一支接管。闸本身留着兜底（手工 jog、配置漂移都还要它）。
    """
    lo = np.array(q_lo, dtype=np.float64, copy=True)
    hi = np.array(q_hi, dtype=np.float64, copy=True)
    lo[1] = max(float(lo[1]), np.radians(JOINT2_MIN_DEG))
    lo[2] = max(float(lo[2]), np.radians(ELBOW3_MIN_DEG))
    return lo, hi


def check_joints(arm, q_now, q_cmd) -> str | None:
    """一步的终点姿态过绝对下限。**只拦「还要往后走」的那一步。**

    回位不受影响：初始位置本身正好等于下限，从更后仰的姿态往回走时 `q_cmd` 不小于
    下限，也不会被判成「更往后」。
    """
    q2_now = float(np.degrees(np.asarray(q_now, dtype=np.float64).reshape(-1)[1]))
    q2_cmd = float(np.degrees(np.asarray(q_cmd, dtype=np.float64).reshape(-1)[1]))
    lim = joint2_home_deg(arm)
    if q2_cmd < lim - 1e-6 and q2_cmd < q2_now - 1e-6:
        return check_joint2_abs(arm, q_cmd)
    # 3 号肘同样只拦「还要往里折」的那一步：从已经折过头的姿态往回开不受影响。
    q3_now = float(np.degrees(np.asarray(q_now, dtype=np.float64).reshape(-1)[2]))
    q3_cmd = float(np.degrees(np.asarray(q_cmd, dtype=np.float64).reshape(-1)[2]))
    if q3_cmd < ELBOW3_MIN_DEG - 1e-6 and q3_cmd < q3_now - 1e-6:
        return check_elbow_abs(arm, q_cmd)
    return None


def _trip(arm, why: str) -> str:
    """越界：立停手标志、记原因、打一行。返回原因给调用方接着传。"""
    arm.safety_why = why
    arm.terminated = True
    print(f"    ⛔ 安全闸：{why} —— 停手，不再往下写", flush=True)
    return why


def check_step(arm, q_now, q_cmd) -> str | None:
    """这一步要写的关节角，有没有哪个关节一次要动太多。"""
    d = np.degrees(np.abs(np.asarray(q_cmd, dtype=np.float64)
                          - np.asarray(q_now, dtype=np.float64)))
    d[5] = 0.0                      # 夹爪是行程端点，不按这条算
    i = int(np.argmax(d))
    if float(d[i]) > MAX_STEP_DEG:
        return _trip(arm, f"一步要动 {float(d[i]):.1f}°（{i + 1} 号关节，"
                          f"上限 {MAX_STEP_DEG:.0f}°）")
    return None


def check_pose(arm, where: str = "") -> str | None:
    """走完读回来这一下，抓取点还在不在工作包络里。"""
    t = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1) * 1000.0
    r = float(np.hypot(t[0], t[1]))
    tag = f"（{where}）" if where else ""
    if r > TCP_R_MAX_MM:
        return _trip(arm, f"抓取点水平半径 {r:.0f}mm{tag}，上限 {TCP_R_MAX_MM:.0f}"
                          f" —— 臂已经跑到工作区外")
    if not (TCP_Z_MIN_MM <= float(t[2]) <= TCP_Z_MAX_MM):
        return _trip(arm, f"抓取点高度 {float(t[2]):.0f}mm{tag}，允许 "
                          f"{TCP_Z_MIN_MM:.0f}~{TCP_Z_MAX_MM:.0f}mm")
    return None
