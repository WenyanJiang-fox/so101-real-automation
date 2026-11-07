"""关节空间运动原语：选关节、受限逆解、走一段、竖直下探、转腕。"""
from __future__ import annotations

from core.params import (ALIGN_CHUNK, ALIGN_MAX_STEPS,            # noqa: E402
                         ALIGN_ROT_SLEW, ALIGN_TOL_DEG, CARRY,
                         GENTLE, GRIP_BIAS, GRIP_K, GRIP_RATE,
                         GRIP_SLACK, LIFT_Z, POS_SCALE,
                         SLOW, SPEED, STUCK_TOL,
                         acc_amax, speed_vmax)
import numpy as np
from . import knobs as _knobs
from .knobs import DESCEND_STEP, GAIN_MAX, GAIN_MIN_ASK_DEG, GAP_FALLBACK_MIN_COS, GAP_FALLBACK_MIN_MM_DEG, GAP_FALLBACK_POOL, REAL_GOTO_TRIES
from .pose import _vertical
def _joint_dir_xy(arm, q_now, j, deg: float = 1.0):
    """只动第 `j` 个关节 `deg` 度，爪尖在平面里往哪走。返回 `(方向单位向量, 毫米/度)`。"""
    q = np.array(q_now, dtype=np.float64, copy=True)
    q[j] = float(q[j]) + np.radians(float(deg))
    a = np.asarray(arm._sim_pose(*arm._fk(q_now))[0], dtype=np.float64)[:2]
    b = np.asarray(arm._sim_pose(*arm._fk(q))[0], dtype=np.float64)[:2]
    d = b - a
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return None, 0.0
    return d / n, n * 1000.0 / abs(float(deg))

def _free_for(arm, need, base):
    """这一步该走 `need`（底座系平面单位向量）—— `base` 那几个关节出不上力就补人。

    `base` 原样保留（用户要的是「解放一个舵机**加入**」，不是换掉）。判据是
    **沿 `need` 的毫米/度**：`base` 里最好的那个不够 `GAP_FALLBACK_MIN_MM_DEG`，就从
    `GAP_FALLBACK_POOL` 里挑分数最高的加进去。五个关节都没力臂就原样返回，并打一行
    ⚠ 出来 —— 那种工位摆法不是改关节能救的。
    """
    if need is None or not base:
        return base
    need = np.asarray(need, dtype=np.float64).reshape(-1)[:2]
    nrm = float(np.linalg.norm(need))
    if nrm < 1e-12:
        return base
    need = need / nrm
    try:
        q_now = np.array(arm._ticks_to_qpos(arm.read_ticks()), dtype=np.float64)
    except Exception:                                               # noqa: BLE001
        return base
    base = tuple(int(v) for v in base)

    def _score(j: int) -> float:
        """第 `j` 个关节**沿 `need` 每度能挪多少毫米**（带号不要，出力不分正反）。"""
        d, mm = _joint_dir_xy(arm, q_now, j)
        if d is None:
            return 0.0
        return float(mm) * abs(float(d @ need))

    def _cos(j: int) -> float:
        """第 `j` 个关节让爪尖走的方向跟 `need` 的夹角余弦（绝对值）。"""
        d, _mm = _joint_dir_xy(arm, q_now, j)
        return 0.0 if d is None else abs(float(d @ need))

    def _ok(j: int) -> bool:
        """这个关节算不算「有力」：**沿这一维的毫米/度**和**方向余弦**两条都得过。"""
        return (_score(j) >= GAP_FALLBACK_MIN_MM_DEG
                and _cos(j) >= GAP_FALLBACK_MIN_COS)

    best_base = max((_score(j) for j in base), default=0.0)
    if any(_ok(j) for j in base):
        return base
    # 挑人时也是「有力」优先，同档才比毫米/度。
    best_j, best = None, (False, best_base)
    for j in GAP_FALLBACK_POOL:
        if int(j) in base:
            continue
        r = (_ok(int(j)), _score(int(j)))
        if r > best:
            best_j, best = int(j), r
    if best_j is None or not best[0]:
        print(f"       ⚠ 这一维五个关节谁都没力臂（最好 {best[1]:.2f}mm/度、"
              f"cos {max((_cos(j) for j in base), default=0.0):.2f}；"
              f"门槛 {GAP_FALLBACK_MIN_MM_DEG:.1f}mm/度、"
              f"{GAP_FALLBACK_MIN_COS:.2f}）—— 这个摆法改关节救不回来")
        return base
    print(f"       {[j + 1 for j in base]} 号沿这一维只出 {best_base:.2f}mm/度、"
          f"cos {max((_cos(j) for j in base), default=0.0):.2f}（出不上力）"
          f" → 补 {best_j + 1} 号进来（{best[1]:.2f}mm/度）")
    return tuple(sorted(set(base) | {best_j}))

def _cmd(arm) -> str:
    """这一步**真正发出去**的位置指令，一行。

    `RealArm` 没有 `last_action`（仿真那支有，记的是最后一次写下去的关节目标）。真机
    发出去的是**末端位移增量**，`arm.vel` 就是它 —— `real/arm.py` 里已经过位置斜率
    限幅、并且已经夹过可达范围的那一维。乘 `POS_SCALE` 换成毫米才看得懂。
    """
    v = getattr(arm, "vel", None)
    if v is None:
        return "指令=?"
    v = np.asarray(v, dtype=np.float64).ravel()[:3] * POS_SCALE * 1000.0
    return ("指令=" + np.array2string(v, precision=3, suppress_small=True,
                                      floatmode="fixed") + "mm")

def _trace(arm):
    """`follow.follow` 的 `verbose` 回调：每 40 步打一行。

    仿真的抬升段挂的就是这个（`skills/pick.py:762`），真机三段动作原来一行都没有。
    挂上它只为一件事：让上一轮那个**没解释的进场 stall**有观测 —— 目标
    `[0.227, 0.068, 0.084]`，实际停在 `[0.166, 0.061, -0.002]`，全程只有 2 号关节
    在动（-38.5° → -18.1°），日志里看不出「发出去的指令到底是什么」。**只加观测，
    不承诺解决。**
    """
    def trace(k, tgt, t, dur):
        if k % 40:
            return
        tgt = np.asarray(tgt, dtype=np.float64)
        print(f"      [{k:4d}] tcp={np.round(arm.tcp(), 3)} "
              f"指定={np.round(tgt, 3)} 进度={t:.1f}/{dur:.1f} "
              f"关节={np.round(arm.angles(), 1)} {_cmd(arm)} "
              f"开合={float(arm.grip_q()):+.3f}")
    return trace

def _ik_free(arm, target_pos, target_quat, qpos0, free, ow=None, iters: int = 40):
    """和 `arm._ik` 解同一个位姿，但**只放 `free` 这几个关节动**，别的钉死。

    做法就是把雅可比里那几个关节的列清零、算出来的 `dq` 也把那几维清零（`arm._ik`
    的注释讲过：`mj_kinematics` 之后必须再来一次 `mj_comPos`，否则整块雅可比是零）。

    迭代次数比 `arm._ik` 的 3 次多：关节一钉死，剩下那几个要走的弧就长了。实测
    （离线，同一姿态）：只放 2+3、`ow=0` 降 20mm，3 次差 0.48mm、12 次差 0.00mm。
    """
    from so101_nexus.kinematics import ee_ik_delta_q, EE_IK_DAMPING   # noqa: PLC0415

    from real import safety                                            # noqa: PLC0415
    lo, hi = safety.clip_bounds(arm.q_lo, arm.q_hi)
    q = np.array(qpos0, dtype=np.float64, copy=True)
    m = np.zeros(6, dtype=bool)
    m[list(free)] = True
    w = float(arm.ow if ow is None else ow)
    for _ in range(int(iters)):
        pos, quat = arm._fk(q)
        jac = np.array(arm._jac(q), dtype=np.float64, copy=True)
        jac[:, ~m] = 0.0
        dq = ee_ik_delta_q(jac, pos, quat,
                           np.asarray(target_pos, dtype=np.float64),
                           np.asarray(target_quat, dtype=np.float64),
                           orientation_weight=w, damping=EE_IK_DAMPING)
        dq[~m] = 0.0
        q = np.clip(q + dq, lo, hi)
    return q

def _step_min_deg(arm, tgt, free, ow=None):
    """走到 `tgt` 这一步，**该按哪个自由关节的度数定步长**。返回 (度, 关节下标, Δz 米)。

    用 `_goto` 同一个 `_ik_free`（`free=None` 时同一个 `arm._ik`）解一遍，拿解出来的
    关节角减此刻读回来的关节角。**不是自己写的运动学** —— 只是把 `_goto` 要解的那个点
    提前解一次，好知道这一步值不值得走。`free=None` 时排除夹爪那一维（它不由 `tgt` 决定）。

    **2026-10-17 改：挑关节的尺子从「度数最大」换成「真的把 TCP 挪得最远」。**
    原来取 `argmax(|Δq|)`，在 `FREE_GAP=(0, 4)` 上被 **5 号腕转**占住 —— 而 5 号在
    下面那张实测表里对「沿缝」的贡献是 **0.00mm/度**：它转多少，爪尖都不动。于是
    「长度往上乘到某个自由关节动 2°」这条把长度乘在了一个不产生位移的关节上。
    实测（`out/` 面板日志 2026-10-17 那一趟微调）：
        算出来该走 0.28mm —— 这一步走 2.84mm（5 号舵机要动 0.28°，至少要 2.0°），方向不变
        走位（关节空间）第 1 趟：[200.3 29.7 9.8]mm → [203.1 29.9 10.2]mm
                             要动 [0.1 0. 0. 0. 0.3 0.5] 度
        走到 [200.3 29.7 9.8]mm  差 2.9mm     ← 读回来和出发时一模一样，臂一步没动
    离线复现（`_probe_free_gap.py`，同一个 `_ik_free`）：free=(0,4) 时度数几乎不随目标
    长度涨（0.28mm→0.20°、20mm→0.83°），最大的一直是 5 号，而目标 20mm 时 TCP 只挪了
    0.63mm。**换成按位移挑，挑出来的就是真走得动缝的那个关节。**
    """
    q_now = np.array(arm._ticks_to_qpos(arm.read_ticks()), dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    if free is None:
        q_goal = arm._ik(*arm._site_pose(tgt, arm.quat()), q_now, ow=ow)
    else:
        q_goal = _ik_free(arm, *arm._site_pose(tgt, arm.quat()), q_now, free, ow=ow)
    d = np.degrees(np.abs(np.asarray(q_goal, dtype=np.float64) - q_now))
    if free is None:
        d[5] = 0.0                  # 夹爪不受 tgt 支配，不参与「够不够大」的判断
        i = int(np.argmax(d))
        return float(d[i]), i, 0.0
    # **按「只动这一个关节、TCP 挪多远」挑**（也就是实测表里那一列 mm/度乘上 Δq）。
    # 每个自由关节只多一次正解，离线，不碰串口。
    #
    # 用户 2026-10-17：「5 号舵机转爪子只会把爪子带的越来越偏，给你开放 5 号舵机是为了
    # 纠正角度偏的，**真正带来横移量的只有 1 号舵机**」。离线实测（`_probe_free_gap.py`，
    # 跑位那一刻的姿态，每 1° 的位移）：
    #     1 号  2.87mm/度   ← 唯一横移的
    #     5 号  0.16mm/度   ← 只是自转，平移≈0
    # 差 18 倍。而上面那一行原来取 `argmax(|Δq|)` —— **5 号拿 0.28°、1 号只拿 0.1°**
    # （逆解为了让姿态对上，转腕比横移更积极），于是「把长度乘到某个自由关节动 2°」
    # 这条把长度一路乘在一个 0.16mm/度 的关节上：乘到 10 倍上限也只有 0.28°、
    # 爪尖横移≈0，臂原地不动（面板日志 2026-10-17 那一趟：读回来一点没变、差 2.9mm）。
    #
    # 换成按位移挑，方向照 `_goto` 的口径取 `_sim_pose`（和 `tgt` 同一个口径）。
    _base = np.asarray(arm._sim_pose(*arm._fk(q_now))[0], dtype=np.float64)
    _best_mm, _best_deg, _best_i, _best_dz = -1.0, 0.0, int(free[0]), 0.0
    for j in free:
        _q = np.array(q_now, dtype=np.float64, copy=True)
        _q[j] = float(q_goal[j])
        _p = (np.asarray(arm._sim_pose(*arm._fk(_q))[0], dtype=np.float64) - _base)
        # **只按水平位移挑人。** 原来算的是三维长度，可 2/3 号是在竖直面里走弧的，
        # 它们一动大部分位移都在 z 上 —— 拿三维长度当分数，等于把一个「往下压」
        # 的关节当成「横着走」的关节，步长照它定，臂就一边横移一边掉。
        # 用户 2026-10-17：「你补进来的关节还在下高度，也要纳入计算」。
        _mm = float(np.linalg.norm(_p[:2]))
        if _mm > _best_mm:
            _best_mm, _best_deg, _best_i = _mm, float(d[j]), int(j)
            _best_dz = float(_p[2])
    return _best_deg, _best_i, _best_dz

def _goto(arm, goal, *, vmax, grip=-1.0, closing=None, tol=0.004,
          free=None, ow=None, tries: int | None = None, quat=None) -> str:
    """走位到 `goal`。**关节空间一步到位**，不走 `core/follow.py` 那条笛卡尔的路。

    **为什么换掉笛卡尔那条。** 实测（COM10 从臂，2026-10-14，回初始位置之后走进场点
    [227, 68, 81]mm）：`follow` 每步的位移由 `POS_SCALE` 定死，逆解出来的关节增量落在
    2/3/4 号的起步死区里。把 `LAG_GAIN`/`POS_SLEW` 放开，2 号 1494→1972、3 号
    2409→1931（各转 42°）—— **手臂确实动起来了**，但最后卡在 **4 号顶到 94.4°**
    （上限 95°），还差 41.7mm。再把逆解里姿态误差的权重往下压，4 号松开、退回 66.1°，
    可 2 号 1492→1451、3 号 2411→2410 **几乎不动了** —— 每步增量又落回死区。
    五个值全 stall，一个都没到，而且压低权重会把爪子弄歪：

        ow=0.05  差 41.7mm  4 号 94.4°  爪子朝下 −0.99（垂直）
        ow=0.02  差 40.1mm  4 号 69.1°  爪子朝下 −0.953
        ow=0.01  差 34.3mm  4 号 65.5°  爪子朝下 −0.933
        ow=0.005 差 30.8mm  4 号 65.5°  爪子朝下 −0.924
        ow=0.00  差 26.9mm  4 号 66.1°  爪子朝下 −0.913（歪 24°）

    也就是说「每步挪一小段、看一步走一步」这个形状在真机上不成立。`real/move_to.py`
    开头记着同一件事，还多一条：读一次要 66ms、`follow` 一步问三次，实测 4.5 步/秒。

    **关节空间为什么行。** `real/joint_move.goto_q` 提前把轨迹算好，一路往下写
    **绝对**关节角、**中途不读**，指令单调推进，自然越过死区；收尾连发几步让舵机追平。
    `real/move_to.py`（真机第一步）和 `skills/home_real.py`（回初始位置）一直靠它到位。

    **姿态怎么保。** 目标关节角是「**保持此刻的朝向**、只解位置」得到的
    （`real/move_to.py:82` 同一写法）。此刻工具轴朝下，解完还是朝下 —— 不夹带姿态
    改变，就没有第二个矛盾。

    **默认 `grip=-1.0` = 走的时候把爪子合着。** 原来默认是 `+1.0`（张开）：第 0 段量
    行程两端要 `measure_open()` 张到底，量完就停在张着的位置，接着整个进场、纠位、转正
    都是张着爪走的 —— 而张开的爪有 55.9mm 宽（合到底 30.9mm），一路上蹭桌沿、蹭别的
    件都得算它头上。现在改成合着走，只在快贴到方块那一小段才张开（`run` 的下探那一段）。
    这条路上 `grip` 不再是「每步增量」而是**全程走完后的落点**，所以直接取行程端点：
    `>0` 张到底、`<0` 合到底、`None` 这一段不动它。

    **走不到就放大。** 每趟走完逐关节量「实际走了 ÷ 要求走了」，下一趟按这个比例的倒数
    放大指令（上限 `GAIN_MAX` 倍）。因为舵机在负载下往一个方向只走三到五成，一趟一趟重解
    同一个点补不齐 —— 原来那条「没更近就停」的判据就是被这一点骗了，见 `GAIN_MAX` 和
    下面那段注释。

    `tol`：走完读回来离 `goal` 还有多远算到了。`vmax` / `closing` 是笛卡尔那条留下的
    参数，关节空间有自己的速度（`joint_move.JVEL`），这两个现在不生效 —— 保留是为了
    不改四个调用方的签名。
    """
    from real import joint_move as jm                              # noqa: PLC0415
    from real import safety                                        # noqa: PLC0415

    goal = np.asarray(goal, dtype=np.float64)
    out = "stall"
    gain = np.ones(6)               # 逐关节的指令放大倍数；第一趟是 1，还没有实测可依
    # **放大可以关掉**（环境变量 `PICK_GAIN_MAX`，用户 2026-10-16：「下一步可不可以不放大」）。
    # 写 1 = 这一趟指令该动多少就动多少，不按实测比例放大。默认还是 `GAIN_MAX`（3）。
    # 关掉的代价：舵机带载只走三到五成，一趟到不了位、要几趟才挪过去；好处是不会猛拽。
    # **2026-10-20 试过把上一趟实测的放大留给下一次走位当首趟起点，回退掉了**：
    # 实测那一下 `collect_gain_034151.log` 补趟数 10 → 10 趟没变，抬升还冲过头
    # （目标 80mm，走到 84.7mm 再退回来）。欠多少取决于方向和负载，不是个常数倍率。
    import os                                                   # noqa: PLC0415

    gmax = float(os.environ.get("PICK_GAIN_MAX", "") or GAIN_MAX)
    if gmax <= 1.0:
        print("      （放大关掉了：指令该动多少就动多少）")
    # **「只跑一趟」**（环境变量 `PICK_GOTO_TRIES`，用户 2026-10-16：「能不能就让机械臂只跑
    # 一下，别反复不放大」）。写 1 = 解一次、走一趟、收工，不重解、不补第二趟。
    # 代价：舵机带载欠的那一截没人补，到点会差几毫米（`REAL_GOTO_TRIES` 那一段的来历）。
    # **调用方还能按「这一次走位」单独指定**（`tries=`，用户 2026-10-16：「下探的时候不准
    # 多抖几下，只准一气呵成」「下探后调节轴偏…一步只准动一下」）—— 给 `tries=1` 的调用
    # 就是「解一次、走一趟」，不看环境变量、也不补第二趟。
    tries = (max(1, int(tries)) if tries is not None
             else max(1, int(os.environ.get("PICK_GOTO_TRIES", "") or REAL_GOTO_TRIES)))
    if tries <= 1:
        print("      （只走一趟：解一次走一趟，不重解）")
    if free is not None:
        _nm = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
        print("      （只放 " + "、".join(_nm[i] for i in free) + " 动，别的钉死）")
    for k in range(tries):
        # **停手了就不再解、不再写**（安全闸 / 时间上限都是立 `terminated`）。
        if getattr(arm, "terminated", False):
            out = "term"
            break
        q_now = arm._ticks_to_qpos(arm.read_ticks())
        # **`_ik` 收的是官方 `gripperframe` 口径的位姿，`goal` 是 TCP 口径的，先换算。**
        # 两者差 `TCP_DELTA_BODY` 那 19.9mm。不换算的话解出来的是「另一个点」，手臂老老
        # 实实走过去、也老老实实差在那儿：实测（离线，从初始位置解 [227, 68, 81]mm）
        #     不换算  差 21.7mm    关节 [-17.7 -18.5  20.5  74.1 -108.4  -8.2]
        #     换算过  差  3.4mm    关节 [-17.7 -20.2  14.5  80.  -108.4  -8.2]
        # 换算这条 `RealArm.drive` 一直在用（`real/arm.py:418`）。姿态也要给 TCP 口径的
        # —— `arm.quat()` 是，`arm._fk(q)[1]` 不是（那是 site 口径）。
        # **姿态默认保此刻**；`quat=` 给了就照给的那个解（2026-10-18：对准之前先把
        # 爪尖摆垂直，再转腕 —— 见 `skill.py` 第 4 段）。两者都是 TCP 口径。
        _q_tgt = arm.quat() if quat is None else np.asarray(quat, dtype=np.float64)
        if free is None:
            q_goal = arm._ik(*arm._site_pose(goal, _q_tgt), q_now, ow=ow)
        else:
            q_goal = _ik_free(arm, *arm._site_pose(goal, _q_tgt), q_now, free, ow=ow)
        q_goal[5] = (q_now[5] if grip is None
                     else (getattr(arm, "grip_open", arm.grip_hi)
                           if float(grip) > 0 else arm.grip_lo))
        # **只放大「这一趟要写下去的位移」，不动 `q_goal`。** `q_goal` 还是逆解算出来的那个
        # 点、夹爪那一维照旧取行程端点；放大的只是从此刻走到它的那一段。横向先夹到关节限位
        # 里面，免得放大把某一维推出限位（`goto_q` 不看限位，写下去就是写下去）。
        # 放大的位移也要夹在**安全下限**里，不能只夹硬件行程：`q_goal` 已经夹过
        # （3 号肘 ≥ +1.2°），再乘 1.3 的放大就会越过它 —— 实测 6.8° 往 1.26° 放大成
        # −0.4°，下一行的 `check_joints` 当场立停手，整趟抬升/摆件全成了空跑。
        _slo, _shi = safety.clip_bounds(arm.q_lo, arm.q_hi)
        q_cmd = np.clip(q_now + (q_goal - q_now) * gain, _slo, _shi)
        q_cmd[5] = q_goal[5]
        # **2026-10-22 试 VLM 的整改（只试这一次）**：1 号关节单步最多 `JOINT1_STEP_DEG`。
        # 超了就按限幅值写下去，下一趟从新位置重新逆解；这样 46.5° 的一大步变成两次 23°。
        _d1 = float(q_cmd[0] - q_now[0])
        # 读模块属性而不是 import 进来的常量：VLM 会在运行时改这个值。
        _lim1 = np.radians(float(_knobs.JOINT1_STEP_DEG))
        if abs(_d1) > _lim1:
            q_cmd[0] = q_now[0] + np.sign(_d1) * _lim1
            print(f"      （1 号关节限幅：{np.degrees(_d1):.1f}° → "
                  f"{np.degrees(_lim1):.1f}°，剩下下一趟再走）")
        # **其它关节的单步上限**（2026-10-22）：默认 `None` = 不额外限，三档策略
        # 走的还是原来那条路；只有 VLM 在安全闸之后给了 `step_deg` 才生效。
        # 做法和 1 号那条一样：超了按上限写下去，下一趟从新位置重新逆解。
        # **只在会重解的那条路上限**（`tries > 1`）。`tries=1` 的调用（竖直下探
        # 那一下）本来就只解一次、不补第二趟：限了它只会停在半路，判据当场报
        # 「没到底」。2026-10-22 实测过这一版：2 号 39.2° 被压到 30°，只走了
        # 13mm/23.5mm。
        _lim_any = getattr(_knobs, "JOINT_STEP_DEG", None)
        if _lim_any and tries > 1:
            _la = np.radians(float(_lim_any))
            _da = q_cmd - q_now
            _da[5] = 0.0
            _hit = np.where(np.abs(_da) > _la)[0]
            if _hit.size:
                q_cmd[_hit] = q_now[_hit] + np.sign(_da[_hit]) * _la
                print("      （关节单步限幅 "
                      + f"{float(_lim_any):.0f}°："
                      + "、".join(f"{int(i) + 1} 号 {np.degrees(_da[i]):.1f}°"
                                  for i in _hit) + "）")
        # **写之前拦一道**（用户 2026-10-20：「机械臂姿态偏离太多」）：一步要动多少度
        # 是有上限的，超了就不写、直接停手。上限和来历见 `real/safety.py` 开头。
        if safety.check_step(arm, q_now, q_cmd):
            out = "safety"
            break
        # **2 号肩抬的绝对下限**（2026-10-24 用户定）：后仰不许比初始位置更激进。
        # 这一步和上面那条是两维：单步闸看「一次动多少」，这条看「绝对到哪」。
        if safety.check_joints(arm, q_now, q_cmd):
            out = "safety"
            break
        print(f"      走位（关节空间）第 {k + 1} 趟：{np.round(arm.tcp() * 1000, 1)}mm"
              f" → {np.round(goal * 1000, 1)}mm   要动 "
              f"{np.round(np.degrees(np.abs(q_cmd - q_now)), 1)} 度")
        # **中间那几趟不要每次收尾都停 0.76 秒。** `goto_q` 的收尾是「同一个目标角连发
        # `hold` 遍，再静 `settle` 秒」，默认 8 遍 + 0.5 秒。三趟各停一次，看着就是
        # 「走一下、定住、再挪一点、又定住」—— 用户看录像的原话是「一顿一顿」。中间那几趟
        # 只补一个很小的角度，短收尾就够读准下一次的起点；**最后一趟照旧用默认值**，
        # 到点了要让舵机真正静下来再返回。
        last = k == tries - 1
        q_after = jm.goto_q(arm, q_cmd, q_from=q_now,
                            **({} if last else {"settle": 0.15, "hold": 3}))
        err = float(np.linalg.norm(goal - arm.tcp()))
        print(f"        走到 {np.round(arm.tcp() * 1000, 1)}mm  差 {err * 1000:.1f}mm"
              f"  关节 {np.round(arm.angles(), 1)}")
        # **走完再看一眼整条臂在哪儿**：单步没超、几趟叠起来跑到工作区外一样要停。
        if safety.check_pose(arm, f"走位第 {k + 1} 趟之后"):
            out = "safety"
            break
        # **一趟不够。** `goto_q` 收尾只是把同一个目标角连发几遍，舵机在重力负载下欠的
        # 那一截没人补：实测（COM10 从臂，2026-10-14，进场 [227, 68, 81]mm）第一趟走到
        # 差 13.0mm，读回来 3 号 16.4°、4 号 81.3°，而指令是 14.5°、80.0° —— 各欠
        # 1.9°、1.3°。从**走到的**姿态再解一次同一个点，就是让逆解把欠的那 2° 补上，
        # 一趟收敛一点，几趟内进 `tol`。
        out = "reached" if err < tol else "stall"
        if out == "reached":
            break
        # **量这一趟的响应比，给下一趟放大用。** 换掉的是原来那条「这趟没比上趟更近，就不
        # 再解下一趟」的判据。那条判据的来历是真的（run 52 微调第 3 步：4.4 → 7.4 → 10.2mm，
        # 越解越远），可它把因果搞反了 —— **越解越远是因为每趟都只走掉一小半**，不是因为
        # 「再解没用」。run 69 抬高度那一步（`out/run_real_69.log` 第 58~64 行）就是这么死的：
        # 指令要把 z 从 −0.8mm 抬到 +0.8mm，第 1 趟差 4.2mm、第 2 趟差 7.4mm，判据当场判死刑。
        want = q_cmd - q_now
        got = np.asarray(q_after, dtype=np.float64) - q_now
        ratio = np.divide(got, want, out=np.ones(6), where=np.abs(want) > 1e-12)
        # **走反了就不放大。** 响应比 ≤ 0 说明这一趟压根没朝指令方向走（重力、摩擦、
        # 别的关节的机械耦合盖过去了）—— 这时候「跟到几成」这个信息**不存在**，取倒数会
        # 给出负值或者二十倍，夹到上限就成了「你越不听话我放大得越狠」。`out/run_real_70.log`
        # 抬高度那一步（第 59~69 行）就是这么甩起来的：−2.2 → −3.2 → −4.0 → **+1.9**，
        # 最后一趟 3 号被顶到 7.4°、冲过目标 1.9mm。**前三趟走反、第三趟猛冲**，两个毛病
        # 是同一个：放大用在了测量不支持它的那一档上。
        ratio[(ratio <= 0.0) | (np.abs(want) <= np.radians(GAIN_MIN_ASK_DEG))] = 1.0
        # 上限夹到 1.0：走得比指令多（从没见过）不反过来缩，没见过的情形不猜。
        ratio = np.clip(ratio, 1.0 / gmax, 1.0)
        ratio[5] = 1.0                  # 夹爪那一维是行程端点，不是「走不到」，不参与
        gain = 1.0 / ratio
        amp = " ".join(f"{j + 1}号×{gain[j]:.1f}"
                       for j in range(6) if gain[j] > 1.01)
        if amp and not last:            # 已经是最后一趟就没有「下一趟」了，别打下去了
            print(f"        ↻ 下一趟按实测放大：{amp}")
    # **`_settle` 不叫了。** 它原来是给笛卡尔那条路收残余速度用的（`follow` 到点是
    # 「离得够近了」就返回，那一刻手臂还在动）。关节空间这条路返回时舵机已经静了
    # （`goto_q` 自己的 `hold` + `settle`），再守 40 步就是白站 3.5 秒 —— 一次 `_goto`
    # 一步没多走，光站在那儿。`grip` 也已经由 `q_goal[5]` 直接走到行程端点了。
    # `_grip_cmd` 是**增量**那一维的状态（`drive` 攒它、`_close_until_stall` 从它出发），
    # 刚才那几趟没经过 `drive`，从读数把它对回来，免得下一步从错的地方开始攒。
    arm._grip_cmd = float(arm.grip_q())
    return out

# 够不着时降高度再走的两个数。低于 5mm 的差不算「够不着」（`_goto` 自己的 tol 是 4mm，
# 舵机 2° 死区那一档也在这个量级）；一次最多降两轮，降不下来就照旧收工。
REACH_SHORT_TOL = 0.005
REACH_LOWER_TRIES = 2

def _goto_reach(arm, goal, *, vmax, floor_z, **kw) -> tuple[str, np.ndarray]:
    """`_goto` 走不到就**降低高度再走**。返回 `(结果, 最后用的目标点)`。

    远处的点在高处够不着：`out/panel_20261020_ppo3.log` 纠位那一趟（目标
    [252.8 19.2 70.0]mm）走了四趟停在 [248.5 20.9 61.8]mm，差 9.4mm；同一趟下探到
    z≈18mm 时同样的水平距离是够得着的 —— 臂伸得越长，能到的高度越低。用户 2026-10-20：
    「2 号舵机稍微开大一点，就能让整个机械臂有更长的抓取范围，只不过起始高度更低罢了」。

    降多少用**实测的水平差**，不按系数猜。最低降到 `floor_z`（调用方给：盒子中心 +
    `APPROACH_MIN_ABOVE_M`），再低就没给下探留出空间。
    """
    p = np.asarray(goal, dtype=np.float64).copy()
    out = "stall"
    for _ in range(REACH_LOWER_TRIES + 1):
        out = _goto(arm, p, vmax=vmax, **kw)
        if out != "stall" or getattr(arm, "terminated", False):
            return out, p
        short = float(np.linalg.norm(np.asarray(arm.tcp(), dtype=np.float64)[:2] - p[:2]))
        if short <= REACH_SHORT_TOL:
            return out, p
        new_z = max(float(p[2]) - short, float(floor_z))
        if new_z >= float(p[2]) - 1e-6:
            return out, p
        print(f"    ↕ 这个高度上够不着（水平还差 {short * 1000:.1f}mm）—— "
              f"高度降 {(p[2] - new_z) * 1000:.1f}mm 再走，降到 {new_z * 1000:+.1f}mm")
        p = np.array([p[0], p[1], new_z], dtype=np.float64)
    return out, p

def _descend(arm, xy, z_top, z_bot, *, closing=None, open_at=0.5, step=DESCEND_STEP,
             tol=0.004) -> tuple[str, np.ndarray]:
    """从 `z_top` 垂直降到 `z_bot`：整根柱子上每一点都对准同一个 `xy`、同一个朝向。

    **为什么不是「一边降一边对准」。** 那正是 `core/follow.py` 那条笛卡尔路在做的事，
    它在真机上输了。实测（`out/run_real_18.log`）指令 x 从头到尾都是 0.239，实际 TCP
    从 237 滑到 204 —— 「每一步都纠」纠不动：一步只纠 1.5mm，而 2 号肩在那段时间里
    自己从 −0.8° 沉到 +8.4°。根子不是纠得不勤，是**每一步都在拿新姿态重解一次逆解**：
    同一个位置，种子姿态变一点、六个关节就重新分一遍，4 号被顶到 94.8°（上限 95°），
    多出来的误差只好由肩去扛。

    **这一支先把整列解好、再一口气走完。** 每一点拿**上一个点的解**当种子（相邻两点
    只差 5mm，解不会跳），走的过程中**不读、不解**，所以肩没有「自己沉」的机会。

    **朝向锁的是「竖直 + 目标侧面法向」，不是「此刻的朝向」。** 竖直这一半是实测逼出来的。
    方位角那一半照仿真：`skills/pick.py:702` 把 `closing=closing` 传进 `follow`，整段
    下探都在往目标法向上拧，转正剩下的残差到合爪时已经被抹平。真机原来只锁竖直，
    方位角是转正那一刻继承的（`run_real_24` 第 1 趟继承 13.7°）。`_align`（转正）
    那 200 步是笛卡尔守位，肩会被姿态目标拖着往上抬，而肩一抬工具轴就跟着歪 ——
    同一个纠位姿态把 2 号一路抬上去，正解出来的工具 z 轴朝下分量：

        2号 −11.8° → −0.979（歪 12°）    +10° → −0.985     +30° → −0.867（歪 30°）

    实测（`out/run_real_19.log` 第 1 趟）下探解到底那点 `[−6.9 42.8 −36.3 95.0 −63.2]`，
    4 号钉在 **95.0° 限位**上，正解出来离目标 **33.9mm** —— 手臂照做，走完读回来
    `[−6.4 43.2 −33.9 94.8 ...]` 正解 `[251.6 30.9 −10.1]mm`，跟日志报的一模一样，
    说明舵机没欠账，是**那个点本身就够不着**。拿同一列换六个种子重解，收敛到同一个
    钉限位的解，也说明不是种子的问题，是**锁了一个歪的朝向**。
    钉竖直之后同一列再解：底点 4 号 51.2°、离限位 43.8°、残差 0.01mm。

    开合度沿柱子线性过渡：`open_at` 之前合到底、之后渐变到张到底。放进同一条柱子里是
    为了让样条连续；单独切一刀会在那儿留个跳变。

    和 `_goto` 一样解 `REAL_GOTO_TRIES` 趟：舵机在重力负载下停在指令角之前（实测 3 号
    欠 1.9°、4 号欠 1.3°），从**走到的**姿态重解整列能把这一截补回来一点。
    """
    from real import joint_move as jm                              # noqa: PLC0415
    from real import safety                                        # noqa: PLC0415

    xy = np.asarray(xy, dtype=np.float64).reshape(2)
    q_before = arm.quat()                  # TCP 口径；`_site_pose` 会换算成官方口径
    quat0, off = _vertical(q_before, closing)
    if closing is not None:
        from scipy.spatial.transform import Rotation            # noqa: PLC0415
        R0 = Rotation.from_quat([q_before[1], q_before[2], q_before[3],
                                 q_before[0]]).as_matrix()
        R1 = Rotation.from_quat([quat0[1], quat0[2], quat0[3],
                                 quat0[0]]).as_matrix()
        a = R0[:, 0] - float(R0[:, 0] @ np.array([0.0, 0.0, -1.0])) * np.array([0.0, 0.0, -1.0])
        b = R1[:, 0] - float(R1[:, 0] @ np.array([0.0, 0.0, -1.0])) * np.array([0.0, 0.0, -1.0])
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        az = (float(np.degrees(np.arccos(np.clip(float(a @ b) / (na * nb), -1.0, 1.0))))
              if na > 1e-6 and nb > 1e-6 else 0.0)
        print(f"      进柱子那一刻的朝向离垂直 {off:.1f}°，方位角比目标侧面法向差 "
              f"{az:.1f}° —— 两个都锁到目标法向上")
    else:
        print(f"      进柱子那一刻的朝向离垂直 {off:.1f}° —— 按「工具 z 轴朝下」锁，"
              f"不锁它")
    goal_bot = np.array([xy[0], xy[1], float(z_bot)])
    g_lo, g_hi = float(arm.grip_lo), float(arm.grip_hi)
    ramp = max(1.0 - float(open_at), 1e-6)      # `open_at` 之后到终点的那一段
    span = abs(float(z_top) - float(z_bot))
    n = max(int(round(span / step)), 1) + 1
    out = "stall"
    for k in range(REAL_GOTO_TRIES):
        # 第一个点是**此刻的**姿态：样条从真身所在的地方起，不是从空中某个点起。
        q0 = arm._ticks_to_qpos(arm.read_ticks())
        q0[5] = g_lo
        q_list = [q0]
        q_seed, hop, worst = q0, 0.0, 0.0
        for i in range(n):
            f = i / (n - 1.0)
            z = float(z_top) + (float(z_bot) - float(z_top)) * f
            goal = np.array([xy[0], xy[1], z])
            q = arm._ik(*arm._site_pose(goal, quat0), q_seed)
            q[5] = (g_lo if f <= open_at
                    else g_lo + (g_hi - g_lo) * min((f - open_at) / ramp, 1.0))
            hop = max(hop, float(np.max(np.abs(np.degrees(q[:5] - q_seed[:5])))))
            # **先看这一列自己解到哪儿了。** 解出来够不着就别走 —— 上一趟就是在这种
            # 列上把臂开过去的（到底那点残差 33.9mm）。
            got = arm._sim_pose(*arm._fk(q))[0]
            worst = max(worst, float(np.linalg.norm(got - goal)))
            q_list.append(q)
            q_seed = q
        print(f"      下探（关节空间）第 {k + 1} 趟：{n} 个点，相邻两点最多差 "
              f"{hop:.1f} 度，逆解自己最大欠 {worst * 1000:.1f}mm；"
              f"到底那点解成 {np.round(np.degrees(q_seed), 1)}")
        if worst > tol:
            print(f"        ✗ 这条列解不到：最大欠 {worst * 1000:.1f}mm，不走")
            return "stall", arm.tcp()
        # **写之前过同一道安全闸**（见 `real/safety.py`）。
        if safety.check_step(arm, q_list[0], q_list[-1]):
            return "safety", arm.tcp()
        # 这一列是**一次解好、中途不再读**的，2 号肩抬的绝对下限得逐点看：
        # 起点、终点都在限制内、中间某一点越过去，一样会把臂带出去。
        _worst = min(q_list, key=lambda q: float(q[1]))
        if safety.check_joint_abs(arm, _worst):
            return "safety", arm.tcp()
        # 3 号肘同样逐点看，取整列里**折得最狠**的那一点：起点终点都合规、中间某点
        # 越过 75° 夹角，一样会把臂折进去。
        _fold = min(q_list, key=lambda q: float(q[2]))
        if safety.check_joint_abs(arm, _fold):
            return "safety", arm.tcp()
        # 中间那几趟短收尾，最后一趟照旧 —— 理由和 `_goto` 同。
        last = k == REAL_GOTO_TRIES - 1
        jm.goto_path(arm, q_list,
                     **({} if last else {"settle": 0.15, "hold": 3}))
        tcp = arm.tcp()
        if safety.check_pose(arm, f"下探第 {k + 1} 趟之后"):
            return "safety", tcp
        err = float(np.linalg.norm(goal_bot - tcp))
        print(f"        走到 {np.round(tcp * 1000, 1)}mm  差 {err * 1000:.1f}mm"
              f"  关节 {np.round(arm.angles(), 1)}")
        out = "reached" if err < tol else "stall"
        if out == "reached":
            break
    arm._grip_cmd = float(arm.grip_q())
    return out, arm.tcp()

def _wrist_turn(arm, deg: float) -> float:
    """**只拧 5 号**（wrist_roll）`deg` 度，走关节空间绝对角。返回实际拧了多少度。

    `real/joint_move.goto_q` 提前把整条轨迹算好、一路写绝对角、中途不读 —— 跟
    `_goto` 同一个机制。其它五个关节原样写回去，所以这一步不动位置、只转腕子。
    5 号自己的行程从 MJCF 的 `jnt_range` 读，不另抄一份数。
    """
    from real import joint_move as jm                               # noqa: PLC0415
    from real.arm import JOINT_ORDER                                # noqa: PLC0415

    idx = list(JOINT_ORDER).index("wrist_roll")
    q0 = np.array(arm._ticks_to_qpos(arm.read_ticks()), dtype=np.float64)
    q = q0.copy()
    q[idx] = float(np.clip(q0[idx] + np.radians(float(deg)),
                           arm.q_lo[idx], arm.q_hi[idx]))
    d = float(np.degrees(q[idx] - q0[idx]))
    if abs(d) < 1e-6:
        return 0.0
    jm.goto_q(arm, q, q_from=q0)
    return d
