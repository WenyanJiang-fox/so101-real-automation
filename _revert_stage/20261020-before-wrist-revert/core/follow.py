"""跟着一条算好的轨迹走，边走边看反馈。

分工是这样的：
    **走哪条路**（拐角绕开底座、最后一段竖直下探）→ `geom.round_route` 把路点铺出来
    **怎么走快**（加速、巡航、提前刹车）    → `trajectory` 交给 toppra 算
    **跟上它**（这一步给什么指令、什么时候算到了）→ 这个文件

**单位说清楚**：toppra 只要求单位一致，不关心是什么单位。这里全程用「每控制步」
而不是「每秒」—— vmax 是米/步，amax 是米/步²，于是 `trajectory.duration` 的单位
就是**步数**，跟控制循环一一对应，不用去猜仿真一步是多少秒。

**进度怎么走**：按步推进（一步一个时间单位），另拿「轨迹上离手臂最近的那一点」
兜底 —— 手臂要是冲在前面，进度就跟着它走，不会落在它后面。**反过来不成立**：
进度不等手臂，手臂跟不上时进度照样往前走，靠失速判据收场。

**指令幅度按位置误差给**，追「再往前一步」那个点：欠多少补多少。既不取轨迹这一刻
的速度（开环，落后会一路攒着，收尾才慢慢蹭回来），也不取到「当前进度那一点」的
距离（起步时那点离手臂不到一毫米，误差趋零就死锁）。
"""

from __future__ import annotations

import numpy as np

from core import trajectory
from core.geom import round_route
from core.params import POS_SCALE, REACH_TOL

# 指令每步最多变多少。**这就是「加减速有多急」**：它和 toppra 的 amax 说的是同一
# 件事（amax 是米/步²，这里是动作量/步），两个得一起调。调小了起步和停住都是慢慢
# 化开的，代价是整段变慢。
# **又砍了一次半**（0.008/0.020 → 0.004/0.010 → 这里）跟着 params.SPEED、ACCEL 一起降
# —— 那两 个数讲的是手臂该走多快，这两个讲的是指令允许多急地变过去，一组不一起降就成了
# 「慢慢走、到点一脚刹死」。这次的原由是用户要「真机架不住大机动，改的温和一点」。
POS_SLEW = 0.002
ROT_SLEW = 0.005

# 落后参考点了，反馈最多补多少（动作单位）。补的是「手臂没跟上轨迹的那一段」，
# 不是「到目标点还有多远」—— 后者在起步时趋零，会死锁。0.25 相当于落后 5mm 时
# 补满。
#
# **不要再往下调。** 试过 0.02：进场在第 86 毫米处就 stall 了，手臂再也挪不动 ——
# 这个数不是「追得多猛」的余量，是**填 PD 稳态误差**用的。手臂实际走出的位移远小于
# 发出的指令，光靠前馈（一个巡航 0.030）补不上这个差，攒着攒着就停住了。
# 末段速度靠的不是压它，是下面那条「刹车封顶」—— 那条只按距离限速，和这个数不冲突。
LAG_GAIN = 0.25


def follow(arm, waypoints, *, vmax, amax, grip=0.0, closing=None,
           final_tol=0.004, max_steps=2000, stall=40, stall_eps=4e-4,
           guard=None, on_step=None, pos_slew=POS_SLEW, rot_slew=ROT_SLEW,
           lag_gain=LAG_GAIN, density=0.01, verbose=None,
           check_reach=True) -> str:
    """沿 waypoints 走一条 toppra 算出来的轨迹。

    vmax / amax : 速度、加速度上限，单位是「米 / 控制步」和「米 / 控制步²」。
    final_tol   : 离最后一个点多近算到了。**只看距离，不看进度** —— 加进度条件的
                  话，手臂被挡在半路时进度先到头，就再也不会返回，只能等失速。
    stall       : 每多少步比一次位置；连续这些步里挪动不到 stall_eps 就判「卡住」。
    guard       : 每步回调一次，返回真就中断（下探时用来盯指垫是不是提前碰上了）。
    on_step     : 每步回调一次，不看返回值（用来每步扫一遍有没有撞到别的方块）。
    lag_gain    : 反馈项封顶（动作单位，模块级 `LAG_GAIN` 那条注释里有它的来由）。
                  真机那一层要传更大的值，见 `skills/pick_real.py` 的同名注释。
    verbose     : 每隔一段回调一次，给日志用。
    check_reach : 出发前问不问 `arm.reach_residual`。真机那一层要传 False，见下面那段。

    返回 'reached' / 'guard' / 'stall' / 'steps' / 'term' / 'bad_path' / 'unreachable'。
    """
    goal = np.asarray(waypoints[-1], dtype=np.float64)
    # **够不着就别出发。** 逆解对够不着的点不报错，它返回「最接近的那一个」：关节夹在
    # 行程上限，手臂走一段就停住，外面只看到「走了一半不动了」——之前中立位就是这么
    # 耗掉几百个控制步、最后判失速的。这里出发前先问一句，够不着当场报出来、一动不动。
    # 不实现 `reach_residual` 的机械臂（真机那一层）自动跳过这一步，和 `dry_run` 一个写法。
    # check_reach=False 关掉这一步。给真机用：那个判断是「从当前姿态出发、保持当前
    # 朝向、解 3 次」的局部判断，手臂停在别的姿态时会给出假否定（拉直姿态下中立位
    # 报 71.6mm，实际走得到 0.8mm），于是这里原地不动。**关掉不等于没有兜底** ——
    # 走不到就由下面的 stall（连续 40 步挪动不到 0.4mm）接住。
    probe = getattr(arm, "reach_residual", None) if check_reach else None
    if probe is not None:
        for w in waypoints:
            gap = float(probe(w))
            print(f"      够得着吗：{np.round(np.asarray(w, dtype=np.float64), 3)}"
                  f"  差 {gap * 1000:.1f}mm")
            if gap > REACH_TOL:
                print(f"    ✗ 这个点够不着（差 {gap * 1000:.0f}mm）："
                      f"不去 —— 去了也是走到半路停住，还多耗几百步")
                return "unreachable"
    try:
        wps = round_route([arm.tcp()] + [np.asarray(w, dtype=np.float64) for w in waypoints])
        traj = trajectory.build(wps, vmax=vmax, amax=amax, density=density)
    except (ValueError, RuntimeError) as exc:
        print(f"    ✗ 这条轨迹走不通：{exc}")
        return "bad_path"

    # 投影用的表：轨迹上每隔一步取一个点，找离手臂最近的那个
    n = max(int(np.ceil(traj.duration)) + 1, 2)
    ts = np.linspace(0.0, traj.duration, n)
    xs = np.array([traj.at(t)[0] for t in ts])

    t = 0.0
    ref = None
    for k in range(max_steps):
        tcp = arm.tcp()
        # **进度按步自己往前走，不靠手臂带动。** 靠手臂带动（取轨迹上离手臂最近的那一点）
        # 在这里是死锁：轨迹从手臂当前位置起算，起步时「轨迹上那一点」离手臂只有零点
        # 几毫米，照它给指令就是 ~0，手臂挪过去就停在原地，进度再也涨不了。所以进度
        # 按步推，另外拿投影兜底（手臂冲在前面时进度跟着它走，不落在它后面）。
        t = max(t, float(ts[int(np.argmin(np.linalg.norm(xs - tcp, axis=1)))]))
        t = min(t + 1.0, traj.duration)
        # **指令按位置误差给**，追「再往前一步」那个点。不按轨迹在这一刻的速度给 ——
        # 那是开环的：手臂的 PD 有稳态误差，每步欠一点会一路攒着，攒到轨迹走完才由
        # 收尾那一下慢慢找补，看着就是「早走完了还在原地蹭」。按误差给，欠多少补多少；
        # 追平了，这一步的位移就等于轨迹自己的速度，两种写法重合。
        tgt, ref_v = traj.at(t)
        # 前馈 + 反馈。前馈是轨迹自己的速度：跟得上时它就等于巡航，和「按速度给」
        # 那版一样稳。反馈按**落后参考点多少**补 —— 手臂的 PD 有稳态误差，光有前馈
        # 每步欠一点会一路攒着，攒到轨迹走完才由收尾那一下慢慢找补。
        # **反馈项要封顶**：不封的话误差一大就顶到满速，配上 0.008/步的斜率限幅，
        # 指令追不上手臂，走近目标时来回蹭（实测进场段从 39 帧涨到 400 帧）。
        # 喂给 drive 的是绝对位置，所以这里先在动作单位上算好，再乘回去。
        want = ref_v / POS_SCALE + np.clip((tgt - tcp) / POS_SCALE,
                                          -lag_gain, lag_gain)
        want = np.clip(want, -1.0, 1.0)
        # **刹得住的封顶。** 上面那条指令是按误差给的：离目标还差 2mm 就发 0.1，
        # 那等于用 2mm/步 去补 2mm 的差 —— 一步冲过头。而减速能力只有 pos_slew/步，
        # 从 0.1 刹到零要滑出二十几毫米（实测守位那 40 步里手臂跑出去 34mm 就是这么
        # 来的）。所以给整条指令再套一层「在这个距离上还刹得住的速度」上限：
        #     |v| ≤ √(2 × pos_slew × 到目标的距离 / POS_SCALE)
        # 这条曲线每一步的斜率恰好等于 pos_slew，斜率限幅跟得上它，两者不打架；
        # 离得远时上限远大于巡航，等于没加；只有最后那十几毫米才咬合。
        room = float(np.linalg.norm(goal - tcp)) / POS_SCALE
        cap = float(np.sqrt(2.0 * float(pos_slew) * max(room, 0.0)))
        mag = float(np.linalg.norm(want))
        if mag > cap > 0.0:
            want = want * (cap / mag)
        # want 是动作单位，乘回 POS_SCALE 就是本步该走的位移；drive 再除回来原样发出
        arm.drive(tcp + want * POS_SCALE, closing=closing, grip=grip,
                  pos_slew=pos_slew, rot_slew=rot_slew)
        if verbose is not None:
            verbose(k, tgt, t, traj.duration)

        if guard is not None and guard():
            return "guard"
        if arm.terminated:
            return "term"
        if on_step is not None:
            on_step()

        now = arm.tcp()
        if np.linalg.norm(goal - now) < final_tol:
            return "reached"
        if ref is None or k % stall == 0:
            if ref is not None and np.linalg.norm(now - ref) < stall_eps:
                return "stall"
            ref = now
    return "steps"


def hold(arm, steps, grip=0.0, closing=None, pos_slew=0.012, rot_slew=0.03,
         at=None, brake=True, on_step=None):
    """位置守住不动，只开合夹爪（合爪 / 松爪）。

    守住的位置不是硬刹：残余速度按同样的斜率收到零，手臂是滑停下来而不是撞停，
    也不会有一下反向的抽动。**另外给指令套了刹车封顶**（`brake`）：守的点不动，
    所以「离点还有多远」就是实打实的剩余路程，可以按「这个距离上刹得住的速度」
    给指令，手臂是走回去的，不是冲过去再退回来。

    at      : 守在哪。不给就守在当前 TCP。**进场到点之后要用它**：`follow` 是「离
              目标进到 final_tol 就说到点了」，那一刻手臂还差几毫米、也还在动，
              想真落到目标点上就得把目标点本身告诉它。
    brake   : 刹车封顶开关，默认开。守点不动时没有理由关。
    `grip` 可以是常数，也可以是每步现算的函数（守位用）。
    """
    hold_at = (arm.tcp().copy() if at is None
               else np.asarray(at, dtype=np.float64).copy())
    for _ in range(int(steps)):
        arm.drive(hold_at, closing=closing, grip=grip,
                  pos_slew=pos_slew, rot_slew=rot_slew, brake=brake)
        if arm.terminated:
            return
        if on_step is not None:
            on_step()
