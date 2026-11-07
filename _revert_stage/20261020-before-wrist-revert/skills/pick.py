"""抓取：中立位 → 标识点上方 → 看一眼量位置和朝向 → 挪到量出来的方块正上方 →
原地把腕子转正 → 竖直下探到侧面 → 合爪挤到位 → 抬到搬运高度。

**每一步都不开环。** 下探带守卫（指垫提前碰上就报出来），合爪走「贴住再挤一点点」
（`close_until_touch`），抬升守着自己合出来的那个开口走 —— 那股恒定的夹持力**有个
下限**（`grip_hold` 看 `params.GRIP_SLACK`）：合到比合爪那一刻紧过一段余量，就反过来
松一点，不让它继续往里挤。不给下限的话方块会被这股持续的力从两指之间一点点挤出去。

下探路上指垫擦到方块**只记录，不截断**。守卫截断下探的后果实测过：固定指垫刚擦到
方块顶沿（触点 z=0.029，方块顶面 0.030，而落点深度是方块中心 0.015）就把这一段掐掉，
合爪合在顶沿上 —— 夹得太浅，抬起时方块滑出去。落点本来就是方块中心高度，爪子到不了
地面；深度该由 `aim` 决定，不该由「刚碰到」决定。擦到的那几个数照旧打进日志。
真到了合不上那一步，`_close_until_touch` 自己会返回不成功。

**抬升路上掉了就回去再抓一次。** 抬升挂着守卫（`_gone_guard`），方块一离开两个指垫
当场停下 —— 不是等抬完才查一次。停的那一刻手臂自己的坐标就是最好的线索：方块是从
那儿掉下去的。按这个坐标重走一遍整个抓取（`group="掉了 → 回来重抓"`，只给一次机会，
`retry=False` 再掉就如实报失败）。这不是防滑 —— 防滑不做，是掉了以后能自己捡回来。

**只管相机，不读方块的真值位姿。** 到标识点上方以后**用腕部相机看一眼，方块在哪、
朝哪都从这一张图里量**（`core.vision.estimate_pose`），不是去问仿真。标识点坐标只用
来带路（走到方块大概在这一带的上方）和当起搜的起点 —— 操作员摆偏了、上一把放偏了，
标识点都不会跟着动。抬完也只看指垫还贴不贴着，不再读方块抬了多高 —— 那个由判官用
真值说。

**要抓哪一块，可以是外面告诉我的，也可以是画面里按指令挑出来的**（`percep` 那一路）。
给了感知，进场第一步就变成「抬头看整张台面，按那句话挑出目标方块、量出它在地面上的
位置」，`index` 跟着改成挑中的那一个 —— 目标是画面上的判断，不再是仿真真值。挑错了
就是抓错东西，判官照实判失败，没有谁替它圆过去。

**看一眼要是没认出来，这把就不抓。** 不给兜底值、不按「摆正」硬来 —— 拿一个自己都
不信的数去转腕子、定落点，比不抓更糟：`core.vision.estimate_pose` 会把「这一眼算不
算数」（剪影重合度）一起报出来，低于线就返回 None，这里当场收场。

**腕子转正在下探之前单独做掉，不塞进下探里。** 以前是边走边转，走到方块边上腕子
还在甩；而且落点是按转之前那个开合轴算的，转的角度一大，落点就偏了。现在先原地
转到位、再按转完的开合轴算落点、然后只剩纯下降。

**落点按「不动的那个爪子」算，不是按开口中心。** 开口中心对着物体中心的话，固定爪
离物体还差半个开口，合爪时活动爪得先把物体横推过去才夹得住 —— 物体是被顶走的，
夹稳的余量全耗在这一推上。改成让固定爪的指面落在物体朝它那一侧的面上（留一条
缝，下探时不刮），活动爪再合过来就是实打实的两面夹。

**抓的是什么形状、多大、从哪儿夹，全问 `core/shapes.py` 那张表。** 这一支以前是照
30mm 方块写死的（投八个角、四条侧面法向、半宽 15mm、落下就停在中心高度）。现在：
剪影由形状给（方块八个角、圆柱两个圆、球面撒点）、合爪方向由形状挑（圆柱怎么转都能
夹、带把手的杯子绕开把手）、沿爪轴鼓出来多少由形状算（`support`）、夹哪个高度由形状
列一串候选。**一个形状给好几种夹法，一种不成换下一种** —— 在原地换，不退回策略层
重来（那要重新进场、重新看一眼）。全试完还没夹住才算「没夹住」。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core import blind, follow, shapes, vision
from core.follow import POS_SLEW, ROT_SLEW
from core.geom import PICK_XY, out_of_reach
from core.interfaces import Skill
from core.params import (AIR_TOL, ALIGN_CHUNK, ALIGN_MAX_STEPS, ALIGN_ROT_SLEW,
                         ALIGN_TOL_DEG, CARRY, FIX_CLEAR, GENTLE, GRIP_BIAS,
                         GRIP_K, GRIP_RATE, GRIP_SLACK, LIFT_Z, SEAT, SETTLE_STEPS, SLOW,
                         OPEN_CLEAR, SPEED, STUCK_TOL, acc_amax, speed_vmax)
from core.registry import register

CUBE_HALF = 0.015   # 30mm 方块的一半。形状表给不出来时用它兜底（老口径）

# 原地那一眼没认出方块时，围着原地换几个机位再看 —— 转几个方向。
# **6 是能保证「挪多远就补多远」的最小值**：机位挪 r 之后，方块离新机位画面中心
# 最坏是 2·r·sin(180°/K)。K=6 时正好等于 r，还落在这一眼视场的内切圆上；
# K=5 时是 1.18r，方块又出去了。
SWEEP_VIEWS = 6

_FAIL_DIR = Path(__file__).resolve().parent.parent / "out"


def _save_fail(cams, color: str) -> None:
    """量不出来的时候，把这一眼**每一路**的画面都留下来。

    「对不上」有几种：框投错了位置，还是画面里那坨色块根本不是方块，还是腕部相机
    自己报的位姿不对。日志里的数字分不出这几种，图能 —— 多留一路就多一个对证的。

    全部留在 out/ 下，和这一轮的记录放一起。
    """
    import cv2

    _FAIL_DIR.mkdir(parents=True, exist_ok=True)
    for name, cam in (cams or {}).items():
        p = _FAIL_DIR / f"_look_fail_{color}_{name}.png"
        cv2.imwrite(str(p), np.asarray(cam.shot())[..., ::-1].astype(np.uint8))
        print(f"      {name} 那一帧 -> {p}")


def _wrist(cams):
    """腕部相机那一路。名字里带 wrist 的那个 —— 换个实现只要还叫这个名就行。"""
    return next((c for n, c in (cams or {}).items() if "wrist" in n), None)


def _shape_of(arm, index: int):
    """第 `index` 件东西是什么形状。

    正常口径：问机械臂（`arm.object_shape`，读的是摆件清单的真值）。
    **盲测口径**（`core.blind.ON`）：走 `blind.shape_of` —— 它读的是**相机估出来的
    那一份**，清单里那些数它一个也拿不到。看一眼之前给的是个中性的 30mm 方块。
    """
    if blind.ON:
        return blind.shape_of(arm, index)
    get = getattr(arm, "object_shape", None)
    return get(index) if get is not None else shapes.Box(CUBE_HALF)


def _z_of(arm, index: int) -> float:
    """第 `index` 件东西躺在地面上时中心的高度。够不着形状表就退回方块那 15mm。"""
    return float(_shape_of(arm, index).rest_z)


def _open_grip(arm, sh, g=None):
    """这一把张爪张到多大。

    - **实心件张满**（1.0 = 每步都往开的方向推，一直推到最大）—— 张得开才容得下，
      夹哪儿都行。
    - **中空件夹外壁**：张到「外径 + `OPEN_CLEAR`」。两个指垫只要夹住那圈壁，张满再
      合，活动爪要多扫一大段空行程，真机上那一下很晃。
    - **中空件内外壁对夹**（`g.wall`，见 `shapes.Grasp.wall`）：张到**活动爪正好落在
      物体中轴上**。它得从口子伸进容器里去，落点又不变（固定爪照样贴着外表面，离中轴
      `support + FIX_CLEAR`），所以「固定爪到中轴的距离」就是该留的开口宽度。

    **返回的要么是常数 1.0，要么是一个「每步现算」的函数。** 因为夹爪那一维收的是
    增量、不是目标角（见 `SimArm.grip_command`），想让它停在某个角上就必须每步问
    一遍还差多少；给个常数只会一路张到满开。

    宽度→角度不自己推，问机械臂自己的 `open_angle`（真机那一层没有这两条，问不到就
    退回张满，跟以前一样）。这一件夹爪圈不住（比最大开口还宽）也是张满。
    """
    if not getattr(sh, "hollow", False):
        return 1.0
    probe = getattr(arm, "open_angle", None)
    cmd = getattr(arm, "grip_command", None)
    if probe is None or cmd is None:
        return 1.0
    half = float(sh.support(0.0, arm.tool_x()))
    if getattr(g, "wall", False):
        width, how = half + FIX_CLEAR, "内外壁对夹：活动爪伸到中轴"
    else:
        width, how = 2.0 * half + OPEN_CLEAR, "外壁：外径加余量"
    q = probe(width)
    if q is None:
        return 1.0
    print(f"      张爪「{how}」{width * 1000:.1f}mm → 停在开合角 {q:+.3f}")
    return lambda: cmd(q)


def _view_radius(cams, z_plane: float):
    """腕部相机这一刻在地面上能看清多大一圈 —— 内切圆半径（米）。算不出来返回 None。

    把画面中心和四条边的中点打回 z_plane 那个水平面，取中心到四条边里最近的那一条。
    这是「换机位该挪多远」的尺子：机位挪 r，原本贴在画面边上的东西正好落到画面正中。

    **内参和位姿都现读**：腕部相机的俯仰、安装位置、视场角每开一局都被重新随机
    （`cam_sim.mount` 那段说的），写死一个数在这种场景里没有意义。
    """
    cam = _wrist(cams)
    mount = cam.mount() if cam is not None else None
    if mount is None:
        return None
    w, h = cam.size()
    k = vision.intrinsics(mount["fovy"], w, h)
    cx, cy = k[2], k[3]
    c0 = vision.backproject(mount["pos"], mount["mat"], k, cx, cy, z_plane)
    if c0 is None:
        return None
    rs = []
    for u, v in ((0.0, cy), (w - 1.0, cy), (cx, 0.0), (cx, h - 1.0)):
        p = vision.backproject(mount["pos"], mount["mat"], k, u, v, z_plane)
        if p is not None:
            rs.append(float(np.linalg.norm(p[:2] - c0[:2])))
    return min(rs) if rs else None


def _report(hits, arm) -> None:
    """把验算结果打出来：第几步、碰上哪个方块、被手臂哪一段碰的。"""
    if not hits:
        print("    验算：这条路先在一份副本上走了一遍，没碰到任何方块")
        return
    names = arm.cube_order
    k, ci, name, p = hits[0]
    print(f"    ⚠ 验算：第 {k} 步就会碰上 {names[ci]} 方块"
          f"（手臂的 {name}，接触点 {np.round(p, 3)}）")
    pairs = []
    for _, c2, n2, _ in hits:
        if (c2, n2) not in pairs:
            pairs.append((c2, n2))
    print(f"      一路上一共 {len(hits)} 处接触，牵涉 "
          + "、".join(f"{names[c2]} 被 {n2}" for c2, n2 in pairs))


def grip_hold(arm, state):
    """夹住之后给的那一维指令：恒定合爪一份，再按「被撑开多少」补回来。

    为什么不能给 0 —— 见 params.GRIP_BIAS。

    **开口有个下限**（`params.GRIP_SLACK`）。合到比合爪那一刻还紧、紧过一段余量之后，
    就反过来松一点，不让它继续往里挤。为什么要这一条：那股恒定的合爪力会把方块从
    两指之间一点点挤下去 —— 光看接触看不出来（指垫一直贴着），爪关节角才是指纹。
    松这一点不会把方块掰开，它只是把那股持续的挤压力卸掉。
    """
    ref = state.grip_ref
    if ref is None:
        return 0.0
    q = float(arm.grip_q())
    if q < ref - GRIP_SLACK:
        return GRIP_BIAS            # 已经比下限更紧了：松回去，别再往里挤
    rel = float(np.clip((ref - q) / GRIP_K, -1.0, 0.0)) * GRIP_RATE
    return rel - GRIP_BIAS


def _gone_guard(arm, state, i, need: int = 3):
    """「方块不在爪里了」的守卫，挂在抬升和搬运那几段 `follow` 上。

    只有一条：连着 `need` 步两个指垫都没有接触才算数 —— 合爪刚停的那一刻接触表可能
    瞬时读到 0，一步就判会把它变成随机的假警报。真的滑出去了是持续为 0，三步足够。

    **曾经加过第二条 `arm.grip_q() < ref - GRIP_SLACK`（爪比合爪那一刻紧过一段余量），
    撤掉了。** 撤的依据是它自己的实测后果：`out/round_01..03.json` 里它响的时候，
    方块两个指垫上还压着 9/10、7/10、0/5 个接触点，方块就在爪里；一响 `place` 就走
    「回去重抓」，那一段第一步是张开爪子 —— 方块是被自己人甩出去的（round_02 里方块
    从 z=0.076 被抛到 z=0.17，落点偏 45mm）。成功率 4/5 → 2/5。两个成功的轮
    （04/05）反而不响，所以它不是「掉了」的判据，是「快掉了」的判据。

    而且那个阈值和 `grip_hold` 里的是同一个 `GRIP_SLACK`：那边到线是往回松一点
    （参见 params.GRIP_BIAS 那一段），这边到线是扔掉重抓 —— 同一个数给出两个相反的
    处置。滑出去这件事的正确处置是前者。

    `state` 现在没用上，留着是因为下一个判据（方块自己掉了多高）还得读它。

    返回给 `follow(guard=...)` 用的回调 —— 返回真就当场中断那一段路，交给调用方处理。
    """
    n = [0]

    def check():
        nf, nm, _ = arm.finger_cube_contacts(i)
        n[0] = n[0] + 1 if (nf == 0 and nm == 0) else 0
        return n[0] >= int(need)

    return check


def _close_until_touch(arm, i, closing, max_steps=600, on_step=None):
    """一小步一小步合爪；两个指垫都贴到之后，再往里挤一点就停，记住这个开口。

    不一路合到底：那样指垫是撞上去的，方块会被推着横移、被顶得跳起来。「刚贴上就
    停」也不行：那一瞬间两面的法向力约等于零，抬起来只靠这点摩擦拉不住，会滑。
    折中是再合一点点，停住之后由 grip_hold 守着这个开口，不继续加力。

    返回 (成没成, 这一步看到的事实)。事实里有四个数：跑完的时候开合角多少、
    两边各贴了几个点、贴上那一刻的开合角是多少、一共走了几步。**两边是不是各贴
    一个点**是分开报的 —— 「一个都没贴到」和「只贴了一侧」原因完全不同。
    """
    hold_at = arm.tcp().copy()
    q0 = arm.grip_q()
    touch_q = None
    q_prev = None
    stuck = 0
    nf = nm = 0
    for step in range(1, max_steps + 1):
        q = arm.grip_q()
        nf, nm, _ = arm.finger_cube_contacts(i)
        if touch_q is None:
            if nf > 0 and nm > 0:
                touch_q = q
                q_prev = q
        else:
            if q <= touch_q - SEAT:
                return True, {"steps": step, "q0": q0, "q": q, "touch_q": touch_q,
                              "nf": nf, "nm": nm, "why": "挤到位"}
            # 方块和爪子都是刚体，挤到某个位置就推不动了。推不动就别再耗着。
            stuck = stuck + 1 if (q_prev is not None and q > q_prev - 1e-5) else 0
            q_prev = q
            if stuck > STUCK_TOL:
                return True, {"steps": step, "q0": q0, "q": q, "touch_q": touch_q,
                              "nf": nf, "nm": nm, "why": "推不动了"}
        arm.drive(hold_at, closing=closing, grip=-GENTLE, pos_slew=0.02, rot_slew=0.03)
        if arm.terminated:
            return False, {"steps": step, "q0": q0, "q": q, "touch_q": touch_q,
                           "nf": nf, "nm": nm, "why": "环境终止"}
        if on_step is not None:
            on_step()
    return False, {"steps": max_steps, "q0": q0, "q": arm.grip_q(), "touch_q": touch_q,
                   "nf": nf, "nm": nm, "why": "合爪步数用完"}


@register("skill", "pick")
class PickSkill(Skill):
    """抓一个方块并抬起来。"""

    name = "pick"

    def _find(self, arm, cams, percep, instruction, guess):
        """抬头看整张台面，按指令挑出目标方块，量出它在地面上的位置。

        返回 (位置 或 None, 挑中的那一块是什么颜色)。**读的是画面，不是方块真值** ——
        用的是固定的俯视相机（看得见整个工位），跟下面 `_look` 那一路腕部相机是两回事：
        这一眼管「说的是哪一块」，腕部那一眼管「它具体在哪、朝哪」。

        `guess` 只递给「按坐标取最近」那一路当依据；拿模型挑的那一路不靠它。
        """
        center, info = percep.locate(instruction, guess=guess)
        print(f"    按指令「{instruction}」找：{info.get('how') or info.get('why')}"
              f"   候选 {info.get('n_cand', '?')} 个")
        if "score" in info:
            print(f"      前三名 {info['score']}（块大小 {info['cand']}）")
        if center is None:
            print(f"    ✗ 没认出指令说的那一块 —— 这一把不抓"
                  f"（认不出来就是认不出来，不拿一个自己都不信的位置去带路）")
            _save_fail(cams, str(info.get("color") or "指令"))
            return None, None
        u, v = info["uv"]
        print(f"    挑中 {info.get('color')} 那块：{np.round(center, 4)}"
              f"   画面里 ({u:.0f},{v:.0f})  {info['area']}px")
        self._facts(info)
        # **这一眼里猜出来的形状记下来**，后面转腕子、算落点、选夹法都用它
        # （`blind.shape_of`）。盲测下这是唯一的来源；不是盲测就是 None，什么也不改。
        blind.note(info.get("est"))
        if info.get("est"):
            e = info["est_info"] or {}
            print(f"      看出来的形状：{e.get('estimate')}"
                  f"（朝向 {e.get('yaw_deg', 0.0):+.0f}°，重合度 "
                  f"{e.get('iou_est', float('nan')):.2f}，"
                  f"前三名 {' / '.join(e.get('top') or [])}）"
                  f"  {'中空' if e.get('hollow') else '实心'}"
                  f"（围住的暗块占 {e.get('dark_in', 0.0) * 100:.1f}%）")
        return center, info.get("color")

    def _look(self, arm, cams, color: str, guess, shape=None):
        """看一眼这东西，量它**在哪、朝哪**。返回 (中心坐标, 朝向弧度)；量不出来 (None, None)。

        用的是**腕部相机**：这时候爪子刚停在它正上方，相机正对着它。取哪个相机
        按名字认（名字里带 wrist 的那个），换个实现只要还叫这个名就行。

        `guess` 是带路用的标识点坐标，也是搜索的起点和兜底 —— **不是答案**。量法见
        `core.vision.estimate_pose`：位置和角度一起拟，拿这个形状的剪影投回画面比重合度。
        `shape` 不说就是 30mm 方块那个老口径。对称的东西（圆柱、球）没有朝向可拟，
        量出来的 `yaw` 是 0。

        这里**不给兜底值**。以前量不出来就按「摆正」算，然后照着 0° 去转腕子 ——
        那是拿一个自己都不信的数当答案往下做。量不出来就是量不出来，交给 `run` 收场。
        """
        cam = _wrist(cams)
        mount = cam.mount() if cam is not None else None
        if cam is None or mount is None:
            print(f"    ！没有能报出位姿的腕部相机，量不了 {color}")
            return None, None
        rgb = cam.shot()
        center, yaw, info = vision.estimate_pose(rgb, mount, color, guess,
                                                 shape=shape, half=CUBE_HALF)
        if yaw is None or center is None:
            print(f"    ！没从画面里认出 {color} 方块（{info['why']}）")
            self._facts(info)
            # **只在量不出来的时候读一次方块真值**，纯粹为了对账：上面那个「该在」
            # 是标识点坐标加地面高度，是算出来的；这里读的是仿真当场给的。两个数
            # 一对，就知道是相机这一路错了，还是方块根本不在我以为的地方。
            # 正常流程（量得出来）一个字都不读它。
            if arm is not None and color in arm.cube_order:
                i = arm.cube_order.index(color)
                print(f"      量这一眼的时候，方块真值在 {np.round(arm.cube_pos(i), 4)}")
            _save_fail(cams, color)
            return None, None
        print(f"    看方块：量到位置 {np.round(center, 4)}"
              f"（带路点 {np.round(np.asarray(guess, dtype=float), 4)}）"
              f"  朝向 {np.degrees(yaw):+.1f}°")
        self._facts(info)
        return center, yaw

    @staticmethod
    def _facts(info) -> None:
        """把这一眼看到的事实打出来。**认出来要看，没认出来更要看** —— 对不上到底是
        框错了位置、还是画面里那坨色块根本不是方块，全在这几行里。"""
        if "iou" not in info:          # 连像素都没分出来，没有别的可说
            return
        print(f"      重合度 {info['iou']:.2f}  {info['px']} 像素"
              f"  画面 {info['img']}  fovy {info['fovy']:.1f}°")
        print(f"      画面里的框 {np.round(info['seen'], 1)}"
              f"  投出来的框 {np.round(info['want'], 1)}")
        print(f"      离光轴：画面里 {info['seen_axis']:.1f}°  模型算 {info['model_axis']:.1f}°"
              f"  画面里色块中心 {np.round(info['seen_c'], 1)}")
        print(f"      质心打回地面 {info['seen_world']}（当搜的起点，不是答案）"
              f"  实际起点 {info['seed']}")
        print(f"      拟出来的方块中心 {info['fit']}  标识点给的 {info['guess']}")
        print(f"      相机 {info['cam']}  视向 {info['view']}")

    def _sweep(self, arm, cams, color, guess, state, bump, percep, instruction, shape=None,
               open_g=1.0):
        """原地那一眼没认出方块时：围着原地换几个机位再看一圈，还不行就换俯视相机再看一眼。

        **换机位先验算再走**（`dry_run`）：会碰到方块的哪条路就直接不试，换下一个方向。
        「不管怎么样都不能碰到别的方块」这句话落在这一步上 —— 腕部相机本来就在方块
        上方 80mm，围着原地挪一个视场半径，扫到别人方块头上的机会是有的。

        返回 (中心, 朝向, 怎么找到的)。换了一圈机位、又换了相机都没认出来就是
        (None, None, None)，交给 `run` 收场。
        """
        guess = np.asarray(guess, dtype=np.float64)
        r = _view_radius(cams, float(guess[2]))
        if r is None:
            print("    ！这一眼的视场多大都算不出来（相机报不出位姿/内参），不换机位")
        else:
            for i in range(SWEEP_VIEWS):
                th = 2.0 * np.pi * i / SWEEP_VIEWS
                cand = guess + np.array([r * np.cos(th), r * np.sin(th), 0.0])
                if out_of_reach(cand[:2]):
                    print(f"    · 第 {i + 1}/{SWEEP_VIEWS} 个机位出了可达环，够不着，跳过")
                    continue
                dry = getattr(arm, "dry_run", None)
                if dry is not None:
                    hits = dry(lambda a, c=cand: self._recenter(a, c, open_g=open_g),
                               ignore=())
                    if hits:
                        _report(hits, arm)
                        print(f"    · 第 {i + 1}/{SWEEP_VIEWS} 个机位这么过去会碰到方块，跳过")
                        continue
                state.seg = "换机位找方块"
                got = self._recenter(arm, cand, on_step=bump, open_g=open_g)
                if got != "reached":
                    print(f"    · 第 {i + 1}/{SWEEP_VIEWS} 个机位没走到（{got}），跳过")
                    continue
                print(f"    ↻ 换第 {i + 1}/{SWEEP_VIEWS} 个机位（离原地 "
                      f"{r * 1000:.0f}mm）再看一眼 {color}")
                center, yaw = self._look(arm, cams, color, guess, shape=shape)
                if center is not None and yaw is not None:
                    return center, yaw, f"换机位（第 {i + 1}/{SWEEP_VIEWS} 个）看出来了"

        if percep is None or instruction is None:
            return None, None, None
        # 第二眼换**俯视相机**：它看得见整张台面，按那句话重新挑一次。挑出来的位置
        # 只拿来带路 —— 到底在哪、朝哪，还是拿腕部相机当场量（挑出来什么颜色这一眼
        # 不看，抓哪一块是进场那一步定下来的，不在半路改）。
        print(f"    ↻ 腕部相机换了一圈机位还是没认出来，换俯视相机再看一眼 "
              f"「{instruction}」")
        found, _ = self._find(arm, cams, percep, instruction, guess)
        if found is None:
            return None, None, None
        found = np.asarray(found, dtype=np.float64)
        if out_of_reach(found[:2]):
            print(f"    · 俯视相机这回说的位置 {np.round(found[:2] * 1000)}mm 出了可达环，不去")
            return None, None, None
        dry = getattr(arm, "dry_run", None)
        if dry is not None:
            hits = dry(lambda a: self._recenter(a, found, open_g=open_g), ignore=())
            if hits:
                _report(hits, arm)
                print("    · 俯视相机这回说的位置，这么过去会碰到方块，不去")
                return None, None, None
        state.seg = "换相机找方块"
        got = self._recenter(arm, found, on_step=bump, open_g=open_g)
        if got != "reached":
            print(f"    · 俯视相机这回说的位置没走到（{got}）")
            return None, None, None
        center, yaw = self._look(arm, cams, color, found, shape=shape)
        if center is not None and yaw is not None:
            return center, yaw, "换俯视相机看出来了"
        return None, None, None

    def _align(self, arm, closing, state, bump, open_g=1.0) -> tuple[int, float]:
        """原地把腕子转到开合轴对上方块侧面。返回 (走了几步, 转完还差几度)。

        位置一步不动，只有旋转在爬，所以旋转斜率可以给得比跟轨迹时更小 —— 转起来
        是缓慢拧过去的，不是甩一下。转到 `ALIGN_TOL_DEG` 以内就收，最多耗
        `ALIGN_MAX_STEPS` 步，别死等。
        """
        for k in range(0, ALIGN_MAX_STEPS, ALIGN_CHUNK):
            left = self._align_deg(arm, closing)
            if left is not None and left <= ALIGN_TOL_DEG:
                return k, left
            follow.hold(arm, ALIGN_CHUNK, grip=open_g, closing=closing,
                        pos_slew=POS_SLEW, rot_slew=ALIGN_ROT_SLEW, on_step=bump)
        left = self._align_deg(arm, closing)
        return ALIGN_MAX_STEPS, (float("nan") if left is None else left)

    @staticmethod
    def _align_deg(arm, closing):
        """开合轴（工具 x 轴的水平投影）离目标方向还差多少度。算不出来返回 None。"""
        xt = arm.tool_x()
        a = np.array([xt[0], xt[1]], dtype=float)
        b = np.array([closing[0], closing[1]], dtype=float)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return None
        c = float(np.clip(float(a @ b) / (na * nb), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def _approach(self, arm, src, on_step=None, open_g=1.0):
        """进场那一段路：从当前位置**直奔**方块上方 80mm，**然后停稳再交出去**。

        **不绕中立位。** 中立位（`core.params.WAYPOINT`，z=0.16）原来是这条路的第一
        个途经点，结果是开局先从复位姿势升到 160mm、再降回 80mm —— 白跑一趟竖直的
        来回，真机上那一下很晃。中立位现在只留在 `home`（换目标、回位）那一步用。

        **单独拎出来是为了同一份路能走两遍**：一遍在一份副本上验算，一遍在真身上走。
        两条路必须是同一份 —— 验算的要是另一条路，那就白验了。

        **末尾这段守位不能省。** follow 是「离目标进了 final_tol(4mm) 就说到了」，
        那一刻手臂还在冲。交出这样一个还在动的腕子去拍照、去转正、去下探，后面每
        一步都是从一个晃着的姿态起算。守位用走路时同一组斜率把残余速度滑着收掉，
        守的位置取目标点本身 —— 不是「差几毫米就算到了」的那一点。
        """
        goal = np.asarray(src, dtype=np.float64) + np.array([0.0, 0.0, 0.08])
        out = follow.follow(arm, [goal],
                            vmax=speed_vmax(SPEED), amax=acc_amax(), grip=open_g,
                            closing=None, final_tol=0.004, on_step=on_step)
        if out == "reached":
            follow.hold(arm, SETTLE_STEPS, grip=open_g, at=goal,
                        pos_slew=POS_SLEW, rot_slew=ROT_SLEW, on_step=on_step)
        return out

    @staticmethod
    def _back_off(arm, hover, on_step=None, open_g=1.0) -> None:
        """从一个没夹成的落点退回到下探起点上方，**并把爪张开**，好接着试下一种夹法。

        张开那一步不能省：上一次合爪没成的时候爪是半合着的，直接横着退，指垫会
        贴着物体侧面一路刮过去。`closing=None` 是「腕子保持现在这个朝向，别转」——
        退回来是为了换方向，不是要把方向丢了。
        """
        follow.follow(arm, [np.asarray(hover, dtype=np.float64)],
                      vmax=speed_vmax(SLOW), amax=acc_amax(), grip=open_g,
                      closing=None, final_tol=0.002, on_step=on_step)

    def _recenter(self, arm, src, on_step=None, open_g=1.0):
        """量完位置以后，挪到**量出来的那个位置**上方，再停稳。

        `_approach` 是照标识点走的 —— 那时候还没看，不知道方块在不在工位上。量出来
        偏了 20mm 的话，爪子就停在离方块正上方 20mm 的地方，下探那一段就从「只剩纯
        下降」变成「一边横移一边下降」：横移是贴着方块顶面扫过去的，守卫（指垫一碰
        上就中断）会中途把这一把掐掉。这里纠回来，下探才是原来设计的那条路。
        """
        goal = np.asarray(src, dtype=np.float64) + np.array([0.0, 0.0, 0.08])
        out = follow.follow(arm, [goal], vmax=speed_vmax(SPEED), amax=acc_amax(),
                            grip=open_g, closing=None, final_tol=0.004, on_step=on_step)
        if out == "reached":
            follow.hold(arm, SETTLE_STEPS, grip=open_g, at=goal,
                        pos_slew=POS_SLEW, rot_slew=ROT_SLEW, on_step=on_step)
        return out

    def run(self, arm, cams, state=None, index: int = 0, src=None, group=None,
            retry: bool = True, percep=None, instruction=None, **kw) -> dict:
        """抓一个方块。返回 {'ok': 成没成, ...}。

        `src`   起手的坐标。不给就取标识点那张表（正常一把）；给了就用它 ——
                掉落之后回来重抓就是这么用的，起手点是「掉下去那一刻自己的坐标」。
        `group` 这一整套动作在日志和视频里归到哪一段名下。给了就是回来重抓，
                顺带**跳过验算**（真机上本来也预演不了）、视频里也只切出一段。
        `retry` 抬起途中又掉了的话，允不允许再抓一次。只往下让一层，不会无限套。
        `percep` 感知那一路。给了它，**去哪儿抓就由那句话说了算**：先抬眼看整张台面、
                按指令挑出目标方块，`index` 跟着改成挑中的那一个。不给就还是老口径 ——
                外面告诉我搬哪块（标识点表），我只负责把它看清楚。
        `instruction` 递给感知读的那句话。
        """
        # 进场这一段只能按已知坐标走 —— 还没看，不知道东西在不在工位上。所以 `src`
        # 现在只是「大概在这一带」，下面量完会换成量出来的那个位置。
        if src is None:
            src = np.array([PICK_XY[index][0], PICK_XY[index][1], _z_of(arm, index)])
            if percep is not None:
                # **指令说了算。** index 是外面（仿真真值）告诉我该抓哪件，只当兜底和
                # 对账用；这里改成画面里跟这句话对上的那一件，后面数接触点、放哪儿
                # 全按它走 —— 挑错了就是抓错东西，判官照实判失败。
                found, picked = self._find(arm, cams, percep, instruction, src)
                if found is None:
                    return {"ok": False, "why": "没按指令认出目标"}
                src = found
                if picked is not None and picked in arm.cube_order:
                    if picked != arm.cube_order[index]:
                        print(f"      注意：这一局真值说该抓 {arm.cube_order[index]}，"
                              f"画面里按指令挑出来的是 {picked}")
                    index = arm.cube_order.index(picked)
        src = np.asarray(src, dtype=np.float64)
        color = arm.cube_order[index]
        # 这一件是什么形状、多大。**尺寸是清单里声明好的，不从画面里量** —— 一个
        # 视角分不清「圆柱粗」和「圆柱远」。问不到（换个没有形状表的机械臂）就退回
        # 30mm 方块，跟以前逐字节一样。
        sh = _shape_of(arm, index)
        # 这一把张爪张到多大 —— 实心件张满，中空件按外径给。**这里按这张形状默认的
        # 那一种算**：进场、纠位、看一眼这几步都还在「选定夹法」之前，手上先摆成这个
        # 形状常用的那个开口。选定夹法之后（转正、下探、退回）改按那一种自己的数张
        # —— 内外壁对夹要张得窄，活动爪才伸得进容器里。
        open_g = _open_grip(arm, sh)
        state.grip_ref = None                       # 这一把重新记开口

        hit = [None]

        def bump():
            state.check_bump(arm)
            # 指垫擦到方块只**记下来**，不截断下探。截断的后果实测过：第一次抓取
            # 固定指垫刚擦到顶沿（触点 z=0.029，方块顶面 0.030）就把这一段掐了，
            # 合爪在顶沿合上，抬起时方块滑出去 —— 夹的深度不够。守卫只报数。
            if hit[0] is None:
                nf, nm, z = arm.finger_cube_contacts(index)
                if nf or nm:
                    hit[0] = (nf, nm, z)

        # 进场这一段还不知道方块朝向（要看一眼才知道），腕子先别绕 z 转
        state.seg, state.ignore = (group or "进场 → 方块上方"), {index}
        if group is None:
            # **先验算，再走。** 同一份路先在一份副本上走一遍，看会不会碰到方块 ——
            # 真身和场上的方块一步没动。现在只把结果报出来，还没有「碰了就不走」那一步。
            # 真机上没有副本可预演（那边得拿模型 + 关节角自己算），这一条先跳过。
            dry = getattr(arm, "dry_run", None)
            if dry is not None:
                _report(dry(lambda a: self._approach(a, src, open_g=open_g), ignore=()), arm)
        approach = self._approach(arm, src, on_step=bump, open_g=open_g)
        print(f"    {group or '进场'}：{approach}  TCP {np.round(arm.tcp(), 3)}"
              f"（该到 {np.round(src + [0, 0, 0.08], 3)}）"
              + (f"  去标记点 pickup#{index}（{color}）" if group is None else ""))

        # 位置和朝向：都只从这一张腕部图里量，真值四元数和实时坐标一次不读。
        # 方块量出来的两条棱等价（差 90°），挑法向那一步会在四条里取最贴近开合轴的。
        guess = src.copy()
        center, yaw = self._look(arm, cams, color, guess, shape=sh)
        if yaw is None or center is None:
            # **丢了目标不马上认输。** 先围着原地换几个机位再看（换机位前先验算，
            # 会碰到东西的方向不走），还认不出来就换俯视相机再看一眼 —— 都不行才算
            # 「找不着」。找不着是收工的理由：同一个现场、同一句话，再来一次还是
            # 认不出来，重试只是把同一段路再走一遍。
            center, yaw, how = self._sweep(arm, cams, color, guess, state, bump,
                                           percep, instruction, shape=sh, open_g=open_g)
            if center is None:
                print("    ✗ 换机位扫了一圈、又换了俯视相机，都没从画面里认出方块在哪/朝哪，"
                      "这一把不抓（宁可不抓，也不能拿一个自己都不信的数去转腕子、定落点）")
                return {"ok": False, "why": "找不着：换机位扫一圈、换相机都没认出来",
                        "give_up": True}
            print(f"    ↻ {how}")
        # **从这里开始，`src` 就是量出来的那个位置。** 下面转正、算落点、下探、抬升
        # 全按它走 —— 标识点只带路，到这儿就不再当答案了。方块摆偏了、上一把放偏了，
        # 都是在这一行纠正回来的。
        src = center

        # 进场是按已知坐标走的，方块偏了就不在正上方了 —— 先挪过去，后面才是纯下降。
        state.seg = group or "挪到方块正上方"
        back = self._recenter(arm, src, on_step=bump, open_g=open_g)
        print(f"    纠位：{back}  TCP {np.round(arm.tcp(), 3)}"
              f"（该到 {np.round(np.asarray(src) + [0, 0, 0.08], 3)}）")

        # ---- 夹法：一个形状给好几种，一种不行换下一种 ----
        # **为什么要在原地试。** 夹不住常常不是「算错了」，是夹的那一处不合适 ——
        # 夹到杯口那圈（两个指垫卡在沿上，一抬就翻）、夹到把手、方块本来就歪着。
        # 退回策略层重来要重新进场、重新看一眼；在原地换一处便宜得多。试的过程全进
        # 日志和视频，回头做数据时把试的那一段剪掉就是了。
        bottom = float(src[2]) - float(sh.rest_z)        # 物体底在哪（躺平就是地面）
        cands = list(sh.grasp_candidates(yaw, arm.tool_x(), bottom_z=bottom))
        if not cands:
            cands = [shapes.Grasp(z=float(src[2]), normal=None, label=f"{sh.word} 中心")]

        why = "没夹住"
        info: dict = {}
        for ci, g in enumerate(cands, start=1):
            # 两个爪子往哪个水平方向合。候选自己没说就用形状挑的那个 —— 方块是四条
            # 侧面法向里转得最少的那条，圆柱直接用现成的开合轴（腕子不用多转），
            # 带把手的杯子会绕开把手那一片。
            closing = (sh.normal(yaw, arm.tool_x()) if g.normal is None
                       else np.asarray(g.normal, dtype=np.float64))

            # 这一种夹法张多开（转正、下探、退回都按它）。**内外壁对夹比外壁夹窄一大
            # 截** —— 活动爪得从这个开口里伸进容器。这里问的是 `support`，中空件
            # （杯、碗）报的都是「沿哪个方向都是这个半径」，跟腕子转到哪没关系，所以
            # 在转正之前算和之后算一样。
            og = _open_grip(arm, sh, g)

            # 转正在下探之前单独做掉：位置不动，只有腕子在慢慢拧。塞进下探里转的话，
            # 走到东西边上腕子还在甩，而且下面这几行是按「转完的开合轴」算落点的，
            # 不先转完就按下探前的轴去算，转个几十度落点就偏了。
            state.seg = group or "腕子对上方块侧面"
            steps, left = self._align(arm, closing, state, bump, open_g=og)
            print(f"    夹法 {ci}/{len(cands)}「{g.label}」：转正 {steps} 步，"
                  f"开合轴离侧面还差 {left:.1f}°   夹在 z={g.z:.3f}")

            # 落点：让不动的那个爪子贴着物体，不是把开口中心对着物体中心。
            xt = arm.tool_x()
            xh = np.array([xt[0], xt[1], 0.0])
            if float(np.linalg.norm(xh)) > 1e-6:
                xh = xh / np.linalg.norm(xh)
            f_tip = arm.fixed_tip_pos()
            side = 1.0 if float(np.dot(f_tip - src, xh)) >= 0.0 else -1.0
            # **沿爪轴物体鼓出来多少由形状自己算。** 方块斜着的时候比半边长宽
            # （15·(|cos|+|sin|)），圆柱就是半径。写死一个 15mm 的话，方块稍微一转
            # 那 1mm 的缝就被吃掉了 —— 实测就是这样压到顶面。
            half = float(sh.support(yaw, xh))
            # 缝是**从指面算**的，不是从指垫球的中点算。`fixed_tip_pos()` 给的是三个
            # 指垫球的中点，球自己还有半个球径在更靠里的地方 —— 少算这一段，`FIX_CLEAR`
            # 那 1mm 实际只剩 0.25mm。中空件下探要贴着口沿那圈最宽的边擦过去，零点几
            # 毫米实测会被顶住、判 `stall` 停在口沿高度上。问机械臂球伸出多少（真机那
            # 一层没有这条，问不到就按 0 算，跟以前一样）。
            face = getattr(arm, "fixed_face_reach", None)
            pad = float(face(xh, side)) if face is not None else 0.0
            ds = side * (half + pad + FIX_CLEAR) - float(np.dot(f_tip - src, xh))
            aim = src + (float(np.dot(arm.tcp(), xh)) + ds - float(np.dot(src, xh))) * xh
            aim[2] = float(g.z)
            if ci == 1:
                print(f"    落点按固定爪贴面算（固定爪在物体 {'+x' if side > 0 else '-x'} 侧）")
            hover = np.array([aim[0], aim[1], src[2] + 0.05])

            state.seg = group or "下探到方块"
            reason = follow.follow(arm, [hover, aim],
                                   vmax=speed_vmax(SLOW), amax=acc_amax(), grip=og,
                                   closing=closing, final_tol=0.002,
                                   on_step=bump)
            state.grasp_z = float(arm.tcp()[2])
            if hit[0] is not None:
                nf, nm, z = hit[0]
                print(f"    ⚠ 下探路上碰到物体：固定指垫 {nf} 点 / 活动指垫 {nm} 点"
                      f"  首次触点 z={z:.3f}  TCP z={state.grasp_z:.3f}"
                      f"  中心 z={src[2]:.3f}  （只记录，不截断下探）")

            # **下探没到底就别合爪。** 深度是 `aim` 定的（这一种夹法要夹的高度），
            # 正常这一路以 `reached` 收尾 —— 全部抓取记录里都是 reached、TCP z 落在
            # 0.016~0.017。2026-10-13 那一轮它中途被判 `stall`、停在 TCP z=0.0223
            # （比正常浅 5mm），合爪就合到了方块**顶部**（指垫接触高度全程比方块中心高
            # 11mm，别的轮只高 2mm），松开时方块没离爪、挂着被带上了天。浅夹就是这条
            # 链的源头，不是后面哪一步没做好：合在这儿，抬起来也是滑掉。
            if reason != "reached":
                self._back_off(arm, hover, bump, open_g=og)
                print(f"    ✗ 下探没到底（{reason}，停在 TCP z={state.grasp_z:.3f}，"
                      f"该到 {aim[2]:.3f}）：合上去只会合在顶上，这一处不合，退回去")
                why = f"下探没到底（{reason}）"
                continue

            closed, info = _close_until_touch(arm, index, closing, on_step=bump)
            if closed:
                break
            print(f"    ✗ 合爪没成（{info['why']}）：跑了 {info['steps']} 步"
                  f"  开合角 {info['q0']:+.3f} → {info['q']:+.3f}"
                  f"  贴上的时刻 {info['touch_q']}"
                  f"  最后这一步：固定指垫触点 {info['nf']} / 活动指垫触点 {info['nm']}")
            if info["nf"] == 0 and info["nm"] == 0:
                print("      两边都没贴到 —— 指垫根本没碰上")
            why = "没夹住"
            if ci < len(cands):
                print(f"      换个夹法再试（这个形状还有 {len(cands) - ci} 种没试）")
                self._back_off(arm, hover, bump, open_g=og)
        else:
            out = {"ok": False, "why": why}
            if info:
                out["close"] = info
            return out

        nf, nm, tz = arm.finger_cube_contacts(index)
        state.held = index
        # **把合爪停住那一刻的开口记下来**，抬升和搬运守的就是它。不记的话
        # `grip_hold` 一进门就 `if ref is None: return 0.0`，而那一维动作是「当前位置
        # + 增量」—— 给 0 等于零刚度，爪面只是贴着方块、一点正压力都没有。实测后果
        # 就是 `params.GRIP_BIAS` 说的那件事：开口 11.24° 一路没变、指令恒为 0.000，
        # 方块以约 3mm/s 匀速滑出去，搬运中途掉地上，放置段读到接触 0/0。
        state.grip_ref = float(arm.grip_q())
        print(f"    合爪成了（{info['why']}，{info['steps']} 步）："
              f"开合角 {info['q0']:+.3f} → {info['q']:+.3f}"
              f"  贴上时 {info['touch_q']:+.3f}"
              f"  下探停因 {reason}  TCP z={state.grasp_z:.3f}"
              f"  固定触点 {nf} / 活动触点 {nm}  接触高度 {tz:.3f}")

        # 抬起来的时候不再一路夹到底，守着自己的开口走
        state.seg = group or "抬升到搬运高度"

        def trace(k, tgt, p, dur):
            # 抬升中途停住的话，只有这里能看出手臂收到什么指令、有没有响应
            if k % 40 == 0:
                print(f"      [{k:4d}] tcp={np.round(arm.tcp(), 3)}"
                      f" 指定={np.round(tgt, 3)} p={p:.1f}/{dur:.1f}"
                      f" 关节={np.round(arm.angles(), 1)}"
                      f" 指令={np.round(arm.last_action[:3], 3)}"
                      f" 开合={float(arm.grip_q()):+.3f}")

        # 抬升这一段挂着守卫：方块一旦滑出去，当场停下并报出来，不是等抬完再看。
        # 停的那一刻手臂自己的坐标就是最好的线索 —— 方块从那儿掉下去的。
        up = follow.follow(arm, [[src[0], src[1], LIFT_Z]], vmax=speed_vmax(CARRY),
                           amax=acc_amax(), grip=lambda: grip_hold(arm, state),
                           closing=closing, final_tol=0.012, on_step=bump,
                           guard=_gone_guard(arm, state, index), verbose=trace)
        nf, nm, _ = arm.finger_cube_contacts(index)
        # 抬完只能凭「指垫还贴不贴着」判断方块在不在手里：方块实际抬了多高，
        # 那是判官拿真值说的，这一步已经不再读方块的实时位置了。
        if nf == 0 and nm == 0:
            p = arm.tcp()
            print(f"    ✗ 抬起途中方块滑出去了（停因 {up}）  TCP {np.round(p, 3)}"
                  f"  爪-方块接触 固定 {nf} / 活动 {nm}")
            if not retry:
                return {"ok": False, "why": "抬起来掉了"}
            print(f"      回去重抓一次：按掉下去那一刻自己的坐标 "
                  f"({p[0] * 1000:.0f},{p[1] * 1000:.0f})mm 一带去找")
            return self.run(arm, cams, state=state, index=index,
                            src=[float(p[0]), float(p[1]), _z_of(arm, index)],
                            group="掉了 → 回来重抓", retry=False)
        if up != "reached":
            print(f"    ✗ 抬升没走完：停因 {up}  TCP z={float(arm.tcp()[2]):.3f}"
                  f"（目标 {LIFT_Z:.2f}）  爪-方块接触 固定 {nf} / 活动 {nm}")
        return {"ok": True, "grasp_z": state.grasp_z}
