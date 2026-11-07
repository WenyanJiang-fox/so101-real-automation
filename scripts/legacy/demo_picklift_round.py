"""一轮演示：两条 pick-and-lift，中间摆一次场。

    G:/anaconda/envs/sim2real/python.exe demo_picklift_round.py --seed 0

一轮的流程（seed = s）：
    抓 seed s 指定的目标方块、抬起来（**第 1 条** pick-and-lift）
    → 摆场：把三个方块都摆到 seed s+1 的开局坐标上（不是摆回各自原来的位置）
    → 抓 seed s+1 指定的目标方块、抬起来（**第 2 条**）
    → 摆场到 seed s+2 → 回原位，本轮结束

摆场用的是**另一个同配置的环境**采样：主环境正在跑，reset 会把机械臂也一起复位。

放方块一律「降到抓取时的那个高度再松爪」，不从高处直接开爪子。抓取时的 TCP 高度
记在 GRASP_Z 里，也就是方块的中心高度 —— 爪子卡在方块正中，放下去时降到同一高度，
方块正好贴地。

动作走 `goto_path`：路点先把拐角抹圆、按弧长重采样，进度再按梯形速度曲线推进
（加速到顶 → 匀速 → 提前算刹车距离减速），指令点从曲线上查。段与段之间靠
VEL / RVEL 保留速度状态，所以「过去 → 抓 → 抬 → 搬 → 放」是连着的，只有夹爪开合
那一下才停。
夹爪的开合轴（工具系 x 轴）在抓取前对齐方块的面法向，两个指面贴在方块的一对侧面
上，不是拿指尖去顶棱。

**轻拿轻放**（不管夹的是目标方块还是摆场时要挪的方块，都走同一套）：
- 贴到方块之后不再一路合到底，只再挤 0.02rad 就停，之后守住这个开口（grip_hold）；
  给 0 了事不行 —— 那一维动作是「当前位置 + 增量」，给 0 等于零刚度，方块会被撑开掉下去。
- 下探到方块、以及搬运时贴地那两段限速 SLOW，不按巡航速度撞过去。
- 松爪是一小步一小步张，不是一步给满弹开。

输出一段 mp4（全局相机 | 腕部相机 左右拼接）。**不写数据集**，只看过程。

场景用 so101_nexus 的 MuJoCoPickLift-v1，场上三个 30mm 方块（红/绿/蓝），没有目标盘。
动作走 pd_ee_delta_pose，参数同 _ref/local_picklift.py。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]

CUBE_ORDER = ("red", "green", "blue")   # 方块池顺序 = slot 0/1/2
N_EPISODES = 2                          # 一轮录两条 pick-and-lift

POS_SCALE = 0.02
ROT_SCALE = 0.1
SPEED_POS = 0.08   # 巡航速度（动作单位）。太快了：整条轨迹和腕子转都按它放
SPEED_ROT = 0.20
OW = 0.05          # ee_orientation_weight：默认 0.01 工具轴太斜，压不下去
FPS = 30
RESET_TOL = 0.012  # 方块离目标坐标超过这个距离，摆场时就要挪

# 中立位：每次靠近方块前先回到这里。实测里每局单起一局能到位、连着跑会卡,
# 差别就是有没有这个中立位 —— 先退出来再过去，不从上一条的收尾姿势直接扑过去。
WAYPOINT = np.array([0.20, 0.0, 0.16])

# ---- 标识点：工位坐标表 ----
# 机械臂的动作一律以这几个**已知坐标**为准，不再去读方块的实时位姿。真机上这张表
# 就是标定/示教的结果，形状不变；场景里同时画出可见的标记盘，以后换成相机识别同
# 一个标记即可。圆心取 shoulder_pan 轴 (0.039, 0)，半径 0.20 落在实测可达区间
# 0.13~0.26 的中段。拾取位和放置位错开 120°，同一侧的相邻两点方位差 40°，
# 半径 0.20 时弦长 0.137——比如今的 min_object_separation 0.12 宽。
MARKER_CENTER = np.array([0.039, 0.0])
MARKER_R = 0.20
PICK_DEG = (20.0, 60.0, 100.0)       # 拾取位：red / green / blue 各一个
PLACE_DEG = (-20.0, -60.0, -100.0)   # 放置位：各自对应的落点
CUBE_Z = 0.015                       # 30mm 方块躺在地面上的中心高度
MARKER_RGBA = {"red": (0.90, 0.15, 0.15), "green": (0.15, 0.75, 0.20),
               "blue": (0.15, 0.30, 0.90)}


def _ring(deg):
    return np.array([[MARKER_CENTER[0] + MARKER_R * np.cos(np.radians(d)),
                      MARKER_CENTER[1] + MARKER_R * np.sin(np.radians(d))] for d in deg])


PICK_XY = _ring(PICK_DEG)
PLACE_XY = _ring(PLACE_DEG)


def marker_xml():
    """标识点的 MJCF：贴地薄圆盘，看得见、不参与碰撞（contype/conaffinity=0）。

    nexus 的 mujoco ``PickEnv`` 没有 ``extra_bodies`` 钩子（只有 ``pick_and_place``
    有），所以调用方把 ``build_object_scene_xml`` 包一层，往 worldbody 末尾注入这段。
    """
    out = []
    for i, (xy, color) in enumerate(zip(np.vstack([PICK_XY, PLACE_XY]), CUBE_ORDER * 2)):
        r, g, b = MARKER_RGBA[color]
        out.append(
            f'    <body name="marker_{i}" pos="{xy[0]:.6f} {xy[1]:.6f} 0.0012">\n'
            f'      <geom type="cylinder" size="0.020 0.0006" '
            f'rgba="{r:.3f} {g:.3f} {b:.3f} 1" contype="0" conaffinity="0"/>\n'
            f'    </body>\n')
    return "".join(out)


def smooth_path(wps, corner=0.02, n_corner=16):
    """折线的拐角切成圆弧，直段保持是直的，返回密集点列。

    追折线的拐角时，误差方向在换路点那一瞬间改向，速度被限斜率慢慢掰过去 ——
    看着就是「走一段、顿一下、再走一段」。把拐角抹圆之后，指令轨迹本身没有尖角，
    限斜率只剩兜底作用。

    只圆角、不整条过样条：整条过 Catmull-Rom 的话，最后一段（垂直下探到方块上方）
    的切向被前一个路点带偏，曲线会先横向鼓出去再荡回来，爪子在方块正上方划个弧才
    落下去 —— 落点就不正了。只把拐角换成二次 Bézier，直段原样保留。
    """
    P = [np.asarray(w, dtype=np.float64) for w in wps]
    if len(P) == 1:
        return P[0][None, :].copy()
    pts = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        u, v = b - a, c - b
        lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if lu < 1e-9 or lv < 1e-9:
            continue
        r = min(corner, 0.4 * lu, 0.4 * lv)     # 圆角半径不超过邻段的 40%
        s, e = b - r * (u / lu), b + r * (v / lv)
        t = np.linspace(0.0, 1.0, n_corner + 1)[:, None]
        pts.append(s)
        pts.extend((1 - t) ** 2 * s + 2 * (1 - t) * t * b + t ** 2 * e)
    pts.append(P[-1])
    out = np.vstack(pts)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(out, axis=0), axis=1) > 1e-9])
    return out[keep]


def arc_table(pts, dens=0.002):
    """按固定弧长间距重采样，返回 (累计弧长, 点列)。

    样条是按参数均匀采的，参数均匀不等于弧长均匀；按弧长重采样之后，「进度走了
    多少」才是真长度，梯形速度曲线才算得对。
    """
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return np.zeros(1), pts[:1].copy()
    u = np.linspace(0.0, s[-1], max(int(s[-1] / dens) + 1, 2))
    xyz = np.column_stack([np.interp(u, s, pts[:, j]) for j in range(3)])
    return u, xyz


def face_normal(obj_quat_wxyz, tool_x):
    """方块四条侧面法向里，跟当前爪开合轴最接近的那一条（单位向量，水平）。

    爪的开合方向就是工具系的 x 轴：固定爪的指面在 TCP 系 x=+0.030，活动爪在
    x=+0.002，两个指面沿 x 分开。所以 x 轴对上某个面法向，两个指面才正好贴在
    方块的一对侧面上；不对上就是拿指尖去顶棱，夹不住。

    方块四棱等价，四条法向（±局部x、±局部y）都能夹，取转得最少的那条。
    """
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat([obj_quat_wxyz[1], obj_quat_wxyz[2],
                            obj_quat_wxyz[3], obj_quat_wxyz[0]]).as_matrix()
    cands = [R[:, 0], -R[:, 0], R[:, 1], -R[:, 1]]
    th = np.array([tool_x[0], tool_x[1], 0.0], dtype=np.float64)
    if np.linalg.norm(th) < 1e-6:
        return cands[0]
    th /= np.linalg.norm(th)
    return max(cands, key=lambda v: float(np.dot(v[:2], th[:2])))


def align_action(q_wxyz, z_des, gain=ROT_SCALE, x_des=None):
    """把工具 z 轴转到 z_des 需要的归一化旋转增量（同 local_picklift.py）。"""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    z = np.asarray(z_des, dtype=np.float64)
    z /= np.linalg.norm(z)
    src = R[:, 0] if x_des is None else np.asarray(x_des, dtype=np.float64)
    x = src - np.dot(src, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = R[:, 1] - np.dot(R[:, 1], z) * z
    x /= np.linalg.norm(x)
    R_des = np.column_stack([x, np.cross(z, x), z])
    return np.clip(Rotation.from_matrix(R_des @ R.T).as_rotvec() / gain, -1.0, 1.0)


def tool_axis(q_wxyz):
    """工具 z 轴（夹爪前进方向）在世界系里的单位向量。"""
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()[:, 2]


def obj_yaw(q_wxyz):
    """绕竖直轴的转角（度）。"""
    from scipy.spatial.transform import Rotation

    return float(np.degrees(Rotation.from_quat(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_euler("zyx")[0]))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="两条 pick-and-lift + 中间复位")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "demo_picklift_round.mp4"))
    ap.add_argument("--stride", type=int, default=2,
                    help="每多少个物理步收一帧。等间隔取样，回放速度才均匀")
    args = ap.parse_args()

    import gymnasium as gym
    import mujoco
    import so101_nexus.mujoco  # noqa: F401  注册 gym 环境
    from so101_nexus import (CubeObject, EndEffectorPose, JointPositions, ObjectPose,
                             OverheadCamera, PickConfig, RobotConfig, WristCamera)
    from so101_nexus.camera_utils import compute_angled_camera_params

    # 把标识点的 geom 注进场景。nexus 的 PickEnv 内部直接调 build_object_scene_xml
    # 且不传 extra_bodies，这里在模块上包一层补上——只改本进程的绑定，不动只读包。
    import so101_nexus.mujoco.pick_env as _pick_env
    _marker_bodies = marker_xml()
    _build_scene = _pick_env.build_object_scene_xml

    def _build_scene_with_markers(*a, **kw):
        kw.setdefault("extra_bodies", _marker_bodies)
        return _build_scene(*a, **kw)

    _pick_env.build_object_scene_xml = _build_scene_with_markers

    cfg = PickConfig(
        objects=[CubeObject(color=c, side_length_mm=30) for c in CUBE_ORDER],
        n_distractors=2,          # 否则另外两个方块被藏到场景外，场上只剩一个
        obs_mode="visual",
        observations=[JointPositions(), EndEffectorPose(), ObjectPose(),
                      WristCamera(), OverheadCamera()],
        robot=RobotConfig(ee_orientation_weight=OW),
        # 半径以 spawn_center 为圆心。圆心取 shoulder_pan 轴的世界坐标 (0.039, 0)，
        # 于是半径就是「离肩部轴多远」。0.13~0.26 是实测可达区间：每个半径单起一局
        # 用控制律驱动，桌面高度(z=0.02)残差 1~4mm、抬到 0.16 实测到 0.157~0.178；
        # 再往里(<0.10)会把方块塞到机身底下。
        spawn_center=(0.039, 0.0), spawn_min_radius=0.13, spawn_max_radius=0.26,
        min_object_separation=0.12,   # 默认 0.04，方块之间只剩 16mm，夹爪下不去
        # 默认 True：方块一抬过 lift_threshold(0.05) 环境就判成功、当场终止这一局。
        # 终止之后 goto_path 每一步立刻返回 "term"，手臂停在半空，后面的搬运/放爪
        # 全在半路上执行 —— 实测就是这样「在空中松开爪子」。这里要的是跑满一整轮，
        # 成功与否由脚本自己判，所以关掉。
        terminate_on_success=False,
    )
    # 默认 max_episode_steps=1024，一轮两条 pick-and-lift 跑不完就被截断
    env = gym.make("MuJoCoPickLift-v1", config=cfg, render_mode="rgb_array",
                   control_mode="pd_ee_delta_pose", max_episode_steps=20000)

    _env = env.unwrapped
    _cam, _comp = _env._overhead_obs_cam, _env._overhead_cam_component
    if _cam is not None and _comp is not None:
        _p = compute_angled_camera_params(
            spawn_center=_env.config.spawn_center,
            spawn_max_radius=_env.config.spawn_max_radius,
            elevation=-34.0, azimuth=205.0,
            fov_deg=_comp.fov_deg, aspect=_comp.width / _comp.height)
        _cam.lookat[:] = _p["lookat"]
        _cam.distance, _cam.elevation, _cam.azimuth = _p["distance"], _p["elevation"], _p["azimuth"]

    # 采样器：另开一个同配置的环境，只用来问「seed N 的方块摆在哪」。
    # 主环境跑着的时候不能 reset —— reset 会把机械臂也一起复位。
    sampler = gym.make("MuJoCoPickLift-v1", config=cfg, render_mode="rgb_array",
                       control_mode="pd_ee_delta_pose", max_episode_steps=20000)

    def layout_of(seed):
        """某个 seed 的开局：三个方块各自的初始坐标 + 它指定的目标。"""
        _, inf = sampler.reset(seed=seed)
        su = sampler.unwrapped
        pos = np.array([su.data.qpos[s.qpos_addr:s.qpos_addr + 3].copy()
                        for s in su._slots])
        return pos, int(inf["target_index"]), str(su.task_description)

    obs, info = env.reset(seed=args.seed)
    u = env.unwrapped
    target_idx = int(info["target_index"])
    cube_addr = [s.qpos_addr for s in u._slots]
    cube_dof = [s.dof_addr for s in u._slots]

    def cube_pos(i):
        return u.data.qpos[cube_addr[i]:cube_addr[i] + 3].copy()

    def cube_quat(i):
        """方块自由关节的四元数（w,x,y,z）。爪子要对的就是它定出来的面法向。"""
        return u.data.qpos[cube_addr[i] + 3:cube_addr[i] + 7].copy()

    ref_pos, ref_tgt, _ = layout_of(args.seed)
    same = (np.allclose(ref_pos, np.array([cube_pos(i) for i in range(3)]), atol=1e-9)
            and ref_tgt == target_idx)
    print(f"任务：{u.task_description}")
    print(f"指定目标 = {CUBE_ORDER[target_idx]}")
    for i in range(3):
        print(f"  方块{i}({CUBE_ORDER[i]}) 初始 {np.round(cube_pos(i), 3)}")
    print(f"采样器复核 seed {args.seed}：{'一致' if same else '不一致（摆场坐标不可信）'}")

    frames: list[np.ndarray] = []

    def st():
        p = np.asarray(info["privileged_state"], dtype=np.float64)
        return p[0:6], p[6:9], p[9:13], p[13:16], p[16:20]

    def tool_x():
        """工具系 x 轴在世界系里的方向 —— 就是爪的开合方向。"""
        from scipy.spatial.transform import Rotation

        q = st()[2]
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()[:, 0]

    def align_err(nrm):
        """爪开合轴和给定法向的水平夹角，用来核实腕子真转到位了。"""
        v = np.asarray(tool_x(), dtype=np.float64)
        v = np.array([v[0], v[1], 0.0])
        if np.linalg.norm(v) < 1e-6:
            return np.pi / 2
        v /= np.linalg.norm(v)
        h = np.asarray(nrm, dtype=np.float64)[:2]
        h = h / np.linalg.norm(h)
        return np.arccos(np.clip(np.dot(v[:2], h), -1.0, 1.0))

    tick = [0]

    def rec():
        """固定步长取样，**不按画面差异丢帧**。

        按差异丢帧会让回放速度忽快忽慢：机械臂慢的地方丢得多、快的地方全留下，
        同一段视频里时间被压缩得不均匀，看起来就是一顿一顿。卡住的死时间由
        goto_path 的 stall 退出砍掉，这里只管等间隔。
        """
        tick[0] += 1
        if tick[0] % args.stride == 0:
            frames.append(np.hstack([obs["overhead_camera"], obs["wrist_camera"]]))

    PARK_RNG = np.random.default_rng(20261005)
    PARK_FRAMES = 60       # 摆件动画的物理步数（按 args.stride 抽样成帧）

    def park_cubes():
        """操作员把三个方块摆到拾取标识点上 —— 不是机械臂干的。

        位置写成标识点的**已知坐标**，yaw 随机：真机上人手一放，位置对得准、朝向对
        不准，这里照这个来。走一段抬起来的弧线，免得平移时从别的方块身上刮过去。
        """
        nonlocal obs, info
        if HELD[0] is not None:            # 上一轮没放下去：操作员从爪子里取走
            grip_move(0.04, 60)
            HELD[0] = None
            GRIP_REF[0] = None
        SEG[0], IGNORE[0] = "摆件（操作员）", {0, 1, 2}
        start = np.array([cube_pos(i) for i in range(3)])
        # 已经在标识点上的方块不动：不然每轮都会把没动过的两个方块抬起来放回去，
        # 视频里看着像无缘无故弹一下。
        move = [i for i in range(3)
                if float(np.linalg.norm(start[i][:2] - PICK_XY[i])) > 0.005]
        if not move:
            print("  摆件：三个方块都已在标识点上，不动")
            return
        for k in range(1, PARK_FRAMES + 1):
            t = k / PARK_FRAMES
            for i in move:
                a, d = cube_addr[i], cube_dof[i]
                yaw = float(PARK_RNG.uniform(-np.pi / 6, np.pi / 6))
                u.data.qpos[a:a + 3] = [start[i][0] + (PICK_XY[i][0] - start[i][0]) * t,
                                        start[i][1] + (PICK_XY[i][1] - start[i][1]) * t,
                                        CUBE_Z + 0.05 * np.sin(np.pi * t)]
                u.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
                u.data.qvel[d:d + 6] = 0.0
            obs, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))
            rec()
        p = np.array([cube_pos(i) for i in range(3)])
        print("  摆件："
              + " / ".join(f"{CUBE_ORDER[i]}({start[i][0] * 1000:.0f},"
                           f"{start[i][1] * 1000:.0f})→({p[i][0] * 1000:.0f},"
                           f"{p[i][1] * 1000:.0f})" for i in move)
              + ("   其余已在标识点" if len(move) < 3 else ""))

    VEL = [np.zeros(3)]    # 跨段保留的平移速度。段与段之间不归零，接得上才不抖
    RVEL = [np.zeros(3)]   # 同上，旋转增量。不限的话误差一大就是满速甩腕

    def arc_waypoints(wps, step=0.04):
        """把相邻两点的**水平**位移改成绕底座的圆弧插值。

        直线连接两个方位角差得远的点，弦会从底座上方切进去：搬运那段从 (0.082,0.212)
        到 (0.215,-0.145)（方位差 103°）的弦离底座只有 0.15，而两端都在 0.23 以上。
        腕关节要折叠到那个半径就顶到 ctrlrange 上限（wrist_flex 钉在 1.658 不动），
        IK 只好放开位置去迁就姿态，手臂甩到外圈走弧线 —— 实测就是这么撞上别的方块的。
        改成按极坐标插值，半径只会在两端之间变化，不再切进去。竖直那几段 r、θ 不变，
        插值出来还是竖直线，不受影响。
        """
        pts = [np.asarray(wps[0], dtype=np.float64)]
        for a, b in zip(wps[:-1], wps[1:]):
            av, bv = np.asarray(a, np.float64), np.asarray(b, np.float64)
            ra, rb = float(np.hypot(av[0], av[1])), float(np.hypot(bv[0], bv[1]))
            ta, tb = float(np.arctan2(av[1], av[0])), float(np.arctan2(bv[1], bv[0]))
            dth = (tb - ta + np.pi) % (2.0 * np.pi) - np.pi
            span = max(abs(dth) * max(ra, rb), abs(rb - ra), abs(bv[2] - av[2]))
            n = max(1, int(np.ceil(span / step)))
            for i in range(1, n):
                t = i / n
                r = ra + (rb - ra) * t
                th = ta + dth * t
                pts.append(np.array([r * np.cos(th), r * np.sin(th),
                                     av[2] + (bv[2] - av[2]) * t]))
            pts.append(bv)
        return pts

    def goto_path(wps, grip, x_des=None, final_tol=0.004, accel=0.012, rslew=0.03,
                  max_steps=2000, stall=40, stall_eps=0.0004, guard=None, speed=None):
        """沿 wps 走一条规划好的轨迹 —— 不追路点，而是让进度沿轨迹滑。

        旧写法每一步追「最近的那个路点」：换点时误差方向瞬间改向，速度被限斜率慢
        慢掰，看着就是一段一段地顿。现在先把 wps 的拐角抹圆并按弧长重采样，再让一个
        弧长进度 p 按梯形速度曲线推进（加速到顶 → 匀速 → 提前算刹车距离减速），指令
        点从曲线上查出来。

        真实起点（手臂现在的位置）会补在 wps 前面：不补的话，从当前位置切到 wps[0]
        这一段不在轨迹里，那个拐角就是硬的，到点必顿一下。

        - accel：每步允许的速度增量（动作单位）。刹车距离按 v²/(2a) 提前算，所以
          是准确停在终点，不是冲过去再折回来。
        - rslew 同理限旋转增量。腕子转角没限速时，一开始对齐面法向那一下是满速甩
          过去的，是忽快忽慢里最明显的一段。
        - final_tol：到最后一个点的允许误差，到了就返回。**只看距离，不看进度**：
          加进度条件的话，手臂被挡在半路时进度先到终点，就再也不会返回，只能等 stall。
          给大一点（比如抬升到顶）是为了让下一段接着走，不在这儿停死。
        - VEL / RVEL 跨段保留，所以「抬起来 → 平移过去 → 放下去」是连着的。
        - guard：每步回调一次，返回真就中断。下探时用它盯指垫是不是提前碰上方块。
        - speed：这一段的限速（动作单位，对应 SPEED_POS）。给 SLOW 用在贴近方块、
          贴地那几段上 —— 那些地方要轻，不能按巡航速度撞过去。
        返回 'reached' / 'guard' / 'stall' / 'term'，调用方据此判断是走到了还是被打断。
        """
        nonlocal obs, info
        wps = arc_waypoints([st()[1]] + [np.asarray(w, dtype=np.float64) for w in wps])
        s_tab, xyz_tab = arc_table(smooth_path(wps))
        L = float(s_tab[-1])
        SEG_ROUTE[0] = (wps[0].copy(), wps[-1].copy())
        spd = SPEED_POS if speed is None else speed
        vmax = spd * POS_SCALE                # 动作单位 → 米/步
        acc = accel * POS_SCALE
        p = 0.0
        v = min(float(np.linalg.norm(VEL[0])) * POS_SCALE, vmax)   # 接上一段的速度
        ref = None
        for k in range(max_steps):
            _, tcp, quat, _, _ = st()
            # 进度跟着手臂在曲线上的投影走，不凭空积分。只积分的话，手臂一旦跟不上
            # （下探被挡、伺服慢），p 会先跑到终点、速度归零，手臂就冻在原地等 stall。
            p = max(p, float(s_tab[int(np.argmin(
                np.linalg.norm(xyz_tab - tcp, axis=1)))]))
            v_allow = np.sqrt(max(0.0, 2.0 * acc * max(L - p, 0.0)))   # 刹车包络
            v = min(vmax, v_allow, v + acc)
            p = min(L, p + v)
            tgt = np.array([np.interp(p, s_tab, xyz_tab[:, j]) for j in range(3)])
            CMD[0], PROG[0] = tgt.copy(), (p, L)
            d = tgt - tcp
            n = float(np.linalg.norm(d))
            if p >= L - 1e-9:
                # 末段进度到头，不能再靠「沿轨迹的速度」：那时 v 已经收到 0，冲过落点
                # 之后指令恒为零，没有任何回正力，手臂就停偏了等 stall。实测停在比落点
                # 低 14mm 的地方，爪子夹在方块下缘，抬不起来。这里直接比例收敛到落点。
                want = np.clip(d / POS_SCALE, -1.0, 1.0) * spd
            else:
                cmd = d / n * v if n > 1e-9 else np.zeros(3)
                want = cmd / POS_SCALE
            VEL[0] = VEL[0] + np.clip(want - VEL[0], -accel, accel)
            rwant = align_action(quat, [0, 0, -1], x_des=x_des) * SPEED_ROT
            RVEL[0] = RVEL[0] + np.clip(rwant - RVEL[0], -rslew, rslew)
            a = np.zeros(7, dtype=np.float32)
            a[:3] = VEL[0]
            a[3:6] = RVEL[0]
            a[6] = grip() if callable(grip) else grip
            obs, _, term, trunc, info = env.step(a)
            rec()
            if TRACE[0] and k % 40 == 0:
                print(f"      [{k:4d}] {SEG[0][:16]:<16} tcp={np.round(tcp, 3)}"
                      f" 指定={np.round(tgt, 3)} p={p:.3f}/{L:.3f}"
                      f" q={np.round(st()[0], 2)}"
                      f" ctrl={np.round(u.data.ctrl[:6], 2)}")
                if HELD[0] is not None:
                    _nf, _nm, _ = finger_cube_contacts(HELD[0])
                    _tilt = float(np.degrees(np.arccos(
                        np.clip(abs(tool_axis(quat)[2]), -1.0, 1.0))))
                    print(f"            爪-{CUBE_ORDER[HELD[0]]} 固定触点 {_nf} 活动触点 {_nm}"
                          f"  方块 {np.round(cube_pos(HELD[0]), 3)}"
                          f"  方块yaw {obj_yaw(cube_quat(HELD[0])):+7.1f}°"
                          f"  工具轴离竖直 {_tilt:4.1f}°"
                          f"  开口 {float(u.data.qpos[GRIP_QADR]):+.3f}"
                          f"  出力 {np.round(u.data.actuator_force, 1)}")
            check_bump()
            if term or trunc:
                return "term"
            if guard is not None and guard():
                return "guard"
            now = st()[1]
            if np.linalg.norm(wps[-1] - now) < final_tol:
                return "reached"
            if ref is None or k % stall == 0:
                if ref is not None and np.linalg.norm(now - ref) < stall_eps:
                    return "stall"
                ref = now
        return "steps"

    def grip_move(angle, n, x_des=None, slew=0.02, rslew=0.03):
        """位置守住不动，只开合夹爪（合爪/松爪）。

        `angle` 可以是常数，也可以是每步现算的函数（守位用 grip_hold）。

        守住的位置不是硬刹：残余速度按同样的 slew 收到零，手臂是滑停下来而不是
        撞停，也不会有一下反向的抽动。
        """
        nonlocal obs, info
        hold = st()[1].copy()
        for _ in range(n):
            _, tcp, quat, _, _ = st()
            want = np.clip((hold - tcp) / POS_SCALE, -1.0, 1.0) * SPEED_POS
            VEL[0] = VEL[0] + np.clip(want - VEL[0], -slew, slew)
            rwant = align_action(quat, [0, 0, -1], x_des=x_des) * SPEED_ROT
            RVEL[0] = RVEL[0] + np.clip(rwant - RVEL[0], -rslew, rslew)
            a = np.zeros(7, dtype=np.float32)
            a[:3] = VEL[0]
            a[3:6] = RVEL[0]
            a[6] = angle() if callable(angle) else angle
            obs, _, term, trunc, info = env.step(a)
            rec()
            if term or trunc:
                return

    # ---- 爪子的接触查询：碰没碰上、夹没夹住，只看接触，不靠算出来的高度 ----

    def _jaw_geoms(prefix):
        """爪子上所有碰撞 geom 的 id。模型里固定爪叫 fixed_jaw_*，活动爪叫
        moving_jaw_*（SO101_menagerie/so101.xml）—— 按名字前缀收，不写死 id。"""
        return {g for g in range(u.model.ngeom)
                if (mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                .startswith(prefix)}

    FIXED_JAW, MOVING_JAW = _jaw_geoms("fixed_jaw"), _jaw_geoms("moving_jaw")
    if not FIXED_JAW or not MOVING_JAW:
        raise RuntimeError(f"模型里没找到固定爪/活动爪的 geom：{FIXED_JAW} {MOVING_JAW}")

    def _tip_geoms(prefix):
        return sorted(g for g in _jaw_geoms(prefix)
                      if "sph_tip" in (mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))

    FIX_TIP, MOV_TIP = _tip_geoms("fixed_jaw"), _tip_geoms("moving_jaw")

    # ---- 撞到别的方块：巡线时每步扫一次接触表 ----
    #
    # 路径规划里根本没读别的方块在哪（WAYPOINT 是一个死点，搬运是直线），所以
    # 蹭上、压上是可能的，而日志里原先一点痕迹都没有。这里不避障，只把撞记下来，
    # 好知道是哪一段撞的。

    GEOM2CUBE = {}
    for _i, _s in enumerate(u._slots):
        for _g in _s.geom_ids:
            GEOM2CUBE[int(_g)] = _i
    # 地面等 worldbody 上的 geom：方块躺在地上也会产生接触，那是正常的，不算撞。
    GROUND_GEOMS = {g for g in range(u.model.ngeom) if int(u.model.geom_bodyid[g]) == 0}

    SEG = [""]        # 现在在走哪一段（调用方在动作之前写进去，只给日志用）
    IGNORE = [set()]  # 这一段的「自己人」：正夹着的、正要去抓的方块，碰了不算撞
    BUMP = {}         # 已经报过的 (段, 方块, geom)，同一件只报一次
    SEG_ROUTE = [None]  # 这一段的 (起点, 终点)，规划出来的真实端点
    CMD = [None]        # 本步让手臂去的点（轨迹曲线上查出来的）
    PROG = [(0.0, 0.0)]  # 本步的 (弧长进度, 全段弧长)
    TRACE = [os.environ.get("TRACE_CARRY") == "1"]   # 逐点打字，只在查跟不对路时开

    def bump_scan():
        """扫接触表，返回 [(方块序号, 撞到它的 geom 名字, 接触点位置)]。

        只看「一边是方块、另一边不是方块也不是地面」的组合 —— 那就是手臂（可能是
        爪子、也可能是臂身）压上去了。方块压方块、方块压地面都跳过。
        """
        out = []
        for k in range(u.data.ncon):
            c = u.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            i1, i2 = GEOM2CUBE.get(g1), GEOM2CUBE.get(g2)
            if (i1 is None) == (i2 is None):
                continue                      # 两边都是方块、或都不是方块
            ci = i1 if i1 is not None else i2
            og = g2 if i1 is not None else g1
            if ci in IGNORE[0] or og in GROUND_GEOMS:
                continue
            name = mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_GEOM, og)
            if not name:
                # 臂身那些碰撞盒在 so101.xml 里没有名字（class="collision" 的 box 都不写
                # name=），只报 geom42 认不出是哪一节。退到所属 body 的名字。
                name = (mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_BODY,
                                         int(u.model.geom_bodyid[og]))
                        or f"geom{og}")
            out.append((ci, name, np.array(c.pos, dtype=float).copy()))
        return out

    def check_bump():
        """走上一步之后调一次。撞上就记一条（同一件只报一次）。"""
        for ci, name, p in bump_scan():
            key = (SEG[0], ci, name)
            if key in BUMP:
                continue
            BUMP[key] = 1
            a, b = SEG_ROUTE[0]
            print(f"    ⚠ 撞到 {CUBE_ORDER[ci]} 方块：{SEG[0]}  撞它的 geom={name}"
                  f"  接触点 {np.round(p, 3)}  TCP {np.round(st()[1], 3)}"
                  f"  方块在 {np.round(cube_pos(ci), 3)}")
            print(f"        这段是 {np.round(a, 3)} → {np.round(b, 3)}"
                  f"  本步指定点 {np.round(CMD[0], 3)}"
                  f"  进度 {PROG[0][0]:.3f}/{PROG[0][1]:.3f}")
        return bool(BUMP)

    def jaw_gap_center():
        """现在这个开口下，合爪时方块该待的位置（世界系，两组指尖球的中点）。

        TCP（gripperframe）不在两个爪子的正中间，固定爪挂在 gripper 体上、离 TCP
        有一截横偏，所以「TCP 对着方块中心」和「开口对着方块中心」是两回事。按 TCP
        去对的话，下探时固定爪会从方块顶面扫过去 —— 实测就是这么撞上的。
        """
        f = u.data.geom_xpos[FIX_TIP].mean(axis=0)
        m = u.data.geom_xpos[MOV_TIP].mean(axis=0)
        return (f + m) / 2.0

    def finger_cube_contacts(i):
        """(固定爪的触点数, 活动爪的触点数, 接触点的平均高度)。

        从 MjData 的接触表里数爪子和这个方块之间的接触。比按「TCP 到多少高度算夹住」
        可靠：高度是算出来的，接触是仿真给的。
        """
        cg = set(u._slots[i].geom_ids)
        nf = nm = 0
        zs = []
        for k in range(u.data.ncon):
            c = u.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in cg:
                jaw = g2
            elif g2 in cg:
                jaw = g1
            else:
                continue
            if jaw in FIXED_JAW:
                nf += 1
            elif jaw in MOVING_JAW:
                nm += 1
            else:
                continue
            zs.append(float(c.pos[2]))
        return nf, nm, (float(np.mean(zs)) if zs else float("nan"))

    def close_until_touch(i, nrm, step=None, seat=0.02, max_steps=600):
        """一小步一小步合爪；两个指垫都贴到之后，再往里挤 `seat` 就停，记住这个开口。

        不一路合到底（-1.0）：那样指垫是撞上去的，方块会被推着横移、被顶得跳起来。
        「刚贴上就停」也不行：那一瞬间两面的法向力约等于零，抬起来方块只靠这点摩擦
        拉不住，会滑。折中是再合一点点 —— 挤压量按夹爪关节角算（跟动作尺度无关），
        停住之后由 grip_hold 守着这个开口，不继续加力。

        返回真表示两指垫都贴上了（GRIP_REF 已写好）。
        """
        nonlocal obs, info
        step = GENTLE if step is None else step
        hold = st()[1].copy()
        touch_q = None           # 刚贴上时的开口
        q_prev = None
        stuck = 0
        for _ in range(max_steps):
            q = float(u.data.qpos[GRIP_QADR])
            if touch_q is None:
                nf, nm, _ = finger_cube_contacts(i)
                if nf > 0 and nm > 0:
                    touch_q = q
                    q_prev = q
            else:
                if q <= touch_q - seat:
                    GRIP_REF[0] = q   # 挤到位，守这个开口
                    return True
                # 方块和爪子都是刚体，挤到某个位置就推不动了。推不动就别再耗着 ——
                # 再合也只是把力加上去，正好是「轻拿轻放」要避免的。
                stuck = stuck + 1 if (q_prev is not None and q > q_prev - 1e-5) else 0
                q_prev = q
                if stuck > 25:
                    GRIP_REF[0] = q
                    return True
            _, tcp, quat, _, _ = st()
            want = np.clip((hold - tcp) / POS_SCALE, -1.0, 1.0) * SPEED_POS
            VEL[0] = VEL[0] + np.clip(want - VEL[0], -0.02, 0.02)
            rwant = align_action(quat, [0, 0, -1], x_des=nrm) * SPEED_ROT
            RVEL[0] = RVEL[0] + np.clip(rwant - RVEL[0], -0.03, 0.03)
            a = np.zeros(7, dtype=np.float32)
            a[:3] = VEL[0]
            a[3:6] = RVEL[0]
            a[6] = -step
            obs, _, term, trunc, info = env.step(a)
            rec()
            if term or trunc:
                return False
        return False

    GRASP_Z = [0.02]    # 合爪那一刻 TCP 的离地高度，放的时候降到同一高度再松爪
    HELD = [None]       # 现在爪里夹着哪个方块

    # 抬起/搬运高度。竖直向下的工具轴是这个 5 关节臂的极限姿势：抬得越高，
    # wrist_flex 越要先折满。实测 z=0.107/0.13 上 wrist_flex 就顶死在 1.658
    # （joint 与 ctrlrange 的上限），位置只剩 4 个关节凑，手臂只能往外鼓到 r≈0.28；
    # 下探时再从 0.28 横着扫回落点，路上就把旁边方块撞了。复位姿势 z=0.061 时
    # wrist_flex=1.29（还剩行程），所以把搬运高度压到这条线以下。
    LIFT_Z = 0.08   # 抬起/搬运高度

    # ---- 「轻拿轻放」：合爪/松爪不再一路给满 ----
    #
    # 合爪那一维动作是**从当前测量位置出发的增量**（base_env.py
    # `current_qpos[-1] + delta[6]`），增量系数 0.2 rad/单位（kinematics.py）。
    # 所以 -1.0 是每步 0.2rad 地压到底（十来步就夹死），1.0 是十来步弹开。
    _grip_j = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    if _grip_j < 0:
        raise RuntimeError("模型里没找到名为 gripper 的关节")
    GRIP_QADR = u.model.jnt_qposadr[_grip_j]

    GENTLE = 0.05       # 合爪每步的指令增量（原来 0.10；松爪原来给 1.0）
    FIX_CLEAR = 0.001   # 下探时固定爪指面和方块侧面之间留的缝，免得刮着方块走
    OPEN_STEP = 0.04    # 松爪每步的指令增量
    SLOW = 0.035        # 贴近方块、贴地那几段的限速（巡航是 SPEED_POS=0.08）
    # 夹着方块走的段落再压一档。方块只是被两片爪面夹着，靠摩擦跟着走；指令速度一快，
    # kp=998 的关节就在跟踪误差里来回抖，抖一次挤出去一点。实测搬运段（SLOW=0.035，
    # 约 21mm/s）方块仍以约 3mm/s 匀速后滑、yaw 每 40 步转 2.7°，接触点 10/11→0/2；
    # 而 15cm 高度的搬运里手臂顶死几乎不动，同一套抓取代码四次都夹得住。
    CARRY = 0.020       # 夹着方块走的限速
    AIR_TOL = 0.010     # 松爪允许高出放置高度的余量：超过它才算「还挂在半空」，不给松

    # 合爪停住之后要**守住那个开口**。不能给 0 了事：给 0 等于目标每步都重置到当前
    # 位置，是个纯阻尼、零刚度，被撑开就回不来，方块滑下去。所以按「撑开了多少」
    # 给一点回收量（跟轨迹末段必须补位置回正同一个道理）。
    GRIP_REF = [None]   # 合爪停住那一刻的夹爪关节角
    GRIP_K = 0.05       # 撑开这么多（rad）就算「撑满」，决定守位的软硬
    GRIP_RATE = 0.12    # 守位时每步最多给多少合爪指令
    # 守位时必须恒定给一份合爪量。只按「被撑开多少」补力的话，爪子停在参考开口上时
    # 指令是 0，而那一维动作是「当前位置 + 增量」—— 给 0 就是零刚度，爪面只是贴着
    # 方块、一点正压力都没有。实测搬运时开口从头到尾钉在 +0.162（方块根本没撑开它），
    # 方块却以约 3mm/s 匀速向后滑出爪子掉地上，接触点 9/7→8/7→4/6→0/2。
    GRIP_BIAS = 0.02    # 守位时恒定给的合爪量（位置误差，不是速度）

    def grip_hold():
        """夹住之后给的那一维指令：恒定合爪一份，再按「被撑开多少」补回来。

        返回值恒在合爪侧（≤0），不会主动去掰开方块。
        """
        ref = GRIP_REF[0]
        if ref is None:
            return 0.0
        q = float(u.data.qpos[GRIP_QADR])
        rel = float(np.clip((ref - q) / GRIP_K, -1.0, 0.0)) * GRIP_RATE
        return rel - GRIP_BIAS

    def pick_up(i) -> bool:
        """中立位 → 方块正上方 → 竖直下探到方块中心 → 合爪（贴住就停）→ 抬到搬运高度。

        下探和合爪都不开环：
        - 下探带 guard，途中指垫一旦碰上这个方块就中断。还没到中心高度就碰上，说明
          指垫是落在方块顶面上的，当场把证据打出来，不接着往下压。
        - 合爪走 close_until_touch，两个指垫同时贴到侧面才收手。

        返回真表示抬起来了（方块跟着 TCP 走）。
        """
        # 起点取自标识点的已知坐标，不读方块实时位置 —— 操作员已经把它摆在那儿了。
        src = np.array([PICK_XY[i][0], PICK_XY[i][1], CUBE_Z])
        GRIP_REF[0] = None                             # 这一把重新记开口
        nrm = face_normal(cube_quat(i), tool_x())      # 四条侧面法向里转得最少的那条
        hit = [None]

        def guard():
            nf, nm, z = finger_cube_contacts(i)
            if nf or nm:
                hit[0] = (nf, nm, z)
                return True
            return False

        SEG[0], IGNORE[0] = "进场 → 方块上方", {i}
        reason = goto_path([WAYPOINT, src + [0, 0, 0.08]],
                           1.0, x_des=nrm, final_tol=0.004)
        # 落点：让**不动的那只爪子**贴着方块，不是把开口中心对着方块中心。
        #
        # 开口中心对方块中心的话，固定爪离方块还差半个开口；合爪时活动爪得先把方块
        # 横推过去才夹得住，方块是被顶走的，夹稳的余量全耗在这一推上。改成让固定爪
        # 的指尖落到方块朝它那一侧的面上（留一点点缝，下探时不刮），活动爪再合过来
        # 就是实打实的两面夹。
        #
        # 腕子这时候才转到位，方向现在才算得对（转腕之前 tool_x 是另一个朝向）。
        xt = tool_x()
        xh = np.array([xt[0], xt[1], 0.0])
        if float(np.linalg.norm(xh)) > 1e-6:
            xh /= np.linalg.norm(xh)
        f_tip = u.data.geom_xpos[FIX_TIP].mean(axis=0)
        side = 1.0 if float(np.dot(f_tip - src, xh)) >= 0.0 else -1.0
        # 方块是斜的，沿爪轴的半宽不是 15mm，是 15·(|cos|+|sin|)（外接盒那一套）。
        # 按 15mm 算的话，方块稍微一转，1mm 的缝就被吃掉了 —— 实测就是这样压到顶面的。
        from scipy.spatial.transform import Rotation
        Rc = Rotation.from_quat([cube_quat(i)[1], cube_quat(i)[2],
                                 cube_quat(i)[3], cube_quat(i)[0]]).as_matrix()
        half = float(np.dot(np.abs(Rc.T @ xh), np.full(3, 0.015)))
        # 固定指尖现在离方块中心多远 → 要挪到「方块那个面 + 一点缝」上
        ds = side * (half + FIX_CLEAR) - float(np.dot(f_tip - src, xh))
        aim = src + (float(np.dot(st()[1], xh)) + ds - float(np.dot(src, xh))) * xh
        aim[2] = src[2]
        print(f"    落点按固定爪贴面算（固定爪在方块 {('+x' if side > 0 else '-x')} 侧）")
        SEG[0] = "下探到方块"
        reason = goto_path([[aim[0], aim[1], src[2] + 0.05], aim],
                           1.0, x_des=nrm, final_tol=0.002, guard=guard, speed=SLOW)
        GRASP_Z[0] = float(st()[1][2])
        if reason == "guard":
            nf, nm, z = hit[0]
            print(f"    ✗ 下探半路就碰上：固定指垫 {nf} 点 / 活动指垫 {nm} 点"
                  f"  触点 z={z:.3f}  TCP z={GRASP_Z[0]:.3f}  方块中心 z={src[2]:.3f}")
            print(f"      下探停因 = {reason}（在方块中心以上就碰到，没进到侧面）")
            return False

        if not close_until_touch(i, nrm):
            print(f"    ✗ 两个指垫没同时贴上方块（只贴了一侧或没贴上），没夹住")
            return False
        nf, nm, tz = finger_cube_contacts(i)
        HELD[0] = i
        print(f"    合爪：下探停因 {reason}  TCP z={GRASP_Z[0]:.3f}"
              f"  固定触点 {nf} / 活动触点 {nm}  接触高度 {tz:.3f}"
              f"  爪轴与面法向 {np.degrees(align_err(nrm)):4.1f}°"
              f"  开口守位 {GRIP_REF[0]:+.3f}")

        # 抬起来的时候不再一路夹到底（原来给 -1.0，十来步就掐死）；守着自己的开口走
        SEG[0] = "抬升到搬运高度"
        up = goto_path([[src[0], src[1], LIFT_Z]], grip_hold, x_des=nrm, final_tol=0.012,
                       speed=CARRY)
        nf, nm, _ = finger_cube_contacts(i)
        if up != "reached":
            # 抬升中途停住（stall / 撞满步数）的话，方块到底跟没跟上来只有这里能看出来。
            # 之前返回值被丢掉，日志里只剩「方块 z 偏低」这一个现象，分不清是没抬上去
            # 还是方块在爪里滑了。
            print(f"    ✗ 抬升没走完：停因 {up}  TCP z={float(st()[1][2]):.3f}"
                  f"（目标 {LIFT_Z:.2f}）  方块 z={float(cube_pos(i)[2]):.3f}"
                  f"  爪-方块接触 固定 {nf} / 活动 {nm}")
        rose = float(cube_pos(i)[2]) - float(src[2])
        if rose < 0.02:
            print(f"    ✗ 抬完之后方块只升了 {rose * 1000:.0f}mm，没夹住")
            return False
        return True

    def carry(i, xy):
        """手里夹着方块，一路搬到 xy 的地面高度再松爪。

        三个点一条路径：目标上方 → 贴地 → 原位不动只松爪。中间不刹车。

        轻拿轻放：搬运时守着自己的开口（grip_hold，不再一路夹到底）；贴地那一段限速
        （speed=SLOW），不是按巡航速度砸下去；松爪是一小步一小步张（OPEN_STEP），
        不是一步给满弹开。

        返回真表示方块确实放下去了。没降到贴地、或者松爪前发现方块已经不在爪里，
        就不张爪（张了就是半空撒手），返回假。
        """
        dst = np.array([xy[0], xy[1], 0.0])
        SEG[0], IGNORE[0] = f"搬运 {CUBE_ORDER[i]} → {np.round(dst[:2], 3)}", {i}
        down = goto_path([[dst[0], dst[1], LIFT_Z], dst + [0, 0, GRASP_Z[0]]],
                         grip_hold, speed=CARRY)
        nf, nm, _ = finger_cube_contacts(i)
        z_tcp = float(st()[1][2])
        # 贴地段几乎都是 stall 收尾（高度到位、xy 差几毫米过不了 final_tol），不算没降下去；
        # 真正要拦的是「还挂在半空」和「方块早就不在爪里了」这两件事。
        if z_tcp > GRASP_Z[0] + AIR_TOL or (nf == 0 and nm == 0):
            # 张爪这一步原先是无条件的，`goto_path` 的返回值被丢掉，所以「空中松爪」
            # 在代码里是真会走到的。这里改成不放：报出来，爪子保持不动。
            print(f"    ✗ 不松爪：贴地段停因 {down}"
                  f"  TCP z={z_tcp:.3f}（放置高度 {GRASP_Z[0]:.3f}，还在上面 ≥{AIR_TOL * 1000:.0f}mm）"
                  f"  爪-方块接触 固定 {nf} / 活动 {nm}"
                  f"  方块在 {np.round(cube_pos(i), 3)}")
            return False
        grip_move(OPEN_STEP, 100)                              # 贴地了才慢慢松爪
        HELD[0] = None
        GRIP_REF[0] = None
        SEG[0] = "放完退开"
        goto_path([[dst[0], dst[1], LIFT_Z]], 1.0, final_tol=0.015, speed=SLOW)  # 退开
        print(f"    放到 {np.round(cube_pos(i)[:2], 3)}")
        return True

    rec()
    tcp_home = st()[1].copy()

    for k in range(N_EPISODES):
        # 每一轮的起点都是标识点：先由「操作员」把三个方块摆上去，机械臂再动手。
        park_cubes()
        _, target_idx, desc = layout_of(args.seed + k)
        print(f"\n第 {k + 1} 条 pick-and-lift：{CUBE_ORDER[target_idx]}"
              f"（seed {args.seed + k} 指定：{desc}）")
        ok = pick_up(target_idx)
        print(f"  {'抬起后方块在' if ok else '没抬起来，方块在'}"
              f" {np.round(cube_pos(target_idx), 3)}")
        if ok:
            placed = carry(target_idx, PLACE_XY[target_idx])
            print(f"    {CUBE_ORDER[target_idx]} → 放置标识点"
                  f"({PLACE_XY[target_idx][0] * 1000:.0f},"
                  f"{PLACE_XY[target_idx][1] * 1000:.0f})"
                  f" {'落地' if placed else '没放下，还在爪里'}")

    SEG[0], IGNORE[0] = "回原位", set()
    goto_path([tcp_home], 1.0, final_tol=0.010)   # 回原位
    if BUMP:
        print(f"\n⚠ 本轮一共撞到别的方块 {len(BUMP)} 次（按「段 + 方块 + geom」去重）：")
        for (seg, ci, name) in BUMP:
            print(f"    {seg}：{CUBE_ORDER[ci]} 方块（geom={name}）")
    else:
        print("\n✓ 全程没碰到不该碰的方块")
    print("\n回到原位，本轮结束")

    # ---- 写视频 ----
    import av
    h, w = frames[0].shape[:2]
    with av.open(args.out, "w", format="mp4") as c:
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "23", "preset": "veryfast"}
        for f in frames:
            for pk in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    print(f"\n{len(frames)} 帧，{len(frames) / FPS:.1f} 秒"
          f"（每 {args.stride} 个物理步一帧，共 {tick[0]} 步）")
    print(f"视频 -> {args.out}  ({Path(args.out).stat().st_size // 1024} KB)")
    env.close()
    sampler.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
