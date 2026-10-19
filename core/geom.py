"""算术工具：工位坐标表、朝向计算。

这些东西不属于任何一块零件 —— 机械臂要算朝向，技能要摆点，将来大模型策略也
用得上。所以放这儿，谁都能调，谁都不拥有。**全部是纯函数**，不看现场、不存状态。

**这里只留「没人做过的小算术」。** 凡是仿真栈 `so101_nexus` 已经有的，一律调它：
旋转误差走 `so101_nexus.kinematics.orientation_error`，不自己再推一遍四元数。
抓取判据（两指是否从两侧对夹）走 `so101_nexus.grasp.opposing_normals_ok`。

**轨迹不在这里。** 拐角、弧长、速度曲线三件事统一交给 toppra，见 `core/trajectory.py`。
"""

from __future__ import annotations

import numpy as np

# ---- 标识点：工位坐标表 ----
# 机械臂的动作一律以这几个**已知坐标**为准，不再去读方块的实时位姿。真机上这张表
# 就是标定/示教的结果，形状不变；场景里同时画出可见的标记盘，以后换成相机识别同
# 一个标记即可。圆心取 shoulder_pan 轴 (0.039, 0)，半径 0.20 落在实测可达区间
# 0.13~0.26 的中段。拾取位和放置位错开 120°，同一侧的相邻两点方位差 40°，
# 半径 0.20 时弦长 0.137——比如今的 min_object_separation 0.12 宽。
CUBE_ORDER = ("red", "green", "blue")   # 方块池顺序 = slot 0/1/2
MARKER_CENTER = np.array([0.039, 0.0])
MARKER_R = 0.20
PICK_DEG = (20.0, 60.0, 100.0)       # 拾取位：red / green / blue 各一个
PLACE_DEG = (-20.0, -60.0, -100.0)   # 放置位：各自对应的落点
CUBE_Z = 0.015                       # 30mm 方块躺在地面上的中心高度
# 标识点圆盘一个中性灰，六个都用它。**不能用方块色**：圆盘直径 40mm、方块才 30mm，
# 圆盘会露在方块外面一圈；同色的话按颜色分出来的就不是方块，是「方块 + 圆盘」那一坨，
# 量朝向就成了量那个圆盘。标识点靠**位置**认，颜色只是为了在画面里看得见。
MARKER_RGBA = (0.55, 0.55, 0.55)

# ---- 摆件清单：桌上摆的是什么 ----
# 每一项 = 一个颜色 + 一个形状 + 尺寸（毫米）。形状表和尺寸口径见 `core/shapes.py`。
# **空 = 老口径**：三块 30mm 方块（`CUBE_ORDER`）。填了就以它为准，长度必须跟它一样。
#
# 这一张表是**唯一的一份**：仿真照它建场景（`impls/arm_sim.py`）、拟合位置照它投剪影
# （`core/vision.py`）、判落地照它算躺平多高（`impls/judge_truth.py`）、感知照它建词表
# （`impls/perception.py`）。改一处，四处一起变。
#
# 例：白瓷茶杯（实心圆柱还是空心的，一个视角看不出来，得在表里说）：
#   OBJECTS = [{"color": "red",   "shape": "cube",     "side_mm": 30},
#              {"color": "white", "shape": "cylinder", "diameter_mm": 60,
#               "height_mm": 60, "cup": True, "handle_deg": 0}]
OBJECTS: list[dict] = []

# nexus 认的全部颜色名（`so101_nexus.constants.COLOR_MAP`）。gray 留给地面，摆件不用它。
# 「把颜色加全」加的就是这一串 —— 白杯子、黄方块、黑骰子都能说得出口。
ALL_COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "black", "white")

ROT_SCALE = 0.1   # 旋转增量归一化的分母（align_action 默认用它）

# ---- 可达环 ----
# 离肩部轴多远算「够得着」。这两个数就是摆方块时允许摆的范围
# （`demo_picklift_round.py` 的 `spawn_min_radius` / `spawn_max_radius`，实测值：
# 桌面高度残差 1~4mm，再往里小于 0.10 会把方块塞到机身底下）。
# 方块出了这个环 = 跑到了当初连摆都不允许摆的地方 = 够不着、也回不来，
# 所以「出界」是收工的理由，不是重试的理由。
REACH_MIN_R = 0.13
REACH_MAX_R = 0.26


def _ring(deg):
    """把一串角度换成工位上的 (x, y) 坐标。"""
    return np.array([[MARKER_CENTER[0] + MARKER_R * np.cos(np.radians(d)),
                      MARKER_CENTER[1] + MARKER_R * np.sin(np.radians(d))] for d in deg])


PICK_XY = _ring(PICK_DEG)
PLACE_XY = _ring(PLACE_DEG)

# ---- 外面给的落点 ----
# **真机上到这一步，没人给你画标记，只有一串数**：「这个点离底座左右多少、前后多少」。
# 放这个字段就是那串数（米，底座坐标系）。给了它就往那儿放，不给就还是用表里的
# `PLACE_XY[i]`。默认 None。
PLACE_TARGET: np.ndarray | None = None

# 额外画出来的盘（米，底座坐标系）。**只画给人看** —— 代码里没有任何一处读它：
# 落点是 `PLACE_TARGET` 那串数说了算，盘只是把它在场景里显出来，让你一眼看出方块
# 落得准不准。真机上没有这个东西，所以它跟落点计算完全无关。
EXTRA_MARKS: list[np.ndarray] = []

# ---- 真机现场的两个常数 ----
# 真机上没有仿真真值，感知要把画面反推成世界坐标，先得知道两件事：桌面在基座系
# 多高（反推的水平面在哪），以及方块大概摆在哪一带（`estimate_pose` 的起搜中心）。
# **由主循环从配置的 `real:` 那一段填**，口径跟 `OBJECTS` / `PLACE_TARGET` /
# `EXTRA_MARKS` 一样：建组件之前先设好，组件起来时读。
# **没填就是 None，读的地方当场抛** —— 跟 `thresholds.json` 那条规矩一样：
# 宁可不干活，也不要拿一个猜的坐标去驱动机械臂。
REAL_TABLE_Z: float | None = None        # 桌面在基座系的 z（米）
REAL_PICK_XY: np.ndarray | None = None   # 方块大致摆在哪（米，基座系 x,y）
# 腕相机反推出来的世界坐标要**扣掉**的固定偏移（米，基座系 x,y）。这是标定误差的补丁，
# 不是模型的一部分：2026-10-15 实测这一支的盒子 x 恒比真实位置大 35.8mm（三次记录一致，
# 见 memory `so101-real-perception-x-offset`），y 不偏。没配就是 None = 不扣。
REAL_PERCEP_OFFSET: np.ndarray | None = None


def real_table_z() -> float:
    """桌面在基座系的 z（米）。配置里没给就抛。"""
    if REAL_TABLE_Z is None:
        raise RuntimeError(
            "没给桌面高度（`core.geom.REAL_TABLE_Z`）。真机上它是反推世界坐标的"
            "水平面，不给就没法算 —— 别用 0 凑，0 是基座原点的高度，不是桌面。"
            "在 configs/real.yaml 的 `real:` 那一段写 `table_z_mm:`。")
    return float(REAL_TABLE_Z)


def real_pick_xy() -> np.ndarray:
    """方块大致摆在哪（米，基座系 x,y）。配置里没给就抛。"""
    if REAL_PICK_XY is None:
        raise RuntimeError(
            "没给方块的摆放位置（`core.geom.REAL_PICK_XY`）。真机上它是"
            "`core.vision.estimate_pose` 的起搜中心（±30mm 网格），不给就没法起搜。"
            "在 configs/real.yaml 的 `real:` 那一段写 `pick_xy_mm: [x, y]`。")
    return np.asarray(REAL_PICK_XY, dtype=np.float64)


def real_percep_offset() -> np.ndarray:
    """腕相机反推的世界坐标要扣掉的固定偏移（米，基座系 x,y）。

    **没配就是 (0, 0)，不抛。** 跟上面两个常数不一样：那两个少了就没法算，
    这个少了只是「不扣」 —— 仿真那一路本来就该是 0。
    """
    if REAL_PERCEP_OFFSET is None:
        return np.zeros(2, dtype=np.float64)
    return np.asarray(REAL_PERCEP_OFFSET, dtype=np.float64).reshape(2)


def object_specs() -> list[dict]:
    """这一局桌上摆的是什么。没人填就是老口径那三块 30mm 方块。

    返回的每一项都补齐了 `shape` 和 `color` 两个键 —— 形状表、感知词表、仿真建场景
    都只认这个格式。
    """
    if OBJECTS:
        out = []
        for o in OBJECTS:
            spec = dict(o)
            spec.setdefault("shape", "cube")
            spec.setdefault("color", "red")
            if spec["shape"] == "cube":
                spec.setdefault("side_mm", 30.0)
            out.append(spec)
        return out
    return [{"color": c, "shape": "cube", "side_mm": 30.0} for c in CUBE_ORDER]


def object_shape(i: int):
    """第 i 件摆件的形状（`core.shapes.Shape`）。越界就取最后一件。"""
    from core import shapes

    specs = object_specs()
    return shapes.from_spec(specs[min(int(i), len(specs) - 1)])


def object_rest_z(i: int = 0) -> float:
    """第 i 件摆件躺在地面上时中心的高度（米）。**判落地按它，别再按 CUBE_Z。**"""
    return float(object_shape(i).rest_z)


def palette() -> tuple[str, ...]:
    """场上这几件的颜色词。感知拿它建词表 —— 场上有几种颜色，就认得几种。"""
    return tuple(str(o["color"]) for o in object_specs())


def place_xy(i: int) -> np.ndarray:
    """第 i 块方块该放到哪：外面给了坐标就用它，否则用表里的放置标识点。"""
    if PLACE_TARGET is None:
        return np.asarray(PLACE_XY[i], dtype=np.float64)
    return np.asarray(PLACE_TARGET, dtype=np.float64)


def reach_radius(xy) -> float:
    """这个平面点离肩部轴多远（米）。"""
    return float(np.hypot(float(xy[0]) - MARKER_CENTER[0],
                          float(xy[1]) - MARKER_CENTER[1]))


def out_of_reach(xy) -> bool:
    """这个平面点是不是出了可达环 —— 够不着了。"""
    r = reach_radius(xy)
    return bool(r < REACH_MIN_R or r > REACH_MAX_R)


def marker_xml(ring: bool = True):
    """标识点的 MJCF：贴地薄圆盘，看得见、不参与碰撞（contype/conaffinity=0）。

    nexus 的 mujoco ``PickEnv`` 没有 ``extra_bodies`` 钩子（只有 ``pick_and_place``
    有），所以调用方把 ``build_object_scene_xml`` 包一层，往 worldbody 末尾注入这段。

    ``ring=False`` 就只画 ``EXTRA_MARKS`` 里的盘、不画工位圆环 —— 单方块试验要的口径
    是「场上没有标识点，只有一个我告诉你的坐标」，圆环一画就把这个口径破了。
    """
    xy_list = [np.asarray(p, dtype=np.float64) for p in EXTRA_MARKS]
    if ring:
        xy_list = [np.asarray(p, dtype=np.float64) for p in np.vstack([PICK_XY, PLACE_XY])] + xy_list

    out = []
    for i, xy in enumerate(xy_list):
        r, g, b = MARKER_RGBA
        out.append(
            f'    <body name="marker_{i}" pos="{xy[0]:.6f} {xy[1]:.6f} 0.0012">\n'
            f'      <geom type="cylinder" size="0.020 0.0006" '
            f'rgba="{r:.3f} {g:.3f} {b:.3f} 1" contype="0" conaffinity="0"/>\n'
            f'    </body>\n')
    return "".join(out)


def round_route(waypoints, step=0.04):
    """把相邻两点的**水平**位移改成绕底座的圆弧插值，竖直段不受影响。

    直线连接两个方位角差得远的点，弦会从底座上方切进去：搬运那段从 (0.082,0.212)
    到 (0.215,-0.145)（方位差 103°）的弦离底座只有 0.15，而两端都在 0.23 以上。
    腕关节要折叠到那个半径就顶到行程上限，逆解只好放开位置去迁就姿态，手臂甩到
    外圈走弧线 —— 实测就是这么撞上别的方块的。按极坐标插值，半径只会在两端之间
    变化，不再切进去。竖直那几段 r、θ 不变，插出来还是竖直线。

    这是「路走哪儿」的问题，不是「怎么走快」的问题 —— 后者交给 toppra。
    """
    pts = [np.asarray(waypoints[0], dtype=np.float64)]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        av = np.asarray(a, dtype=np.float64)
        bv = np.asarray(b, dtype=np.float64)
        ra, rb = float(np.hypot(av[0], av[1])), float(np.hypot(bv[0], bv[1]))
        ta = float(np.arctan2(av[1], av[0]))
        tb = float(np.arctan2(bv[1], bv[0]))
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


def _matrix(q_wxyz):
    """四元数 [w,x,y,z] → 3x3 旋转矩阵。"""
    from scipy.spatial.transform import Rotation

    q = np.asarray(q_wxyz, dtype=np.float64)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def tool_axis(q_wxyz):
    """工具 z 轴（夹爪前进方向）在世界系里的单位向量。"""
    return _matrix(q_wxyz)[:, 2]


def obj_yaw(q_wxyz):
    """绕竖直轴的转角（度）。"""
    from scipy.spatial.transform import Rotation

    q = np.asarray(q_wxyz, dtype=np.float64)
    return float(np.degrees(Rotation.from_quat(
        [q[1], q[2], q[3], q[0]]).as_euler("zyx")[0]))


def face_normal(obj_quat_wxyz, tool_x, shape=None):
    """该往哪个水平方向合爪 —— 跟当前爪开合轴最接近的那条物体表面法向。

    爪的开合方向就是工具系的 x 轴：固定爪的指面在 TCP 系 x=+0.030，活动爪在
    x=+0.002，两个指面沿 x 分开。所以 x 轴对上某个面法向，两个指面才正好贴在
    方块的一对侧面上；不对上就是拿指尖去顶棱，夹不住。

    `shape=None`（默认）就是原来口径：**方块**四棱等价，四条法向（±局部x、±局部y）
    都能夹，取转得最少的那条。给了 `core.shapes.Shape` 就问那个形状自己 —— 圆柱
    怎么转都能夹（用现成的开合轴，腕子不用多转），有把手的杯子还会绕开把手那一片。
    见 `core/shapes.py` 的 `Shape.normal`。
    """
    if shape is not None:
        from core import shapes as _shapes

        return _shapes.from_spec(shape).normal(np.radians(obj_yaw(obj_quat_wxyz)), tool_x)

    R = _matrix(obj_quat_wxyz)
    cands = [R[:, 0], -R[:, 0], R[:, 1], -R[:, 1]]
    th = np.array([tool_x[0], tool_x[1], 0.0], dtype=np.float64)
    if np.linalg.norm(th) < 1e-6:
        return cands[0]
    th /= np.linalg.norm(th)
    return max(cands, key=lambda v: float(np.dot(v[:2], th[:2])))


def align_action(q_wxyz, z_des, gain=ROT_SCALE, x_des=None):
    """把工具 z 轴转到 z_des 需要的归一化旋转增量。

    目标姿态怎么造是我们的事（「工具轴对准某个方向、同时让开合轴尽量贴住给定的
    x 参考」）；**从当前姿态到目标姿态的旋转误差不自己算，调 nexus**。
    """
    from scipy.spatial.transform import Rotation

    from so101_nexus.kinematics import orientation_error

    R = _matrix(q_wxyz)
    z = np.asarray(z_des, dtype=np.float64)
    z = z / np.linalg.norm(z)
    src = R[:, 0] if x_des is None else np.asarray(x_des, dtype=np.float64)
    x = src - np.dot(src, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = R[:, 1] - np.dot(R[:, 1], z) * z
    x = x / np.linalg.norm(x)
    R_des = np.column_stack([x, np.cross(z, x), z])

    quat_xyzw = Rotation.from_matrix(R_des).as_quat()          # [x,y,z,w]
    q_des_wxyz = np.array([quat_xyzw[3], quat_xyzw[0],
                           quat_xyzw[1], quat_xyzw[2]])
    err = orientation_error(np.asarray(q_wxyz, dtype=np.float64), q_des_wxyz)
    return np.clip(err / gain, -1.0, 1.0)
