"""仿真机械臂：把 nexus 的 MuJoCo 环境包成一个 Arm。

外面只看到 `tip()` / `drive()` / `angles()` / `contacts()` 四个动作。里面所有跟
MuJoCo 打交道的细节（qpos 地址、接触表、哪个 geom 是爪子）都锁在这个文件里 ——
换真机的时候，这个文件整个换掉，别的一个字不动。

**动作怎么发**：nexus 这个环境收的是 `pd_ee_delta_pose` 七个数的增量
（[dx,dy,dz, rx,ry,rz, 夹爪]），其中旋转是「归一化的旋转增量」，不是角度。
所以 `drive()` 收的是「爪子尖去哪」，这里再换算成那个七维增量：
    位置差 → 除以 POS_SCALE 压到 ±1 → 乘速度
    朝向   → 交给 core.geom.align_action 算出旋转增量
两条都过一道限斜率（每步最多变多少），免得误差一大就是满速甩腕。

**限斜率的状态（VEL / RVEL）存在这个对象里**：段与段之间不归零，接得上才不抖。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from core.geom import ROT_SCALE, align_action, marker_xml
from core.interfaces import Arm
from core.registry import register

# 动作尺度。和 nexus 的 pd_ee_delta_pose 口径一致，改这里等于改整台机器的快慢。
POS_SCALE = 0.02    # 一个动作单位对应多少米
SPEED_POS = 0.08    # 巡航速度（动作单位）。太快了：整条轨迹和腕子转都按它放
SPEED_ROT = 0.15    # 腕子转的上限（动作单位）。和速度档一起压低，转起来才不生硬
OW = 0.05           # ee_orientation_weight：默认 0.01 工具轴太斜，压不下去
CUBE_Z = 0.015      # 30mm 方块躺在地面上的中心高度（老口径；形状一多就按形状自己算）
MAX_STEPS = 20000
PAD_FRICTION = 4.0  # 爪指垫的滑动摩擦。模型给的 1.0 夹不住方块，见 _make_env 里的说明

# ---- 域随机化：让每一局的视觉条件都不一样 ------------------------------------
# 现在每一局的光和桌面是**一模一样**的。模型在这种数据上练出来的，很可能只是
# 「我认得这张桌子、认得这个光」，换个灯换张桌布就废了。每局都换一套，它才会去
# 抓真正有用的线索。
#
# **只动仿真里写死的那两样：灯光和地面。** 另外两样刻意不碰：
#   方块颜色 —— 任务指令的文字里带着方块颜色（`config.describe_pick_target` 那个
#               `{target!r}`），方块一变色，指令就跟画面对不上。
#   干扰物数量 —— 编译期定死，每局要变得重建整个场景，代价太大。
#
# 下面是保守值。要收紧或放宽，只改这一处。
LIGHT_SCALE = (0.55, 1.25)     # 所有灯的亮度乘这个系数，连续抽
LIGHT_TINT = (                 # 色温三档：暖 / 白 / 冷
    (1.00, 0.72, 0.45),
    (1.00, 1.00, 1.00),
    (0.80, 0.92, 1.00),
)
FLOOR_COLORS = ("gray", "white", "black", "blue", "green", "orange")  # 用 nexus 自己那套色名
FLOOR_GEOM = "floor"           # 场景里地面 geom 的名字（nexus 生成的）
REF_PARK_XY = (3.0, 3.0)       # 拍空工位参考图时把方块暂时挪到这儿 —— 画面外，不挡着

SIM_DIR = Path(r"G:\sim2real\so101-smolvla-sim")

# 摆件序号 → 网格缩放。扫描网格（GSO 那种）是**真尺寸**，想在仿真里抓它得先缩到
# 夹爪开得下的尺寸。缩放**加在生成的场景 XML 上**，不动只读的包、也不动
# `~/.cache/so101_nexus` 里那份原网格（改了缓存，别的用同一件东西的地方也跟着变）。
MESH_SCALE: dict[int, float] = {}


def _scale_meshes(xml: str) -> str:
    """给指定序号的摆件网格补上 `scale=`。只认 nexus 生成的那两种 mesh 名字。"""
    import re

    def one(pat: str, xml_: str) -> str:
        def sub(m):
            s = MESH_SCALE.get(int(m.group(1)))
            if s is None:
                return m.group(0)
            return m.group(0).replace("/>", f' scale="{s} {s} {s}"/>')
        return re.sub(pat, sub, xml_)

    xml = one(r'<mesh name="pick_coll_(\d+)_\d+" file="[^"]+"\s*/>', xml)
    return one(r'<mesh name="pick_vis_(\d+)" file="[^"]+"\s*/>', xml)


# 盲测那几件 nexus 造不出来的形状。**不碰只读的包**，改的是生成出来的那几行 XML。
# 键是摆件序号，跟 `MESH_SCALE` 一个口径，`SimArm.__init__` 里按清单填。
BOX_HALF: dict[int, tuple[float, float, float]] = {}   # → 三个半长（米，x y z）
CUP_WALL: dict[int, tuple[float, float, float]] = {}   # → (外半径, 半高, 壁厚)（米）
CUP_SEGS = 24        # 杯壁拼多少片。24 片的外接多边形跟圆差 0.3%（剪影上看不出来）
CUP_SEG_MARGIN = 1.30   # 每片切向多给三成，让相邻两片搭上 —— 不留缝，多出来的重叠
                        # 部分在同一个刚体里、互相不产生接触


def _reshape(xml: str, objects) -> str:
    """把长方体和中空圆柱的几何改出来。

    **为什么在 XML 上做。** nexus 的摆件只有「正方体（一个半长）、圆柱（高=直径）」，
    长方体和中空件它造不出来（`CubeObject` 只有 `half_size` 一个标量；`CylinderObject`
    是实心的）。而 MuJoCo 的碰撞几何**必须是凸的**，一个真中空的网格塞进去等于塞了
    个实心 —— 所以中空只能拿一圈凸块拼出来（nexus 自己对扫描件也是这么干的：coacd
    拆成若干凸块）。这里改的只是自己进程里生成出来的那几行，包和缓存一个字节没动。

    长方体：`size` 三个数各是各的（原来三个都是同一个半长），body 的初始摆放高度
    跟着改成竖边的一半（`spawn_z` 是 nexus 按 `half_size` 定的，我们造的时候就把
    `half_size` 传成竖边的一半，所以这里只需要改 geom 的 size）。
    中空圆柱：底留一块（**名字还是 `pick_slot_{i}_geom`**，nexus 按这个名字找碰撞
    geom，改了名它就找不到、`_slots` 直接报错），外面一圈薄片当壁，名字另起。
    """
    import re

    from so101_nexus.constants import COLOR_MAP

    def one(i):
        r, g, b, a = COLOR_MAP[str(objects[i].color)]
        return f'"{r} {g} {b} {a}"'

    def geom(i, name, size, extra=""):
        return (f'<geom name="{name}" type="box" size="{size}"\n'
                f'            rgba={one(i)} {extra}'
                f'contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"\n'
                f'            solref="0.01 1" solimp="0.95 0.99 0.001"/>')

    for i, (hx, hy, hz) in BOX_HALF.items():
        xml = re.sub(rf'<geom name="pick_slot_{i}_geom".*?/>',
                     geom(i, f"pick_slot_{i}_geom", f"{hx} {hy} {hz}",
                          f'mass="{float(objects[i].mass)}" '),
                     xml, count=1, flags=re.S)

    for i, (r_out, hz, t) in CUP_WALL.items():
        mass = float(getattr(objects[i], "mass", 0.01))
        r_mid = max(1e-4, r_out - t / 2.0)
        cut = 2.0 * r_out * math.tan(math.pi / CUP_SEGS) * CUP_SEG_MARGIN   # 每片切向全长
        segs = [f'      <geom name="pick_slot_{i}_geom" type="cylinder"'
                f' size="{r_mid} {t / 2.0}" pos="0 0 {-hz + t / 2.0}"\n'
                f'            rgba={one(i)} mass="{mass * 0.5}"'
                f' contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"\n'
                f'            solref="0.01 1" solimp="0.95 0.99 0.001"/>']
        for k in range(CUP_SEGS):
            ang = 2.0 * math.pi * k / CUP_SEGS
            segs.append(
                f'      <geom name="pick_slot_{i}_wall_{k}" type="box"'
                f' size="{t / 2.0} {cut / 2.0} {hz}"'
                f' pos="{r_mid * math.cos(ang)} {r_mid * math.sin(ang)} 0"'
                f' quat="{math.cos(ang / 2.0)} 0 0 {math.sin(ang / 2.0)}"\n'
                f'            rgba={one(i)} mass="{mass * 0.5 / CUP_SEGS}"'
                f' contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"\n'
                f'            solref="0.01 1" solimp="0.95 0.99 0.001"/>')
        # 原来那一行圆柱整个换掉：底那块**沿用名字**，nexus 靠它找这一件的碰撞 geom
        xml = re.sub(rf'<geom name="pick_slot_{i}_geom".*?/>',
                     "\n".join(segs), xml, count=1, flags=re.S)
    return xml


def _prepare_scene(ring: bool = True, markers: bool = True):
    """包住 nexus 的场景 XML 生成，补上标识盘。

    nexus 的 PickEnv 内部直接调 `build_object_scene_xml` 且不传 extra_bodies，
    这里在模块上包一层补上 —— 只改本进程的绑定，不动只读的包。

    `ring=False` 只画 `EXTRA_MARKS`（外面给的那个落点），不画工位圆环。
    `markers=False` 一个盘都不画（单方块试验要的就是这个口径）。
    """
    import so101_nexus.mujoco.pick_env as pick_env

    if getattr(pick_env.build_object_scene_xml, "_harness_wrapped", False):
        return
    original = pick_env.build_object_scene_xml

    def wrapped(*a, **kw):
        if markers:
            kw.setdefault("extra_bodies", marker_xml(ring))
        xml = original(*a, **kw)
        # 几何**在生成之后**改：长方体和中空圆柱 nexus 造不出来，改它生成出来的那几行
        if BOX_HALF or CUP_WALL:
            xml = _reshape(xml, list(a[0]) if a else [])
        # 缩放同理：生成时期望的是真尺寸的网格，只有落进场景那一刻要缩。
        return _scale_meshes(xml) if MESH_SCALE else xml

    wrapped._harness_wrapped = True
    pick_env.build_object_scene_xml = wrapped


def _nexus_object(spec: dict):
    """按摆件清单的一行造一个 nexus 场景物体。

    **尺寸对不上的组合当场报错，不悄悄凑合。** nexus 的圆柱和球是「填满同一个立方体」
    的 —— `CylinderObject(diameter_mm=d)` 造出来直径和**高**都是 d，没有单独给高的
    参数。所以清单里写了 `height_mm` 又跟直径不一样，仿真这边造不出来，直接说清楚。
    （真机那边不受这条限制，`core.shapes.Cylinder` 半径和高是分开的。）

    `cup=True` 也只是**抓法**上的意思（顶上那截口沿不许夹），仿真里照样是个实心圆柱
    —— 一个视角看不出里面是空的，实心圆柱和杯子的剪影一模一样。
    **但清单里另写了 `wall_mm` 就不一样了**：那是「这一件真的是个薄壁杯子」，仿真里
    得真造出内腔来（`_reshape` 拿底加一圈薄片拼），不然里面看得见的那点阴影就不存在，
    视觉判空心这一条根本没法考。老配置里 `cup: true`（不带 `wall_mm`）照旧是实心的。
    """
    from so101_nexus import CubeObject, CylinderObject, SphereObject

    kind = str(spec.get("shape", "cube")).lower()
    color = str(spec.get("color", "red"))
    # **扫描网格**（`gso:` 给了数据集的 id）先走这一条：它是真扫描件、真的有中空，
    # 碰撞几何是 coacd 拆出来的凸块（nexus 的网格管线），所以爪子能伸进碗口里侧。
    # 尺寸改不了 —— 要抓得下只能在场景里缩（`MESH_SCALE`），缩放和质量都在这里定。
    if spec.get("gso"):
        from so101_nexus.constants import GSO_MASSES
        from so101_nexus.objects import GSOObject

        mid = str(spec["gso"])
        s = float(spec.get("scale") or 1.0)
        mass = spec.get("mass_kg")
        if mass is None:
            # 质量按体积缩 —— 扫出来的碗是 120g，缩到一半边长就只剩八分之一重。
            mass = float(GSO_MASSES.get(mid, 0.1)) * s ** 3
        return GSOObject(mid, mass_override=float(mass))
    if kind == "cube":
        return CubeObject(color=color, side_length_mm=float(spec.get("side_mm", 30.0)))
    if kind in ("box", "cuboid"):
        # 长方体。nexus 的 `CubeObject` 只有一个标量半长，三个半长各是各的造不出来 ——
        # 这里先按**竖边**造一个正方体（`spawn_z` 拿的就是它，物件得靠这个高度落到
        # 地面上），三个半长的差别交给 `_reshape` 在 XML 上改。
        hx, hy, hz = (float(v) for v in spec["size_mm"])
        return CubeObject(color=color, half_size=hz / 2000.0)
    if kind in ("cylinder", "sphere"):
        d = float(spec.get("diameter_mm", 60.0))
        h = float(spec.get("height_mm", d))
        if abs(h - d) > 1e-6:
            raise ValueError(
                f"仿真里造不出高 {h}mm、直径 {d}mm 的{'圆柱' if kind == 'cylinder' else '球'}"
                f"：nexus 这一支的{'圆柱/球' if kind == 'cylinder' else '球'}就是「填满同一个"
                f"立方体」，高必须等于直径。要么把 height_mm 去掉（就按直径），"
                f"要么这件只在真机上跑。")
        cls = CylinderObject if kind == "cylinder" else SphereObject
        return cls(color=color, diameter_mm=d)
    raise ValueError(f"仿真这边没有这种形状：{kind!r}（有 cube / cylinder / sphere）。"
                     f"清单里那一行：{spec}")


def _make_env(objects, markers: bool = True, ring: bool = True):
    """造一个 MuJoCoPickLift 环境。参数全部沿用调过的那一套。

    `objects` 是摆件清单（`core.geom.object_specs()` 那个格式）：一个颜色 + 一个形状
    + 尺寸。场上摆几件就是几个物体，除了目标那几个全算干扰项。

    `markers=False` 时不把标识盘注进场景。标识点在真机上就是「我知道这个坐标」，
    不是场景里一个看得见、相机认得出的东西 —— 单方块试验要的就是这个口径。
    `ring=False` 时连工位圆环也不画，只画 `EXTRA_MARKS` 里那几个盘。
    """
    import gymnasium as gym
    import so101_nexus.mujoco  # noqa: F401  注册 gym 环境
    from so101_nexus import (EndEffectorPose, JointPositions, ObjectPose, OverheadCamera,
                             PickConfig, RobotConfig, WristCamera)
    from so101_nexus.camera_utils import compute_angled_camera_params

    if markers:
        _prepare_scene(ring, markers)

    cfg = PickConfig(
        objects=[_nexus_object(o) for o in objects],
        # 物体池里除了目标的那几个全放上场：三件就是 2 个干扰项，一件就是 0
        #（0 = 单目标场景，官方支持）。
        n_distractors=len(objects) - 1,
        obs_mode="visual",
        observations=[JointPositions(), EndEffectorPose(), ObjectPose(),
                      WristCamera(), OverheadCamera()],
        robot=RobotConfig(ee_orientation_weight=OW),
        # 半径以 spawn_center 为圆心；圆心取 shoulder_pan 轴的世界坐标 (0.039, 0)
        spawn_center=(0.039, 0.0), spawn_min_radius=0.13, spawn_max_radius=0.26,
        min_object_separation=0.12,   # 默认 0.04，方块之间只剩 16mm，夹爪下不去
        # 默认 True：方块一抬过 lift_threshold 环境就判成功、当场终止这一局，
        # 后面的搬运/放爪全在半路上执行。这里要跑满一整轮，成功与否由判官说
        terminate_on_success=False,
    )
    env = gym.make("MuJoCoPickLift-v1", config=cfg, render_mode="rgb_array",
                   control_mode="pd_ee_delta_pose", max_episode_steps=MAX_STEPS)

    u = env.unwrapped

    # **爪指垫的滑动摩擦提到 4。** 模型里 `fixed_jaw_*` / `moving_jaw_*` 两个 class 都是
    # `friction="1 5e-3 5e-4"`（so101.xml:34-39），滑动系数 1.0，和方块一样。实测后果：
    # 搬运前 32 帧夹爪关节角锁死在 +7.66° —— 说明指垫确实夹住、位置伺服合不动了 ——
    # 第 323 帧一失去接触就自由合到硬限位 −10°，方块直坠落地，放置段读到接触 0/0。
    # 同一个机器人在 mcp_robot 那轮也栽在这上面（爪面摩擦 1→4，滑落 190mm→44mm），
    # 所以先动这一处。改的是运行时模型，不动只读包里的 xml。
    import mujoco

    for g in range(u.model.ngeom):
        name = mujoco.mj_id2name(u.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith("fixed_jaw") or name.startswith("moving_jaw"):
            u.model.geom_friction[g][0] = PAD_FRICTION

    # 俯视相机的机位按可达区算出来，不然角度偏了看不到整个工位
    cam, comp = u._overhead_obs_cam, u._overhead_cam_component
    if cam is not None and comp is not None:
        p = compute_angled_camera_params(
            spawn_center=u.config.spawn_center,
            spawn_max_radius=u.config.spawn_max_radius,
            elevation=-34.0, azimuth=205.0,
            fov_deg=comp.fov_deg, aspect=comp.width / comp.height)
        cam.lookat[:] = p["lookat"]
        cam.distance, cam.elevation, cam.azimuth = p["distance"], p["elevation"], p["azimuth"]
    return env


@register("arm", "sim")
class SimArm(Arm):
    """仿真机械臂。"""

    name = "sim"

    def __init__(self, render: bool = True, objects=None,
                 cube_order=("red", "green", "blue"),
                 markers: bool = True, ring: bool = True, randomize: bool = False):
        """`objects` 是摆件清单（`core.geom.object_specs()` 那个格式）。不给就按
        `cube_order` 摆一圈 30mm 方块 —— 老口径，跟以前逐字节一样。"""
        import mujoco

        self._mujoco = mujoco
        if objects:
            self.objects = [dict(o) for o in objects]
            for o in self.objects:
                o.setdefault("shape", "cube")
                o.setdefault("color", "red")
        else:
            self.objects = [{"color": str(c), "shape": "cube", "side_mm": 30.0}
                            for c in cube_order]
        # 颜色词这一串还是「第几件是谁」的名字，日志和证据里到处在用，不动它。
        self.cube_order = tuple(str(o["color"]) for o in self.objects)
        # 默认关。关着的时候这个类跟以前逐字节一样，现有实验和判官都不受影响。
        self.randomize = bool(randomize)

        # 缩放在**建场之前**填好：`_prepare_scene` 包住的建场函数在生成 XML 时读它，
        # 把 `scale="s s s"` 追加到这件网格的资产行上。nexus 自己不管缩放 —— 网格是按
        # 真实尺寸生成、也按真实尺寸算离地高的，只有这一步让它变小。
        MESH_SCALE.clear()
        for i, o in enumerate(self.objects):
            if o.get("scale"):
                MESH_SCALE[i] = float(o["scale"])
        # 长方体和中空圆柱的几何，跟缩放一个时机填：建场那一步在生成 XML 时读它们。
        BOX_HALF.clear()
        CUP_WALL.clear()
        for i, o in enumerate(self.objects):
            kind = str(o.get("shape", "cube")).lower()
            if kind in ("box", "cuboid") and o.get("size_mm"):
                hx, hy, hz = (float(v) / 2000.0 for v in o["size_mm"])
                BOX_HALF[i] = (hx, hy, hz)
            if kind == "cylinder" and o.get("wall_mm"):
                d = float(o["diameter_mm"])
                CUP_WALL[i] = (d / 2000.0,
                               float(o.get("height_mm", d)) / 2000.0,
                               float(o["wall_mm"]) / 1000.0)
        self.env = _make_env(self.objects, markers, ring)
        # 另开一个同配置的环境，只用来问「seed N 的东西摆在哪」。
        # 主环境跑着的时候不能 reset —— reset 会把机械臂也一起复位。
        self.sampler = _make_env(self.objects, markers, ring)

        self.obs = None
        self.info = None
        self.terminated = False
        self.last_action = np.zeros(7, dtype=np.float32)   # 最近一步发出去的七维指令
        self.vel = np.zeros(3)      # 跨段保留的平移速度
        self.rvel = np.zeros(3)     # 同上，旋转增量

        u = self.env.unwrapped
        self.u = u

        # 随机化用的「原样」：亮度每次都从这一份算。不这么做，一局局会叠着乘下去，
        # 跑上十局全场就黑了 —— 跟每次改颜色都从 `base_rgba` 算回来是同一个道理。
        self._base_lights = u.model.light_diffuse.copy()
        fid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
        if self.randomize and fid < 0:
            raise RuntimeError(f"模型里没找到名为 {FLOOR_GEOM!r} 的地面 geom，"
                               f"随机化没法换桌面颜色")
        self._floor_geom = int(fid) if fid >= 0 else None

        self.cube_addr = [s.qpos_addr for s in u._slots]
        self.cube_dof = [s.dof_addr for s in u._slots]
        # **一件东西的全部 geom，不只是 nexus 报的那一个。** 中空件是「底 + 一圈薄片」
        # 拼出来的（见 `_reshape`），nexus 只认名字里那一个（它按 `pick_slot_{i}_geom`
        # 找，多几何体的槽位它这一支不支持）。只认那一个的话，爪子夹在**杯壁**上的
        # 接触一个都数不到 —— 而「夹住了没有」正是按接触点在数的。按 body 收全部：
        # 一个刚体里的 geom 互相不产生接触，多收的只有这一件自己的。
        def _geoms_of(slot):
            bid = int(u.model.geom_bodyid[slot.geom_id])
            return {g for g in range(u.model.ngeom)
                    if int(u.model.geom_bodyid[g]) == bid}

        self.cube_geom = [_geoms_of(s) for s in u._slots]

        self.fixed_jaw = self._jaw_geoms("fixed_jaw")
        self.moving_jaw = self._jaw_geoms("moving_jaw")
        if not self.fixed_jaw or not self.moving_jaw:
            raise RuntimeError(f"模型里没找到固定爪/活动爪的 geom："
                               f"{self.fixed_jaw} {self.moving_jaw}")
        self.fix_tip = self._tip_geoms("fixed_jaw")
        self.mov_tip = self._tip_geoms("moving_jaw")

        grip_j = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        if grip_j < 0:
            raise RuntimeError("模型里没找到名为 gripper 的关节")
        self.grip_qadr = int(u.model.jnt_qposadr[grip_j])

        # 方块 geom → 方块序号；地面那些 worldbody geom 单独收一份，躺在地上不算撞
        # （跟 `cube_geom` 同一个口径：一件东西的全部 geom，不是 nexus 报的那一个）
        self.geom2cube = {}
        for i, gs in enumerate(self.cube_geom):
            for g in gs:
                self.geom2cube[int(g)] = i
        self.ground_geoms = {g for g in range(u.model.ngeom)
                             if int(u.model.geom_bodyid[g]) == 0}

        # 离屏 renderer 必须同一线程创建 + 使用（Windows WGL）。nexus 的 renderer 是
        # 懒建的，第一次 reset 时建 —— 在这里先 reset 一次，把它们建在当前线程。
        if render:
            self.env.reset(seed=0)

    # ---- 造环境时要用的小工具 ----

    def _jaw_geoms(self, prefix):
        mujoco = self._mujoco
        m = self.u.model
        return {g for g in range(m.ngeom)
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                .startswith(prefix)}

    def _tip_geoms(self, prefix):
        mujoco = self._mujoco
        m = self.u.model
        return sorted(g for g in self._jaw_geoms(prefix)
                      if "sph_tip" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))

    def _geom_name(self, gid):
        mujoco = self._mujoco
        m = self.u.model
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name:
            return name
        # 臂身那些碰撞盒在 so101.xml 里不写 name，退到所属 body 的名字
        return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gid]))
                or f"geom{gid}")

    # ---- Arm 接口 ----

    def reset(self, seed: int) -> dict:
        """开一局。返回这一局的真值：要抓哪个方块、任务描述是什么。"""
        self.obs, self.info = self.env.reset(seed=seed)
        if self.randomize:
            self._randomize_look(seed)
            # **必须重出一帧观测。** reset 返回的那份是在换光和换桌面之前渲染的，
            # 直接拿去喂策略，第一帧看到的还是上一局的环境。
            self.obs, self.info = self.u._observe()
        self.vel = np.zeros(3)
        self.rvel = np.zeros(3)
        return {"target_index": int(self.info["target_index"]),
                "task": str(self.u.task_description)}

    def _randomize_look(self, seed: int) -> None:
        """按这一局的 seed 抽一套视觉条件：灯光 + 桌面颜色。

        RNG 从 seed 派生，所以**同一个 seed 开出来光和桌面是同一套** —— 跟这个仓库
        里别的地方一个口径：同一个 seed 就是同一局，能复现，不会跑一次一个样。

        改的是运行时模型（跟改 `geom_rgba` 是同一个手法），不重建场景：`geom_rgba`
        和 `light_diffuse` 都只影响画面，改完不用 `mj_forward`，下一帧渲染直接生效。
        """
        from so101_nexus.constants import COLOR_MAP

        rng = np.random.default_rng(seed)
        m = self.u.model

        scale = float(rng.uniform(*LIGHT_SCALE))
        tint = np.asarray(LIGHT_TINT[int(rng.integers(len(LIGHT_TINT)))], dtype=np.float64)
        for i in range(int(m.nlight)):
            m.light_diffuse[i] = np.clip(self._base_lights[i] * scale * tint, 0.0, 1.0)

        name = FLOOR_COLORS[int(rng.integers(len(FLOOR_COLORS)))]
        m.geom_rgba[self._floor_geom] = np.asarray(COLOR_MAP[name], dtype=np.float64)

    def tip(self) -> tuple[np.ndarray, np.ndarray]:
        """现在爪子尖在哪、朝哪个方向。方向给的是四元数 [w,x,y,z]。"""
        _, pos, quat, _, _ = self.state()
        return pos, quat

    def drive(self, to_position, to_direction=None, closing=None,
              grip=None, pos_slew=0.012, rot_slew=0.03, brake=False) -> None:
        """朝这个目标走一小步，然后换一拍。

        to_position : 爪子尖要去的点
        to_direction: 工具 z 轴（夹爪前进方向）要朝哪，默认竖直向下
        closing     : 爪的开合轴要朝哪（抓之前用来对上方块的面法向）
        grip        : 夹爪那一维的指令。可以是常数，也可以是每步现算的函数
        pos_slew / rot_slew : 每步允许变多少。越小越稳、越慢，贴近方块时压低它
        brake       : 给指令再套一层「在这个距离上还刹得住」的上限（见 `hold`）。
                      **默认关。** 追一个每步都在动的点时（`follow` 就是这么用的）
                      目标离手臂永远只有一步的距离，套上限等于把指令压到接近零、
                      手臂当场冻住。只有守一个不动的点才该开。

        **指令是按误差比例给的**，不是按速度积分。所以到点之后误差归零，手臂自己
        停住 —— 不需要额外补一段「位置回正」（以前按速度积分的写法会冲过落点停偏）。

        **单位说清楚**：`to_position` 给的是「本步该走到哪」，`to_position - tcp` 就是
        本步该走的位移（米/步）。除以 POS_SCALE 就是该发的动作量。**这里不能再乘
        SPEED_POS** —— 那是巡航速度的绝对值；轨迹上前一步那个点离手臂本来就只有
        一步的距离，再乘一次等于把指令压小 12.5 倍，手臂爬不到 0.4mm/40步 就被判失速。
        """
        _, tcp, quat, _, _ = self.state()
        d = np.asarray(to_position, dtype=np.float64) - tcp
        want = np.clip(d / POS_SCALE, -1.0, 1.0)
        if brake:
            # 和 core/follow.py 里那条封顶同一个式子。守点不动时这条才成立：
            # 离点还有 2.6mm 就按 2.6mm/2cm=0.13 给指令，手臂会以 2.6mm/步 往回冲，
            # 而减速只有 pos_slew/步，冲过头再慢慢爬回来。套上「这个距离上刹得住的
            # 速度」，指令一开始就只有 0.03 上下，是走回去的不是冲回去的。
            room = float(np.linalg.norm(d)) / POS_SCALE
            cap = float(np.sqrt(2.0 * float(pos_slew) * max(room, 0.0)))
            mag = float(np.linalg.norm(want))
            if mag > cap > 0.0:
                want = want * (cap / mag)
        self.vel = self.vel + np.clip(want - self.vel, -pos_slew, pos_slew)

        z_des = np.array([0.0, 0.0, -1.0]) if to_direction is None else np.asarray(to_direction, float)
        rwant = align_action(quat, z_des, gain=ROT_SCALE, x_des=closing) * SPEED_ROT
        self.rvel = self.rvel + np.clip(rwant - self.rvel, -rot_slew, rot_slew)

        a = np.zeros(7, dtype=np.float32)
        a[:3] = self.vel
        a[3:6] = self.rvel
        a[6] = grip() if callable(grip) else (0.0 if grip is None else float(grip))
        self.step_action(a)

    def angles(self) -> np.ndarray:
        """现在六个关节各多少度。"""
        return np.degrees(self.state()[0])

    def contacts(self) -> int:
        """现在两个爪子和方块之间一共几个接触点。"""
        n = 0
        jaws = self.fixed_jaw | self.moving_jaw
        for k in range(self.u.data.ncon):
            c = self.u.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in jaws and g2 in self.geom2cube) or (
                    g2 in jaws and g1 in self.geom2cube):
                n += 1
        return n

    # ---- 底层：一步动作、和读状态 ----

    def step_action(self, action) -> None:
        self.last_action = np.asarray(action, dtype=np.float32)
        self.obs, _, term, trunc, self.info = self.env.step(action)
        self.terminated = bool(term or trunc)
        if self.on_step is not None:
            self.on_step()

    def step_idle(self) -> None:
        """什么都不动，走一步（摆件动画用）。"""
        self.step_action(np.zeros(7, dtype=np.float32))

    def state(self):
        """privileged_state 切好的五段：关节角 / TCP 位置 / 姿态四元数 / …"""
        p = np.asarray(self.info["privileged_state"], dtype=np.float64)
        return p[0:6], p[6:9], p[9:13], p[13:16], p[16:20]

    def tcp(self) -> np.ndarray:
        return self.state()[1]

    def quat(self) -> np.ndarray:
        return self.state()[2]

    def tool_x(self) -> np.ndarray:
        """工具系 x 轴在世界系的方向 —— 就是爪的开合方向。"""
        from scipy.spatial.transform import Rotation

        q = self.quat()
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()[:, 0]

    def grip_q(self) -> float:
        """夹爪关节现在的角度（rad）。合爪就是在把它压小。"""
        return float(self.u.data.qpos[self.grip_qadr])

    def grip_command(self, q: float) -> float:
        """夹爪那一维的指令：**发成「要去到这个角度」**，不是「每步张开多少」。

        **这一维的动作是增量，不是目标角。** nexus 的 `pd_ee_delta_pose` 里
        （`mujoco/base_env.py::_action_to_ctrl`）夹爪目标是
        `现在量到的角 + action[6] × ee_delta_scale[6]`。所以给一个恒定的正数等于
        每步都往开的方向推，一路积到行程上限 —— 想让它停在某个角上，就得每步现算
        「还差多少」再折回指令。实测过：给常数时起始开合角永远是满开 +1.745，
        「按外径算出来的那个角」一次都没到过。

        `open_angle` 给的是关节角，和这里配套用：`grip_command(open_angle(w))`。
        """
        scale = float(self.u._ee_delta_scale[-1])
        return float(np.clip((float(q) - self.grip_q()) / scale, -1.0, 1.0))

    def reach_residual(self, position) -> float:
        """这个点够不够得着：返回「够过去以后离目标还差多少米」，0 就是够得着。

        **不自己写逆解** —— 调的是 nexus 环境自己的 `_solve_ee_ik`（`pd_ee_delta_pose`
        那一档每步都在用它）。它够不着时不报错，只返回「最接近的那一个」，所以只能
        反着验：把返回的关节角正推回 TCP 位置，量一下和目标差多少。差得多 = 够不着。

        姿态沿用**现在**这个（`drive` 走路时工具轴本来就一直朝下、不变），所以问的是
        「保持这个朝向，手能不能挪到那个点」。这是从这个姿态出发的局部判断 —— 换言之
        它答的是「逆解能不能收敛过来」，不是几何上的全局可达集。

        全程算在 `_ik_data` 这份草稿上，真身的 `data` 一步没动。
        """
        u = self.u
        mujoco = self._mujoco
        ik = getattr(u, "_ik_data", None)
        if ik is None:
            return 0.0                      # 不是 EE 控制模式，问不了，当够得着
        target = np.asarray(position, dtype=np.float64)
        _, _, quat, _, _ = self.state()
        q = u._solve_ee_ik(target, np.asarray(quat, dtype=np.float64), self.grip_q())
        addrs = u._arm_qpos_addrs
        ik.qpos[addrs] = q[:len(addrs)]
        mujoco.mj_kinematics(u.model, ik)
        got = np.asarray(ik.site_xpos[u._tcp_site_id], dtype=np.float64)
        return float(np.linalg.norm(got - target))

    # ---- 场景查询 ----

    @property
    def n_cubes(self) -> int:
        """场上几件东西。技能里别再写死 3 —— 单件试验场上只有一个。"""
        return len(self.cube_order)

    def object_shape(self, i: int):
        """第 i 件东西是什么形状（`core.shapes.Shape`）。越界就取最后一件。

        尺寸是清单里声明好的，不从画面里量 —— 一个视角分不清「圆柱粗」和「圆柱远」。
        """
        from core import shapes

        return shapes.from_spec(self.objects[min(int(i), len(self.objects) - 1)])

    def cube_pos(self, i: int) -> np.ndarray:
        a = self.cube_addr[i]
        return self.u.data.qpos[a:a + 3].copy()

    def cube_quat(self, i: int) -> np.ndarray:
        """方块自由关节的四元数 [w,x,y,z]。爪子要对的就是它定出来的面法向。"""
        a = self.cube_addr[i]
        return self.u.data.qpos[a + 3:a + 7].copy()

    def place_cube(self, i: int, xy, z: float, yaw: float) -> None:
        """把方块直接摆到某处（「操作员」摆件用，不是机械臂干的）。"""
        a, d = self.cube_addr[i], self.cube_dof[i]
        self.u.data.qpos[a:a + 3] = [float(xy[0]), float(xy[1]), float(z)]
        self.u.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.u.data.qvel[d:d + 6] = 0.0

    def fixed_tip_pos(self) -> np.ndarray:
        """固定爪两个指垫球的中点。"""
        return self.u.data.geom_xpos[self.fix_tip].mean(axis=0)

    def fixed_face_reach(self, axis, side: float) -> float:
        """固定爪**越过指垫中点、朝物体那一侧**还伸出多少米。

        落点和 `pad_gap` 一直按「指垫球的中点」算，可真正能压到东西上的是球的
        **表面**：球半径 0.75mm，`FIX_CLEAR` 是 1mm，于是「指面离物体侧面留 1mm」
        实际只剩 0.25mm。中空件最宽的一圈在容器口，而下探又必须贴着口沿擦过去
        （锥面越往下越窄），零点几毫米的缝实测会被顶住 —— 下探当场判 `stall`、
        停在口沿高度上。

        `side` 是固定爪在开合轴的哪一侧（`+1/-1`，跟落点那个 `side` 是同一个）：
        朝物体那一侧就是 `-side`。
        """
        xh = np.asarray(axis, dtype=np.float64)
        xh = xh / max(float(np.linalg.norm(xh)), 1e-12)
        m, d = self.u.model, self.u.data
        mid = float(np.dot(self.fixed_tip_pos(), xh))
        out = 0.0
        for g in self.fix_tip:
            c = float(np.dot(d.geom_xpos[g], xh))
            out = max(out, float(side) * (mid - c) + float(m.geom_size[g][0]))
        return float(out)

    def pad_gap(self, q: float | None = None) -> float:
        """两个指垫中点之间多宽（米）。

        给了 `q` 就按**那个夹爪关节角**算，算在 `_ik_data` 那份草稿上，真身一步没动。
        指垫间距只跟夹爪这一个关节有关（两个指垫挂在同一条腕链上，手臂别的关节怎么
        动，它俩一起动，间距不变），所以草稿里别的关节是多少不影响这个数。
        """
        data = self.u.data
        if q is not None:
            data = getattr(self.u, "_ik_data", None)
            if data is None:
                return float("nan")
            data.qpos[:] = self.u.data.qpos
            data.qpos[self.grip_qadr] = float(q)
            self._mujoco.mj_kinematics(self.u.model, data)
        f = data.geom_xpos[self.fix_tip].mean(axis=0)
        m = data.geom_xpos[self.mov_tip].mean(axis=0)
        return float(np.linalg.norm(f - m))

    def open_angle(self, width: float) -> float | None:
        """张到「两个指垫之间宽 `width` 米」需要给夹爪多大的角度指令。

        **不自己推几何**：在夹爪关节的可指令行程上采一串点，每个点用 MuJoCo 自己的
        正向运动学量一次指垫间距（`pad_gap`），再在相邻两点之间线性插值求反。
        指垫是绕铰链转的，间距对关节角不是线性的，所以不拿两端硬拉直线。

        张到最满还不到 `width` 就返回 None —— 这一件夹爪根本圈不住，调用方自己决定
        怎么办（现在是退回张满）。
        """
        lo = float(self.u._target_low[-1])      # 夹爪这一维的可指令行程
        hi = float(self.u._target_high[-1])
        n = 12
        qs = np.linspace(lo, hi, n)
        gaps = np.array([self.pad_gap(float(q)) for q in qs], dtype=np.float64)
        if not np.all(np.isfinite(gaps)):
            return None
        if width >= gaps[-1]:
            return None
        if width <= gaps[0]:
            return float(qs[0])
        k = int(np.searchsorted(gaps, width))
        g0, g1 = float(gaps[k - 1]), float(gaps[k])
        q0, q1 = float(qs[k - 1]), float(qs[k])
        t = 0.0 if g1 <= g0 else (width - g0) / (g1 - g0)
        return float(q0 + (q1 - q0) * t)

    def finger_cube_contacts(self, i: int):
        """(固定爪触点数, 活动爪触点数, 接触点平均高度)。

        从接触表里数爪子和这个方块之间的接触。比按「TCP 到多少高度算夹住」可靠：
        高度是算出来的，接触是仿真给的。
        """
        cg = self.cube_geom[i]
        nf = nm = 0
        zs = []
        for k in range(self.u.data.ncon):
            c = self.u.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in cg:
                jaw = g2
            elif g2 in cg:
                jaw = g1
            else:
                continue
            if jaw in self.fixed_jaw:
                nf += 1
            elif jaw in self.moving_jaw:
                nm += 1
            else:
                continue
            zs.append(float(c.pos[2]))
        return nf, nm, (float(np.mean(zs)) if zs else float("nan"))

    def bump_scan(self, ignore=()):
        """扫接触表，返回 [(方块序号, 撞它的 geom 名字, 接触点)]。

        只看「一边是方块、另一边既不是方块也不是地面」的组合 —— 那就是手臂压上去了。
        方块压方块、方块压地面都跳过。
        """
        out = []
        for k in range(self.u.data.ncon):
            c = self.u.data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            i1, i2 = self.geom2cube.get(g1), self.geom2cube.get(g2)
            if (i1 is None) == (i2 is None):
                continue
            ci = i1 if i1 is not None else i2
            og = g2 if i1 is not None else g1
            if ci in ignore or og in self.ground_geoms:
                continue
            out.append((int(ci), self._geom_name(og),
                        np.array(c.pos, dtype=float).copy()))
        return out

    def dry_run(self, plan, ignore=()):
        """把一段动作在一份**副本**上先走一遍，只回报路上碰到了什么；真身一步不动。

        这就是仿真的「出发前验算」。碰撞不用自己写几何 —— MuJoCo 自己就有：副本
        和真身是同一个模型、同一个控制器、同一个起点，所以**它走出来的轨迹和真身
        一模一样**，它碰到的就是真身会碰到的。区别只在场上的方块：真身走一遍会把
        方块推走，副本走一遍推的是副本里的方块，摆完就扔。

        plan   : 收一个「接口和真身一样」的机械臂，想怎么走就怎么走。
        ignore : 这些方块不算（比如手里正夹着的那个）。
        返回   : [(第几步, 方块序号, 撞它的 geom 名, 接触点)]，一路干净就是空表。
        """
        scratch = self.sampler
        # 副本要先复位过才准 step（gym 的规矩）；另外它自己也计步，TimeLimit 到点
        # 会 trunc，那这条路就走一半停了，攒够也复位一次。复位只清计数器和方块
        # 摆位，物理状态紧跟着被真身整份覆盖，验算结果不受影响。
        # 没 reset 过时 `_elapsed_steps` 是 None，当 0 看。
        done = getattr(scratch, "_elapsed_steps", 0) or 0
        if done == 0 or done > 0.8 * MAX_STEPS:
            scratch.reset(seed=0)

        mu, su = self.env.unwrapped, scratch.unwrapped
        su.data.qpos[:] = mu.data.qpos
        su.data.qvel[:] = mu.data.qvel
        su.data.time = mu.data.time
        self._mujoco.mj_forward(su.model, su.data)

        real = (self.env, self.u, self.obs, self.info, self.terminated,
                self.vel, self.rvel, self.step_action, self.on_step)
        step0 = self.step_action                 # 绑在真身上的那个（下面要换掉）
        hits: list = []
        n = [0]

        def spy(action):
            step0(action)                        # 这一步已经落到副本上了
            n[0] += 1
            for ci, name, p in self.bump_scan(ignore):
                hits.append((n[0], int(ci), name, np.array(p, dtype=float)))

        try:
            self.env, self.u = scratch, su
            self.vel = np.zeros(3)
            self.rvel = np.zeros(3)
            self.terminated = False
            self.on_step = None                  # 预演不录帧、不进视频
            # 借 `_observe`（step 内部用的也是它）把副本这一刻的 info 取出来，TCP
            # 从这里读。**不能改成 step**：step 会把副本往前推一格，起点就跟真身
            # 差开了，那样验算出来的碰撞就不是真身会碰的那个。
            self.obs, self.info = su._observe()
            self.step_action = spy
            plan(self)
        finally:
            (self.env, self.u, self.obs, self.info, self.terminated,
             self.vel, self.rvel, self.step_action, self.on_step) = real
        return hits

    def layout_of(self, seed: int):
        """某个 seed 的开局：三个方块各自的坐标 + 它指定的目标方块。

        用另一个环境采样 —— 主环境正在跑，reset 会把机械臂一起复位。
        """
        _, info = self.sampler.reset(seed=seed)
        su = self.sampler.unwrapped
        pos = np.array([su.data.qpos[s.qpos_addr:s.qpos_addr + 3].copy()
                        for s in su._slots])
        return pos, int(info["target_index"]), str(su.task_description)

    def ref_frame(self) -> np.ndarray:
        """空工位的参考图（俯视那一张）：把方块临时挪出画面渲一张，再原样放回来。

        `core.vision.foreground_mask` 是靠「跟一张什么都没有的工位比」把方块挑出来的
        —— 真机上就是开机拍的那张空桌面，仿真里得现造一张。**只有固定的相机才有这
        张图可言**：腕部相机跟着手走，每拍一张背景都不一样，参考图对它没有意义。

        **只在开完一局、机械臂还没动的时候叫。** 它会把方块的 qpos 挪走再放回，
        放回的是原值、但中途 `mj_forward` 过；手臂正在路上时叫等于往 physics 里插一帧。
        """
        u = self.u
        keep = {}
        try:
            for i in range(self.n_cubes):
                a, d = self.cube_addr[i], self.cube_dof[i]
                keep[i] = (u.data.qpos[a:a + 3].copy(), u.data.qpos[a + 3:a + 7].copy())
                # 挪到画面外时按**这一件自己**躺平的高度放 —— 拿 30mm 方块那 15mm 去放
                # 一根 60mm 的圆柱，它会一半埋进地面里。
                u.data.qpos[a:a + 3] = [REF_PARK_XY[0], REF_PARK_XY[1],
                                        self.object_shape(i).rest_z]
                u.data.qvel[d:d + 6] = 0.0
            self._mujoco.mj_forward(u.model, u.data)
            rend = u._overhead_obs_renderer
            rend.update_scene(u.data, camera=u._overhead_obs_cam)
            return np.asarray(rend.render()).copy()
        finally:
            for i, (p, q) in keep.items():
                a, d = self.cube_addr[i], self.cube_dof[i]
                u.data.qpos[a:a + 3] = p
                u.data.qpos[a + 3:a + 7] = q
                u.data.qvel[d:d + 6] = 0.0
            self._mujoco.mj_forward(u.model, u.data)

    def close(self) -> None:
        self.env.close()
        self.sampler.close()
