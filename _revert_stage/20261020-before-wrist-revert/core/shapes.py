"""形状表：桌上那个东西是什么形状、多大、该从哪儿夹。

**为什么单立这张表。** 「它是个 30mm 立方体」这件事以前写死在链路的三处：拟合位置
时按一个半边长投八个角（`core/vision.py::estimate_pose`）；算爪子贴哪面时从四条侧面
法向里挑（`core/geom.py::face_normal`）；判落地时按 15mm 这个高度
（`impls/judge_truth.py`）。抽成一张表以后，这三处都改成问表，加一种形状只加一段。

**每个形状回答三件事：**
    剪影  `silhouette_points(center, yaw)` —— 从随便哪个角度看，你在画面里的轮廓是
          哪几个点的凸包。拟合位置就是拿它投回画面、跟画面里那个色块比重合度。
    对称  `yaw_period_deg` —— 绕竖轴转多少度跟自己重合。90 = 正方形截面，180 =
          长方形截面，None = 圆柱/球（怎么转都一样，角度干脆不拟合，省一整维搜索）。
    夹法  `normal()` 往哪个水平方向合、`support()` 沿那个方向表面离中心多远、
          `grasp_candidates()` 一串候选高度（和绕开把手的方向），顺序就是先试哪个。

**尺寸是清单里给的，不是从画面里量出来的。** 一个视角分不清「圆柱粗」和「圆柱远」
—— 尺寸和距离在这张图里本来就是同一件事；而且真机那两路相机没有深度，「已知尺寸 +
剪影」是唯一能原样搬过去的做法。尺寸的地位跟工位坐标表一样：摆件的人知道桌上放的
是什么、多大，把数填进来（真杯子拿卡尺量一次）。

**都躺着放。** 中心高度 = `rest_z`（长方体/圆柱半高、球半径）。这一版不支持侧躺、
倒扣、叠放。

**抓法为什么给一串而不是一个。** 夹不住的原因经常不是「算错了」，是「夹的那一处不
合适」—— 夹到杯口那圈、夹到把手、方块本来歪着。与其退回策略层从头走一遍（重新进场、
重新看一眼），不如在原地换一处再夹。这些都会进日志和视频，回头做数据时把试的那一段
剪掉就是了。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

# 夹爪两个指垫最多能张开多宽（米）。实测行程 0°（闭，30.9mm）到 180°（开，55.9mm），
# 取整到 56mm。**只有长方体用得上它**：挑夹哪一对边的时候要把张不开的那一对排掉。
MAX_JAW_SPAN = 0.056

# 一个形状最多给几种候选**高度**。再多不如退回策略层重来（那边会重新看一眼）。
# 中空件另有一个「内外壁对夹」的候选排在最前面 —— 那不是同一个高度的另一种夹法，
# 是另一种夹法本身，所以不占这个额度，`grasp_candidates` 会多给一个。
GRASP_TRIES = 3

# 圆柱的剪影用上下两个圆各采这么多点。圆的凸包是内接多边形，48 个点的面积误差
# 约 0.1%（1 - sin(π/n)·n/π），比重合度那道 0.5 的线小两个数量级。
RIM_POINTS = 48
# 球的剪影同理，按黄金螺旋在球面上撒点。
SPHERE_POINTS = 96

# 杯子：顶上这一截是口沿，不许夹（夹上去两个指垫容易卡在沿上，一抬就翻）。
CUP_RIM_MARGIN = 0.25
# 有把手时，把手周围多大一片不许夹。
HANDLE_SPAN_DEG = 70.0


def _h_axis(v) -> np.ndarray:
    """把任意向量压到水平面上再归一化。压不出方向就退回 +x。"""
    a = np.asarray(v, dtype=np.float64)
    h = np.array([float(a[0]), float(a[1]), 0.0]) if a.size >= 2 else np.array([1.0, 0.0, 0.0])
    n = float(np.linalg.norm(h))
    return h / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _yaw_matrix(yaw: float) -> np.ndarray:
    """绕竖直轴转 yaw 弧度的 3×3 旋转矩阵。"""
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Grasp:
    """一种夹法。

    `z`       夹爪该停在的高度（世界系，米）。不是物体中心高度，是这一把要夹的那
              一处的高度 —— 杯子不许夹口沿，靠它往下挪。
    `normal`  水平单位向量，两个爪子合拢的方向。None = 就用「离当前开合轴最近」的
              那个方向（对称物体都是这样）。
    `label`   日志里好认的名字。
    `wall`    True = 这一把是**内外壁对夹**：一个指垫在壁外面、另一个钻到壁里面，
              把那一层薄壁夹在两个指垫中间。只有中空件（`Shape.hollow`）才有这一种。

              为什么值得单立一种：只夹外壁的话，两个指垫都压在同一侧，锥面（碗壁、
              盅壁都是下窄上宽）上那股往里挤的力会分出一个往下/往外的分量，东西是
              被「挤」出去的 —— 实测抬到一半就滑掉。对夹把压力分到内外两侧，两个
              指垫把壁夹住，滑不出去。

              它跟外壁那几种的差别只在**张多开**（落点规则一样：固定爪照样贴外表面）：
              活动爪要留出「从口子伸进容器里」的宽度，所以张到活动爪正好落在物体
              中轴上为止。伸不进去（口径比爪指还窄、或者够不着）下探就会卡住，那时
              按候选顺序退回外壁那一套。
    """

    z: float
    normal: np.ndarray | None
    label: str
    wall: bool = False


class Shape(ABC):
    """桌上一个东西的形状。尺寸是构造时给死的，不从画面里量。"""

    word: ClassVar[str] = "?"

    @property
    @abstractmethod
    def rest_z(self) -> float:
        """躺在地面上时，中心离地多高（米）。"""

    @property
    @abstractmethod
    def yaw_period_deg(self) -> float | None:
        """绕竖轴转多少度和自己重合。None = 怎么转都一样，角度不用拟合。"""

    @abstractmethod
    def silhouette_points(self, center, yaw: float) -> np.ndarray:
        """(N,3) 世界点。它们的凸包就是这东西在画面里的轮廓。"""

    @abstractmethod
    def normal(self, yaw: float, tool_xy) -> np.ndarray:
        """两个爪子该往哪个水平方向合。"""

    @abstractmethod
    def support(self, yaw: float, axis_xy) -> float:
        """沿某个水平方向，物体表面离中心最远多少（米）。

        **不是「法向那一条」，是「沿这个方向鼓出来多少」。** 腕子转正之后开合轴跟
        理想法向差着几度是常有的事，用被转过的那个轴去算落点，才不会被那几度吃掉
        缝（以前立方体写死成 `half·(|cos|+|sin|)` 就是这个意思）。
        """

    def grasp_candidates(self, yaw: float, tool_xy, *, bottom_z: float = 0.0) -> list[Grasp]:
        """一串候选夹法，先试哪个排在前面。默认只有「正中间那一处」。"""
        return [Grasp(z=float(self.rest_z), normal=None, label=f"{self.word} 中心")]

    @property
    def hollow(self) -> bool:
        """是不是**中空的**（杯、碗、盅）。

        只影响一件事：张爪张到多大。中空件两个指垫只要夹住那圈**壁**，张到「外径
        + 一点余量」就够；张满再合，活动爪要多扫一大段空行程，真机上那一下很晃。
        实心件照旧张满 —— 张得开才容得下，夹哪儿都无所谓。
        """
        return False


# ---------------------------------------------------------------- 长方体

class Box(Shape):
    """长方体。三个半长各给一个就是长方体，给一个就是立方体。"""

    word = "cube"

    def __init__(self, half_x: float, half_y: float | None = None,
                 half_z: float | None = None):
        self.half_x = float(half_x)
        self.half_y = float(half_x if half_y is None else half_y)
        self.half_z = float(half_x if half_z is None else half_z)
        self.half = np.array([self.half_x, self.half_y, self.half_z])

    @property
    def rest_z(self) -> float:
        return self.half_z

    @property
    def yaw_period_deg(self) -> float | None:
        # 正方形截面转 90° 就重合了；长方形要转 180°。
        return 90.0 if abs(self.half_x - self.half_y) < 1e-9 else 180.0

    def silhouette_points(self, center, yaw: float) -> np.ndarray:
        """八个角点。跟以前 `vision.cube_corners` 逐字节一样，只是三个轴各用各的半长。"""
        c, s = math.cos(float(yaw)), math.sin(float(yaw))
        cx, cy, cz = (float(v) for v in center)
        px, py, pz = np.meshgrid([-self.half_x, self.half_x],
                                 [-self.half_y, self.half_y],
                                 [-self.half_z, self.half_z], indexing="ij")
        return np.column_stack([(cx + px * c - py * s).ravel(),
                                (cy + px * s + py * c).ravel(),
                                (cz + pz).ravel()])

    def _side_normals(self, yaw: float) -> list[np.ndarray]:
        """四条侧面法向（±局部 x、±局部 y）。四棱等价，取哪条都行。"""
        R = _yaw_matrix(yaw)
        return [R[:, 0], -R[:, 0], R[:, 1], -R[:, 1]]

    def normal(self, yaw: float, tool_xy) -> np.ndarray:
        """两个爪子往哪个方向合。

        **先排掉夹爪张不开的那一对。** 四个方向的法向里，±x 那一对隔 `2·half_x`、
        ±y 那一对隔 `2·half_y`；隔得超过 `MAX_JAW_SPAN` 的那一对，两个指垫本来就
        合不到一起（长方体的长边能到 100mm，夹爪最大开口才 56mm），挑上去合爪只会
        把东西推着走。

        **剩下两对都塞得进时，取跨度最小的那一对。** 用户 2026-10-17 定的口径：
        「细长的棍子总不能夹两个圆面吧，得抓棍子长边」—— 棍子（200×10）那一对跨度
        200mm 本来就超 56mm、被上面滤掉了，所以这条只在**矮胖长方体**上起作用
        （40×30、50×20 那类）。跨小的那一维 = 两个指垫贴在**面积最大**的那一对面上。
        原来是「挑最贴近当前开合轴的那条」（腕子转得最少），`_probe_box_normal.py`
        在 40×30 上量过 30 组姿态，只有 14 组落在小跨度那一对。

        跨度相同时（正方体、以及两对都超限的退回支）再按「离当前开合轴最近」定 ——
        这一条必须用**绝对值**点积（`|cos|` 而不是 `cos`）：`±` 那一半是不定的，
        拿带符号的点积去比会在这两对之间挑反，正方体在 yaw ≥ 45° 时就会从
        「转 ±y」变成「转 ±x」—— 对正方体无害（四面等价），但**腕子转的方向跟原版
        就不是一回事了**，那是这版已经跑通的东西，不许动。取 `|cos|` 之后正方体
        逐组跟老口径一致（`_probe_box_normal.py` 那 30 组）。
        """
        want = _h_axis(tool_xy)
        R = _yaw_matrix(yaw)
        pairs = ((2.0 * self.half_y, R[:, 1]), (2.0 * self.half_x, R[:, 0]))
        fits = [p for p in pairs if p[0] <= MAX_JAW_SPAN]
        if fits:
            _span, n = min(fits, key=lambda p: (p[0], -abs(float(
                np.dot(p[1][:2], want[:2])))))
            return n if float(np.dot(n[:2], want[:2])) >= 0.0 else -n
        # 两对都张不开：老口径照旧（在四个 ± 法向里挑离当前开合轴最近的那个），
        # 在这儿的选法一个字没动 —— 拒绝「张不开的物体」是另一件事，这一版没做。
        out = []
        for _span, v in pairs:
            out += [v, -v]
        return max(out, key=lambda v: float(np.dot(v[:2], want[:2])))

    def support(self, yaw: float, axis_xy) -> float:
        R = _yaw_matrix(yaw)
        axis = np.array([float(axis_xy[0]), float(axis_xy[1]), 0.0])
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
            return float(max(self.half))
        return float(np.dot(np.abs(R.T @ (axis / n)), self.half))


# ---------------------------------------------------------------- 圆柱

class Cylinder(Shape):
    """圆柱。`cup=True` 表示这是个杯子：顶上那截口沿不许夹。

    实心圆柱和杯子的**剪影一模一样** —— 一个视角看不出里面是空的。差别只在「夹哪儿
    合法」：杯子夹到口沿那一段容易让指垫卡在沿上，一抬就翻。`handle_deg` 再排掉
    把手那一片方位。
    """

    word = "cylinder"

    def __init__(self, radius: float, half_z: float, *, cup: bool = False,
                 handle_deg: float | None = None, handle_span_deg: float = HANDLE_SPAN_DEG):
        self.radius = float(radius)
        self.half_z = float(half_z)
        self.cup = bool(cup)
        self.handle_deg = None if handle_deg is None else float(handle_deg)
        self.handle_span_deg = float(handle_span_deg)

    @property
    def height(self) -> float:
        return 2.0 * self.half_z

    @property
    def hollow(self) -> bool:
        return bool(self.cup)

    @property
    def rest_z(self) -> float:
        return self.half_z

    @property
    def yaw_period_deg(self) -> float | None:
        # 绕竖轴转多少度都一样（把手除外，但把手只影响夹哪儿、不影响剪影）。
        return None

    def silhouette_points(self, center, yaw: float) -> np.ndarray:
        """上下两个圆各采一圈。圆柱侧面是直纹面，轮廓上的极值点全在这两个圆上。"""
        cx, cy, cz = (float(v) for v in center)
        ang = np.linspace(0.0, 2.0 * math.pi, RIM_POINTS, endpoint=False)
        ring = np.column_stack([self.radius * np.cos(ang), self.radius * np.sin(ang)])
        out = []
        for dz in (-self.half_z, self.half_z):
            out.append(np.column_stack([cx + ring[:, 0], cy + ring[:, 1],
                                        np.full(RIM_POINTS, cz + dz)]))
        return np.vstack(out)

    def normal(self, yaw: float, tool_xy) -> np.ndarray:
        """旋转对称，哪个方向都能夹 —— 用离当前开合轴最近的那个，腕子就不用多转。"""
        return _h_axis(tool_xy)

    def support(self, yaw: float, axis_xy) -> float:
        return self.radius

    def _allowed_z(self) -> tuple[float, float]:
        """允许夹的高度区间（世界系）。"""
        bottom, top = 0.0, self.height
        if self.cup:
            # 口沿那一段排掉，往下留出握手的那一带。
            top = bottom + (1.0 - CUP_RIM_MARGIN) * self.height
            bottom = bottom + 0.25 * self.height
        return bottom, top

    def _away_from_handle(self, base: np.ndarray) -> np.ndarray:
        """把手那一片方位排掉：真转到了，就往边上让开半个跨度。"""
        if self.handle_deg is None:
            return base
        have = math.degrees(math.atan2(float(base[1]), float(base[0]))) % 360.0
        gap = (have - self.handle_deg) % 360.0
        if min(gap, 360.0 - gap) >= self.handle_span_deg / 2.0:
            return base
        want = math.radians(self.handle_deg + self.handle_span_deg / 2.0)
        return np.array([math.cos(want), math.sin(want), 0.0])

    def grasp_candidates(self, yaw: float, tool_xy, *, bottom_z: float = 0.0) -> list[Grasp]:
        """夹哪儿：**杯子先试内外壁对夹，再按中部/偏低/偏高的外壁**；方位一律先躲开把手。

        高度按**世界系**给（`bottom_z + 相对高度`）—— 物体躺在地上，底就是地面。
        """
        lo, hi = self._allowed_z()
        mid = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo)
        heights = [mid, mid - 0.6 * span, mid + 0.6 * span]
        base = self._away_from_handle(_h_axis(tool_xy))
        names = ("中部", "偏低", "偏高")
        out = []
        for h, name in zip(heights[:GRASP_TRIES], names):
            if h <= 0.0:
                continue
            out.append(Grasp(z=float(bottom_z + h), normal=base,
                             label=f"{'杯壁' if self.cup else '圆柱'} {name}"))
        if self.cup:
            # 杯壁那圈薄壁夹在内外两侧。**排在外壁那一套前面**：只夹外壁时两个指垫
            # 都压在同一侧，圆柱面还好，杯壁稍微有点锥度就会被挤出去。口径太窄、
            # 活动爪伸不进去的话，下探会卡住，再退回外壁那一套。
            out.insert(0, Grasp(z=float(bottom_z + mid), normal=base,
                                 label="杯壁 内外对夹", wall=True))
        return out


# ---------------------------------------------------------------- 碗 / 盅

class Bowl(Shape):
    """碗、盅、带喇叭口的杯子：下窄上宽，**原点在底面**。

    **为什么单立一种，不套圆柱。** 扫描网格（GSO/YCB 那种）自己的坐标就是把底放在
    z=0 —— 物体原点在底面上，不在重心上。`Cylinder` 那套「原点在中心、`rest_z`
    是半高」套上去，判落地（`judge_truth`）和夹取高度（`pick`）全会差半个高。
    喇叭口也一样要单说：剪影上下两个圆**半径不同**，拿同一个半径去投，重合度会被
    这 4mm 的差吃掉（实测这只盅：底半径 18.3mm、顶半径 22.1mm）。

    尺寸还是清单里声明的，不从画面里量。`cup=True` 时顶上那一段口沿不许夹 ——
    拿碗口那圈一夹，两个指垫骑在沿上，一抬就翻。
    """

    word = "ramekin"

    def __init__(self, r_bottom: float, r_top: float, height: float, *,
                 cup: bool = True):
        self.r_bottom = float(r_bottom)
        self.r_top = float(r_top)
        self.height = float(height)
        self.cup = bool(cup)

    @property
    def rest_z(self) -> float:
        """躺在地上时**原点**离地多高 —— 底在原点，所以是 0。"""
        return 0.0

    @property
    def hollow(self) -> bool:
        return True            # 碗就是中空的，没有实心的碗

    @property
    def yaw_period_deg(self) -> float | None:
        return None            # 旋转对称，角度不拟合

    def silhouette_points(self, center, yaw: float) -> np.ndarray:
        """上下两个圆，半径各用各的。`center[2]` 是底面高度。"""
        cx, cy, cz = (float(v) for v in center)
        ang = np.linspace(0.0, 2.0 * math.pi, RIM_POINTS, endpoint=False)
        out = []
        for r, dz in ((self.r_bottom, 0.0), (self.r_top, self.height)):
            out.append(np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang),
                                        np.full(RIM_POINTS, cz + dz)]))
        return np.vstack(out)

    def normal(self, yaw: float, tool_xy) -> np.ndarray:
        return _h_axis(tool_xy)

    def support(self, yaw: float, axis_xy) -> float:
        """沿开合轴鼓出来多少 —— 取顶上那一圈，比按当地半径算略保守（留一点缝）。"""
        return self.r_top

    def grasp_candidates(self, yaw: float, tool_xy, *, bottom_z: float = 0.0) -> list[Grasp]:
        """夹哪儿：**先试内外壁对夹**，再按碗壁中部/偏低/偏高夹外面那圈。高度从**底面**算起。

        对夹排在最前面，理由见 `Grasp.wall`：碗壁是下窄上宽的锥面，只夹外壁时那股
        往里挤的力会把碗往下赶（实测抬到一半就滑掉），对夹把力分到内外两侧。
        活动爪伸不进口子的话，下探会卡住，再退回外壁那一套。
        """
        lo = 0.25 * self.height
        hi = (1.0 - CUP_RIM_MARGIN) * self.height if self.cup else 0.85 * self.height
        mid = 0.5 * (lo + hi)
        span = 0.5 * (hi - lo)
        heights = [mid, mid - 0.6 * span, mid + 0.6 * span]
        base = _h_axis(tool_xy)
        names = ("碗壁 中部", "碗壁 偏低", "碗壁 偏高")
        out = [Grasp(z=float(bottom_z + h), normal=base, label=name)
               for h, name in zip(heights[:GRASP_TRIES], names)]
        if self.cup:
            out.insert(0, Grasp(z=float(bottom_z + mid), normal=base,
                                 label="碗壁 内外对夹", wall=True))
        return out


# ---------------------------------------------------------------- 球

class Sphere(Shape):
    """球。剪影就是个圆，抓哪儿都一样，用半径当开口。"""

    word = "sphere"

    def __init__(self, radius: float):
        self.radius = float(radius)

    @property
    def rest_z(self) -> float:
        return self.radius

    @property
    def yaw_period_deg(self) -> float | None:
        return None

    def silhouette_points(self, center, yaw: float) -> np.ndarray:
        """球面上撒点（黄金螺旋）。投影后的凸包是内接多边形，密度够高误差就够小。"""
        cx, cy, cz = (float(v) for v in center)
        i = np.arange(SPHERE_POINTS, dtype=np.float64)
        z = 1.0 - 2.0 * (i + 0.5) / SPHERE_POINTS
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        phi = math.pi * (1.0 + 5.0 ** 0.5) * i
        return np.column_stack([cx + self.radius * r * np.cos(phi),
                                cy + self.radius * r * np.sin(phi),
                                cz + self.radius * z])

    def normal(self, yaw: float, tool_xy) -> np.ndarray:
        return _h_axis(tool_xy)

    def support(self, yaw: float, axis_xy) -> float:
        return self.radius


# ---------------------------------------------------------------- 从清单造

def from_spec(spec) -> Shape:
    """按清单里一行造一个形状。已经是个 Shape 就原样返回。

    清单里的尺寸一律**毫米**（跟 nexus 的 `side_length_mm` / `diameter_mm` 对齐），
    进来就换成米。种类名跟 nexus 的 `shape_name` 对齐，因为任务指令就是它生成的。
    """
    if isinstance(spec, Shape):
        return spec
    kind = str(spec.get("shape", "cube")).lower()
    mm = 0.001
    if kind == "cube":
        return Box(float(spec["side_mm"]) / 2.0 * mm)
    if kind in ("box", "cuboid"):
        hx, hy, hz = (float(v) for v in spec["size_mm"])
        return Box(hx / 2.0 * mm, hy / 2.0 * mm, hz / 2.0 * mm)
    if kind == "cylinder":
        return Cylinder(float(spec["diameter_mm"]) / 2.0 * mm,
                        float(spec.get("height_mm", spec["diameter_mm"])) / 2.0 * mm,
                        cup=bool(spec.get("cup", False)),
                        handle_deg=spec.get("handle_deg"))
    if kind == "sphere":
        return Sphere(float(spec["diameter_mm"]) / 2.0 * mm)
    if kind in ("bowl", "ramekin"):
        return Bowl(float(spec["bottom_diameter_mm"]) / 2.0 * mm,
                    float(spec["top_diameter_mm"]) / 2.0 * mm,
                    float(spec["height_mm"]) * mm,
                    cup=bool(spec.get("cup", True)))
    raise ValueError(f"形状表里没有这种形状：{kind!r}"
                     f"（有 cube / box / cylinder / sphere / bowl）")


def word_of(spec) -> str:
    """这东西在指令里叫什么词（跟 nexus 生成的句子对得上）。"""
    if isinstance(spec, Shape):
        return spec.word
    return str(spec.get("shape", "cube")).lower()
