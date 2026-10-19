"""轨迹：给一串路点，交给 toppra 算出「什么时候在哪、走多快」。

**这一层是为了不手搓。** 以前是自己干三件事：把折线拐角抹圆、按弧长重排点、
再手写梯形速度曲线并提前算刹车距离。这三件事合起来就是「时间最优参数化」，
有人做好了 —— RoboTwin 用 mplib 规划出路径后，就是交给 toppra 做这一步的。

toppra 的分工：你给它一条路径（几何形状）和速度/加速度上限，它算出沿这条路径
每个时刻的位置和速度。它不管形状好不好看，只管「在限制内最快走完」。

所以本模块只做两件事：
    1. 把路点变成一条光滑路径（`SplineInterpolator`，三次样条过路点）
    2. 按速度/加速度上限算时间最优轨迹（`TOPPRA` + `seidel` 求解器）

**一个已知的坑**：三次样条是「过路点」的，最后一段（垂直下探到方块上方）会被
前一个路点带偏，先鼓出去再荡回来。以前只抹拐角不整条过样条就是为了躲这个。
现在的对策是在下探段的起点和终点之间多插几个共线路点，把样条钉直。
这件事由调用方负责 —— 见 `pin_straight()`。

单位：坐标是米，速度米/秒，加速度米/秒²。toppra 不关心坐标是什么意思，
所以同一条路走笛卡尔空间还是关节空间由调用方决定。
"""

from __future__ import annotations

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint


class Trajectory:
    """一条算好的轨迹。问它「第 t 秒在哪」就行。"""

    def __init__(self, traj, dim: int):
        self._traj = traj
        self.dim = dim

    @property
    def duration(self) -> float:
        """走完整条要多久（秒）。"""
        return float(self._traj.duration)

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """第 t 秒的位置和速度。超过末尾就钉在终点（速度归零）。"""
        t = min(max(float(t), 0.0), self.duration)
        pos = np.asarray(self._traj(t), dtype=np.float64).reshape(self.dim)
        vel = np.asarray(self._traj(t, 1), dtype=np.float64).reshape(self.dim)
        return pos, vel

    def sample(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按固定时间间隔铺一遍，返回 (时刻, 位置, 速度)。

        控制循环常要固定节拍发指令，这个函数一次给齐。
        """
        n = max(int(np.ceil(self.duration / dt)) + 1, 2)
        ts = np.linspace(0.0, self.duration, n)
        pos = np.asarray(self._traj(ts), dtype=np.float64)
        vel = np.asarray(self._traj(ts, 1), dtype=np.float64)
        return ts, pos, vel


def pin_straight(waypoints, index: int, extra: int = 2) -> np.ndarray:
    """在某个路点前后补共线点，把样条钉直，不让它鼓包。

    用法：整条路里最后那段「垂直下探」不想被拐弯带偏，就把下探段的起点下标传进来，
    它会在起点和下一个点之间插几个点。插的点在同一条线上，样条走过它们时就只能走直线。

    `extra` 是插几个点，2 一般够；不够就往上加。
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if index < 0 or index + 1 >= len(wp):
        raise ValueError(f"pin_straight: 下标 {index} 不在 0..{len(wp) - 2} 之间")
    a, b = wp[index], wp[index + 1]
    ts = np.linspace(0.0, 1.0, extra + 2)[1:-1]
    fill = a[None, :] + ts[:, None] * (b - a)[None, :]
    return np.vstack([wp[: index + 1], fill, wp[index + 1:]])


def build(waypoints, vmax: float, amax: float, *, density: float = 0.002) -> Trajectory:
    """把路点变成一条时间最优轨迹。

    waypoints : (N, D) 数组，N ≥ 2。D 是几维就几维（我们一般是 3 维位置）。
    vmax      : 每个坐标的速度上限（绝对值），米/秒。
    amax      : 每个坐标的加速度上限（绝对值），米/秒²。
    density   : 送进样条前把路点按弧长加密到这个间距，路点太稀样条会甩出去。

    路径参数用累计直线距离（弦长），这样「走了多少」就是真长度 —— 以前自己写的
    `arc_table` 干的就是这件事，现在 toppra 内部处理。
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.ndim != 2 or len(wp) < 2:
        raise ValueError(f"waypoints 要是 (N, D)、N≥2 的数组，收到 {wp.shape}")

    wp = _densify(wp, density)
    dim = wp.shape[1]

    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    ss = np.concatenate([[0.0], np.cumsum(seg)])
    if ss[-1] < 1e-9:
        raise ValueError("waypoints 全是同一个点，没有路可走")

    path = ta.SplineInterpolator(ss, wp)

    vlim = np.column_stack([np.full(dim, -vmax), np.full(dim, vmax)])
    alim = np.column_stack([np.full(dim, -amax), np.full(dim, amax)])
    instance = algo.TOPPRA(
        [constraint.JointVelocityConstraint(vlim),
         constraint.JointAccelerationConstraint(alim)],
        path,
        solver_wrapper="seidel",
    )
    traj = instance.compute_trajectory(0.0, 0.0)
    if traj is None:
        raise RuntimeError(
            f"toppra 说这条路径在 vmax={vmax} / amax={amax} 下走不通，"
            "把限制放宽或者把路点改顺")
    return Trajectory(traj, dim)


def _densify(wp: np.ndarray, density: float) -> np.ndarray:
    """把路点按直线距离加密，间距不超过 density。路点太稀的话样条会甩出去。

    **短于 density 的那一段，加密点一个都不插，但终点必须留下。** 以前 `n == 1`
    走的是 `continue`，把这一段的终点一起丢了 —— 于是「两个路点、相距 6mm」直接
    塌成一个点，`build` 当场抛 `waypoints 全是同一个点`。`Mover.to` 拿关节角弧度
    调它（`density=0.01`），凡是小于 0.573° 的关节移动都踩这个坑，而「臂本来就快
    到位了」的那些收尾微调恰好全在这个量级。
    """
    if density <= 0:
        return wp
    out = [wp[0]]
    for a, b in zip(wp[:-1], wp[1:]):
        seg = float(np.linalg.norm(b - a))
        n = max(int(np.ceil(seg / density)), 1)
        if n == 1:
            out.append(b)
            continue
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        out.extend(a[None, :] + t * (b - a)[None, :])
    return np.vstack(out)
