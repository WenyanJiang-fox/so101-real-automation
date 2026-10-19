"""位姿换算：水平投影、把工具轴压成竖直、视角半径、世界点投画面、像素每毫米。"""
from __future__ import annotations

from core import follow, geom, shapes, vision
import numpy as np
def _h(a) -> np.ndarray:
    """一个三维向量的水平投影，单位化。水平分量太小就返回零向量。"""
    v = np.array([float(a[0]), float(a[1]), 0.0])
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v

def _align_deg(arm, closing) -> float | None:
    """开合轴（工具 x 轴的水平投影）离目标方向还差多少度。算不出来返回 None。"""
    a, b = _h(arm.tool_x()), _h(closing)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.degrees(np.arccos(
        float(np.clip(float(a @ b) / (na * nb), -1.0, 1.0)))))

def _fold90(d: float) -> float:
    """角度差折到 [−45, 45]。方块是方的，两条侧面差 90° 等价。"""
    return float((float(d) + 45.0) % 90.0 - 45.0)

def _declared_half(yaw, axis_xy) -> float | None:
    """配置里申报的那些东西，沿开合轴鼓出多少（米）。一件都取不到就 None。

    和 `sh.support` 同一个口径（斜着放的方块比半边长宽），只是把来源从「画面估出来的
    形状」换成「配置 `objects:` 里申报的那一行」。**取所有申报件里最大的那个** ——
    单目标场景里就是那件；万一配置写了好几行，取大也还是危险的那一侧。
    """
    best: float | None = None
    for spec in geom.object_specs():
        try:
            h = float(shapes.from_spec(spec).support(yaw, axis_xy))
        except Exception:                                        # noqa: BLE001
            continue
        best = h if best is None else max(best, h)
    return best

def _vertical(quat, x_des=None) -> tuple[np.ndarray, float]:
    """把一个 TCP 口径的朝向摆正成「工具 z 轴朝下」，开合轴（工具 x）尽量保住。

    返回 (摆正后的四元数, 原朝向离垂直差多少度)。做法：工具 z 钉成 (0,0,−1)，工具 x
    投影到水平面当新的 x，y = z × x。

    给了 `x_des`（要贴着的那条侧面法向）就**连方位角一起纠到它**，不再保留原来那个。
    仿真那支就是这么干的：`skills/pick.py:702` 的下探把 `closing=closing` 传给
    `follow`，整段下探都在往目标法向上拧。真机原来没传，方位角是转正那一刻继承的
    ——`run_real_24` 第 1 趟继承的就是转正剩下的 13.7°，一路带进合爪。
    """
    from scipy.spatial.transform import Rotation                # noqa: PLC0415

    q = np.asarray(quat, dtype=np.float64)
    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()   # scipy 吃 xyzw
    down = np.array([0.0, 0.0, -1.0])
    off = float(np.degrees(np.arccos(np.clip(float(R[:, 2] @ down), -1.0, 1.0))))
    x = R[:, 0] - float(R[:, 0] @ down) * down
    if float(np.linalg.norm(x)) < 1e-6:        # 开合轴本来竖直：换 y 轴来投
        x = R[:, 1] - float(R[:, 1] @ down) * down
    n = float(np.linalg.norm(x))
    x = x / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    if x_des is not None:
        xd = np.asarray(x_des, dtype=np.float64) - float(
            np.asarray(x_des, dtype=np.float64) @ down) * down
        nd = float(np.linalg.norm(xd))
        if nd > 1e-6:
            xd = xd / nd
            if float(xd @ x) < 0.0:            # 开合轴没有正负，取跟现在同侧的那一头
                xd = -xd
            x = xd
    out = Rotation.from_matrix(np.column_stack([x, np.cross(down, x), down])).as_quat()
    return np.array([out[3], out[0], out[1], out[2]]), off

def _wrist(cams):
    """腕部相机那一路：名字里带 `wrist` 的那个。"""
    return next((c for n, c in (cams or {}).items() if "wrist" in n), None)

def _view_radius(cams, z_plane: float):
    """腕相机这一刻在 `z_plane` 那个水平面上能看清多大一圈 —— 内切圆半径（米）。

    把画面中心和四条边的中点打回 `z_plane`，取中心到最近那条边的距离。**这是「前后左右
    该挪多远」的尺子**：机位挪 r，原本贴在画面边上的东西正好落到画面正中。算不出来
    返回 None。内参和位姿都现读，写死一个数没意义。

    只读 `mount()` / `size()`，两样 `RealWristCamera` 都有。
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

def _project_px(cam, p_world):
    """世界点投到像素。投不出来（没有腕相机 / 没机位 / 点在相机背后）返回 `None`。

    内参那条岔路跟 `_px_per_mm` **一模一样**（真机那份标定带着 `k`，仿真那份只有 `fovy`）
    —— 两处口径不许各写一份。
    """
    if cam is None:
        return None
    mount = cam.mount()
    if mount is None:
        return None
    w, h = (int(v) for v in cam.size())
    k = mount.get("k")
    if k is None:
        k = vision.intrinsics(float(mount["fovy"]), w, h)
    return vision.project(mount["pos"], mount["mat"], k, np.asarray(p_world, dtype=np.float64))

def _px_per_mm(cam, center):
    """这一点上「画面像素 / 底座毫米」的**带号** 2×2 矩阵 `J = ∂(u,v)/∂(x,y)`。

    投不出来（没有腕相机 / 没机位 / 点在相机背后）给 `None`。`_solve_delta` 和
    `_set_clearance` 的 `B` 初值都用它 —— 只此一处，别在两处各写一遍内参那条岔路。
    """
    if cam is None:
        return None
    mount = cam.mount()
    if mount is None:
        return None
    # 内参：真机那份标定里带着 `k`，仿真那份只有 `fovy` —— 跟 `_view_radius` 一个口径。
    w, h = (int(v) for v in cam.size())
    k = mount.get("k")
    if k is None:
        k = vision.intrinsics(float(mount["fovy"]), w, h)
    jac = vision.px_per_m_xy(mount, k, np.asarray(center, dtype=np.float64))
    return None if jac is None else np.asarray(jac, dtype=np.float64)
