"""从腕部相机那张图里量方块在哪、朝哪 —— 「看」的那一步。

**为什么要看**：标识点只给得出「方块大概在这一带」，给不出它到底在哪。操作员摆到
工位上、机械臂自己放下去，两边都可能偏，实测按半个夹爪开口（约 25mm）算。所以
**位置和朝向都得从画面里量**，标识点坐标只当起搜的起点，不当答案。

**一次也没读方块的真值位姿。** 用的是相机自己的位姿和内参（相机标定 + 手眼标定的
结果，真机上也照样有这一份）；方块的四元数和实时坐标全程没碰。

约定（和 `mcp_robot/sources/verify_backproject.py` 一致，那份对着真值验过）：
    `cam_xmat` 是 cam→world；相机看向自身 -z；
    图像 x 向右 = 相机 +x，图像 y 向下 = 相机 -y；
    针孔内参从 `cam_fovy`（垂直视场角，度）和画面高宽算，方形像素。

**量法是「猜一组未知量，看它该长什么样」**（见 `estimate_pose`）：未知量是方块中心
的水平两个坐标加绕竖直轴的转角，拿相机位姿把八个角点投回画面，跟画面里那个色块的
凸包比重合度，取最像的那组。不量图像里的矩形边 —— 相机一斜，图像里的边就不是方块
棱的投影了（腕部相机的俯仰、位置、视场角每次 reset 都被 nexus 重新随机，实测同一个
方块量出过差 70° 的值）。

**朝向量的是「90° 倍数线」不是「正负号」。** 方块四个面等价，量出来的方向差 90°
无所谓 —— 用它的那个 `face_normal()` 会在四条法向里挑最接近当前开合轴的那条。

**量什么形状由 `core/shapes.py` 那张表说了算。** 投回画面的不是写死的八个角，是
「这个形状从哪个角度看轮廓是哪几个点」——方块是八个角、圆柱是上下两个圆、球是
球面上撒的点。对称的（圆柱、球）连角度都不拟合。尺寸从表里来，不从画面里量：
一个视角分不清「圆柱粗」和「圆柱远」。
"""

from __future__ import annotations

import math

import numpy as np


def intrinsics(fovy_deg: float, width: int, height: int) -> tuple[float, float, float, float]:
    """针孔内参 (fx, fy, cx, cy)。主点取画面正中，方形像素。"""
    fy = (height / 2.0) / math.tan(math.radians(float(fovy_deg)) / 2.0)
    return fy, fy, width / 2.0, height / 2.0


def backproject(cam_pos, cam_mat, k, u: float, v: float, z_plane: float):
    """像素 (u,v) 这条视线射到世界水平面 z=z_plane 上，返回那个世界点（射不到返回 None）。"""
    fx, fy, cx, cy = k
    d_cam = np.array([(u - cx) / fx, (cy - v) / fy, -1.0])
    d_world = np.asarray(cam_mat, dtype=np.float64) @ d_cam
    if abs(float(d_world[2])) < 1e-9:
        return None
    t = (float(z_plane) - float(np.asarray(cam_pos)[2])) / float(d_world[2])
    if t <= 0.0:
        return None
    return np.asarray(cam_pos, dtype=np.float64) + t * d_world


def color_mask(rgb, color: str) -> np.ndarray:
    """把这个颜色的东西从画面里挑出来。掩膜支持的颜色，见 `core.geom.ALL_COLORS`。

    **有彩色**（红黄蓝绿紫）靠「哪个通道高、另外的低」互相分开 —— 这跟光照强弱无关，
    灯亮灯暗都成立（`--randomize` 每局换亮度，靠的就是这一条）。
    **中性色**（白/黑）没有色相可分，只能靠「亮不亮 + 灰不灰」：白 = 亮且三个通道
    挨得近，黑 = 三个通道都低。

    白这条会有多靠谱，取决于**背景是不是又亮又灰**：腕部相机看的是深色桌面和地面，
    实测画面里「亮且不饱和」的像素占比 0.000~0.001（`out/_look_fail_*.png` 那两张），
    所以背景进不来。要是哪天背景换成白桌面，这条就会把桌面也圈进来 —— 那时候靠
    `estimate_pose` 的重合度那道线（`min_iou`）把它挡掉。

    **橙色故意不支持。** 机械臂自己就是橙色 (1.0,0.5,0.0)，腕部相机里它一直在画面里，
    挑橙色等于把爪子当成目标。写了橙色就当场报错，别让它悄悄挑错。
    """
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"要 HxWx3 的图，拿到 {a.shape}")
    r = a[..., 0].astype(np.int16)
    g = a[..., 1].astype(np.int16)
    b = a[..., 2].astype(np.int16)
    if color == "red":
        return (r > 120) & (g < 100) & (b < 100)
    if color == "green":
        return (g > 120) & (r < 100) & (b < 100)
    if color == "blue":
        return (b > 120) & (r < 100) & (g < 100)
    if color == "yellow":
        return (r > 150) & (g > 150) & (b < 110)
    if color == "purple":
        return (r > 70) & (b > 70) & (g < 60)
    if color == "white":
        hi = np.minimum(np.minimum(r, g), b)
        lo = np.maximum(np.maximum(r, g), b)
        return (lo > 190) & (lo - hi < 45)
    if color == "black":
        return (np.maximum(np.maximum(r, g), b) < 60)
    if color == "orange":
        raise ValueError(
            "橙色不能用颜色掩膜挑：机械臂自己就是橙色 (1.0,0.5,0.0)，腕部相机里它一直"
            "在画面里，挑橙色等于把爪子当成目标。换一个颜色。")
    raise ValueError(f"不认识的方块颜色：{color!r}")


def project(cam_pos, cam_mat, k, p_world):
    """世界点投到像素上。在相机背后（或贴得太近）返回 None。

    `backproject` 的逆：相机看向自身 -z，所以深度是 -z；图像 y 向下，世界-相机 y 取负。
    """
    fx, fy, cx, cy = k
    d = np.asarray(p_world, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    p = np.asarray(cam_mat, dtype=np.float64).T @ d      # world → cam
    depth = -float(p[2])
    if depth <= 1e-6:
        return None
    return np.array([cx + fx * float(p[0]) / depth, cy - fy * float(p[1]) / depth])


def cube_corners(center, half: float, yaw: float) -> np.ndarray:
    """方块八个角点在世界里的坐标。center 是方块中心，yaw 是绕竖直轴转过的角（弧度）。"""
    c, s = math.cos(yaw), math.sin(yaw)
    cx, cy, cz = (float(v) for v in center)
    e = np.array([-float(half), float(half)], dtype=np.float64)
    px, py, pz = np.meshgrid(e, e, e, indexing="ij")
    return np.column_stack([(cx + px * c - py * s).ravel(),
                            (cy + px * s + py * c).ravel(),
                            (cz + pz).ravel()])


def project_many(cam_pos, cam_mat, k, pts):
    """一次把好几个世界点投到像素上；有一个在相机背后就返回 None。

    `project` 的批量版 —— 量位置要在几千组候选上反复投同一批角点，一个一个调
    numpy 开销太大。
    """
    fx, fy, cx, cy = k
    d = np.asarray(pts, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    p = d @ np.asarray(cam_mat, dtype=np.float64)      # world → cam（每行一个点）
    depth = -p[:, 2]
    if bool(np.any(depth <= 1e-6)):
        return None
    return np.column_stack([cx + fx * p[:, 0] / depth, cy - fy * p[:, 1] / depth])


def estimate_pose(rgb, mount: dict, color: str, guess, *, shape=None,
                  half: float = 0.015, span: float = 0.030, coarse_m: float = 0.006,
                  coarse_deg: float = 5.0, min_px: int = 120, min_iou: float = 0.5,
                  mask=None):
    """从一张图里量方块在哪、朝哪。返回 (中心坐标 或 None, 朝向弧度 或 None, 事实字典)。

    做法是**反过来**：不量画面里的边，而是拿相机位姿把方块八个角点按候选的（位置,
    角度）投回画面，跟画面里那个色块的凸包比重合度，取最像的那一组。

    为什么这样对：方块是凸的，八个角点的凸包就是它在画面里的剪影，不多不少。所以
    只要相机标定准、候选的位置和角度就是真的那个，投影出来的形状和画面里的形状本来
    就是同一个东西 —— 相机斜不斜、装在腕子的哪个角度上，都不影响。

    剪影取凸包还有一个好处：手指或者别的东西挡住一块，凸包会把它填回来。

    解三个未知量（水平两维 + 一个角度）不是硬搜到底，分三步：
        起点   画面里色块的质心打回地面。**质心不是方块中心**（俯视时顶面把它往相机
               那边拽），所以只当起点，后面靠重合度挪过去。打不回去就退回 `guess`。
        粗扫   在起点周围 ±span 铺一层网格（粗步长 `coarse_m` / `coarse_deg`），取最像的。
        细收   在粗扫那颗最像的附近，按「一次只动一维、动不了就缩一半」的方式收到
               `coarse_m/2**14`，不会再变了为止。

    guess  : 起搜的起点（标识点坐标）。**只用来定 z 平面和兜底**，不是答案。
    span   : 网格铺多大（米，单边）。按半个夹爪开口（约 25mm）留余量，取 30mm。
    shape  : 这是什么形状（`core.shapes.Shape` 或它的一行清单）。给了它，投回画面的
              就是那个形状的剪影，角度那几维按它自己的对称性来 —— 圆柱/球怎么转都一样
              （`yaw_period_deg is None`），**角度整个不拟合**，省掉一整维搜索，量得也
              更准（没有角度可以拟合错）。不给就是老口径：30mm 方块。
    half   : `shape` 不给时用的方块半边长（米）。**给了 shape 就不看它。**
    min_iou: 重合度低于这个就**判为没量出来**（返回 None）。这是「这一眼算不算数」
              的那条线：够了才拿出来用，不够就当没看见。

    事实字典（会打进日志）：
        px    分出多少个像素。太少就是没认出来
        iou   最好那一组的剪影重合度（0~1）。**这是这一量可不可信的判据**，
              低于 min_iou 就返回 None
        fit   量出来的方块中心（世界坐标）
        guess 起搜的起点。两个并排看，就知道这一眼把方块挪了多少

    掩膜从哪来：不传 `mask` 就按 `color` 挑（`color_mask`）。传了就用外面递进来的那份 ——
    这一段里的每一步只认「哪几个像素是这东西」，不关心它怎么来的。所以把掩膜换成
    `foreground_mask`（跟空工位的参考图比）就得到一条完全不看颜色的量法。
    """
    import cv2

    from core import shapes as _shapes

    # 形状表：不传就是老口径那块 30mm 方块，逐字节一样。
    sh = _shapes.from_spec(shape) if shape is not None else _shapes.Box(float(half))
    period = sh.yaw_period_deg       # None = 绕竖轴怎么转都一样，角度这一维不拟合

    if mask is None:
        mask = color_mask(rgb, color)
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    n = int(np.count_nonzero(mask))
    if n < int(min_px):
        return None, None, {"why": f"画面里 {color} 像素只有 {n} 个", "px": n}

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    seen = cv2.convexHull(c).astype(np.float32)
    if len(seen) < 3:
        return None, None, {"why": "方块轮廓太小", "px": n}
    area_seen = float(cv2.contourArea(seen))
    if area_seen <= 0.0:
        return None, None, {"why": "方块轮廓退化了", "px": n}

    # 内参优先用标定出来的那份（`mount["k"]`）。**不能一律按「视场角 + 画面中心」算**：
    # 腕部相机的主点正好在画面正中，这两套是同一个数，一直没露馅；俯视相机的主点偏左
    # 25 像素（实测 cx=295 不是 320），按居中那套算，量出来的位置会被顶偏 36mm。
    k = mount.get("k")
    k = (np.asarray(k, dtype=np.float64) if k is not None
         else intrinsics(mount["fovy"], rgb.shape[1], rgb.shape[0]))
    cam_pos = np.asarray(mount["pos"], dtype=np.float64)
    cam_mat = np.asarray(mount["mat"], dtype=np.float64)
    z0 = float(np.asarray(guess, dtype=np.float64)[2])
    sc_u = float(seen[:, :, 0].mean())
    sc_v = float(seen[:, :, 1].mean())

    seed = backproject(cam_pos, cam_mat, k, sc_u, sc_v, z0)
    if seed is None:
        seed = np.asarray(guess, dtype=np.float64)
    sx, sy = float(seed[0]), float(seed[1])

    def fit(x, y, deg):
        """这组（位置, 角度）投出来的剪影和画面里的重合多少。投不出来就算 None。"""
        pts = project_many(cam_pos, cam_mat, k,
                           sh.silhouette_points((x, y, z0), math.radians(deg)))
        if pts is None:
            return None, None
        want = cv2.convexHull(pts.astype(np.float32))
        inter, _ = cv2.intersectConvexConvex(seen, want)
        union = area_seen + float(cv2.contourArea(want)) - float(inter)
        return (float(inter) / union if union > 0.0 else None), want

    bx, by, bdeg, best_iou, best_want = sx, sy, 0.0, -1.0, None
    offs = np.arange(-float(span), float(span) + 1e-9, float(coarse_m))
    # 角度那一维只在这个形状**转一圈能跟自己重合**（`period` 有值）的时候才扫。
    # 圆柱、球转多少度都一样，扫它等于拿几十倍的算力去拟合一个不存在的东西。
    degs = ([0.0] if period is None
            else list(np.arange(0.0, float(period), float(coarse_deg))))
    for deg in degs:
        for ox in offs:
            for oy in offs:
                v, want = fit(sx + float(ox), sy + float(oy), float(deg))
                if v is not None and v > best_iou:
                    bx, by, bdeg, best_iou, best_want = sx + float(ox), sy + float(oy), float(deg), v, want
    if best_want is None:
        return None, None, {"why": "投到画面外了（相机没对着它？）", "px": n}

    # 细收：一次只动一维（x / y / 角度各试正负），所有方向都不动了就把步长砍一半。
    # 不缩步长直接按梯度走会来回跳；网格再密一档的话粗扫那边就要多十几倍的开销。
    step_m, step_deg = float(coarse_m) * 0.5, float(coarse_deg) * 0.5
    for _ in range(14):
        moves = [(step_m, 0.0, 0.0), (-step_m, 0.0, 0.0),
                 (0.0, step_m, 0.0), (0.0, -step_m, 0.0)]
        if period is not None:
            moves += [(0.0, 0.0, step_deg), (0.0, 0.0, -step_deg)]
        for _ in range(20):
            moved = False
            for dx, dy, dd in moves:
                v, want = fit(bx + dx, by + dy, bdeg + dd)
                if v is not None and v > best_iou:
                    bx, by, bdeg = bx + dx, by + dy, bdeg + dd
                    best_iou, best_want = v, want
                    moved = True
            if not moved:
                break
        step_m *= 0.5
        step_deg *= 0.5

    def box(h):
        a = h.reshape(-1, 2)
        return (float(a[:, 0].min()), float(a[:, 1].min()),
                float(a[:, 0].max()), float(a[:, 1].max()))

    center = np.array([bx, by, z0], dtype=np.float64)

    # 自检：模型说方块离光轴多少度 vs 画面里方块实际离画面中心多少度。
    # 两个数对不上，就说明这套内参/位姿跟渲染这张图的相机不是一回事。
    fx, fy, cx, cy = k
    seen_axis = math.degrees(math.atan(math.hypot((sc_u - cx) / fx, (cy - sc_v) / fy)))
    view = -cam_mat[:, 2]                                   # 相机看的方向（世界系）
    to_cube = center - cam_pos
    cosang = float(view @ to_cube) / (float(np.linalg.norm(view)) * float(np.linalg.norm(to_cube)))
    model_axis = math.degrees(math.acos(float(np.clip(cosang, -1.0, 1.0))))

    # 角度折到一个基本区间里，好看一点。**折法是这个形状自己说了算的**：方形截面
    # 90° 一循环（四个面等价），长方形 180°，圆柱球根本没有角度（`period is None`）。
    # 折错了不影响结果（等价的那些面怎么取都对），但日志里的数会难看。
    if period is None:
        deg = 0.0
    else:
        deg = float(bdeg) % float(period)
        if deg >= float(period) / 2.0:
            deg -= float(period)
    info = {
        "px": n, "iou": best_iou, "yaw": deg,
        "img": (int(rgb.shape[0]), int(rgb.shape[1])),
        "seen": box(seen), "want": box(best_want),
        "seen_area": area_seen, "want_area": float(cv2.contourArea(best_want)),
        "fovy": float(mount["fovy"]),
        "seen_axis": seen_axis, "model_axis": model_axis,
        "seen_c": (sc_u, sc_v),
        "seen_world": np.round(backproject(cam_pos, cam_mat, k, sc_u, sc_v, z0), 4)
        if backproject(cam_pos, cam_mat, k, sc_u, sc_v, z0) is not None else None,
        "seed": np.round(seed, 4),
        "fit": np.round(center, 4),
        "guess": np.round(np.asarray(guess, dtype=float), 4),
        "cam": np.round(cam_pos, 4),
        "view": np.round(view, 3)}

    # **算得出来不等于算得对。** 重合度就是这一眼可不可信的判据：真认准了在 0.7 往上，
    # 前提里有一条不成立（相机位姿不对、画面里那坨色块根本不是方块）就掉到 0 附近。
    # 低了返回 None —— 交给调用方当场停住，别拿它去转腕子、也别拿它当落点。
    # 线取 0.5：本地实测认准的那几把是 0.72~0.90，认错的那把是 0.02，中间空着一大段。
    if best_iou < float(min_iou):
        return None, None, {**info, "why": f"量出来的形状跟画面对不上（重合度 {best_iou:.2f}）"}
    return center, math.radians(deg), info


# ---------------------------------------------------------------- 猜形状和尺寸

# 盲测那一支的尺寸梯子（毫米，全长）。**跟 `core.blind` 的抽样范围对齐**：那边抽
# 20~40mm，这边就 5mm 一档扫过去 —— 梯子上最坏差半档（2.5mm），比 `FIX_CLEAR` 大
# 一点、比 `OPEN_CLEAR` 小。不按「画面里量出来多大」直接定死一个数：相机是斜俯视，
# 顶面比底面离相机近得多，面积反推的尺寸会被这个视差顶偏三成。
BLIND_SIDES_MM = (20.0, 25.0, 30.0, 35.0, 40.0)
BLIND_HEIGHTS_MM = (16.0, 24.0, 32.0, 40.0)     # 长方体那条竖边
DARK_IN_MIN = 0.06       # 「被亮的一圈围住的暗块」占到整块面积的这个比例，就当中空
# **方形剪影不许判成圆柱**（2026-10-18）。圆的 `rect_fill` 恒为 π/4≈0.785、
# 圆度≈1；方形的 `rect_fill`≈1、圆度≈π/4≈0.785。两条里任一条说「方」，圆柱候选就不上。
# 阈值留了抗锯齿/JPEG 边毛的余量：实测那个方块 rect_fill 0.93、圆度约 0.8。
POLY_FILL_MIN = 0.85
POLY_CIRC_MIN = 0.88


def px_per_m(mount, k, point, step: float = 0.005) -> float | None:
    """那个位置上「1 米对应多少像素」。投不出来返回 None。

    **要连位置一起给**：相机斜着看，同样 1 米，画面边上和正中差着的像素数不少
    （实测俯视相机近处 1 米 500 多像素、远处 300 多），拿一个常数当尺子量尺寸，
    边上那件就会量小。
    """
    p = np.asarray(point, dtype=np.float64)[:3].copy()
    c0 = project(mount["pos"], mount["mat"], k, p)
    if c0 is None:
        return None
    rs = []
    for d in ((step, 0.0, 0.0), (0.0, step, 0.0)):
        q = project(mount["pos"], mount["mat"], k, p + np.array(d))
        if q is not None:
            rs.append(float(np.linalg.norm(q - c0)) / float(step))
    return float(np.mean(rs)) if rs else None


def px_per_m_xy(mount, k, point, step: float = 0.005):
    """那个位置上「**世界那一点**沿 x / y 各走 1 米，画面里各挪多少像素」——带号的 2×2。

    返回 `[[du/dx, du/dy], [dv/dx, dv/dy]]`（列 = 走 x / 走 y），投不出来返回 None。

    跟 `px_per_m` 只差一点：那个取的是模长，±号被 `norm` 吃掉了，所以「画面里差几个像素
    该往哪边挪」它答不了 —— 真机上这一点是有代价的：`out/run_real_37.log` 里让开量 14
    像素、按模长算出来「往回缩 1.1mm」，量回来是 16 像素，指令和读数方向是反的。

    **用的时候注意符号**：这里量的是「世界那一点自己动」的像素响应。臂动 δ 的时候，那一点
    在世界里没动（腕相机和爪子一起走了），画面里动的是**相机** —— 等价于「点动了 −δ」，
    所以点在图里的像素位移 ≈ `−J δ`，反解 δ 时要带这个负号（见 `pick_real._nudge`）。
    """
    p = np.asarray(point, dtype=np.float64)[:3].copy()
    c0 = project(mount["pos"], mount["mat"], k, p)
    if c0 is None:
        return None
    cols = []
    for d in ((step, 0.0, 0.0), (0.0, step, 0.0)):
        q = project(mount["pos"], mount["mat"], k, p + np.array(d))
        if q is None:
            return None
        cols.append((np.asarray(q, dtype=np.float64) - np.asarray(c0, dtype=np.float64))
                    / float(step))
    return np.column_stack(cols)


def estimate_shape(rgb, mount, mask, guess, *, sides=BLIND_SIDES_MM,
                   heights=BLIND_HEIGHTS_MM, span: float = 0.030,
                   coarse_m: float = 0.010, coarse_deg: float = 15.0,
                   min_iou: float = 0.35):
    """桌上这件东西**是什么形状、多大**，从这一张图里猜。返回 (清单一行 或 None, 事实)。

    做法跟量位置是同一套（`estimate_pose`：拿候选形状的剪影投回画面比重合度），只是
    候选从一个变成一个梯子：
        第一轮   正方体（边长 20~40mm 五个档）和圆柱（直径同样五档）各投一遍。
                 这一轮把位置和朝向定下来，顺带把「这东西多高」也试出来一个初值。
        第二轮   长方体：用第一轮那个高度当尺子，把画面里那块色块的**外接矩形**换算
                 成米，得到两条平面内边长；再拿它当梯子，竖边扫四档。
    谁的重合度最高就是谁。**外接矩形只在第二轮用、只当梯子** —— 它不是答案（斜俯视
    下它被视差顶偏），真正的判据始终是「投出来的剪影跟画面里那块对不对得上」。

    空心还是实心：**剪影分不出来**（两个外轮廓一模一样，前景掩膜连成一片）。所以另
    看一项：**有没有一块暗的、被亮的一圈围在中间**（`dark_in`，占整块面积多少）。
    敞口的容器，口沿一圈是亮的、把里面的暗腔围住；实心件最暗的那一片一定贴着轮廓的
    边缘。判法写在下面那一段，判成什么都进日志，好对着真值看这条判据立不立得住。

    `mask` 是**这一件自己的**掩膜（0/255，填满的连通域）：位置、尺寸、空心全从它来。
    """
    import cv2

    from core import shapes as _shapes

    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    n = int(np.count_nonzero(m))
    if n < 120:
        return None, {"why": f"这一块只有 {n} 像素，太小了，猜不出形状"}
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c).astype(np.float32)
    if len(hull) < 3:
        return None, {"why": "轮廓退化，猜不出形状"}
    area_px = float(cv2.contourArea(hull))
    (rcx, rcy), (rw, rh), _ang = cv2.minAreaRect(c)
    # **这块剪影是不是圆的**：两个旋转不变的量，从轮廓本身量，不依赖拟合。
    #   `rect_fill` = 面积 ÷ 最小外接旋转矩形面积：圆恒为 π/4≈0.785，四边形≈1；
    #   `circ`      = 4πA/P²：圆≈1，方形≈π/4≈0.785。
    # 用户 2026-10-18 问「为啥会估成圆柱」：那天一帧 rect_fill 0.93（明显是方的），
    # 圆柱候选照样以 0.67 赢了 —— 拟合本身分不清，得靠这两个量先把圆排除掉。
    rect_fill_px = area_px / max(float(rw) * float(rh), 1.0)
    _per = float(cv2.arcLength(hull.reshape(-1, 1, 2), True))
    circ_px = (4.0 * math.pi * area_px / (_per * _per)) if _per > 1e-6 else 0.0
    _is_poly = bool(rect_fill_px >= POLY_FILL_MIN or circ_px < POLY_CIRC_MIN)

    k = mount.get("k")
    k = (np.asarray(k, dtype=np.float64) if k is not None
         else intrinsics(mount["fovy"], rgb.shape[1], rgb.shape[0]))
    g = np.asarray(guess, dtype=np.float64)[:3].copy()

    def fit_one(sh, name):
        """按这个候选形状拟一次，返回 (重合度, 中心, 朝向度, 这一眼的事实)。

        **高度传的是这个候选自己的 `rest_z`**（中心离地多高），不是外面那个猜测值。
        这一步非做不可：剪影是按 `guess` 的 z 投回画面的，相机又是斜着看下来的，
        高度差 10mm 就能把水平位置顶偏十几毫米（`10mm/tan(34°)≈15mm`），而这个
        搜索只扫水平两维、扫不到高度那一维去。所以每试一个尺寸，就按那个尺寸应该
        在多高去投 —— 高度错的候选自然在水平上也对不上，重合度自己会把它压下去。
        """
        gg = np.array([g[0], g[1], float(sh.rest_z)], dtype=np.float64)
        center, yaw, info = estimate_pose(rgb, mount, None, gg, shape=sh, mask=m,
                                          span=span, coarse_m=coarse_m,
                                          coarse_deg=coarse_deg, min_iou=-1.0)
        if center is None:
            return -1.0, None, 0.0, info
        deg = float(np.degrees(yaw or 0.0))
        return float(info.get("iou", -1.0)), center, deg, info

    tried: list[tuple[float, str]] = []
    best = (-1.0, None, None, 0.0, "")

    # ---- 第一轮：正方体 + 圆柱 ----
    for s in sides:
        sh = _shapes.Box(s / 2000.0)
        iou, center, deg, _ = fit_one(sh, "cube")
        tried.append((iou, f"正方体 边长{s:.0f}mm"))
        if iou > best[0]:
            best = (iou, sh, center, deg, f"正方体 边长{s:.0f}mm")
    for d in sides:
        if _is_poly:
            break
        sh = _shapes.Cylinder(d / 2000.0, d / 2000.0)
        iou, center, deg, _ = fit_one(sh, "cyl")
        tried.append((iou, f"圆柱 直径{d:.0f}mm"))
        if iou > best[0]:
            best = (iou, sh, center, deg, f"圆柱 直径{d:.0f}mm")

    # 第一轮最好的那个有多高，拿它当尺子去换算外接矩形 —— 同一个位置、不同的高度，
    # 1 米对应多少像素能差三成，尺子用错档，长方体的两条边长就整片偏。
    hi_m = float(getattr(best[1], "half_z", 0.015))
    scale = px_per_m(mount, k, [g[0], g[1], hi_m])
    if scale is None:
        scale = px_per_m(mount, k, [g[0], g[1], 0.015])
    if scale is None:
        return None, {"why": "算不出这个位置 1 米多少像素", "iou": best[0]}

    # ---- 第二轮：长方体。两条平面内边长从外接矩形换算，竖边扫梯子 ----
    a_mm = float(rw) / scale * 1000.0
    b_mm = float(rh) / scale * 1000.0
    # 外接矩形还带着视差和阴影撑出来的那一圈，往下收一成半再当梯子。
    a_mm, b_mm = max(15.0, a_mm * 0.85), max(15.0, b_mm * 0.85)
    for h in heights:
        for (ax, bx) in ((a_mm, b_mm), (b_mm, a_mm)):
            sh = _shapes.Box(ax / 2000.0, bx / 2000.0, h / 2000.0)
            iou, center, deg, _ = fit_one(sh, "box")
            tried.append((iou, f"长方体 {ax:.0f}×{bx:.0f}×{h:.0f}mm"))
            if iou > best[0]:
                best = (iou, sh, center, deg, f"长方体 {ax:.0f}×{bx:.0f}×{h:.0f}mm")

    # ---- 空心还是实心：找「被亮的一圈包围着的暗块」 ----
    #
    # **不能只比「里圈比外圈暗多少」。** 第一版就是这么写的，实心长方体当场被判成中空
    # （实测那一件里圈比外圈暗 15.6 个单位）：斜俯视下看到的轮廓里本来就带着侧面，
    # 顶面在中间、**暗的侧面在边缘**，里亮外暗是实心件的常态，跟空不空心没关系。
    #
    # 真正的区别是**拓扑**：敞口的容器，那圈口沿是亮的、把里面的暗腔**围在中间**；
    # 实心件最暗的那一片（侧面、或者背光那半边）一定**贴着轮廓的边缘**，围不起来。
    # 所以：取暗的那一半，丢掉所有碰到轮廓边缘的连通块，剩下的若是够大一块 —— 中空。
    lb = lit_normalized(rgb)
    hollow = False
    dark_in = 0.0
    try:
        v = lb[m > 0]
        if v.size >= 200:
            lo, hi = (float(x) for x in np.percentile(v, (5.0, 95.0)))
            span = max(1e-6, hi - lo)
            dark = ((lb < lo + 0.5 * span) & (m > 0)).astype(np.uint8)
            edge = (m > 0) & (cv2.erode(m, np.ones((5, 5), np.uint8)) == 0)
            n_lab, lab = cv2.connectedComponents(dark, 8)
            keep = 0
            for c_i in range(1, n_lab):
                blob = lab == c_i
                if np.any(blob & edge):
                    continue                    # 碰到边缘的暗块 = 侧面那一圈，不是腔
                keep = max(keep, int(np.count_nonzero(blob)))
            dark_in = float(keep) / float(max(1, int(np.count_nonzero(m))))
            hollow = bool(dark_in >= DARK_IN_MIN)
    except Exception:                            # noqa: BLE001  判空心失败不该毁掉这一眼
        hollow = False

    iou, sh, center, deg, name = best
    tried.sort(key=lambda t: -t[0])
    info = {
        "estimate": name, "iou_est": iou,
        "rect_fill": round(rect_fill_px, 3), "circularity": round(circ_px, 3),
        "cyl_skipped": bool(_is_poly),
        "top": [f"{nm} {v:.2f}" for v, nm in tried[:3]],
        "dark_in": dark_in, "hollow": hollow,
        "size_hint_mm": [round(a_mm, 1), round(b_mm, 1)], "scale_px_m": scale,
        "yaw_deg": round(deg, 1), "fit": None if center is None else np.round(center, 4),
    }
    if sh is None or iou < float(min_iou):
        return None, {**info, "why": f"一个候选都对不上（最好的是「{name}」，重合度 {iou:.2f}）"}
    spec = _spec_of(sh, hollow)
    info["spec"] = spec
    return spec, info


def _spec_of(sh, hollow: bool) -> dict:
    """把一个 `core.shapes.Shape` 翻回清单一行（毫米）。"""
    from core import shapes as _shapes

    if isinstance(sh, _shapes.Cylinder):
        return {"shape": "cylinder", "diameter_mm": sh.radius * 2000.0,
                "height_mm": sh.height * 1000.0, "cup": bool(hollow)}
    if isinstance(sh, _shapes.Box):
        return {"shape": "box",
                "size_mm": [sh.half_x * 2000.0, sh.half_y * 2000.0, sh.half_z * 2000.0]}
    raise TypeError(f"估出来的形状不认：{type(sh)}")


# ---------------------------------------------------------------- 不看颜色：跟参考图比

def lit_normalized(rgb) -> np.ndarray:
    """把画面按自己的中位数和离散度归一：减掉中位数，再除以 MAD。

    整张图一起变亮、变暗、灯变色温，落在灰度上就是「乘一个系数、再加一个常数」。减中位数
    消掉那个常数，除以 MAD（绝对中位差 × 1.4826，抗离群的离散度）消掉那个系数。剩下的是
    「这一片比全场典型值亮多少」，跟光照无关。用中位数不用均值：方块、机械臂这些只占一小块，
    拉不动中位数。

    **为什么不用局部对比度（减自己的高斯模糊）。** 试过了：平坦的面上局部对比度恒等于 0，
    方块顶面正中一片全是 0 —— 拿它跟参考图比，比出来只有方块**四周一圈轮廓**，中间是空的。
    这种空心掩膜拿去跟完整方块的投影比重合度，最多到 0.45，永远过不了线。要比就得比明暗本身。

    用三通道平均当灰度，不查彩色转灰的权重表：参考图和当前图是同一套约定，权重取什么都
    一致，少一个能出错的地方。
    """
    g = np.asarray(rgb, dtype=np.float32)[..., :3].mean(axis=-1)
    med = float(np.median(g))
    spread = 1.4826 * float(np.median(np.abs(g - med)))
    return (g - med) / spread if spread > 1e-6 else g - med


def foreground_mask(rgb, ref, *, thr: float = 3.0,
                    open_px: int = 3, close_px: int = 7) -> np.ndarray:
    """跟一张「空工位」的参考图比，把多出来的那一片挑出来。返回 0/255 的掩膜。

    参考图要在工位上什么都没有的时候拍，相机不能动 —— 这两条是它的前提，
    真机上也一样（相机被碰过就得重拍）。

    thr  : 两张图归一化之后差多少算「多出来了」，单位是 MAD 的倍数。
    """
    import cv2

    d = np.abs(lit_normalized(rgb) - lit_normalized(ref))
    m = (d > float(thr)).astype(np.uint8) * 255
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_px), int(open_px)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)       # 抹掉孤立噪点
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_px), int(close_px)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)      # 补上一块东西内部的小洞
    return m


def blobs_of(mask, *, min_area: int = 40, border: int = 4):
    """掩膜里的连通域，面积大的在前：[(面积, (质心u, 质心v), (x, y, w, h), 标签号)]。

    **碰到画面边的丢掉。** 从画面外伸进来的东西（机械臂、人的手）一定连着边，摆在台上的
    一个方块不连着边。这一条替掉了以前「靠颜色把橙色手臂排除在外」那个作用 —— 不看颜色
    之后，手臂就是最大的干扰源，得有这么一条把它的量挡掉。

    标签号是 `connectedComponentsWithStats` 给那一块的编号，拿它可以只留下这一块
    （`labels == 标签号`），把一个候选单独喂给 `estimate_pose`。
    """
    import cv2

    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < int(min_area):
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if x <= border or y <= border or x + bw >= w - border or y + bh >= h - border:
            continue
        out.append((a, (float(cent[i][0]), float(cent[i][1])), (x, y, bw, bh), int(i)))
    out.sort(key=lambda b: -b[0])
    return out
