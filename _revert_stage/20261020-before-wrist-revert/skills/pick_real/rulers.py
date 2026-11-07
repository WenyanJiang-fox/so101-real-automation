"""两条尺子（画面像素 / 3D 毫米）和爪子轮廓。**判据只在这儿定义一次。**"""
from __future__ import annotations

from core import follow, geom, shapes, vision
import numpy as np
from . import knobs
from .knobs import AXIS_TOL_PX, CLEAR_3D_ENFORCE, CLEAR_3D_TOL_MM, CLEAR_BAND_HI_PX, CLEAR_BAND_LO_PX, CLEAR_BAND_ON, CLEAR_EXIT_EPS_PX, CLEAR_TARGET_PX, JAW_RAW_IMG, JAW_SHAPE_JSON, LOOK_IMG
from .pose import _fold90, _project_px, _px_per_mm, _wrist
def _jaw_axis_deg(cam) -> float | None:
    """合爪轴在**画面里**的角度（度）。没描过轮廓、画幅对不上就返回 None。"""
    try:
        jaw = _jaw_shape(cam)
    except RuntimeError:
        return None
    v = np.asarray(jaw["axis_dir_px"], dtype=np.float64).reshape(-1)[:2]
    return float(np.degrees(np.arctan2(float(v[1]), float(v[0]))))

def _red_rect_deg(cam, percep) -> float | None:
    """红区剪影的最小外接旋转矩形的角度（度）。就地拍一张；量不出来返回 None。

    掩膜那条链跟 `_jaw_check` 逐字一样（同一份 HSV 区间、同一个最小面积、同一个矩形度
    门槛），不然量的是另一块红区。
    """
    if cam is None or percep is None:
        return None
    import cv2                                                  # noqa: PLC0415

    from real.perception import brown_mask, candidates           # noqa: PLC0415

    bgr = cv2.cvtColor(np.asarray(cam.shot()), cv2.COLOR_RGB2BGR)
    mask = brown_mask(bgr, percep.ranges)
    cands = [d for d in candidates(mask, percep.min_area)
             if d["rect_fill"] >= percep.fill_min]
    if not cands:
        return None
    return float(cands[0]["rect"][2])

def _box_deg_px(cam, percep) -> float | None:
    """**画面里**「方块侧面」和「合爪轴」差多少度。量不出来返回 None。

    这条为什么作数：相机和两爪在**同一个刚体**上 —— 合爪轴在画面里的角度跟臂怎么转都
    无关（`assets/camera_calibration/real_wrist_jaw_shape.json` 那几圈轮廓敢写死就是
    这个道理）。所以画面里的这个夹角**就是**世界系里开合轴离方块侧面差多少度，中间
    不过「量 yaw → 算侧面法向 → 正解 tool_x」那一串模型量 —— 那一串哪一环歪了，它都
    不会知道。

    2026-10-17 实测：画面上 27.6°（红区剪影 `ang=−63.43°`，116.7×109.6px，填充 0.895）
    vs `_align_deg` 报的 1.8°。用户先看出来的（「方块在你的画面里面都成了 45° 了」）。
    """
    a = _red_rect_deg(cam, percep)
    ax = _jaw_axis_deg(cam)
    if a is None or ax is None:
        return None
    return _fold90(a - ax)

def _yaw_from_px(sh, center, yaw, cam, percep):
    """拿画面里红区的角度把 `yaw` 纠回来。返回 `(纠过的 yaw, 画面角度, 模型角度)`。

    `estimate_pose` 给的 `yaw` 是模型量：它歪多少，`sh.normal` 算出来的侧面法向就跟着
    歪多少 —— 而转正挑哪一面、下探的落点全挂在它上面。画面这条不经过它。

    比法两边**同一个量法**才说得通：模型这边把八个角点投到画面、取凸包，再用同一个
    `minAreaRect` 量一次，跟红区那个角度对。`yaw` 由粗到细扫一遍取最贴的那个。

    量不出来（没相机 / 没红区 / 投不出来）返回 None —— 调用方按老口径走。
    """
    if cam is None or percep is None:
        return None
    import cv2                                                  # noqa: PLC0415

    a_meas = _red_rect_deg(cam, percep)
    if a_meas is None:
        return None
    ctr = np.asarray(center, dtype=np.float64)

    def _pred(y):
        pts = np.asarray(sh.silhouette_points(ctr, float(y)), dtype=np.float64)
        uv = [q for q in (_project_px(cam, p) for p in pts) if q is not None]
        if len(uv) < 4:
            return None
        hull = cv2.convexHull(np.asarray(uv, dtype=np.float32).reshape(-1, 1, 2))
        return float(cv2.minAreaRect(hull)[2])

    def _cost(y):
        m = _pred(y)
        if m is None:
            return 1e9
        # 90° 一循环，用 4θ 比才不会在折角上跳
        return 1.0 - float(np.cos(np.radians(4.0 * (m - a_meas))))

    best = float(yaw)
    # **步长和跨度都是弧度**（`yaw` 就是弧度）—— 2026-10-17 第一版写成 1.0/45.0，
    # 那是「1 弧度一步、扫 ±45 弧度」，扫了七八圈才回来，返回一个等价于原值的
    # 12.38 rad（打印出来 +709.4°）。
    for step, span in ((np.radians(1.0), np.radians(45.0)),
                       (np.radians(0.1), np.radians(1.0))):
        grid = np.arange(best - span, best + span + 1e-9, step)
        best = float(min(grid, key=_cost))
    pred = _pred(best)
    if pred is None:
        return None
    return best, float(a_meas), float(pred)

def _jaw_axis_model_deg(arm, cam):
    """**模型这边**「合爪轴在画面里的角度」：两个爪尖投到画面，连线取角。

    跟 `_jaw_axis_deg`（`real_wrist_jaw_shape.json` 里描死的那条）对一下 —— 两个都该
    说同一条线。对不上，就说明画面那条尺子拿的不是模型这条轴，`_box_deg_px` 的读数
    不能直接当「世界系里差多少度」用。
    """
    if cam is None or arm is None:
        return None
    try:
        fixed, moving = arm.tips()
    except Exception:                                           # noqa: BLE001
        return None
    a = _project_px(cam, np.asarray(fixed, dtype=np.float64))
    b = _project_px(cam, np.asarray(moving, dtype=np.float64))
    if a is None or b is None:
        return None
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if float(np.linalg.norm(d)) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(float(d[1]), float(d[0]))))

def _jaw_shape(cam=None) -> dict:
    """读 `_trace_jaw.py` 描好的爪子轮廓。没有、或者画幅对不上就当场抛。

    **画幅对不上不许凑合**。那几圈是在某一路相机的某个分辨率上点的像素，换一路、
    换分辨率之后再拿它去量「爪子离目标多远」，量的是别的东西。
    """
    import json                                                 # noqa: PLC0415

    # 相机自己能报就听它的。**真机这一路没有这个函数**，走下面那份描好的 JSON；
    # 留这个口子是为了不把「相机自己会算」的那类实现堵死。
    f = getattr(cam, "jaw_shape", None)
    if f is not None:
        return f()
    if not JAW_SHAPE_JSON.exists():
        raise RuntimeError(
            f"没有 {JAW_SHAPE_JSON} —— 机械爪在腕相机画面里的轮廓没描过。跑 "
            "`G:/anaconda/envs/sim2real/python.exe _trace_jaw.py "
            "--img out/jaw_frame.jpg` 描一次。")
    d = json.loads(JAW_SHAPE_JSON.read_text(encoding="utf-8"))
    if cam is not None:
        w, h = (int(v) for v in cam.size())
        if [int(v) for v in d.get("image_size", ())] != [w, h]:
            raise RuntimeError(
                f"{JAW_SHAPE_JSON} 里那三圈是在 {d.get('image_size')} 的画面上描的，"
                f"此刻这一路是 {[w, h]} —— 对不上，重新描一次。")
    return d

def _contour_pts(cnt, step_px: float = 2.0) -> np.ndarray:
    """把轮廓折线按 ≤`step_px` 的间隔重采样成点列（连收口那条边一起）。

    **不能直接拿轮廓顶点去量距离。** `real/perception.py` 那边用的是
    `CHAIN_APPROX_SIMPLE`，一条直边只留两个端点 —— 拿顶点量出来的「离爪子多远」其实只
    量了四个角点，边中间那一截（方块在画面里基本全是直边）全漏掉。
    """
    p = np.asarray(cnt, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return p
    p = np.vstack([p, p[:1]])
    out = [p[0]]
    for a, b in zip(p, p[1:]):
        n = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / step_px)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.asarray(out, dtype=np.float64)

def _gap_px(poly, pts_uv) -> tuple[float, np.ndarray]:
    """红区轮廓那一串点里**最靠里**的那一点，到某一圈爪子轮廓的**带号**距离（像素）。

    **符号跟 `cv2.pointPolygonTest` 一致**：正 = 红区那一点落在爪子轮廓**里面**（压上了），
    负 = 落在外面（留了缝）。这里是「拿红区上的点去问爪子」，**别记串**。
    `_set_clearance` 用的是取负之后的那个数（「离爪子外面多少像素」，越大越好）。

    **取的是最大值，不是最小值。** 判「有没有接触」要看**最靠里的那一个红点**，所以比的是
    谁更靠里（带号值更大）。取最小值拿到的是**离爪子最远**的那个点，等于一直在看方块另一
    边 —— 2026-10-15 就是这么漏掉的：`out/real_37_preclose_f82.jpg` 上真正最近的红点是
    (1063,414)、离轮廓 0.7 像素（贴上了），这段代码报的是 81 像素「缝很宽」，判据过关、
    微调也没被触发。
    """
    import cv2                                                  # noqa: PLC0415

    best: tuple[float, np.ndarray] | None = None
    for p in pts_uv:
        d = float(cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), True))
        if best is None or d > best[0]:
            best = (d, np.asarray(p, dtype=np.float64))
    return best if best is not None else (float("nan"), np.zeros(2))

def _exit_px(polys, pts_uv, u) -> float:
    """沿 `u` 把红区整体推开，推多少像素才**一个红点都不剩在爪子轮廓里**。

    返回 0 = 现在就没有红点压在轮廓里。`_gap_px` 量的是「**最近的那一个点**」的距离，
    方块大半压在轮廓里时它给不出「退出去还要多少」：实测（`out/clear_step.json` 第一眼）
    红区最深只压进 0.7 像素，那条尺子就只要求横移 8.7 像素，而按
    `assets/camera_calibration/real_wrist_jaw_shape.json` 的几何，要让整块红区离开
    爪轮廓（占 x 1023~1106）得挪 60~85 像素。用户 2026-10-16 的原话是「把爪子轮廓
    完整移出去」—— 这个函数就是补那一半的尺子。

    `u` 取单位向量（这里是合爪轴方向）。二分：单调沿一个方向平移，块离开轮廓之后
    不会再压回去（轮廓是楔形、没有把方块包住的凹口）。
    """
    import cv2                                                  # noqa: PLC0415

    u = np.asarray(u, dtype=np.float64).reshape(2)
    pts = np.asarray(pts_uv, dtype=np.float64).reshape(-1, 2)
    polys = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2) for p in polys]
    if pts.size == 0 or not polys:
        return 0.0

    def clean(t: float) -> bool:
        q = pts + float(t) * u
        for poly in polys:
            for p in q:
                if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) > 0:
                    return False
        return True

    if clean(0.0):
        return 0.0
    lo, hi = 0.0, 256.0
    while hi < 1.0e6 and not clean(hi):
        lo, hi = hi, hi * 2.0
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if clean(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)

def _box_half_m(percep) -> float:
    """方块**半个边长**（米），给「缝不够时一步至少走半个边长」那条下限用。

    从感知那份清单里取：**优先用「往下真正在用的那一份」**（`percep.spec_used` ——
    2026-10-17 用户定：画面估出来过就用估出来的，估不出来感知那边才退回配置那一行，
    两者它在 `locate` 里并排打出来了）。再按形状取长度：`cube` 用 `side_mm`，`box` 用
    `size_mm` 三个全长里最长的那个，`cylinder` 用 `diameter_mm` / `height_mm` 里大的那个
    （跟 `box` 同一个口径 —— 「取大的那一维」是安全的那个方向）。取不到就退回 18mm。

    **改这里之前先想清楚**：原来这一段明写「不用画面估出来的那一份」，理由是下限只是
    「别挪一眼几乎不动」、估小了也只是步子小一点。但同一份值也喂给 `_place_real` 当
    「方块下底面」算松爪高度，那一处估小了是要命的（松爪时方块已经贴桌面）。所以改成
    跟其余落点算式同一个来源。
    """
    spec = (getattr(percep, "spec_used", None)
            or getattr(percep, "spec", None) or {})
    v = spec.get("side_mm")
    if v is None:
        # `box` 给三个全长；`cylinder` 给直径和高。两种都是「几个长度里取最大的那个」。
        # **2026-10-17 补的圆柱一支**：原来只认 `side_mm` / `size_mm`，圆柱那两行
        # (`diameter_mm` / `height_mm`) 一个都读不到，直接退到 18mm —— 直径 40 的圆柱
        # 会被当成 18mm 的小方块算半个身位。
        sz = spec.get("size_mm")
        if sz is None and spec.get("diameter_mm") is not None:
            sz = [x for x in (spec.get("diameter_mm"), spec.get("height_mm"))
                  if x is not None]
        v = max(float(x) for x in sz) if sz else 18.0
    return 0.5 * float(v) / 1000.0

def _gap_want_px() -> float:
    """沿合爪轴那一步的**目标**（像素）。带开着取带中值；关掉就是老的单边门槛。"""
    if CLEAR_BAND_ON:
        return 0.5 * (CLEAR_BAND_LO_PX + CLEAR_BAND_HI_PX)
    return CLEAR_TARGET_PX

def _gap_ok(gap_px: float) -> bool:
    """这一眼的缝算不算过。带开着要落在 `[LO, HI]` 里；关掉还是老的「≥ 门槛」。"""
    if CLEAR_BAND_ON:
        return bool(CLEAR_BAND_LO_PX <= float(gap_px) <= CLEAR_BAND_HI_PX)
    return bool(float(gap_px) >= CLEAR_TARGET_PX)

def _gap_need_px(gap_px: float, gap_exit_px: float) -> float:
    """沿合爪轴还差多少像素、往哪边（**带号**：正 = 沿合爪轴推出去，负 = 把方块拉回来）。

    **带开着的时候两头都管**（用户 2026-10-16：「必须要把目标放在两个爪子中间」）：
      · 缝 > `CLEAR_BAND_HI_PX` → 方块退出两爪了，返回负值 —— 这是原来那条单边门槛
        漏掉的一半，`out/run_real_89.log` 末态缝 +46 像素就是这么判过、然后空抓的；
      · 缝 < `CLEAR_BAND_LO_PX` → 压上了，返回正值；**还要跟「整块退出去」那条取大的**
        （`_exit_px` 量的是「推开多少像素才一个红点都不剩在爪轮廓里」，压得深时它更大）；
      · 落在带里 → 0，这一维不用动，去对轴。
    """
    g = float(gap_px)
    if not CLEAR_BAND_ON:
        return max(float(gap_exit_px), CLEAR_TARGET_PX - g)
    if g > CLEAR_BAND_HI_PX:
        return _gap_want_px() - g
    if g < CLEAR_BAND_LO_PX:
        return max(float(gap_exit_px), CLEAR_BAND_LO_PX - g)
    return 0.0

def _gap_leaving(need_px: float) -> bool:
    """这一步算不算「缝」那一段（该走沿合爪轴这条、还是去对轴）。

    带开着时**两头都算** —— 退出去和压上来都是「缝」那一段。关掉时还是老口径：只有
    「还差多少推出去」为正才算，负值（方块已经退出去了）掉进对轴那一段。
    """
    if CLEAR_BAND_ON:
        return abs(float(need_px)) > CLEAR_EXIT_EPS_PX
    return float(need_px) > CLEAR_EXIT_EPS_PX

def _gap_axis_px(polys, pts_uv, u) -> float:
    """**沿合爪轴 `u` 看**，爪子轮廓和红区之间还剩多少像素净空。负 = 沿这条轴压在一起。

    `_gap_px` 量的是「红区里最靠里的那一点到爪轮廓的**最近**距离」—— 方向不限。爪子跑到
    方块**上方**（不在合爪轴那一侧）时，最近的那一点是**上边**那一段，它会报一个正的
    「缝」，而沿合爪轴（该留缝的那一侧）其实一点缝都没有。用户 2026-10-16 的原话：
    「爪子根本没到位置，它在图像上部，并没有到方块的右边制造缝隙，你检测的留出的缝隙是在
    上面的缝隙是错误的」。

    所以这里把两个点集都投到 `u` 上，看两段区间**沿这条轴**离多远：
    不相交 = 净空（正），相交 = 压进深度（负）。`u` 取合爪轴方向（`axis_dir_px`）。
    """
    u = np.asarray(u, dtype=np.float64).reshape(2)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    pts = np.asarray(pts_uv, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0 or not polys:
        return float("nan")
    q = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polys],
                       axis=0)
    r0, r1 = float((pts @ u).min()), float((pts @ u).max())
    j0, j1 = float((q @ u).min()), float((q @ u).max())
    if r1 < j0:
        return j0 - r1
    if j1 < r0:
        return r0 - j1
    return -min(r1 - j0, j1 - r0)

def _box_center_px(cnt, bgr=None) -> np.ndarray | None:
    """方块**顶面**（用户说的「目标那个面」）中心在画面里那一个像素。

    **不能拿整块的重心。** 相机是斜着往下看的，剪影是「顶面四边形 + 正面那一条下沿」，
    重心被正面整个往下拽（`out/jaw_raw.jpg` 这一帧：重心 (1047,463)，顶面中心 (1050,441)）。

    **不能靠剪影的角去凑。** 2026-10-15 试过「凸包 → 剔共线 → 去掉最下面两个角」：
    · 凸包给 19 个点，而剔共线那个循环的停止条件写的是「剩 4 个」不是「没有共线点了」，
      于是它把顶沿一带的点剔到只剩 4 个就停，接着「去掉 y 最大的两个」去掉的是两个
      **顶面**的角，报 (1051.5, 394.5)；
    · 就算循环改对了，这办法也够不着真值：顶面的前沿左角 T4 **不是剪影的极值点**，
      它落在左边那条直线上，任何「取剪影的角」的规则都看不见它。
    旧版（`approxPolyDP` 按 y 取最上面四个）在这一帧取到的是 T1、T2、**爪左边缘压上顶沿
    的交点**、**爪尖切出来的凹口**，报 (1034,399)，跟真值差 42 像素（`out/run_real_63.log`）。

    **改用亮暗。** （2026-10-15 那版用的是「它自己的颜色」= BGR 的红通道，2026-10-17 换成
    V 通道，换的理由和回退办法见 `BOX_CENTER_CHAN`；下面那两个峰的数字是**红通道**下量的，
    V 通道下峰位会变、结论不变。）顶面朝上受光、正面背光 —— 这一帧红区里 R 通道是两个清清楚楚的
    峰：正面 ≈120（1900 像素）、顶面 ≈180（2000 像素），中间 130~139 只有 300 来个像素的
    谷。顶面 = 红区里亮的那一半，两半的**分界线就是前沿那条棱**（用户量的那条：列 x=1050
    上亮暗交界在 y≈492）。亮块取**凸包** —— 爪尖压出来的凹口被凸包自然补回来 —— 剩下
    3~4 个角就是顶面的角，均值即中心。

    **假设**：盒子顶面和正面亮度分得开（受光方向固定就该如此）。分不干净（亮块太小 /
    简化后不是 3~4 个角）返回 `None`，不猜。
    """
    import cv2                                                  # noqa: PLC0415

    p = np.asarray(cnt, dtype=np.float32).reshape(-1, 1, 2)
    if len(p) < 4 or bgr is None:
        return None
    h, w = bgr.shape[:2]
    full = np.zeros((h, w), np.uint8)
    cv2.drawContours(full, [p.astype(np.int32)], -1, 255, -1)
    area0 = int((full > 0).sum())
    if area0 < 50:
        return None
    # 亮暗分界用 Otsu 现算，不写死阈值 —— 光照变了自己跟着变。
    # 用哪个通道见 `BOX_CENTER_CHAN`：默认 V（HSV 亮度，与颜色无关），换回红通道也在这里。
    if knobs.BOX_CENTER_CHAN == "r":
        ch = bgr[:, :, 2]
    else:
        ch = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    t, _ = cv2.threshold(ch[full > 0].reshape(-1, 1), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    top = np.zeros((h, w), np.uint8)
    top[(full > 0) & (ch >= float(t))] = 255
    # 亮的可能不止一块（反光、桌面上的红别的东西）；取最大那块。
    n, lab, stats, _ = cv2.connectedComponentsWithStats(top, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[k, cv2.CC_STAT_AREA]) < 0.15 * area0:
        return None
    one = np.zeros((h, w), np.uint8)
    one[lab == k] = 255
    cs, _ = cv2.findContours(one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    per = float(cv2.arcLength(hull, True))
    if per < 1.0:
        return None
    ap = None
    for f in (0.02, 0.03, 0.045, 0.06):        # 容差逐级放大，直到简化出 3~4 个角
        c = cv2.approxPolyDP(hull, f * per, True).reshape(-1, 2).astype(np.float64)
        if len(c) == 4:
            return c.mean(axis=0)
        if len(c) == 3:
            ap = c
            break
    if ap is None:
        return None
    # 只认出三个角：顶面在画面里是平行四边形，缺的那个 = A + C - B。哪一对是**对角**
    # 不靠顺序定 —— 把三种配法都补一遍，补出来的点**落在红区里**的那一对才是对角
    # （另外两种配法补出来的点会跑到方块外面去）。中心 = 对角的中点。
    best: tuple[float, np.ndarray] | None = None
    for i in range(3):
        a, b, cc = ap[i], ap[(i + 1) % 3], ap[(i + 2) % 3]
        d = a + cc - b
        if cv2.pointPolygonTest(np.asarray(cnt).reshape(-1, 1, 2).astype(np.int32),
                                (float(d[0]), float(d[1])), False) < 0:
            continue
        span = float(np.linalg.norm(a - cc))
        if best is None or span > best[0]:
            best = (span, (a + cc) / 2.0)
    return None if best is None else best[1]

def _put_lines(img, lines, col=(0, 255, 255), bottom: bool = False) -> None:
    """在画幅左上角（`bottom=True` 时左下角）写几行 ASCII，**字号按最长那行现算**。

    `cv2.putText` 不认中文，所以这里只有 ASCII。原来每个拦门的地方都写死 0.85：
    `tries 4   gap +61px (want 20..75 = between jaws)   axis -43px (tol 35)   along +119
    (pixel ruler: reference only, NOT the gate)` 那一行比 1920 还宽，右边的字被切在画外。
    用户 2026-10-17：「字搞的大一点」。

    现算 = 拿最长那行在 1.0 号下的实测宽度抬到「两边各留 16px」为止，再卡在 [0.8, 1.6]。
    行距跟着字号走，行多了不会叠在一起。

    `bottom=True` 是给 `_jaw_check` 用的：它先往 `jaw_check.jpg` 上写一段，`_confirm_step`
    再读同一张图、往**顶上**写自己的几行 —— 两段都在顶上就压在一起了。
    """
    import cv2                                                    # noqa: PLC0415

    if not lines:
        return
    (_tw, _th), _ = cv2.getTextSize(max(lines, key=len), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    sc = min(1.6, max(0.8, (float(img.shape[1]) - 32.0) / max(float(_tw), 1.0)))
    dy = int(round(float(_th) * sc * 1.5)) + 10
    y0 = int(img.shape[0]) - 24 - dy * (len(lines) - 1) if bottom else 34
    for i, txt in enumerate(lines):
        org = (16, y0 + dy * i)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)

def _draw_measures(img, pts, polys, ref, u, box_uv, gap_px, axis_px,
                   gap_gated: bool = True) -> None:
    """把「缝」和「轴偏」这两个数**是怎么量出来的**画在画面上（用户 2026-10-19）。

    只画量法，不改任何判据，也不改返回的读数：

    · **缝（黄）**：把方块轮廓和爪子轮廓都投到合爪轴 `u` 上，看两段区间沿这条轴离多远
      （就是 `_gap_axis_px` 报的那个数）。图上画成**尺寸线**：过「方块对着爪子那一端」
      和「爪子对着方块那一端」各拉一条 ⊥`u` 的刻线，再在偏出去的位置拉一条 ∥`u` 的
      双箭头 —— **箭头的长度就是那个数**，不是两个点之间的直线距离（两点连起来不一定
      平行于轴，那样量出来的是另一回事）。两段重叠 = 压上了，画成红色，长度是压进去
      的深度。
    · **轴偏（绿）**：方块顶面中心 `box_uv` 到合爪轴那条直线的**垂足**，连一条线 +
      直角记号 —— 长度就是 `_jaw_check` 报的 `axis`，正负只看在轴的哪一侧。

    文字全 ASCII（`cv2.putText` 不认中文）。
    """
    import cv2                                                   # noqa: PLC0415

    def _p(v) -> tuple[int, int]:
        return int(round(float(v[0]))), int(round(float(v[1])))

    def _text(s, org, col):
        # **贴边就往回收**：字比画幅宽时 `cv2.putText` 直接画到框外，右边那几个字没了。
        (tw, _th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x = min(max(8, int(org[0])), max(8, img.shape[1] - tw - 12))
        y = min(max(_th + 8, int(org[1])), img.shape[0] - 12)
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)

    u = np.asarray(u, dtype=np.float64).reshape(2)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    n_hat = np.array([-u[1], u[0]], dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2)
                        for p in polys], axis=0)
    if pts.size == 0 or q.size == 0:
        return

    # ---- 缝：两端对着对方的点，连一条沿合爪轴的线 ----
    if gap_px is not None and np.isfinite(float(gap_px)):
        s_b, s_j = pts @ u, q @ u
        r0, r1 = float(s_b.min()), float(s_b.max())
        j0, j1 = float(s_j.min()), float(s_j.max())
        if r1 < j0:                       # 方块在轴负侧、爪子正侧，中间是缝
            p_b, p_j, col = pts[int(np.argmax(s_b))], q[int(np.argmin(s_j))], (0, 220, 220)
        elif j1 < r0:                     # 反过来
            p_b, p_j, col = pts[int(np.argmin(s_b))], q[int(np.argmax(s_j))], (0, 220, 220)
        else:                             # 压上了：画重叠段，红的
            if (r1 - j0) <= (j1 - r0):
                p_b, p_j = pts[int(np.argmax(s_b))], q[int(np.argmin(s_j))]
            else:
                p_b, p_j = pts[int(np.argmin(s_b))], q[int(np.argmax(s_j))]
            col = (0, 0, 255)
        if not gap_gated:
            # 合爪轴根本没穿过方块：这个数是「离得多远」，不是缝。**照样画出来**
            # （用户 2026-10-19：画面上要看得到是怎么算的），换个颜色、字里写明作废。
            col = (0, 140, 255)
        # 过两端的 ⊥u 刻线；尺寸线偏出去 90px、方向平行于 u —— 它才是「沿轴的距离」。
        # 两端沿 n̂ 各偏多远都一样长（两条刻线平行，沿轴错开的量是定值），所以取同一个偏移。
        for p in (p_b, p_j):
            cv2.line(img, _p(p - 90.0 * n_hat), _p(p + 90.0 * n_hat), col, 2, cv2.LINE_AA)
        q_b = p_b + 90.0 * n_hat
        q_j = q_b + float(np.dot(p_j - p_b, u)) * u
        if float(np.linalg.norm(q_j - q_b)) >= 2.0:
            cv2.arrowedLine(img, _p(q_b), _p(q_j), col, 3, cv2.LINE_AA, tipLength=0.12)
            cv2.arrowedLine(img, _p(q_j), _p(q_b), col, 3, cv2.LINE_AA, tipLength=0.12)
        mid = 0.5 * (q_b + q_j)
        org = _p(mid + 30.0 * n_hat)
        _text(f"gap {float(gap_px):+.0f}px" + ("" if gap_gated else " (RULER OFF)"),
              (org[0], org[1]), col)

    # ---- 轴偏：方块中心到合爪轴的垂足 ----
    if axis_px is not None and np.isfinite(float(axis_px)):
        c = np.asarray(box_uv, dtype=np.float64).reshape(2)
        ref = np.asarray(ref, dtype=np.float64).reshape(2)
        foot = ref + float(np.dot(c - ref, u)) * u
        cv2.line(img, _p(c), _p(foot), (0, 200, 0), 4, cv2.LINE_AA)
        cv2.circle(img, _p(c), 6, (0, 200, 0), -1, cv2.LINE_AA)
        # 直角记号：垂足处两个方向各回 14px 画个小方框
        d1 = foot - c
        d1 = d1 / max(float(np.linalg.norm(d1)), 1e-12)
        d2 = u if float(np.dot(u, c - foot)) >= 0.0 else -u
        a = foot - 14.0 * d1
        b = a + 14.0 * d2
        e = foot + 14.0 * d2
        cv2.polylines(img, [np.round(np.array([a, b, e])).astype(np.int32)], False,
                      (0, 200, 0), 3, cv2.LINE_AA)
        mid = 0.5 * (c + foot)
        org = _p(mid - 30.0 * u)
        _text(f"axis {float(axis_px):+.0f}px", (org[0], org[1]), (0, 200, 0))


_ASCII_TAGS = {
    "下探前": "DESCEND",
    "下探到底（合爪前）": "BEFORE CLOSE",
    "下探之后（只修轴偏）": "AFTER DESCEND",
    "重放": "REPLAY",
    "对正之后停住": "HOLD AFTER ALIGN",
}


def _ascii_tag(tag: str) -> str:
    """图上那行字只能是 ASCII（cv2 没有 freetype，中文全渲染成 `?`）。终端日志不改。"""
    import re                                                   # noqa: PLC0415

    t = str(tag)
    if t.isascii():
        return t
    hit = _ASCII_TAGS.get(t)
    if hit is not None:
        return hit
    for pat, fmt in ((r"^(.*)（微调第 (\d+) 次后）$", "{} after fine-tune {}"),
                     (r"^(.*)（手动第 (\d+) 步后）$", "{} after manual move {}")):
        m = re.match(pat, t)
        if m:
            return fmt.format(_ascii_tag(m.group(1)), m.group(2))
    return re.sub(r"[^\x20-\x7e]", "", t).strip() or "CHECK"


def _jaw_check(cam, percep, arm, tag: str, axis_only: bool = False) -> dict:
    """合爪前那一眼：红区离**整圈爪子轮廓**多少像素、目标中心离合爪轴多少像素。只读。

    · `gap`：`real/perception.py` 选中的那块红区的轮廓里**最靠里的那一个点**，到描好的爪子
      轮廓（`fixed_jaw` 必有，`moving_closed` 描了就算）的带号距离，取负值 ——
      **正 = 整块红区都在爪子外面（留了缝），负 = 有红像素探进爪子里（接触了）**。
      **哪一圈被探得最狠就报哪一圈** —— 压上去的可能是固定爪，也可能是动爪合过来那一下
      落的地方。
    · `axis`：**方块顶面中心**（`_box_center_px`，不是整块重心）离合爪轴
      （`_trace_jaw.py` 拉的那条线）的带号垂直距离。正负只表示在轴的哪一侧。
    · `along`：同一个重心沿轴的方向走了多远（打印出来给人看，不参与判据）。

    量不出来（没有够格的红区 / 轮廓的文件不在 / 画幅对不上）返回 `{'ok': False,
    'why': ...}` —— 调用方按「这一趟判失败」办，不再像原来那样「量不了也照常合爪」。
    """
    import cv2                                                  # noqa: PLC0415

    from real.perception import brown_mask, candidates           # noqa: PLC0415

    if cam is None:
        return {"ok": False, "why": "没有腕相机这一路"}
    try:
        jaw = _jaw_shape(cam)
    except RuntimeError as e:
        return {"ok": False, "why": str(e)}

    rgb = np.asarray(cam.shot())
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(JAW_RAW_IMG), bgr)          # 画之前先留一张干净的，离线复算用
    # 掩膜口径跟感知那条链必须一样（同一份 HSV 区间、同一个最小面积、同一个矩形度门槛），
    # 不然量的是「另一块红区」。
    mask = brown_mask(bgr, percep.ranges)
    cands = [d for d in candidates(mask, percep.min_area)
             if d["rect_fill"] >= percep.fill_min]
    if not cands:
        cv2.imwrite(str(knobs.JAW_CHECK_IMG), bgr)
        return {"ok": False, "why": "画面里没有过得去的红区"}

    cnt = cands[0]["contour"]
    pts = _contour_pts(cnt)
    polys = {k: np.asarray(jaw[k], dtype=np.float32).reshape(-1, 1, 2)
             for k in ("fixed_jaw", "moving_open", "moving_closed") if k in jaw}
    if "fixed_jaw" not in polys:
        return {"ok": False, "why": f"{JAW_SHAPE_JSON} 里没有 `fixed_jaw`"}
    raw, near, which = None, None, ""
    for name, poly in polys.items():
        d, q = _gap_px(poly, pts)
        if raw is None or d > raw:          # 哪一圈被探得最狠就报哪一圈
            raw, near, which = d, q, name
    # `_gap_px` 给的是 `pointPolygonTest` 的原值（正 = 红区探进爪子轮廓里）。这里翻成
    # 「红区离爪子轮廓外面多少像素」—— 正 = 留了缝。判据要的是它**落在带里**（见
    # `CLEAR_BAND_*`），只有 `CLEAR_BAND_ON = False` 时才是「越大越好」。
    gap = -float(raw)

    m = cv2.moments(cnt)
    if abs(float(m["m00"])) < 1e-9:
        return {"ok": False, "why": "红区退化成一个点，量不出中心"}
    c_uv = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float64)
    box_uv = _box_center_px(cnt, bgr)
    if box_uv is None:
        return {"ok": False, "why": "红区里亮暗分不开（或亮块太小），量不出方块顶面中心"}
    band_uv = np.concatenate([polys[k].reshape(-1, 2) for k in polys],
                             axis=0).mean(axis=0)
    ref = np.asarray(jaw["axis_ref_px"], dtype=np.float64)
    axis_dir = np.asarray(jaw["axis_dir_px"], dtype=np.float64)
    axis_dir = axis_dir / max(float(np.linalg.norm(axis_dir)), 1e-12)
    n_hat = np.array([-axis_dir[1], axis_dir[0]], dtype=np.float64)
    rel = box_uv - ref
    axis = float(np.dot(rel, n_hat))
    along = float(np.dot(rel, axis_dir))
    # **第二条缝的尺子：整块退出去还要多少像素。** 见 `_exit_px` 的说明 —— `gap` 只看
    # 最近的那一个点，方块大半压在轮廓里时它会说「只差几个像素」。
    gap_exit = _exit_px(list(polys.values()), pts, axis_dir)
    # **拦门的那条缝换成「沿合爪轴的净空」**（用户 2026-10-16 定：量上面那段缝是错的）。
    # 最近距离那条（`gap`）照量照打，只留作参考。
    gap_axis = _gap_axis_px(list(polys.values()), pts, axis_dir)
    gap_raw = gap_axis             # 轴没穿过方块时拦门那条作废，但图上照样标这个数
    gap_near = gap

    # **量缝之前先看合爪轴有没有真的对着方块**（用户 2026-10-17）：「首先得要让夹取方向
    # 在目标像素里面，咱不能离目标太远开始量缝吧」。把红区投到 ⊥合爪轴 那一维上，看
    # 合爪轴自己那条线（偏移 0）在不在方块横跨的那一段里。不在 = 轴根本没穿过它，这时
    # 沿轴量出来的那个数是「离得多远」，不是「缝」—— 这条缝不认，按「量不出来」办。
    _lat = (pts - ref) @ n_hat
    axis_in = bool(float(_lat.min()) <= 0.0 <= float(_lat.max()))
    if not axis_in:
        print(f"    {tag}缝不量：合爪轴没穿过方块（方块横跨 "
              f"{float(_lat.min()):+.0f}~{float(_lat.max()):+.0f} 像素，合爪轴在 0 那一条）"
              f" —— 离得太远，沿轴量出来的不是缝")
        gap_axis = None

    # **`axis_only=True`（下探之后那一趟）缝这条不拦门**（用户 2026-10-17：「下探后的微调
    # 无需调整缝隙……微调如果第一眼就符合就不用调整了」）。判据只剩轴偏这一条，缝照量照打。
    #
    # 原来这里一律 `缝 且 轴偏` 一起判，`ok` 到 `_clearance_loop` 里再被改回来 —— 于是
    # **打出来的那一行说的是错的**。实测 `out/` 那一轮的面板日志（2026-10-17 03:3x）：
    # `下探之后（只修轴偏）轮廓：缝 +10 像素（要落在 [20, 75] 里）  轴偏 +9 像素（容 ≤35）
    # → 不过` —— 轴偏 +9 明明在容内，那一行却报「不过」，就是因为缝在替它拦门。
    gap_ok = bool(gap_axis is not None and _gap_ok(gap_axis))
    ok = bool(abs(axis) <= AXIS_TOL_PX) if axis_only else bool(
        gap_ok and abs(axis) <= AXIS_TOL_PX)
    # **缝这条要落在带里**（2026-10-16 用户定）：低于下限 = 爪面压上方块，高于上限 = 方块
    # 已经退出两爪之间。老口径只有下限，所以退得越远越「合格」—— 那正是空抓的那一半。
    want = (f"要落在 [{CLEAR_BAND_LO_PX:.0f}, {CLEAR_BAND_HI_PX:.0f}] 里＝方块在两爪之间"
            if CLEAR_BAND_ON else f"要 ≥{CLEAR_TARGET_PX:.0f}")
    if axis_only:
        want = f"只报数（这一趟不拦门；拦门的是轴偏 ≤{AXIS_TOL_PX:.0f}）"
    _gap_txt = "—（合爪轴没穿过方块）" if gap_axis is None else f"{gap_axis:+.0f}"
    print(f"    {tag}轮廓：缝 {_gap_txt} 像素（{want}，**沿合爪轴**"
          f"量；最近距离那条 {gap_near:+.0f}、最近的是 {which}；整块退出去还要 {gap_exit:.0f}）"
          f"  轴偏 {axis:+.0f} 像素（容 ≤{AXIS_TOL_PX:.0f}）"
          f"  沿轴 {along:+.0f}  方块中心 ({box_uv[0]:.0f},{box_uv[1]:.0f})"
          f"  → {'过' if ok else '不过'}")

    # 画一张：这一刻的画面 + 描好的轮廓 + 合爪轴 + 红区轮廓 + 读数。
    # **固定文件名，覆盖**。检查没过、或者事后想知道「那一眼看成什么样」，看它。
    #
    # **这一张不画任何「中心」**（用户 2026-10-17）：「其他点比如顶面中心，其他心啥的都不
    # 用标在图上」。原来这里画了三个 —— 绿十字（画面那条尺量的顶面中心）、灰小圆（红区质
    # 心）、白圈（红区轮廓离爪最近那一点）。要看的那个中心只有一个：3D 那条尺子冻住的方块
    # 中心，由 `_mark_frozen` 在这一张上画成白色小三角。图上少三个点，剩下的白三角反而好找。
    col = {"fixed_jaw": (255, 255, 0), "moving_open": (0, 255, 0),
           "moving_closed": (0, 165, 255)}
    for name, poly in polys.items():
        cv2.polylines(bgr, [poly.astype(np.int32)], True, col[name], 3, cv2.LINE_AA)
    far = 0.8 * float(bgr.shape[1])
    cv2.line(bgr, tuple(np.round(ref - far * axis_dir).astype(int)),
             tuple(np.round(ref + far * axis_dir).astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
    cv2.drawContours(bgr, [cnt], -1, (0, 0, 255), 2)
    # **把「缝」「轴偏」怎么量的画出来**（用户 2026-10-19：以后画面里都要标）。黄 = 缝
    # 那一段（沿合爪轴、方块边到爪边）、绿 = 轴偏那一段（顶面中心到合爪轴的垂线），
    # 跟 `clear_step` 那张图的颜色口径一致。只画量法，读数照旧。
    _draw_measures(bgr, pts, list(polys.values()), ref, axis_dir, box_uv,
                   gap_raw, axis, gap_gated=gap_axis is not None)
    # **`gap_axis` 可能是 `None`**（合爪轴没穿过方块那一眼，见上面那段）—— 按浮点格式化
    # 会直接抛 `TypeError: unsupported format string passed to NoneType.__format__`，
    # 把整趟技能带走。这一行跟上面那行 `_gap_txt` 一个口径：量不出来就写 `--`。
    _gap_img = "--" if gap_axis is None else format(float(gap_axis), "+.0f")
    txt = (f"{_ascii_tag(tag)}  gap {_gap_img}px "
           f"({'want ' + format(CLEAR_BAND_LO_PX, '.0f') + '..' + format(CLEAR_BAND_HI_PX, '.0f') + ' = between jaws' if CLEAR_BAND_ON else 'want >=' + format(CLEAR_TARGET_PX, '.0f')}, along the jaw axis)"
           f"   axis {axis:+.0f}px (tol {AXIS_TOL_PX:.0f})   {'PASS' if ok else 'FAIL'}")
    # **写在下边**：`_confirm_step` 会读同一张图、往顶上写它那几行（见 `_put_lines`）。
    # **图例**（用户 2026-10-19：图上的缝/轴偏要写清楚是怎么算的）。图上那两个数旁边
    # 只放数值，量法集中在底下这一行 —— 放段上会把字撑出画幅、两条也会叠在一起。
    _gap_drawn = gap_raw is not None and bool(np.isfinite(float(gap_raw)))
    legend = ("axis (green) = box top-face center -> jaw axis, perpendicular"
              + ("   |   gap (yellow) = jaw edge -> box edge, along jaw axis"
                 if _gap_drawn and gap_axis is not None else "")
              + ("   |   gap (orange) = same ruler, but jaw axis misses the box"
                 " -> NOT used" if _gap_drawn and gap_axis is None else ""))
    _put_lines(bgr, [txt, legend], (0, 200, 0) if ok else (0, 0, 255), bottom=True)
    cv2.imwrite(str(knobs.JAW_CHECK_IMG), bgr)

    return {"ok": ok, "gap": None if gap_axis is None else float(gap_axis),
            "axis_in": axis_in, "gap_near": float(gap_near),
            "gap_exit": float(gap_exit), "which": which,
            "axis": axis, "along": along,
            "box_uv": box_uv, "c_uv": c_uv, "band_uv": band_uv, "near_uv": near,
            "axis_ref_px": ref, "axis_dir_px": axis_dir}

def _mark_frozen(cam, p_world) -> None:
    """**离轴要对准的那个中心** —— 冻住的 3D 方块中心，投影到画面里画成**白色小三角**。

    它就是 `_jaw_check_3d` 量离轴用的那个点（`p = (center - tcp)·n`，目标 0 = 把方块中心
    摆到合爪轴上）。用户 2026-10-17：「你把离轴要对准的目标中心用小三角形标出来，用白色，
    字搞的大一点，其他点比如顶面中心，其他心啥的都不用标在图上给我看了」。

    所以这张图上**只有这一个点**：青/绿/橙三圈是爪轮廓、品红长线是合爪轴、红描边是红区
    轮廓 —— 都是线段不是「中心」；绿十字／灰小圆／白圈那三个中心已经撤掉（见 `_jaw_check`）。
    投不出来就不画。
    """
    import cv2                                                  # noqa: PLC0415

    uv = _project_px(cam, p_world)
    if uv is None:
        return
    img = cv2.imread(str(knobs.JAW_CHECK_IMG))
    if img is None:
        return
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    # 先黑后白：盒子是深棕、桌面也偏暗，只用白的边容易糊掉。
    for col, w in (((0, 0, 0), 6), ((255, 255, 255), 2)):
        cv2.drawMarker(img, (u, v), col, cv2.MARKER_TRIANGLE_UP, 16, w, cv2.LINE_AA)
    for col, w in (((0, 0, 0), 6), ((255, 255, 255), 2)):
        cv2.putText(img, "OFF-AXIS TARGET", (u + 20, v - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, w, cv2.LINE_AA)
    cv2.imwrite(str(knobs.JAW_CHECK_IMG), img)

def _save_look(cams, facts) -> None:
    """把「看一眼」那一帧存下来（带 `real.perception.draw` 那层标注），写 `out/look_see.jpg`。

    只写图，**不动任何判据**。这一眼就是冻结那个 3D 中心的来源（`run` 里进场之后、纠位
    之前那一档），事后要复盘「中心是按哪张图定的」只能看它。
    """
    import cv2                                                  # noqa: PLC0415

    from real.perception import draw                            # noqa: PLC0415

    cam = _wrist(cams)
    if cam is None:
        print("    （存这一眼的画面：没有腕相机这一路）")
        return
    try:
        bgr = cv2.cvtColor(np.asarray(cam.shot()), cv2.COLOR_RGB2BGR)
        LOOK_IMG.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(LOOK_IMG), draw(bgr, facts or {}))
    except Exception as e:                                      # noqa: BLE001
        # 存图失败不该把这一趟弄死，但**不许闷掉** —— 把原文打出来。
        print(f"    （存这一眼的画面失败：{type(e).__name__}: {e}）")
        return
    print(f"    这一眼的画面（带标注）-> {LOOK_IMG}")

def _jaw_mm(arm, center, clear, tag: str, *, extra: str = "", cam=None) -> dict:
    """合爪轴（`arm.tips()` 正解）跟**给定的**一个方块中心（3D、底座系）摆到一起。

    返回 `p`（离轴，米）、`ds`（沿轴 − 该在的 `clear`，米）、`delta`（该把 TCP 挪多少，
    水平、米）、`ok`。**这是 2026-10-15 起拦门的那条尺子**（`CLEAR_3D_ENFORCE`）：
    `|p| <= CLEAR_3D_TOL_MM` 且 `|ds| <= CLEAR_3D_TOL_MM`。

    `center` 由调用方给。微调那一支给的是**冻住的那一个**（下探之前那一眼量到的），
    **不再从可见红区轮廓上现取** —— 那个点会被爪子挡住而乱跳（实测同一趟里 34 像素）。
    合爪轴只用关节角正解，跟画面无关，挡不挡都一样。
    """
    f, m = arm.tips()
    u = np.asarray(m, dtype=np.float64) - np.asarray(f, dtype=np.float64)
    u[2] = 0.0
    nu = float(np.linalg.norm(u))
    if nu < 1e-9:
        return {"ok": False, "no_world": True, "why": "两爪尖水平投影重合，量不出合爪轴"}
    u = u / nu
    n = np.array([-u[1], u[0], 0.0], dtype=np.float64)

    center = np.asarray(center, dtype=np.float64).reshape(-1)[:3]
    tcp = np.asarray(arm.tcp(), dtype=np.float64)
    r = center - tcp
    s = float(np.dot(r, u))                     # 沿合爪轴，从固定爪尖算起（目标 +clear）
    p = float(np.dot(r, n))                     # 离轴（目标 0）
    s_want = float(clear)                       # 进嘴那一侧 = +u，**不乘 side**（见下面那条）
    ds = s - s_want
    tol = CLEAR_3D_TOL_MM / 1000.0
    # **3D 这边只拦「离轴」那一维**（用户 2026-10-17：「新版离轴+旧版的缝」）。
    # 沿轴那维照量照打、**不在这里拦** —— 缝的判据换回**旧版画面那条单边下限**
    # （`gap >= CLEAR_TARGET_PX`，无上限），在 `_clearance_loop` 里跟这条并起来判。
    ok = bool(abs(p) <= tol)
    print(f"    {tag} 3D：离轴 {p * 1000:+.1f}mm（容 ≤{CLEAR_3D_TOL_MM:.0f}）  "
          f"沿轴 {s * 1000:+.1f}mm（该 {s_want * 1000:+.1f}，差 {ds * 1000:+.1f}）"
          f"  方块中心 {np.round(center * 1000, 1)}mm{extra}"
          f"  → 离轴这边{'过' if ok else '不过'}"
          + ("" if CLEAR_3D_ENFORCE else "（**只报数，拦门的是画面那两条**）"))
    if cam is not None:
        # 让这一眼用的那个「中心」在图上有个标记 —— 不画的话没人知道它落在盒子哪儿。
        _mark_frozen(cam, center)
    delta = p * n + ds * u
    return {"ok": ok, "s": s, "p": p, "ds": ds, "s_want": s_want, "delta": delta,
            "center": center, "u": u, "n": n, "tcp": tcp}

def _jaw_check_3d(arm, cam, percep, instruction, guess, clear, tag: str) -> dict:
    """下探到底那一刻的 **3D 判据**：方块中心离实测出来的合爪轴、沿轴各差多少毫米。

    三个输入都不是新东西：
      · **合爪轴** —— 读回六个关节角（`arm.tips()` 自己读），取两个爪尖的水平连线当
        `u`（固定爪 → 动爪）、它的水平垂线当 `n`；
      · **方块中心** —— 走线上同一条感知链 `percep.locate`（它内部扣掉
        `real.percep_offset_mm`，所以跟 `arm.tcp()` 在同一个坐标系里）；
      · **该在哪** —— 就是下探那一段已经算好的 `aim`：`(center − tcp)` 沿轴该等于
        `+clear`、离轴该是 0。**不另发明目标点。**

    **沿轴那个目标点就是 `+clear`，不带符号。** `u` 按定义从固定爪尖指向动爪尖，方块该
    待在**嘴里面**、固定爪尖前 `clear`（半边长 + 爪面探出 + 缝）—— 就是 `aim` 那条算式
    `center + side·clear·xt` 挪出来的那一点。**2026-10-15 第一版这里写成 `−side·clear`，
    修正量整个发反了**：这台机器上 `arm.tool_x()` 跟 `u` 反向（`xt·u = −1`），乘上去才等于
    `aim`；不乘就成了「把方块放到固定爪背后」。日志证据：第一眼打「该 −10.8」，而按几何
    该是 `+10.8`，于是 TCP 朝抓取方向走了 5.5mm，实际要朝反方向退 16.1mm。

    `guess` 要的是**已经扣过偏移的**方块中心（上一眼量出来的那个），本函数内部把偏移加
    回去再喂给 `locate` —— `locate` 的种子在扣偏移之前那一侧（`real/perception.py:351-359`），
    拿扣过的直接当种子会差那 35.8mm。

    量不出来（方块被爪挡掉 / 没有够格的棕区）返回 `{'ok': False, 'no_world': True}`，
    调用方不当回事 —— 它只报数。
    """
    if cam is None:
        return {"ok": False, "no_world": True, "why": "没有腕相机这一路"}
    seed = np.asarray(guess, dtype=np.float64).copy() if guess is not None else None
    if seed is not None:
        seed[:2] = seed[:2] + geom.real_percep_offset()
    center, facts = percep.locate(instruction, guess=seed)
    if center is None:
        why = facts.get("no_world_why") or facts.get("why") or "没认出来"
        return {"ok": False, "no_world": True, "why": why, "iou": facts.get("iou")}
    # **每眼现量**的那一版。微调那一支不用它 —— 用 `_jaw_mm` 加冻住的那个中心。
    out = _jaw_mm(arm, center, clear, tag, cam=cam,
                  extra=f"  重合度 {facts.get('iou')}  估的形状 {facts.get('size')}")
    out["facts"] = facts
    return out

def _solve_delta(cam, center, m):
    """按画面里的偏差反解出「臂该动多少毫米」。返回 `(Δp 像素, δ 毫米)`，算不出来给 `None`。

    `Δp = ref − 方块顶面中心`：把方块顶面中心挪回描轮廓时点的那一点（`axis_ref_px`，
    「方块该被夹住的那个中心」）。沿合爪轴那一段把它拉回理想中心（爪朝画面**右边**退），
    垂直那一段把它拉回轴上。两段都到 0，就是「方块顶面中心正好落在该被夹住的那个中心上」。

    **画面里怎么换到臂上。** 臂一动，爪子跟相机是一起走的 —— 画面里动的只有目标，位移
    ≈ `−J·δ`（`J` = `vision.px_per_m_xy` 在目标那一点上的**带号** 2×2）。反解
    `δ = −J⁻¹·Δp`。**这正是原来那版栽过的地方**：它拿 `px_per_m` 的模长当尺子、方向靠
    `side` 猜，`out/run_real_37.log` 里读数 14 → 16 → 17 一路往外跑，指令和读数方向是反的。

    这个是**解出来的量**，自动那一路只拿它当方向（`_nudge` 用定步长走），手动那一路把它
    当建议值摆进 `out/manual_state.jsonl` 给操作员看。
    """
    dp = (np.asarray(m["axis_ref_px"], dtype=np.float64)
          - np.asarray(m["box_uv"], dtype=np.float64))
    if float(np.linalg.norm(dp)) < 1e-6:
        return None
    want = _want_from_dp(cam, center, dp)
    if want is None or float(np.linalg.norm(want)) < 1e-9:
        return None
    return dp, want


def _want_from_dp(cam, center, dp):
    """`dp` 像素 → 底座系该走的米：`δ = −J⁻¹·dp`。

    **全仓只此一份。** 技能里的 `_solve_delta` 和面板里的 `panel/measure._want_from_dp`
    都走它 —— 面板说的「该往左 5mm」和机械臂真走的那 5mm 必须是同一个 5mm。
    """
    jac = _px_per_mm(cam, center)
    if jac is None:
        return None
    try:
        inv = np.linalg.inv(jac)
    except np.linalg.LinAlgError:
        return None
    return -inv @ np.asarray(dp, dtype=np.float64)


def _axis_seg(dp, axis_dir_px):
    """`dp` 里**只留垂直合爪轴那一段**。

    `n_hat` 是垂直于 `axis_dir_px` 的单位向量；`(dp·n_hat)·n_hat` 把轴偏留下、
    沿轴那一段丢掉。面板「到底之后只修轴偏」和技能里拼 `seg_axis` 用的是同一条。
    """
    u = np.asarray(axis_dir_px, dtype=np.float64)
    n = np.array([-u[1], u[0]], dtype=np.float64)
    return float(np.dot(np.asarray(dp, dtype=np.float64), n)) * n
