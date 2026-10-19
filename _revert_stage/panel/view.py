"""画那一层：把 `panel/measure.py` 算出来的东西标到画面上。**只画，不算。**

这一层不读硬件、不下命令、不判阈值。它拿一份快照（`Rig.snap()` 那个 dict），把里面
已经算好的读数画成看得见的东西 —— 画错了不影响机械臂，量错了才影响。

**画面上的字全是 ASCII（英文）。** `cv2.putText` 只认 ASCII，写中文会画成问号；
这套 cv2 里也没有 `freetype`（试过：`hasattr(cv2, "freetype")` 是 False），没有别的
写字路径。所以标签一律用英文，中文说法在 `panel/ui.py` 的窗口中不出现、只在终端日志里。
关节名直接用 `JOINT_ORDER` 里那套 —— 仓库里本来就是英文（`shoulder_pan` 等）。

标什么，以及每一样为什么值得标：

    `fixed_jaw` / `moving_open` / `moving_closed`  爪子在画面里的三圈轮廓。
        缝和轴偏全是相对它们量的，不画出来就没法核对读数是不是对的。
    `axis_ref_px` / `axis_dir_px`                 合爪轴的**中心**和**方向**。
        方向就是「夹取方向」（`_trace_jaw.py` 第二下点的那条）；中心是「方块该被夹在
        这儿的那个点」。`axis` 就是方块顶面中心到这条轴的垂直距离。
    方块轮廓 + 顶面中心                                感知选中的那一块和它顶面的中心。
        **判据用的是顶面中心，不是整块重心**（斜视下差 22 像素，见 `_box_center_px`）。
    缝的标尺                                        沿合爪轴，爪轮廓近端 → 方块近端。
        两端的刻度就是 `gap`；再往前标出目标带 `[LO, HI]` —— 方块近端该落在那两条
        虚线之间。这是整个面板最该看的一条：**空抓就是它不在带里。**
    垂直修正段                                      方块顶面中心 → 合爪轴上的垂足。
        这一段就是 `axis`，也是 `dp_px` 的方向。
"""

from __future__ import annotations

import cv2
import numpy as np

from skills import pick_real as PR
from panel import measure

# 颜色全按 BGR。
C_BOX = (0, 235, 0)          # 绿：方块轮廓 / 顶面中心
C_PTS = (110, 255, 110)      # 浅绿：重采样过的轮廓点列
C_JAW_FIX = (255, 190, 0)    # 蓝青：固定爪
C_JAW_OPEN = (0, 190, 255)   # 橙：动爪（张着那圈）
C_JAW_CLOSED = (220, 60, 220)  # 紫：动爪（合上那圈）
C_AXIS = (0, 235, 235)       # 黄：合爪轴（那条线）
C_REF = (255, 0, 255)        # 洋红：合爪轴上的参考中心
C_GRASP = (255, 255, 255)    # 白：夹取方向
C_GOOD = (0, 235, 0)
C_BAD = (60, 60, 255)
C_WARN = (0, 200, 255)
C_BAND = (0, 235, 235)
C_3D = (235, 190, 0)         # 青：3D 那条尺子比的那个点（体中心投回画面）

PHASE_EN = {
    measure.PHASE_LOST: ("LOST", C_BAD),
    measure.PHASE_SEE: ("SEE", C_WARN),
    measure.PHASE_ALIGN: ("ALIGN", C_WARN),
    measure.PHASE_DESCEND: ("DESCEND", C_AXIS),
    measure.PHASE_CLOSE: ("CLOSE", C_GOOD),
    measure.PHASE_LIFT: ("LIFT", C_GOOD),
}


# ---- 画字的底座 ----

def text(img, s: str, org, color, scale: float = 0.55, thick: int = 2) -> None:
    """黑描边 + 本色，压在相机画面上也读得清。"""
    for col, w in (((0, 0, 0), thick + 3), (color, thick)):
        cv2.putText(img, s, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, col, w, cv2.LINE_AA)


def text_block(img, org, lines, *, line_h: int = 20, scale: float = 0.52) -> None:
    """左上角那一块读数。**先铺一层半透明黑** —— 画面是实拍的，不铺底色字会糊。"""
    lines = [(t, c) for t, c in lines if t]
    if not lines:
        return
    w = max(len(t) for t, _ in lines)
    pad, x0, y0 = 8, int(org[0]), int(org[1])
    x1 = x0 + int(w * scale * 13) + 2 * pad
    y1 = y0 + line_h * len(lines) + pad
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, roi)
    for i, (t, c) in enumerate(lines):
        text(img, t, (x0 + pad, y0 + pad + line_h * (i + 1) - 5), c, scale)


# ---- 零件 ----

def dot(img, p, color, r: int = 5) -> None:
    cv2.circle(img, (int(round(float(p[0]))), int(round(float(p[1])))), r, color, -1,
               cv2.LINE_AA)


def star(img, p, color, size: int = 13) -> None:
    q = (int(round(float(p[0]))), int(round(float(p[1]))))
    cv2.drawMarker(img, q, color, cv2.MARKER_STAR, size * 2, 2, cv2.LINE_AA)
    cv2.circle(img, q, size, color, 2, cv2.LINE_AA)


def double_arrow(img, p, u, half: float, color, thick: int = 2) -> None:
    """在 `p` 处沿 `u` 拉一条双头箭头 —— 用来示「夹取方向」。"""
    a, b = p - u * half, p + u * half
    a, b = a.astype(int), b.astype(int)
    cv2.arrowedLine(img, tuple(a), tuple(b), color, thick, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(img, tuple(b), tuple(a), color, thick, cv2.LINE_AA, tipLength=0.18)


def _u_coord_line(img, through, u, c: float, half: float, color, *, dashed=False,
                  thick: int = 2) -> None:
    """在「沿 `u` 的坐标 = `c`」那个位置，画一条垂直于 `u` 的线。

    `through` 只用来定坐标原点（任何一个点都行），长度是 `half` 往两边各伸。
    """
    u = np.asarray(u, dtype=np.float64)
    p0 = np.asarray(through, dtype=np.float64)
    p = p0 + (float(c) - float(np.dot(p0, u))) * u
    n = np.array([-u[1], u[0]], dtype=np.float64)
    a, b = p - n * half, p + n * half
    if not dashed:
        cv2.line(img, a.astype(int), b.astype(int), color, thick, cv2.LINE_AA)
        return
    seg = np.linalg.norm(b - a)
    k = max(int(seg / 10.0), 2)
    for i in range(0, k, 2):
        s = a + (b - a) * (i / k)
        e = a + (b - a) * (min(i + 1, k) / k)
        cv2.line(img, s.astype(int), e.astype(int), color, thick, cv2.LINE_AA)


# ---- 主入口 ----

def draw(bgr: np.ndarray, snap: dict) -> np.ndarray:
    """把这一份快照画到这一帧上。**返回的是新图**，原帧不动。"""
    img = np.array(bgr, copy=True)
    m, box, step = snap.get("m"), snap.get("box"), snap.get("step")

    if snap.get("moving"):
        # 走位途中：算出来的读数是上一处机位的，标上去是骗人。
        text_block(img, (12, 12), [("MOVING", C_AXIS),
                                   ("metrics frozen - camera is on the move", C_WARN)])
        _draw_footer(img, snap)
        return img

    if m and m.get("ok"):
        _draw_jaws(img, m["polys"])
        _draw_axis(img, m)
    if m and m.get("ok"):
        cv2.drawContours(img, [m["contour"]], -1, C_BOX, 2, cv2.LINE_AA)
        pts = m["pts"].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [pts], False, C_PTS, 1, cv2.LINE_AA)
        _draw_gap(img, m)
        # **两条尺子各比各的点，两个都标，但标小**（用户 2026-10-17：「不用把顶面中心
        # 标记的那么大，你比什么就标记什么」）：
        #   · 绿色的 = **画面那条尺子**比的那个点（`_box_center_px` 分出来的顶面中心，
        #     `axis` 就是从这个点量到合爪轴的）；
        #   · 青色的 = **3D 那条尺子**比的那个点（`percep.locate` 给的体中心投回画面，
        #     `p`/`ds` 是从它量的），由 `panel/rig.py` 投好放进 `box.uv3d`。
        star(img, m["box_uv"], C_BOX, 7)
        text(img, f"TOP CENTER (picture ruler) {m['box_uv'][0]:.0f},{m['box_uv'][1]:.0f}",
             m["box_uv"] + np.array([14.0, -12.0]), C_BOX, 0.45, 1)
        dot(img, m["c_uv"], (180, 180, 180), 3)      # 整块重心，对照用
        text(img, "bbox centroid", m["c_uv"] + np.array([14.0, 18.0]),
             (180, 180, 180), 0.45, 1)
    elif box is not None and snap.get("facts", {}).get("uv") is not None:
        uv = np.asarray(snap["facts"]["uv"], dtype=np.float64)
        dot(img, uv, C_WARN, 6)
        text(img, "box (no metrics)", uv + np.array([14.0, -14.0]), C_WARN, 0.5, 1)

    _draw_center3d(img, snap)
    _draw_hud(img, snap, m, step)
    _draw_footer(img, snap)
    return img


def _draw_center3d(img, snap) -> None:
    """**3D 那条尺子比的那个点**（体中心投回画面）。只画，不算。

    `p`（离轴）和 `ds`（沿轴）都是从它量出来的；画面那条尺子比的却是`_box_center_px`
    给的**顶面中心**。两个点在画面里不是同一个像素（腕机斜看，顶面中心往相机那头偏），
    所以两个都标 —— 只看一个的话，很难看出两边的读数为什么会对不上。

    `uv3d` 由 `panel/rig.py` 投好放进快照；没有（投不出来 / 没认出来）就不画。
    """
    uv = (snap.get("box") or {}).get("uv3d")
    if uv is None:
        return
    uv = np.asarray(uv, dtype=np.float64).reshape(-1)[:2]
    dot(img, uv, C_3D, 5)
    cv2.circle(img, (int(round(float(uv[0]))), int(round(float(uv[1])))), 9, C_3D, 1,
               cv2.LINE_AA)
    text(img, f"3D CENTER (mm ruler) {uv[0]:.0f},{uv[1]:.0f}",
         uv + np.array([13.0, 22.0]), C_3D, 0.45, 1)


def _draw_jaws(img, polys) -> None:
    spec = {"fixed_jaw": (C_JAW_FIX, "FIXED JAW"),
            "moving_open": (C_JAW_OPEN, "MOVING OPEN"),
            "moving_closed": (C_JAW_CLOSED, "MOVING CLOSED")}
    for k, poly in polys.items():
        col, name = spec.get(k, (C_JAW_FIX, k.upper()))
        p = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [p], True, col, 1, cv2.LINE_AA)
        text(img, name, p.reshape(-1, 2)[0] + np.array([6.0, -8.0]), col, 0.7, 2)


def _draw_axis(img, m) -> None:
    """合爪轴：过参考中心的一条长线 + 参考中心 + 夹取方向箭头。"""
    ref = np.asarray(m["axis_ref_px"], dtype=np.float64)
    u = np.asarray(m["axis_dir_px"], dtype=np.float64)
    L = 0.45 * float(max(img.shape[:2]))
    cv2.line(img, (ref - u * L).astype(int), (ref + u * L).astype(int), C_AXIS, 1,
             cv2.LINE_AA)
    double_arrow(img, np.asarray(m["band_uv"], dtype=np.float64), u, 70.0, C_GRASP, 2)
    text(img, "GRASP DIR", np.asarray(m["band_uv"], dtype=np.float64) + u * 74.0
         + np.array([0.0, -6.0]), C_GRASP, 0.5, 1)
    dot(img, ref, C_REF, 5)
    text(img, "AXIS CENTER", ref + np.array([10.0, 16.0]), C_REF, 0.45, 1)
    foot = ref + u * float(np.dot(np.asarray(m["box_uv"], dtype=np.float64) - ref, u))
    ok = abs(float(m["axis"])) <= _axis_tol()
    col = C_GOOD if ok else C_BAD
    cv2.line(img, np.asarray(m["box_uv"], dtype=np.float64).astype(int),
             foot.astype(int), col, 2, cv2.LINE_AA)
    text(img, f"axis {m['axis']:+.0f}px", (foot + np.asarray(m["box_uv"],
             dtype=np.float64)) / 2.0 + np.array([8.0, -16.0]), col, 0.8, 2)


def _draw_gap(img, m) -> None:
    """缝的标尺：沿合爪轴，爪轮廓近端 → 方块近端，再标出目标带。

    **只画，判据不在这儿。** 这里只做一次投影把两端找出来（按两端在轴上的中位数谁大
    谁小判方块在哪一侧），带号距离算出来应当跟 `m['gap']` 一致 —— 不一致说明画错了，
    那也只是画错了。
    """
    u = np.asarray(m["axis_dir_px"], dtype=np.float64)
    pts = np.asarray(m["pts"], dtype=np.float64).reshape(-1, 2)
    q = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2)
                        for p in m["polys"].values()], axis=0)
    r0, r1 = float((pts @ u).min()), float((pts @ u).max())
    j0, j1 = float((q @ u).min()), float((q @ u).max())
    side = 1.0 if float((pts @ u).mean()) >= float((q @ u).mean()) else -1.0
    j_near = j1 if side > 0 else j0
    r_near = r0 if side > 0 else r1
    thru = np.asarray(m["box_uv"], dtype=np.float64)
    span_r = 0.5 * float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    span_j = 0.5 * float(np.linalg.norm(q.max(axis=0) - q.min(axis=0)))
    _u_coord_line(img, thru, u, j_near, max(span_j, 40.0), C_JAW_FIX, thick=2)
    _u_coord_line(img, thru, u, r_near, max(span_r, 30.0), C_BOX, thick=2)
    # 目标带：从爪轮廓近端量起 LO~HI
    for c, name in ((j_near + side * float(PR.CLEAR_BAND_LO_PX), "LO"),
                    (j_near + side * float(PR.CLEAR_BAND_HI_PX), "HI")):
        _u_coord_line(img, thru, u, c, max(span_j, 40.0) * 1.3, C_BAND, dashed=True,
                      thick=2)
        p = thru + (c - float(np.dot(thru, u))) * u
        text(img, name, p + np.array([6.0, -10.0]), C_BAND, 0.7, 2)
    mid = thru + (((j_near + r_near) / 2.0) - float(np.dot(thru, u))) * u
    col = C_GOOD if PR._gap_ok(float(m["gap"])) else C_BAD
    text(img, f"gap {m['gap']:+.0f}px (want {PR.CLEAR_BAND_LO_PX:.0f}.."
              f"{PR.CLEAR_BAND_HI_PX:.0f})", mid + np.array([8.0, 56.0]), col, 0.8, 2)


def _draw_hud(img, snap, m, step) -> None:
    lines: list[tuple[str, str]] = []
    src = snap.get("angles_src") or "?"
    if step is not None:
        name, col = PHASE_EN.get(step.get("phase"), (str(step.get("phase")), C_WARN))
        lines.append((f"STEP   {name}", col))
        if m and m.get("ok"):
            g_ok = float(m["gap"]) if "gap" in m else float("nan")
            lines.append((f"gap    {g_ok:+.0f}px", C_GOOD if _gap_ok(g_ok) else C_BAD))
            a_ok = abs(float(m["axis"])) <= float(_axis_tol())
            lines.append((f"axis   {m['axis']:+.0f}px", C_GOOD if a_ok else C_BAD))
        if step.get("dxy_mm") is not None:
            dx, dy = step["dxy_mm"]
            lines.append((f"MOVE   dx {dx:+.1f}  dy {dy:+.1f} mm", C_AXIS))
        if step.get("dz_mm") is not None:
            lines.append((f"MOVE   dz {step['dz_mm']:+.1f} mm", C_AXIS))
        if step.get("at_depth"):
            # 到底之后：缝那一条不再拦门，只看轴偏（`measure.next_step` 第 2 段）。
            lines.append(("AT DEPTH - gap not gated, axis only", C_AXIS))
        lines.append((f"(angles {src})", (170, 170, 170)))
    elif snap.get("box") is not None and snap["box"].get("center") is None:
        # **不画 `why`。** `measure.next_step` 那句话是中文，`cv2.putText` 画出来是
        # 一串问号。中文那一路走终端日志（`panel/rig.py` 在相位变化时打一行）。
        lines.append(("STEP   LOST - no box in view", C_BAD))
    text_block(img, (12, 12), lines)


def _axis_tol() -> float:
    return float(PR.AXIS_TOL_PX)


def _gap_ok(v: float) -> bool:
    return bool(PR._gap_ok(v))


def _draw_footer(img, snap) -> None:
    """画面底下那一条：六个关节角 + TCP + 6 号（爪）角度和判定。"""
    h, w = img.shape[:2]
    y = h - 50                     # 两行，底下留 8 像素，别让第二行掉出画幅
    rows = []
    a = snap.get("angles_deg")
    if a is not None:
        rows.append("J1..J6  " + "  ".join(f"{v:6.1f}" for v in np.asarray(a)[:6])
                    + f"   [{snap.get('angles_src') or '?'}]")
    tcp = snap.get("tcp")
    if tcp is not None:
        t = np.asarray(tcp) * 1000.0
        rows.append(f"TCP     x {t[0]:7.1f}  y {t[1]:7.1f}  z {t[2]:7.1f} mm"
                    + (f"    grip {snap['grip_deg']:.1f} deg  {snap.get('verdict') or ''}"
                       if snap.get("grip_deg") is not None else ""))
    if not rows:
        return
    roi = img[y - 6:min(h, y + 20 * len(rows)), 0:w]
    if roi.size:
        cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, roi)
    for i, r in enumerate(rows):
        text(img, r, (12, y + 16 + 20 * i), (240, 240, 240), 0.55, 1)
