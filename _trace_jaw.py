"""把机械爪在腕相机画面里的轮廓和合爪方向描一遍，存成常数。

**为什么能是常数。** 腕相机和两个爪子都长在 `gripper` 那个体上（手眼标定
`real_wrist_mount.json` 就是把相机定死在 `gripperframe` 上），两者相对位置是死的 ——
所以爪子的轮廓在画面里永远是同一圈像素，跟臂怎么动、腕子怎么转无关。描一次就够。

**只描两样，因为只看得见这两样**（2026-10-15 实看 `out/round_01_完整.mp4`）：
腕机是俯视着往下看的，**爪子一张开，动爪整个跑出画面上边**（f58/f66/f74/f78/f86 都只
看得见固定爪那一根），张开那些帧里根本描不到动爪。所以：

    1  固定爪的轮廓        —— 合爪前它不许压到目标（就是原来那几趟栽的地方）
    2  动爪的轮廓（可选）  —— 合到底那一下它落在哪。合到底那张图上两个爪子都在画面里
                              （f130），要描就描这一张上的
    3  合爪轴（两下）      —— 第一下点「方块该被夹住的那个中心」，第二下沿开合方向点远
                              一点；方向 = 第 1 点 → 第 2 点，过这两点的直线就是轴

描完存 `assets/camera_calibration/real_wrist_jaw_shape.json`，`skills/pick_real.py`
合爪前那两道判据只吃这一份。存盘时另外写一张 `out/jaw_trace_check.jpg`（三样画在一起），
回头看一眼对不对。

用法：

    G:/anaconda/envs/sim2real/python.exe _trace_jaw.py
    G:/anaconda/envs/sim2real/python.exe _trace_jaw.py --img out/别的帧.jpg

- 左键 = 在当前那一样上加一个点；`b` = 撤掉最后一点；`r` = 清掉当前这一样
- `1` / `2` / `3` = 切到要描的那一样
- `t` = 边缘增强开关（爪子是白的、纸也是白的，开了好找那条棱）
- 滚轮 = 以光标为中心缩放；`0` = 复位
- `q` = 存盘退出（固定爪 ≥3 点、合爪轴 2 点）；`ESC` = 不存退出
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(r"G:\sim2real\so101-real-harness")
OUT_JSON = _ROOT / "assets" / "camera_calibration" / "real_wrist_jaw_shape.json"
CHECK_IMG = _ROOT / "out" / "jaw_trace_check.jpg"
DEFAULT_IMG = _ROOT / "out" / "jaw_frame.jpg"

# 显示尺寸。**不能用 WINDOW_AUTOSIZE**：屏幕 125% 缩放，1080p 会被撑成 2400x1350 顶出
# 屏幕（memory `opencv-window-dpi-scaling`）。鼠标事件也只在 `waitKey` 泵消息时进来。
DISP_W, DISP_H = 1280, 720
NAME = "trace jaw: 1 fixed / 2 moving / 3 closing axis   q=save"

FIXED, MOVING, AXIS = 1, 2, 3
KEY_OF = {FIXED: "fixed_jaw", MOVING: "moving_closed"}
COLOR = {FIXED: (255, 255, 0), MOVING: (0, 165, 255), AXIS: (255, 0, 255)}
TITLE = {FIXED: "1 fixed jaw outline", MOVING: "2 moving jaw outline (optional)",
         AXIS: "3 closing axis (click 2 points)"}


def _centroid(pts) -> np.ndarray:
    """面积重心（`cv2.moments`）；点太少或退化时退回顶点平均。"""
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    m = cv2.moments(p)
    if abs(float(m["m00"])) < 1e-9:
        return np.asarray(pts, dtype=np.float64).mean(axis=0)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float64)


def _edge(img):
    """边缘增强：把 Canny 出来的棱涂成蓝的。白爪子压在白纸上，不用这个找不着边。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 30, 80)
    out = img.copy()
    out[e > 0] = (255, 0, 0)
    return out


def _draw(img, polys, axis_pts, cur, footer):
    """把已经描的东西画到**原图**上（缩放由外层做，这里一律用原图坐标）。"""
    view = img.copy()
    for k, pts in polys.items():
        pts_i = [tuple(np.round(p).astype(int)) for p in pts]
        if not pts_i:
            continue
        on_cur = (k == cur)
        col = COLOR[k]
        if len(pts_i) >= 2:
            cv2.polylines(view, [np.asarray(pts_i, np.int32)],
                          on_cur and len(pts_i) >= 3, col, 4 if on_cur else 2, cv2.LINE_AA)
        if len(pts_i) >= 3:
            cv2.line(view, pts_i[-1], pts_i[0], col, 1, cv2.LINE_AA)
        for i, p in enumerate(pts_i):
            cv2.circle(view, p, 6 if on_cur else 3, col, -1)
            if on_cur:
                cv2.putText(view, str(i + 1), (p[0] + 10, p[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    if axis_pts:
        col = COLOR[AXIS]
        for i, p in enumerate(axis_pts):
            cv2.circle(view, tuple(np.round(p).astype(int)), 8, col, -1)
        if len(axis_pts) >= 2:
            a = np.asarray(axis_pts[0], dtype=np.float64)
            d = np.asarray(axis_pts[1], dtype=np.float64) - a
            d = d / max(float(np.linalg.norm(d)), 1e-9)
            far = a + 0.9 * img.shape[1] * d
            cv2.arrowedLine(view, tuple(np.round(a - 40 * d).astype(int)),
                            tuple(np.round(far).astype(int)), col, 3, cv2.LINE_AA,
                            tipLength=0.05)

    for i, (k, txt) in enumerate(
            ((FIXED, f"{TITLE[FIXED]} ({len(polys[FIXED])} pts)"),
             (MOVING, f"{TITLE[MOVING]} ({len(polys[MOVING])} pts)"),
             (AXIS, f"{TITLE[AXIS]} ({len(axis_pts)}/2)"))):
        cv2.putText(view, txt, (30, 80 + 46 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(view, txt, (30, 80 + 46 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    COLOR[k], 2, cv2.LINE_AA)
    cv2.putText(view, footer, (30, view.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(view, footer, (30, view.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return view


def main() -> int:
    args = list(sys.argv[1:])
    img_path = DEFAULT_IMG
    if "--img" in args:
        i = args.index("--img")
        img_path = Path(args[i + 1])
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"读不到 {img_path}")
        return 1
    h, w = img.shape[:2]
    print(f"底图 {w}x{h}：{img_path}")
    print(f"窗口按 {DISP_W}x{DISP_H} 显示，点下去记的是**原图**坐标。")

    polys: dict[int, list[tuple[float, float]]] = {FIXED: [], MOVING: []}
    axis_pts: list[tuple[float, float]] = []
    # **先把现在这一份摆上来**（用户 2026-10-16：「把原来的线加进去啊」）。原来是从空白
    # 开局，改一个点就得把固定爪 14 个点、合爪轴那两个点整个重描一遍。
    if OUT_JSON.exists():
        try:
            _old = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:                                       # noqa: BLE001
            _old = {}
        for _k, _name in KEY_OF.items():
            for _p in (_old.get(_name) or []):
                polys[_k].append((float(_p[0]), float(_p[1])))
        _a, _d = _old.get("axis_ref_px"), _old.get("axis_dir_px")
        if _a and _d:
            _n = float(_old.get("axis_len_px") or 211.2)
            axis_pts.append((float(_a[0]), float(_a[1])))
            axis_pts.append((float(_a[0]) + float(_d[0]) * _n,
                             float(_a[1]) + float(_d[1]) * _n))
        print(f"已经把 {OUT_JSON.name} 现在这一份摆上来了：固定爪 {len(polys[FIXED])} 点、"
              f"动爪 {len(polys[MOVING])} 点、合爪轴 {len(axis_pts)} 点")
        print("  （要清空就按 `r`；要重描就按 `b` 一点一点撤）")
    cur = FIXED
    zoom = 0.0                      # 0 = 铺满窗口（不放大）
    cx, cy = w / 2.0, h / 2.0
    edges = False

    def scale() -> float:
        """显示像素 / 原图像素。`zoom` 是相对「整幅铺满窗口」的倍数。"""
        return (zoom if zoom > 0 else 1.0) * (DISP_W / float(w))

    def to_orig(wx: float, wy: float) -> tuple[float, float]:
        sc = scale()
        return (cx + (wx - DISP_W / 2.0) / sc, cy + (wy - DISP_H / 2.0) / sc)

    def redraw() -> None:
        sc = scale()
        view = _draw(_edge(img) if edges else img, polys, axis_pts, cur,
                     f"base {img_path.name}   zoom {sc:.2f}x   edges {'on' if edges else 'off'}"
                     f"   drawing: {TITLE[cur]}")
        m = np.array([[sc, 0.0, DISP_W / 2.0 - sc * cx], [0.0, sc, DISP_H / 2.0 - sc * cy]])
        disp = cv2.warpAffine(view, m, (DISP_W, DISP_H), flags=cv2.INTER_AREA)
        cv2.imshow(NAME, disp)

    def on_mouse(event, x, y, flags, _param) -> None:
        nonlocal cur, zoom, cx, cy
        if event == cv2.EVENT_MOUSEWHEEL:
            ox, oy = to_orig(x, y)
            zoom = float(np.clip((zoom if zoom > 0 else 1.0)
                                 * (1.25 if flags > 0 else 1 / 1.25), 1.0, 8.0))
            sc_new = scale()
            # 光标底下那一点不动
            cx = float(np.clip(ox - (x - DISP_W / 2.0) / sc_new, 0.0, w))
            cy = float(np.clip(oy - (y - DISP_H / 2.0) / sc_new, 0.0, h))
            redraw()
        elif event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = to_orig(x, y)
            if not (0 <= ox < w and 0 <= oy < h):
                return
            if cur == AXIS:
                if len(axis_pts) >= 2:
                    axis_pts.clear()
                axis_pts.append((float(ox), float(oy)))
                print(f"  合爪轴第 {len(axis_pts)} 个点：原图 ({ox:.0f}, {oy:.0f})", flush=True)
            else:
                polys[cur].append((float(ox), float(oy)))
                print(f"  {TITLE[cur]} 第 {len(polys[cur])} 个点：原图 ({ox:.0f}, {oy:.0f})",
                      flush=True)
            redraw()

    cv2.namedWindow(NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NAME, DISP_W, DISP_H)
    cv2.setMouseCallback(NAME, on_mouse)
    redraw()
    # **窗口「已经出现过」之前不算它关**。原来第一圈 `waitKey(20)` 之后立刻问
    # `getWindowProperty(WND_PROP_VISIBLE)`，20 毫秒里窗口还没建出来，它返回 0，
    # 程序当场判「窗口关了，不存退出」退出 —— 实测 2026-10-16 从后台起来就是这个症状：
    # 打印三行就没了。改成见过一次之后才认「关」。
    _seen_win = False
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            print("不存退出")
            return 0
        if k == ord("q"):
            missing = []
            if len(polys[FIXED]) < 3:
                missing.append("固定爪轮廓（至少 3 个点）")
            if len(axis_pts) < 2:
                missing.append("合爪轴（要 2 个点）")
            if missing:
                print(f"  ✗ 还差：{'、'.join(missing)}")
                continue
            break
        if k in (ord("1"), ord("2"), ord("3")):
            cur = int(chr(k))
            redraw()
        elif k == ord("t"):
            edges = not edges
            redraw()
        elif k == ord("0"):
            zoom, cx, cy = 0.0, w / 2.0, h / 2.0
            redraw()
        elif k == ord("b"):
            if cur == AXIS:
                if axis_pts:
                    axis_pts.pop()
            elif polys[cur]:
                polys[cur].pop()
            redraw()
        elif k == ord("r"):
            if cur == AXIS:
                axis_pts.clear()
            else:
                polys[cur].clear()
            redraw()
        if cv2.getWindowProperty(NAME, cv2.WND_PROP_VISIBLE) >= 1:
            _seen_win = True
        elif _seen_win:
            print("窗口关了，不存退出")
            return 0
    cv2.destroyAllWindows()

    a = np.asarray(axis_pts[0], dtype=np.float64)
    d = np.asarray(axis_pts[1], dtype=np.float64) - a
    n = float(np.linalg.norm(d))
    if n < 20.0:
        print(f"✗ 合爪轴两个点只差 {n:.0f} 像素 —— 方向定不准，不存")
        return 1
    out = {"image_size": [w, h],
           "image": str(img_path),
           **{KEY_OF[k]: [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in polys[k]]
              for k in KEY_OF if polys[k]},
           "axis_ref_px": [round(float(a[0]), 1), round(float(a[1]), 1)],
           "axis_dir_px": [round(float(d[0] / n), 4), round(float(d[1] / n), 4)],
           "axis_len_px": round(n, 1),
           "what": "机械爪在腕相机画面里的轮廓和合爪轴（`_trace_jaw.py` 描）。相机与两爪"
                   "同在一个刚体上，这些像素跟臂怎么动无关，描一次就够。"
                   "`skills/pick_real.py` 合爪前的检查吃这一份。"}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"存到 {OUT_JSON}")
    print(f"  固定爪 {len(polys[FIXED])} 点"
          f"  动爪 {len(polys[MOVING])} 点（0 = 没描）")
    print(f"  合爪轴 过 {np.round(a, 1)} 方向 {np.round(d / n, 3)}（画面里的单位向量）")
    cv2.imwrite(str(CHECK_IMG), _draw(img, polys, axis_pts, FIXED, "check"))
    print(f"  对照图 {CHECK_IMG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
