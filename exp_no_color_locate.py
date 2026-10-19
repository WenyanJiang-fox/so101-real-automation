"""实验：找方块不看颜色，改成「跟一张空工位的参考图比，多出来的那块就是它」。

**为什么要试这个。** 现在 `core/vision.py::color_mask` 是一条写死的 RGB 阈值（红要
R>120 且 G、B<100）。换灯、换材质、方块跟地面颜色接近，这条线就断了 —— 而真机上这三件
事都会发生。改成几何判据以后，「是不是方块」由**形状对不对得上**说了算，颜色不参与。

**两路都用同一个形状拟合**（`vision.estimate_pose`）：把方块八个角点按候选位姿投回画面
跟剪影比重合度。两路唯一的差别是**剪影从哪来**：
    旧路  按颜色挑像素（`color_mask`）
    新路  跟空工位的参考图比（`vision.foreground_mask`）
所以这个实验测的就是「把取色那一步换掉，结果会怎样」。

**为什么拿俯视相机做。** 参考图这条路要求**相机不动** —— 腕部相机是跟着手走的，背景
每拍一张都不一样，参考图对它没有意义。固定的俯视相机才是这条路的适用场景。

**真值只用来打分。** 定位那两路一个数都不读方块真值；这里的 `arm.cube_pos(0)` 只出现在
算误差那一行。

跑：G:/anaconda/envs/sim2real/python.exe G:\\sim2real\\so101-real-harness\\exp_no_color_locate.py
出图：G:\\sim2real\\so101-real-harness\\out\\no_color_locate\\
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import impls.arm_sim                                # noqa: E402,F401  导入即挂上牌子
import impls.cam_sim                                # noqa: E402,F401
from core import vision                             # noqa: E402
from core.geom import CUBE_Z, PLACE_XY, PICK_XY     # noqa: E402
from core.registry import make                      # noqa: E402
from measure_stereo import calibrate_overhead, intr, overhead_mount  # noqa: E402

OUT = HERE / "out" / "no_color_locate"
CUBE_HALF = CUBE_Z            # 30mm 方块的半边长
DISC_Z = 0.0018               # 标识盘上表面高度（跟 measure_stereo 里同一个数）
PARK_XY = (3.0, 3.0)          # 把方块临时挪到这儿 —— 在画面外，拍参考图用

# 四种刁难 + 一个正常。改的都是运行时模型，不动只读包里那份 xml。
# 文件名的前缀用 ASCII —— cv2.imwrite 在 Windows 上写不了非 ASCII 路径，会出来乱码名。
CONDS = [
    ("1_normal",    "正常",       dict(scale=1.0,  tint=(1.0, 1.0, 1.0),  rgba=None)),
    ("2_dark",      "变暗",       dict(scale=0.35, tint=(1.0, 1.0, 1.0),  rgba=None)),
    ("3_warm",      "暖光",       dict(scale=1.0,  tint=(1.0, 0.55, 0.25), rgba=None)),
    ("4_graycube",  "方块涂灰",   dict(scale=1.0,  tint=(1.0, 1.0, 1.0),  rgba=(0.55, 0.55, 0.55, 1.0))),
    ("5_greencube", "方块涂绿",   dict(scale=1.0,  tint=(1.0, 1.0, 1.0),  rgba=(0.05, 0.60, 0.05, 1.0))),
]


# ---------------------------------------------------------------- 拍/摆

def shot(arm) -> np.ndarray:
    """现渲一张俯视图。

    **不走 `arm.obs`，也不走 env.step。** 这里要的是「改了灯和材质之后立刻看一张」，
    step 一步会让方块掉、让手臂动；直接问渲染器要，改完就能看。
    """
    rend = arm.u._overhead_obs_renderer
    rend.update_scene(arm.u.data, camera=arm.u._overhead_obs_cam)
    return np.asarray(rend.render())


def park_cubes(arm, park: bool, keep: dict) -> None:
    """把方块临时挪到画面外（拍参考图），用原样放回。

    `place_cube` 按 yaw 重算四元数，会把方块原来的姿态抹掉，所以存的是 qpos 原值，
    放回时直接写回去。改完 qpos 必须 `mj_forward` —— 渲染器读的是 `geom_xpos`，
    不 forward 就是上一拍的旧位置。
    """
    import mujoco

    u = arm.u
    for i in range(arm.n_cubes):
        a, d = arm.cube_addr[i], arm.cube_dof[i]
        if park:
            keep[i] = (u.data.qpos[a:a + 3].copy(), u.data.qpos[a + 3:a + 7].copy())
            u.data.qpos[a:a + 3] = [PARK_XY[0], PARK_XY[1], CUBE_Z]
            u.data.qvel[d:d + 6] = 0.0
        else:
            p, q = keep[i]
            u.data.qpos[a:a + 3] = p
            u.data.qpos[a + 3:a + 7] = q
            u.data.qvel[d:d + 6] = 0.0
    mujoco.mj_forward(u.model, u.data)


def set_lights(model, base, scale: float = 1.0, tint=(1.0, 1.0, 1.0)) -> None:
    """所有灯一起改亮度/色温。`base` 是场景原样，每次从它算，不叠着上一次改。"""
    tint = np.asarray(tint, dtype=np.float64)
    for i in range(int(model.nlight)):
        model.light_diffuse[i] = np.clip(np.asarray(base[i], dtype=np.float64) * float(scale) * tint,
                                         0.0, 1.0)


def paint_cubes(arm, base: dict, rgba) -> None:
    """把方块的表面颜色换掉（模拟换材质）。rgba=None 就是恢复原样。"""
    for i in range(arm.n_cubes):
        for g in arm.cube_geom[i]:
            arm.u.model.geom_rgba[g] = (np.asarray(base[i][g], dtype=np.float64)
                                        if rgba is None else np.asarray(rgba, dtype=np.float64))


# ---------------------------------------------------------------- 两路定位

def path_color(rgb, mount, guess):
    """旧路：按颜色挑像素。"""
    return vision.estimate_pose(rgb, mount, "red", guess, half=CUBE_HALF)


def path_shape(rgb, ref, mount, guess, k):
    """新路：跟空工位参考图比 → 在候选里挑离带路点最近的那块 → 同一个形状拟合。

    为什么要挑：不看颜色以后，画面里**任何**跟参考图不一样的东西都会成为候选（手臂、
    影子、挪过的托盘）。带路点（标识点坐标）本来就是干这个的 —— 它说得出「方块大概在
    这一带」，那就取最靠近它的那个候选。真机上也是同一个道理。
    """
    mask_all = vision.foreground_mask(rgb, ref)
    cands = vision.blobs_of(mask_all)
    if not cands:
        return None, None, {"why": "跟参考图比完，一个像样的候选都没有", "px": 0}, mask_all, cands, None

    gu = vision.project(mount["pos"], mount["mat"], k, guess)
    if gu is None:
        return None, None, {"why": "带路点根本投不进画面", "px": 0}, mask_all, cands, None
    pick = min(cands, key=lambda b: float(np.hypot(b[1][0] - gu[0], b[1][1] - gu[1])))

    import cv2
    n_lab, lab = cv2.connectedComponents(np.asarray(mask_all, np.uint8), 8)
    one = (lab == pick[3]).astype(np.uint8) * 255
    center, yaw, info = vision.estimate_pose(rgb, mount, "red", guess, half=CUBE_HALF, mask=one)
    return center, yaw, info, mask_all, cands, pick


# ---------------------------------------------------------------- 画

def zoom(rgb, uv, half: int = 32, mag: int = 6) -> np.ndarray:
    """把某一处裁出来放大 —— 看「认不出来」到底是什么样子。"""
    import cv2

    a = np.asarray(rgb)
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    y0, y1 = max(0, v - half), min(a.shape[0], v + half)
    x0, x1 = max(0, u - half), min(a.shape[1], u + half)
    return cv2.resize(a[y0:y1, x0:x1], None, fx=mag, fy=mag, interpolation=cv2.INTER_NEAREST)


def annotate(rgb, mount, k, truth, got_color, got_shape) -> np.ndarray:
    """把三样画上去：真值（绿）、旧路结果（品红）、新路结果（青）。"""
    import cv2

    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR).copy()
    W = bgr.shape[1]

    def cross(p, color, label, dy):
        uv = vision.project(mount["pos"], mount["mat"], k, p)
        if uv is None:
            return
        u, v = int(round(uv[0])), int(round(uv[1]))
        cv2.drawMarker(bgr, (u, v), color, cv2.MARKER_CROSS, 26, 2)
        cv2.putText(bgr, label, (min(u + 8, W - 120), v + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cross(truth, (0, 200, 0), "truth", -10)
    if got_color is not None:
        cross(got_color, (255, 0, 255), "color", 14)
    if got_shape is not None:
        cross(got_shape, (255, 255, 0), "shape", 34)
    return bgr


def scanline(rgb, ref, u: int, v0: int, v1: int, step: int = 2) -> None:
    """从画面上竖着切一刀打印像素值 —— 「认不出来」到底卡在哪一条上，看这个。

    打三样：当前图的 RGB、参考图的 RGB、两张图局部对比度差的绝对值（新路用的就是它，
    超过 thr 就算「多出来了」）。顺带标一下这一行在颜色阈值里过不过。
    """
    a = np.asarray(rgb, dtype=np.int16)
    b = np.asarray(ref, dtype=np.int16)
    dn = np.abs(vision.lit_normalized(rgb) - vision.lit_normalized(ref))
    print(f"    u={u} 竖着扫一刀   当前RGB / 参考RGB / |归一化后的差| / 过不过颜色线(红)")
    for v in range(v0, v1 + 1, step):
        r_, g_, bl = (int(x) for x in a[v, u][:3])
        r2, g2, b2 = (int(x) for x in b[v, u][:3])
        red_ok = "过" if (r_ > 120 and g_ < 100 and bl < 100) else "—"
        print(f"      v={v:3d}  ({r_:3d},{g_:3d},{bl:3d})  ({r2:3d},{g2:3d},{b2:3d})"
              f"   {dn[v, u]:5.2f}   {red_ok}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if OUT.exists():
        shutil.rmtree(OUT)                       # 上一次的图不留着
    OUT.mkdir(parents=True)

    arm = make("arm", "sim", cube_order=("red",))
    try:
        arm.reset(seed=0)
        u = arm.u
        base_lights = u.model.light_diffuse.copy()
        base_rgba = {i: {g: u.model.geom_rgba[g].copy() for g in arm.cube_geom[i]}
                     for i in range(arm.n_cubes)}

        mount = overhead_mount(arm)
        img0 = shot(arm)
        size = (img0.shape[1], img0.shape[0])     # 跟 measure_stereo 一样按 (宽, 高) 传
        truth = np.asarray(arm.cube_pos(0), dtype=np.float64)
        guess = np.array([float(PICK_XY[0][0]), float(PICK_XY[0][1]), float(CUBE_Z)])
        print(f"方块真值 {np.round(truth, 4)}   带路点（标识点）{np.round(guess, 4)}")

        # ---- 参考图：把方块挪到画面外，拍一张空工位 ----
        keep: dict = {}
        park_cubes(arm, True, keep)
        ref = shot(arm)
        park_cubes(arm, False, keep)
        import cv2
        cv2.imwrite(str(OUT / "0_参考图_空工位.png"), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))

        # ---- 标定这台相机的内参（用标识点盘，世界坐标事先知道）----
        marks = np.column_stack([np.vstack([PICK_XY, PLACE_XY]),
                                 np.full(len(PICK_XY) + len(PLACE_XY), DISC_Z)])
        k_fit, k_info = calibrate_overhead(ref, mount, marks)
        if k_fit is not None:
            mount["k"] = np.asarray(k_fit, dtype=np.float64)
        k = intr(mount, size)
        print(f"这台相机的内参 {np.round(k, 2)}"
              f"{'（标定后）' if k_fit is not None else '（没标出来，用没标定的）'}")

        # 地面参考像素：一个工位上没有东西的点（放料区标记点），z 取 0
        floor_uv = vision.project(mount["pos"], mount["mat"], k,
                                  [float(PLACE_XY[0][0]), float(PLACE_XY[0][1]), 0.0])
        top_uv = vision.project(mount["pos"], mount["mat"], k, truth + np.array([0.0, 0.0, CUBE_HALF]))

        rows = []
        for key, label, cond in CONDS:
            set_lights(u.model, base_lights, cond["scale"], cond["tint"])
            paint_cubes(arm, base_rgba, cond["rgba"])
            rgb = shot(arm)

            c_col, y_col, i_col = path_color(rgb, mount, guess)
            c_shp, y_shp, i_shp, mask_all, cands, pick = path_shape(rgb, ref, mount, guess, k)

            def err(p):
                if p is None:
                    return None, None
                return (float(np.hypot(p[0] - truth[0], p[1] - truth[1])), float(p[2] - truth[2]))

            e_col, z_col = err(c_col)
            e_shp, z_shp = err(c_shp)
            rows.append((label, e_col, z_col, i_col, e_shp, z_shp, i_shp))

            cube_px = rgb[int(round(top_uv[1])), int(round(top_uv[0]))] if top_uv is not None else None
            floor_px = rgb[int(round(floor_uv[1])), int(round(floor_uv[0]))] if floor_uv is not None else None
            print(f"\n【{label}】灯={cond['scale']}x{tint_str(cond['tint'])}  方块涂色={cond['rgba']}")
            print(f"    方块顶面那一像素 {cube_px}   地面那一像素 {floor_px}")
            print(f"    候选 {len(cands)} 个：" + ("，".join(
                f"{a}px@({c[0]:.0f},{c[1]:.0f})" for a, c, _b, _l in cands[:4]) or "无")
                + (f"  → 选中 {pick[0]}px@({pick[1][0]:.0f},{pick[1][1]:.0f})" if pick else ""))
            print(f"    旧路（颜色）：{_line(c_col, e_col, z_col, i_col)}")
            print(f"    新路（形状）：{_line(c_shp, e_shp, z_shp, i_shp)}")
            for tag, info in (("旧路", i_col), ("新路", i_shp)):
                print(f"      {tag} 面积：画面里那块 {info.get('seen_area') or 0:.0f}px "
                      f"{_box(info.get('seen'))}   投影出来 {info.get('want_area') or 0:.0f}px "
                      f"{_box(info.get('want'))}   重合度 {info.get('iou')}")
            if key == "1_normal" and top_uv is not None:
                scanline(rgb, ref, int(round(top_uv[0])), int(round(top_uv[1])) - 34,
                         int(round(top_uv[1])) + 24, 2)

            cv2.imwrite(str(OUT / f"{key}_overhead.png"),
                        annotate(rgb, mount, k, truth, c_col, c_shp))
            cv2.imwrite(str(OUT / f"{key}_mask.png"), np.asarray(mask_all))
            if top_uv is not None:
                cv2.imwrite(str(OUT / f"{key}_zoom.png"),
                            cv2.cvtColor(zoom(rgb, top_uv), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(OUT / f"{key}_zoom_mask.png"), zoom(mask_all, top_uv))

        set_lights(u.model, base_lights)
        paint_cubes(arm, base_rgba, None)

        print("\n结果（水平误差 = 量出来的点跟真值在 x-y 平面上的距离）：")
        print(f"  {'条件':<14s}{'旧路 认出来':>14s}{'水平误差':>10s}{'高度差':>9s}"
              f"{'新路 认出来':>14s}{'水平误差':>10s}{'高度差':>9s}")
        for name, e_col, z_col, i_col, e_shp, z_shp, i_shp in rows:
            print(f"  {name:<14s}{_ok(e_col, i_col):>14s}{_mm(e_col):>10s}{_mm(z_col, signed=True):>9s}"
                  f"{_ok(e_shp, i_shp):>14s}{_mm(e_shp):>10s}{_mm(z_shp, signed=True):>9s}")

        print(f"\n图在 {OUT}")
    finally:
        try:
            arm.close()
        except Exception as exc:                    # noqa: BLE001
            print(f"关环境时报错（不影响上面的结果）：{exc}")
    return 0


def tint_str(t) -> str:
    return "(" + ",".join(f"{v:.2f}" for v in t) + ")"


def _line(center, e, z, info) -> str:
    if center is None:
        return f"没认出来（{info.get('why')}）"
    return (f"量到 {np.round(center, 4)}  水平误差 {e * 1000:.1f}mm  高度差 {z * 1000:+.1f}mm"
            f"  重合度 {info.get('iou', float('nan')):.2f}")


def _ok(e, info) -> str:
    return "是" if e is not None else "否"


def _mm(v, signed: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v * 1000:+.1f}" if signed else f"{v * 1000:.1f}"


def _box(b) -> str:
    if b is None:
        return "外框 —"
    return f"外框 ({b[0]:.0f},{b[1]:.0f})-({b[2]:.0f},{b[3]:.0f})"


if __name__ == "__main__":
    raise SystemExit(main())
