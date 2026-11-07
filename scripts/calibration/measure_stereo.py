"""一次性测量：三角测量量物体位置，两种取景方式各有多准。

**这是给「真机上只有普通 RGB 相机、拿不到深度」那个问题量的一把尺子。** 比的是两条：

    A 腕部挪两下   同一个腕部相机，机械臂横移一小段，前后各拍一张
    B 腕部 + 全局  腕部相机拍一张，固定俯视相机拍一张

两条原理是同一个：两个视角看同一个点，它在两张图里挪了多少就能推出远近。差别只在
「两个视角隔多远」和「两张图差多少」。

**量的是三角测量本身，不是整套流程。** 两个相机的位姿在这儿是从仿真模型里现读的
（真机上这份得靠标定，标定误差不在这把尺子里）。所以下面那些误差是**下限** ——
真机上只会更差。

取的点是**色块剪影的质心**：两张图里各找一次红色区域的质心，各拉一条视线出去，
取两条视线的最近点当答案。质心在两张图里对应的不是同一个三维点（看到的面不一样），
这个偏差就包含在结果里 —— 它就是「配对误差」的真身，实打实算进去而不是假设为零。

跑法（在 G:\\sim2real\\so101-real-harness 下）：
    G:/anaconda/envs/sim2real/python.exe measure_stereo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import impls.arm_sim                                # noqa: E402,F401  导入即挂上牌子
import impls.cam_sim                                # noqa: E402,F401
from core import follow, vision                      # noqa: E402
from core.geom import CUBE_Z, PLACE_XY, PICK_XY      # noqa: E402
from core.params import SPEED, acc_amax, speed_vmax  # noqa: E402
from core.registry import make                       # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "out" / "measure_stereo"
DISC_Z = 0.0018      # 标识点圆盘上表面的高度（core.geom.marker_xml：盘心 z=1.2mm、厚 1.2mm）
CUBE_HALF = CUBE_Z   # 30mm 方块的半边长，正好等于它躺平时中心的高度


# ---------------------------------------------------------------- 几何

def overhead_mount(arm):
    """俯视那个自由相机的世界位姿 + 内参 —— **直接问渲染器要**。

    俯视相机是按 lookat/distance/elevation/azimuth 摆的自由相机，模型里没有它的
    `cam_xpos`。但渲染这张图的那个 `Renderer` 手上就有一份现成的解算结果
    （`renderer.scene.camera[0]`），那是像素真正由它算出来的那个机位。

    **不要自己拿 lookat/distance 推三角函数。** 试过了：按文档公式推出来的机位和
    渲染器实际用的差约 3°（整整 20~40 像素），白白给测量结果塞进一个假误差。

    场景相机只给 pos / forward / up，没有现成的 3x3 矩阵，得自己拼：相机看向 -z，
    所以 z 轴 = 视线方向取反；x 轴向右 = y × z。视场角同理，从视锥的
    top/near 反解（`fovy = 2·atan(top/near)`），这才是 mjr_render 真正用的那个。
    """
    rend = arm.u._overhead_obs_renderer
    rend.update_scene(arm.u.data, camera=arm.u._overhead_obs_cam)
    c = rend.scene.camera[0]

    z = -np.asarray(c.forward, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.cross(np.asarray(c.up, dtype=float), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    near = float(c.frustum_near)
    fovy = float(np.degrees(2.0 * np.arctan(float(c.frustum_top) / near)))
    print(f"    [俯视] 场景相机0 pos={np.round(c.pos,4)} fovy={fovy:.4f}°"
          f" 场景内相机数={len(rend.scene.camera)}"
          f" 观测图={np.asarray(arm.obs['overhead_camera']).shape}")
    # 现渲一张，和观测里那张比：一样，就说明「我读的位姿」和「渲染用的位姿」是同一个
    fresh = rend.render()
    got = np.asarray(arm.obs["overhead_camera"])
    diff = int(np.count_nonzero(np.any(np.abs(fresh.astype(int) - got.astype(int)) > 2, axis=-1)))
    print(f"    现渲一张 vs 观测那张：{diff} 个像素不一样（共 {got.shape[0]*got.shape[1]}）")
    return {"pos": np.asarray(c.pos, dtype=float).copy(),
            "mat": np.column_stack([x, y, z]),
            "fovy": fovy}


def blob_centroid(rgb, color="red", *, verbose=False):
    """画面里那个色块的质心（像素）和它占多少像素。

    **按连通域取最大的一块，不是所有红像素一锅端。** 画面里往往不止一处红：机械臂上
    的红色零件、方块自己的影子边缘都可能过阈值，一锅端出来的质心落在半空。真机上更
    是这样 —— 桌布、别的红色物体。取最大的一块，同时把每一块都打出来备查。
    """
    import cv2

    mask = vision.color_mask(rgb, color).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    blobs = [(int(stats[i, cv2.CC_STAT_AREA]),
              (float(cent[i][0]), float(cent[i][1])),
              tuple(int(v) for v in stats[i, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                              cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]))
             for i in range(1, n)]
    blobs.sort(key=lambda b: -b[0])
    if verbose:
        print("    红色连通域（面积 / 质心 / 外框 x,y,w,h）：")
        for a, c, b in blobs[:5]:
            print(f"      {a:6d}  ({c[0]:6.1f},{c[1]:6.1f})  {b}")
    if not blobs:
        return None, 0
    return np.array(blobs[0][1]), blobs[0][0]


def view_ray(mount, uv, size):
    """像素 → 世界系里的视线（起点=相机原点，方向=单位向量）。"""
    fx, fy, cx, cy = intr(mount, size)
    d_cam = np.array([(uv[0] - cx) / fx, -(uv[1] - cy) / fy, -1.0])
    dw = np.asarray(mount["mat"], float) @ d_cam
    return np.asarray(mount["pos"], float), dw / np.linalg.norm(dw)


def intr(mount, size):
    """这台相机的内参。`k` 有值就用标定出来的那份，没有就按视场角算。"""
    if mount.get("k") is not None:
        return np.asarray(mount["k"], dtype=np.float64)
    return vision.intrinsics(mount["fovy"], size[0], size[1])


def triangulate(p1, d1, p2, d2):
    """两条视线取最近点。返回 (中点, 两视线夹角°, 两条视线隔多远)。"""
    w0 = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    dd, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-12:
        return None, 0.0, float("nan")       # 两条视线平行，三角测量没定义
    t1 = (b * e - c * dd) / den
    t2 = (a * e - b * dd) / den
    q1, q2 = p1 + t1 * d1, p2 + t2 * d2
    ang = float(np.degrees(np.arccos(np.clip(float(d1 @ d2), -1.0, 1.0))))
    return 0.5 * (q1 + q2), ang, float(np.linalg.norm(q1 - q2))


def detect_discs(rgb, *, verbose=False):
    """画面里那些标识点圆盘（中性灰、贴地、直径 40mm）。

    判据是「不带色的亮斑」：三个通道彼此差得少（灰），且整体亮。地是暗灰、机械臂是
    黄的、方块是红的，都不会进来。影子是暗灰，被亮度门槛挡掉。
    """
    import cv2

    a = np.asarray(rgb).astype(np.int16)
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    # 盘面灰约 178、地面约 155，差 20 来个灰阶，阈值卡在中间；顺便要求「不带色」
    mask = ((mx - mn) <= 30) & (mn >= 168)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (120 <= area <= 3000):
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        # 圆盘被压扁成椭圆（俯视 34°），长宽比要像椭圆，不像长条杂物
        if not (1.0 <= max(w, h) / max(1, min(w, h)) <= 3.5):
            continue
        sel = _lab == i
        out.append({"uv": np.array([float(cent[i][0]), float(cent[i][1])]),
                    "area": area, "box": (x, y, w, h),
                    "rgb": np.round(a[sel].mean(axis=0), 1)})
    out.sort(key=lambda d: -d["area"])
    if verbose:
        print(f"    圆盘候选 {len(out)} 个：")
        for d in out[:12]:
            print(f"      {d['area']:5d}px  质心 ({d['uv'][0]:6.1f},{d['uv'][1]:6.1f})"
                  f"  外框 {d['box']}  平均色 {d['rgb']}")
    return out


def calibrate_overhead(rgb, mount, world_pts, *, verbose=True):
    """用标识点把俯视相机的内参标出来。

    标识点的世界坐标是**事先知道的**（`core.geom` 里那圈半径 200mm 的盘，绕底座摆
    在 ±20°/±60°/±100°）—— 这就是真机上贴在工位上的标定靶：位置已知、拿来解相机。

    位姿不动，只解内参（fx, fy, cx, cy）。理由：竖直方向本来就是对上的（真值方块顶面
    投回去和画面里的方块差 0.3 像素），错的只在水平方向那一项；位姿和焦距一起解的话，
    靶点全在一个平面上，两个未知量会互相当替身，解不稳。

    返回 (标定好的内参 或 None, 事实字典)。
    """
    discs = detect_discs(rgb, verbose=verbose)
    if not discs:
        return None, {"why": "画面里没找到标识点圆盘"}

    k0 = intr(mount, (rgb.shape[1], rgb.shape[0]))
    pos = np.asarray(mount["pos"], dtype=np.float64)
    mat = np.asarray(mount["mat"], dtype=np.float64)
    wpts = np.asarray(world_pts, dtype=np.float64)

    def project(k, p_world):
        fx, fy, cx, cy = k
        p = mat.T @ (p_world - pos)
        depth = -float(p[2])
        if depth <= 1e-6:
            return None
        return np.array([cx + fx * float(p[0]) / depth, cy - fy * float(p[1]) / depth])

    def match(k):
        """拿一份内参把已知点投出去，跟画面里找到的圆盘就近配。"""
        ps = []
        for wp in wpts:
            pred = project(k, wp)
            if pred is None:
                continue
            best, bd = None, 1e9
            for d in discs:
                dd = float(np.linalg.norm(pred - d["uv"]))
                if dd < bd:
                    best, bd = d, dd
            if best is not None and bd < 60.0:
                ps.append((wp, best["uv"]))
        return ps

    def solve(pairs):
        """闭式解。

        u = cx + fx·(x/z)、v = cy − fy·(y/z)：位姿定了之后，`x/z`、`y/z` 是已知数，
        所以 u 对 (fx, cx) 是**线性**的，v 对 (fy, cy) 也是。直接最小二乘一次解出来，
        不用迭代 —— 迭代那个上次掉进局部极小，六个点残差到了 42 像素。
        """
        su, sv, uo, vo = [], [], [], []
        for wp, uv in pairs:
            p = mat.T @ (wp - pos)
            depth = -float(p[2])
            su.append(float(p[0]) / depth)
            sv.append(-float(p[1]) / depth)
            uo.append(float(uv[0]))
            vo.append(float(uv[1]))
        A = np.column_stack([su, np.ones(len(su))])
        fx, cx = np.linalg.lstsq(A, np.array(uo), rcond=None)[0]
        A = np.column_stack([sv, np.ones(len(sv))])
        fy, cy = np.linalg.lstsq(A, np.array(vo), rcond=None)[0]
        return np.array([fx, fy, cx, cy])

    pairs = match(k0)
    if verbose:
        print(f"    配上 {len(pairs)} 对：已知世界点 对 画面里的盘")
    if len(pairs) < 4:
        return None, {"why": f"只配上 {len(pairs)} 对，不够解内参"}

    k = solve(pairs)
    err = np.asarray([np.linalg.norm(np.asarray(project(k, wp)) - uv) for wp, uv in pairs])
    # 再配对一次：上次配对是按没标定的内参配的，标定完可能配得更准，重配重解
    pairs2 = match(k)
    if len(pairs2) >= 4:
        k2 = solve(pairs2)
        err2 = np.asarray([np.linalg.norm(np.asarray(project(k2, wp)) - uv)
                           for wp, uv in pairs2])
        if err2.max() < err.max():
            pairs, k, err = pairs2, k2, err2
    info = {"n": len(pairs), "k": np.round(k, 2), "k0": np.round(k0, 2),
            "err_px": np.round(err, 2).tolist(),
            "err_max": float(err.max()) if len(err) else float("nan")}
    if verbose:
        print(f"    没标定 {np.round(k0,2)}  →  标定后 {np.round(k,2)}")
        print(f"    各点残差 {np.round(err,2).tolist()} 像素，最大 {info['err_max']:.2f}")
    # 残差里含「圆盘质心本身估歪」的那一两个像素（盘面有一侧在阴影里），
    # 5 像素（约 6mm）以内就算标上了
    return (k if info["err_max"] < 5.0 else None), info


# ---------------------------------------------------------------- 一拍

def look(arm, cams, label, mounts):
    """拍一眼：腕部和俯视各记一张、各找一次色块。返回这一拍的记录。"""
    rec = {}
    for name in ("sim_wrist", "sim_overhead"):
        rgb = np.asarray(cams[name].shot())
        c, npx = blob_centroid(rgb, verbose=(label.endswith("1") and name == "sim_overhead"))
        rec[name] = {"rgb": rgb, "uv": c, "px": npx, "mount": mounts[name]()}
    print(f"  {label}：腕部 {rec['sim_wrist']['px']} 像素 / 俯视 {rec['sim_overhead']['px']} 像素")
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    arm = make("arm", "sim", cube_order=("red",))
    try:
        arm.reset(seed=0)
        cams = {n: make("camera", n, source=arm) for n in ("sim_wrist", "sim_overhead")}

        u = arm.u
        mounts = {
            "sim_wrist": cams["sim_wrist"].mount,
            "sim_overhead": lambda: overhead_mount(arm),
        }

        truth = np.asarray(arm.cube_pos(0), dtype=float)
        print(f"方块真值 {np.round(truth, 4)}")

        def walk(pts):
            return follow.follow(arm, pts, vmax=speed_vmax(SPEED), amax=acc_amax(),
                                 grip=1.0, final_tol=0.004)

        # 走到方块上方 80mm —— 跟 pick 里进场那一段一样高
        above = truth + np.array([0.0, 0.0, 0.08])
        print("进场：", walk([above]), " TCP", np.round(arm.tcp(), 3))
        a = look(arm, cams, "腕部第 1 个位置", mounts)

        # 横移 30mm，同一个相机换个位置再看一眼
        side = above + np.array([0.03, 0.0, 0.0])
        print("横移：", walk([side]), " TCP", np.round(arm.tcp(), 3))
        b = look(arm, cams, "腕部第 2 个位置", mounts)

        # 手臂抬走挪开，让俯视相机看得到方块（手臂就在方块正上方，会挡住）
        print("抬走：", walk([truth + np.array([0.0, 0.0, 0.26])]), " TCP", np.round(arm.tcp(), 3))
        print("挪开：", walk([[truth[0] + 0.22, truth[1] + 0.22, 0.26]]), " TCP", np.round(arm.tcp(), 3))
        c = look(arm, cams, "手臂让开", mounts)
        arm.step_idle()          # 让 obs 跟上最后一步

        # ---- 标定俯视相机：用手臂让开这一拍，画里圆盘最多 ----
        print("\n标定俯视相机（标识点圆盘，世界坐标事先知道）：")
        marks = np.column_stack([np.vstack([PICK_XY, PLACE_XY]),
                                 np.full(len(PICK_XY) + len(PLACE_XY), DISC_Z)])
        k_fit, k_info = calibrate_overhead(c["sim_overhead"]["rgb"],
                                           c["sim_overhead"]["mount"], marks)
        if k_fit is None:
            print(f"    标不出来：{k_info.get('why')} —— 下面俯视那一路还是用没标定的内参，"
                  f"数字不可信")
        else:
            for rec in (a, b, c):
                rec["sim_overhead"]["mount"]["k"] = k_fit

        # ---- 先对账：把真值投进每张画面，看落不落在色块上 ----
        print("\n对账（把方块真值投回画面 vs 画面里色块质心，差了说明相机位姿/内参不对）：")
        for tag, rec in (("A", a), ("B", b), ("C", c)):
            for name in ("sim_wrist", "sim_overhead"):
                rgb = rec[name]["rgb"]
                if rec[name]["uv"] is None:
                    continue
                k = intr(rec[name]["mount"], (rgb.shape[1], rgb.shape[0]))
                want = vision.project(rec[name]["mount"]["pos"], rec[name]["mount"]["mat"], k, truth)
                rec[name]["proj"] = want
                if want is None:
                    print(f"  {tag}-{name[4:]:9s} 真值投不出画面（在相机背后）")
                    continue
                d = float(np.linalg.norm(np.asarray(want) - rec[name]["uv"]))
                print(f"  {tag}-{name[4:]:9s} 真值投在 {np.round(want,1)}"
                      f"  色块质心 {np.round(rec[name]['uv'],1)}  差 {d:5.1f} 像素")
        import cv2
        for tag, rec in (("A", a), ("B", b), ("C", c)):
            for name in ("sim_wrist", "sim_overhead"):
                pic = np.asarray(rec[name]["rgb"]).copy()
                for uv, col in ((rec[name]["uv"], (0, 0, 255)),                     # 红叉=色块质心
                                (rec[name].get("proj"), (0, 255, 0))):              # 绿叉=真值投回来
                    if uv is None:
                        continue
                    p = (int(round(float(uv[0]))), int(round(float(uv[1]))))
                    cv2.drawMarker(pic, p, col, cv2.MARKER_CROSS, 21, 2)
                # 真值方块的八条棱投到画面上：线框套住了画里的方块，就说明这套
                # 相机位姿 + 内参跟渲染这张图的那台是同一台
                mnt = rec[name]["mount"]
                kk = intr(mnt, (pic.shape[1], pic.shape[0]))
                cor = vision.project_many(mnt["pos"], mnt["mat"], kk,
                                          vision.cube_corners(truth, 0.015, 0.0))
                if cor is not None:
                    for i in range(8):
                        for j in (1, 2, 4):
                            q = i ^ j
                            if q > i:
                                cv2.line(pic, tuple(np.round(cor[i]).astype(int)),
                                         tuple(np.round(cor[q]).astype(int)), (0, 255, 0), 1)
                cv2.imwrite(str(OUT / f"{tag}_{name}.png"),
                            cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))
                msk = (vision.color_mask(rec[name]["rgb"], "red") * 255).astype(np.uint8)
                cv2.imwrite(str(OUT / f"{tag}_{name}_mask.png"), msk)

        # ---- 比两组 ----
        def pair(rec1, name1, rec2, name2):
            size1 = rec1[name1]["rgb"].shape[1::-1]          # (w,h)
            size2 = rec2[name2]["rgb"].shape[1::-1]
            if rec1[name1]["uv"] is None or rec2[name2]["uv"] is None:
                return None
            p1, d1 = view_ray(rec1[name1]["mount"], rec1[name1]["uv"], size1)
            p2, d2 = view_ray(rec2[name2]["mount"], rec2[name2]["uv"], size2)
            mid, ang, gap = triangulate(p1, d1, p2, d2)
            if mid is None:
                return None
            # 同一个色块在两台相机里各是多少像素 —— 差得多说明「配对」这件事有多难
            return {"base": float(np.linalg.norm(p2 - p1)), "ang": ang, "gap": gap,
                    "mid": mid, "err": float(np.linalg.norm(mid - truth)),
                    "uv1": rec1[name1]["uv"], "uv2": rec2[name2]["uv"]}

        rows = [("A 腕部挪两下", pair(a, "sim_wrist", b, "sim_wrist")),
                ("B 腕部 + 全局", pair(a, "sim_wrist", c, "sim_overhead"))]

        # 量到的那个点是**方块顶面中心**，不是方块中心：色块是可见的那一片，
        # 俯视看下去主要就是顶面。所以除了对方块中心，也报一列对顶面中心差多少 ——
        # 那一列才是「这一眼量得准不准」，另一列差的是 15mm 的定义差（顶面比中心高半个方块）。
        top = truth + np.array([0.0, 0.0, CUBE_HALF])
        print(f"\n结果（方块中心真值 {np.round(truth,4)}，顶面中心 {np.round(top,4)}）：")
        print(f"  {'方案':<16s}{'基线':>8s}{'夹角':>8s}{'视线间距':>9s}"
              f"{'算出来的点':>24s}{'对顶面':>9s}{'对方块中心':>11s}")
        for tag, r in rows:
            if r is None:
                print(f"  {tag:<16s} 算不出来")
                continue
            e_top = float(np.linalg.norm(r["mid"] - top))
            print(f"  {tag:<16s}{r['base']*1000:7.0f}mm{r['ang']:7.1f}°"
                  f"{r['gap']*1000:8.1f}mm"
                  f"   {str(np.round(r['mid'], 4)):>20s}"
                  f"{e_top*1000:8.1f}mm{r['err']*1000:10.1f}mm")

        print(f"\n画面存这儿：{OUT}")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
