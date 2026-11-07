"""验 `RealWristCamera.mount()` 反推出来的世界坐标对不对。**不动机械臂。**

**为什么能验。** 桌上那块 ChArUco 靶标是**已知几何**：8x6 个格子，每格边长写在
`assets/camera_calibration/real_wrist.json` 的 `square_m`（0.028 m）。所以把画面里
认到的角点按 `mount()` 给的机位反推回桌面那个平面，量出来的相邻角点间距**必须还是
28mm**。差了就是这条链上有东西错了。

**这一条同时卡住三样**（它们乘在一起才决定像素落在哪）：

    · 内参（fx/fy 决定这条射线有多陡）
    · 机位（手眼 + 这一下模型正解给出的相机在哪、朝哪）
    · 桌面那个平面在基座系里的高度

**反着还能解一次。** 反过来问「平面压在哪个高度、量出来的间距才正好是 28mm」，
那个高度就是桌面在基座系里的 z。它跟手眼顺手给出来的板面高度（两块独立场次分别
是 −0.0095 / −0.0082 m）应该对得上 —— 这是第二条互相独立的证据。

跑法（在 G:\\sim2real\\so101-real-harness 下）：
    G:/anaconda/envs/sim2real/python.exe real/mount_check.py
出图：G:\\sim2real\\so101-real-harness\\out\\mount_check_<时间戳>.png（整幅，不裁）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import vision                                       # noqa: E402
from core.registry import make                                # noqa: E402
from real import cameras as _cameras                          # noqa: E402
from real.arm import RealArm                                  # noqa: E402
from servo_bridge import open_controller                      # noqa: E402

OUT = _ROOT / "out"

# 找「间距正好等于格子边长」那个平面时的扫描范围（基座系 z，米）。板子摊在臂基座
# 平面上下一点的地方，给足余量，但别给到相机后面去。
Z_LO, Z_HI = -0.30, 0.05
Z_STEP = 0.001


def spacing_stats(ids, uvs, mount, z_plane):
    """把角点反推到 z_plane 那个平面，量相邻角点的间距（米）。反推不出来返回 None。

    ChArUco 角点编号是 `行 x 7 + 列`（8x6 的板子每行 7 个角点）。所以「同一行隔壁」
    是 +1、但列到 6 就得停（+1 会跳到下一行第一个）；「下一行同一个」是 +7。
    """
    pts = {}
    for cid, uv in zip(ids, uvs):
        p = vision.backproject(mount["pos"], mount["mat"], mount["k"],
                               float(uv[0]), float(uv[1]), z_plane)
        if p is None:
            return None
        pts[int(cid)] = p
    ds = []
    for cid, p in pts.items():
        for nb in (cid + 1, cid + 7):
            if nb not in pts:
                continue
            if nb == cid + 1 and cid % 7 == 6:
                continue                     # 行尾不跟下一行的第一个连
            ds.append(float(np.linalg.norm(pts[nb] - p)))
    if not ds:
        return None
    return np.asarray(ds), pts


def solve_plane_z(ids, uvs, mount, square, lo=Z_LO, hi=Z_HI, step=Z_STEP):
    """反求「让相邻角点间距正好等于 square」的那个平面高度。解不出返回 None。

    间距随平面高度单调变（平面离相机越远，同样像素跨的物理距离越大），所以先粗扫
    找变号的那一段，再二分。**不假设方向**，两个方向都认。
    """
    zs = np.arange(lo, hi + step, step)
    prev = None
    bracket = None
    for z in zs:
        got = spacing_stats(ids, uvs, mount, float(z))
        if got is None:
            continue
        med = float(np.median(got[0]))
        if prev is not None and (prev[1] - square) * (med - square) <= 0.0:
            bracket = (prev[0], float(z))
            break
        prev = (float(z), med)
    if bracket is None:
        return None
    a, b = bracket
    for _ in range(40):
        m = 0.5 * (a + b)
        got = spacing_stats(ids, uvs, mount, m)
        if got is None:
            return None
        med = float(np.median(got[0]))
        got_a = spacing_stats(ids, uvs, mount, a)
        if (float(np.median(got_a[0])) - square) * (med - square) <= 0.0:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


def undisto(uvs, k3, dist):
    """把像素点按标定的畸变系数拉回理想针孔像素。

    `core/vision.backproject` 是个**纯针孔**，不吃畸变。这颗镜头在 1920x1080 下
    k2 = −0.04，画面边上畸变很大（归一化半径 1.1 处径向就压掉 8%）。所以拿原始
    像素去喂它，越靠画面边越偏。这一步是把这个差补掉。
    """
    if k3 is None or dist is None:
        return uvs
    return cv2.undistortPoints(np.asarray(uvs, dtype=np.float32).reshape(-1, 1, 2),
                               k3, dist, P=k3).reshape(-1, 2).astype(np.float64)


def replay(arm, cal, det, board, square, z_plane, views: Path, limit: int,
           skip: int = 0, k3=None, dist=None, undistort: bool = False):
    """拿已存的标定图离线复算：同一个物理点，各姿态反推出来该是同一个坐标。

    **这是「坐标对不对」最硬的那条。** 上面那个间距只能卡住标度（内参、机位高度、
    机位朝向），卡不住机位整体的横移 —— 相机平移到别处，格子照样是刚性 28mm。这条
    换 23 个差得很开的姿态，把同一块没动过的板子各反推一次：横移错了，板心就会
    随着姿态飘。

    位姿走的是 `real/cameras.py::pose_from_fk`，**和线上 `mount()` 同一个函数**。
    """
    recs = json.loads(views.read_text(encoding="utf-8"))
    if skip > 0:
        recs = recs[skip:]
    if limit > 0:
        recs = recs[:limit]
    rows, lost = [], 0
    for r in recs:
        img = cv2.imread(str(views.parent / r["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            lost += 1
            continue
        ch_c, ch_i, _m_c, _m_i = det.detectBoard(img)
        if ch_c is None or len(ch_c) < 6:
            lost += 1
            continue
        ids = [int(v) for v in np.asarray(ch_i).reshape(-1)]
        uvs = np.asarray(ch_c, dtype=np.float64).reshape(-1, 2)
        if undistort:
            uvs = undisto(uvs, k3, dist)
        # **用记下来的实测 raw_ticks，不用当时发的指令值** —— 舵机有静差。
        q = arm._ticks_to_qpos({int(k): int(v) for k, v in r["raw_ticks"].items()})
        mnt = _cameras.pose_from_fk(cal, *arm._fk(q))
        got = spacing_stats(ids, uvs, mnt, z_plane)
        if got is None:
            lost += 1
            continue
        ds, pts = got
        # **两条路线对一对。** 同一个角点，一路走 `solvePnP`（板子位姿 → 基座系），
        # 一路走「像素射线 × 平面」。角点真在板上、板子真在那个平面，两条就该给出
        # 同一个点。这一列是**这批图最直接的可信度**：差多少，就是这套反推在那种
        # 角度下能信到多少。
        gap = None
        if k3 is not None and dist is not None:
            obj, imp = board.matchImagePoints(ch_c, ch_i)
            if obj is not None and len(obj) >= 6:
                ok, rvec, tvec = cv2.solvePnP(
                    (obj.reshape(-1, 3) * square).astype(np.float32),
                    imp.astype(np.float32), k3, dist)
                if ok:
                    p_cv = (cv2.Rodrigues(rvec)[0]
                            @ (obj.reshape(-1, 3) * square).T).T + tvec.reshape(3)
                    # cv2 相机系 -> MuJoCo 相机系，再摆到基座系。
                    p_b = (mnt["mat"] @ (p_cv * np.array([1.0, -1.0, -1.0])).T).T + mnt["pos"]
                    ref = np.array([pts[i] for i in ids])
                    gap = float(np.median(np.linalg.norm(p_b - ref, axis=1)))
        # **每张图反求一遍「间距正好是格子边长」的平面高度。** 板子要是真水平、位姿
        # 又是对的，这一列该处处等于板面高度。它散开 = 板子本身是斜的（那这条复算
        # 的前提就不成立），或者这一张的角度太刁、打平面打不准。
        z_i = solve_plane_z(ids, uvs, mnt, square)
        # 相机这一刻离竖直线多少度。复算精度最吃这个：光轴越斜，打平面越刁。
        tilt = float(np.degrees(np.arccos(np.clip(-mnt["mat"][2, 2], -1.0, 1.0))))
        # **角点在画面里离主点多远（归一化半径）。** 镜头畸变就是按这个半径来的，而
        # `core/vision.backproject` 是个纯针孔、根本不吃畸变。这一列是用来分清
        # 「误差是角度引起的」还是「是畸变引起的」：角度和半径在这一批图里是高度
        # 重合的两个量，只有把半径单列出来才看得出谁跟着谁走。
        if k3 is not None:
            rad = float(np.median(np.hypot((uvs[:, 0] - k3[0, 2]) / k3[0, 0],
                                           (uvs[:, 1] - k3[1, 2]) / k3[1, 1])))
        else:
            rad = float("nan")
        rows.append({"file": r["file"], "pts": pts, "med": float(np.median(ds)),
                     "z": z_i, "tilt": tilt, "gap": gap, "rad": rad})
    if len(rows) < 4:
        print(f"\n复算只活下来 {len(rows)} 张，太少，下不了结论（丢了 {lost} 张）。")
        return None

    # **按角点编号逐个看，不要看「一张图里所有角点的平均」。** 每张认到的角点数目
    # 差得很远（极端姿态下只认到板子一角），平均值跟着认到几个就走，那不是坐标错。
    # 编号是钉死在板子上的：21 号角点在任何一张图里都是**同一个物理点**。
    per_id: dict[int, list[np.ndarray]] = {}
    for row in rows:
        for k, p in row["pts"].items():
            per_id.setdefault(k, []).append(p)
    devs = {}
    for k, ps in per_id.items():
        if len(ps) < 6:
            continue
        arr = np.array(ps)
        devs[k] = float(np.linalg.norm(arr - arr.mean(axis=0), axis=1).max())

    med = np.array([row["med"] for row in rows])
    zs = np.array([np.nan if row["z"] is None else row["z"] for row in rows])
    tilts = np.array([row["tilt"] for row in rows])
    rads = np.array([row["rad"] for row in rows])
    gaps = np.array([np.nan if row["gap"] is None else row["gap"] for row in rows])
    print(f"\n离线复算（{len(rows)} 张已存图，丢了 {lost} 张）：")
    print(f"  相机离竖直（180° = 正下方）   {tilts.min():.0f} ~ {tilts.max():.0f}°")
    print(f"  角点离主点的归一化半径        {np.nanmin(rads):.2f} ~ {np.nanmax(rads):.2f}")
    if devs:
        wid = max(devs, key=devs.get)
        print(f"  同一个角点跨图反推的散度  中位 {np.median(list(devs.values())) * 1000:.2f}"
              f" mm   最差 角点#{wid} {devs[wid] * 1000:.2f} mm"
              f"（{len(devs)} 个角点各出现 >=6 次）")
    print(f"  各图量出来的格子间距      中位 {med.min() * 1000:.2f} ~ "
          f"{med.max() * 1000:.2f} mm（期望 {square * 1000:.1f}）")
    print(f"  各图反求的平面高度        {np.nanmin(zs) * 1000:+.1f} ~ "
          f"{np.nanmax(zs) * 1000:+.1f} mm（手眼给的板面 {z_plane * 1000:+.1f}）"
          f"   标准差 {np.nanstd(zs) * 1000:.1f} mm")
    if not np.all(np.isnan(gaps)):
        print(f"  两条路线（PnP vs 打平面）的出入  中位 "
              f"{np.nanmedian(gaps) * 1000:.2f} mm   最大 "
              f"{np.nanmax(gaps) * 1000:.2f} mm")
    # 谁跟着谁走：两路差 跟 半径 更相关 = 是畸变；跟 倾角 更相关 = 是角度。
    ok = ~np.isnan(gaps) & ~np.isnan(rads)
    if int(ok.sum()) >= 4:
        print(f"  相关系数            跟半径 {np.corrcoef(gaps[ok], rads[ok])[0, 1]:+.3f}"
              f"   跟倾角 {np.corrcoef(gaps[ok], tilts[ok])[0, 1]:+.3f}")
    print("\n  逐张（按两路差最大排，列前 6）：")
    order = np.argsort(-np.nan_to_num(gaps, nan=0.0))
    for k in order[:6]:
        row = rows[int(k)]
        zt = "解不出" if row["z"] is None else f"{row['z'] * 1000:+.1f}"
        gt = "—" if row["gap"] is None else f"{row['gap'] * 1000:.2f}"
        print(f"    {row['file']}  斜 {row['tilt']:4.0f}°  半径 {row['rad']:.2f}"
              f"  间距 {row['med'] * 1000:6.2f}mm  反求 z {zt:>7} mm"
              f"  两路差 {gt:>6} mm")
    return float(max(devs.values())) if devs else None


def main() -> int:                                             # noqa: C901
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="验真机腕相机反推出来的世界坐标")
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--index", type=int, default=1, help="摄像头序号，屏幕上是 camera 1")
    ap.add_argument("--squares", default="8x6")
    ap.add_argument("--views", type=Path, default=OUT / "calib_views.json",
                    help="拿这批已存图离线复算一遍（走的是线上同一个函数，见 "
                         "`real/cameras.py::pose_from_fk`）")
    ap.add_argument("--skip", type=int, default=0, help="复算时跳过前 N 张（分段用）")
    ap.add_argument("--limit", type=int, default=23,
                    help="复算只用前 N 张。**默认 23 = 只有远距那一轮** —— 那轮板子"
                         "全程没动过；近距那轮采之前板子被挪过，混进来就没可比性了")
    ap.add_argument("--no-replay", action="store_true", help="只做实时那一项")
    ap.add_argument("--undistort", action="store_true",
                    help="角点先按标定的畸变系数拉回理想针孔再反推。**用来验「偏差是不是"
                         "畸变引起的」**：反推那条链是纯针孔，不吃畸变，补上这一步两路"
                         "之间的差该塌下去")
    args = ap.parse_args()

    import make_charuco as mc                                  # noqa: PLC0415

    intr = json.loads(_cameras.INTRINSICS_JSON.read_text(encoding="utf-8"))
    square = float(intr["square_m"])
    he = json.loads(_cameras.HAND_EYE_JSON.read_text(encoding="utf-8"))

    sx, sy = (int(v) for v in args.squares.lower().split("x"))
    _, board = mc.build(sx, sy)
    det = cv2.aruco.CharucoDetector(board)

    ctrl = open_controller(args.port)
    arm = RealArm(ctrl=ctrl, speed=0)          # speed 用不上：这一步一步都不发指令
    cam = make("camera", "real_wrist", source=arm, index=args.index)
    print(f"相机 {cam.size()[0]}x{cam.size()[1]}   格子边长 {square * 1000:.1f}mm")
    print(f"手眼那份是 {he['method']}，用了 {he['n_views']} 张，"
          f"解出来的板心在基座系 "
          f"{np.round(he['validated']['target_pos_in_base_m'], 4)}（z 就是板面高度）\n")

    try:
        mount = cam.mount()
        if mount is None:
            print("**`mount()` 返回了 None，什么都验不了。** 见上面那行提示："
                  "要么标定文件缺，要么相机分辨率跟标定对不上。")
            return 1
        print(f"这一刻的机位（模型基座系）  pos {np.round(mount['pos'], 4)}")
        print(f"                            光轴 {-np.round(mount['mat'][:, 2], 4)}"
              f"（MuJoCo 的视线是 −z）")
        print(f"                            内参 k {np.round(mount['k'], 2)}  "
              f"视场角 {mount['fovy']:.2f}°")

        rgb = cam.shot()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ch_c, ch_i, m_c, _m_i = det.detectBoard(bgr)
        n_c = 0 if ch_c is None else len(ch_c)
        n_m = 0 if m_c is None else len(m_c)
        print(f"\n这一帧认出 {n_c} 个角点 / {n_m} 个码")
        if n_c < 6:
            print("**角点太少，验不了。** 靶标得在画面里、别糊。"
                  "（调机械臂去拍它之前先说一声，这一步本来不动臂。）")
            return 1

        ids = [int(v) for v in np.asarray(ch_i).reshape(-1)]
        uvs = np.asarray(ch_c, dtype=np.float64).reshape(-1, 2)
        if args.undistort:
            uvs = undisto(uvs, np.asarray(intr["k"], dtype=np.float64),
                          np.asarray(intr["dist"], dtype=np.float64))

        z_he = float(he["validated"]["target_pos_in_base_m"][2])
        got = spacing_stats(ids, uvs, mount, z_he)
        if got is None:
            print("\n**按手眼给的板面高度反推，射线打不到那个平面。**"
                  "多半是机位的朝向反了（相机系没换成 MuJoCo 那套），或者平面选错了。")
            return 1
        ds, pts = got
        med = float(np.median(ds))
        print(f"\n按手眼给的板面高度 z = {z_he:+.4f} m 反推：")
        print(f"  相邻角点间距   中位 {med * 1000:7.2f} mm   期望 {square * 1000:.1f} mm"
              f"   （差 {med / square - 1:+.1%}）")
        print(f"                 范围 {ds.min() * 1000:.2f} ~ {ds.max() * 1000:.2f} mm"
              f"   标准差 {ds.std() * 1000:.2f} mm")
        cen = np.mean(np.array(list(pts.values())), axis=0)
        print(f"  板心在基座系   {np.round(cen, 4)}")

        z_fit = solve_plane_z(ids, uvs, mount, square)
        print(f"\n反过来求「间距正好 {square * 1000:.1f}mm」的平面高度：")
        if z_fit is None:
            print("  扫描范围内没找到那个平面 —— 说明这条链的**标度**不对"
                  "（内参或机位高度错了），不是差一点。")
        else:
            print(f"  反求 z = {z_fit:+.4f} m     手眼给的板面 z = {z_he:+.4f} m"
                  f"     差 {(z_fit - z_he) * 1000:+.1f} mm")

        img = bgr.copy()
        cv2.aruco.drawDetectedCornersCharuco(img, ch_c, ch_i, (0, 255, 0))
        for cid, uv in zip(ids, uvs):
            cv2.putText(img, str(cid), (int(uv[0]) + 4, int(uv[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        p = OUT / f"mount_check_{int(time.time())}.png"
        cv2.imwrite(str(p), img)
        print(f"\n整幅原图（画了认到的角点）：{p}")

        if not args.no_replay and args.views.exists():
            cal = _cameras.load_mount_cal()
            replay(arm, cal, det, board, square, z_he, args.views, args.limit,
                   args.skip, k3=np.asarray(intr["k"], dtype=np.float64),
                   dist=np.asarray(intr["dist"], dtype=np.float64),
                   undistort=args.undistort)
        elif not args.no_replay:
            print(f"\n没有 {args.views}，跳过离线复算。")
    finally:
        cam.close()
        ctrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
