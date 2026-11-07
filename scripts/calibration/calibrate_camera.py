"""从 `out/calib_views.json` 那批视图解相机内参。

**内参跟格子边长没关系。** fx/fy/cx/cy 是纯像素量，`--square` 给多大多小都不影响
它们，只影响解出来的板子距离和畸变系数的标度。所以「2.8cm 到底量的是格子还是码」
这件事**卡不住内参这一步**，只影响畸变系数和三阶段里后面那些要真实尺度的量。

**harness 现在没地方放畸变。** `core/interfaces.py` 的 `mount()` 只有
`pos/mat/fovy`，`impls/perception.py::_intr` 优先读 `mount["k"]`（3x3），
没有畸变系数的槽。所以这份输出里畸变单独记一份，接进 harness 时先只用 k ——
1920x1080 下这条臂的畸变有多大，看下面的 `畸变` 一行自己判断值不值得加槽。

跑法（在 G:\\sim2real\\so101-real-harness 下）：
    G:/anaconda/envs/sim2real/python.exe calibrate_camera.py
    G:/anaconda/envs/sim2real/python.exe calibrate_camera.py --square 0.028 --drop 20
出参：G:\\sim2real\\so101-real-harness\\assets\\camera_calibration\\real_wrist.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 图像尺寸。**必须和采图时那一档一致** —— 内参按像素算，换分辨率就作废。
CAM_W, CAM_H = 1920, 1080


def detect(board, path: Path):
    """在一张图上找板子，返回 (角点, 角点编号) 或 None。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    det = cv2.aruco.CharucoDetector(board)
    ch_c, ch_i, _m_c, _m_i = det.detectBoard(img)
    if ch_c is None or ch_i is None or len(ch_c) < 4:
        return None
    return ch_c, ch_i, img.shape


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="解真机腕相机内参")
    ap.add_argument("--views", type=Path, default=ROOT / "out" / "calib_views.json")
    ap.add_argument("--squares", default="8x6", help="板子的格子数，要和 make_charuco 出图时一致")
    ap.add_argument("--square", type=float, default=0.028,
                    help="一个格子的物理边长（米）。**不影响 fx/fy/cx/cy**，见文件头")
    ap.add_argument("--drop", type=float, default=0.0,
                    help="单张重投影误差超过这个像素数就剔掉重解一轮。0 = 不剔")
    ap.add_argument("--use", default="free", choices=["free", "pp", "aspect", "both"],
                    help="写进 json 的那一组。四个都解一遍打表，见那张表再定")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "assets" / "camera_calibration" / "real_wrist.json")
    args = ap.parse_args()

    import make_charuco as mc                                    # noqa: PLC0415
    sx, sy = (int(v) for v in args.squares.lower().split("x"))
    _, board = mc.build(sx, sy)

    recs = json.loads(args.views.read_text(encoding="utf-8"))
    objp: list = []
    imgp: list = []
    used: list[str] = []
    for r in recs:
        p = args.views.parent / r["file"]
        got = detect(board, p)
        if got is None:
            print(f"  {r['file']}  再检测失败，丢")
            continue
        ch_c, ch_i, _shape = got
        o, i = board.matchImagePoints(ch_c, ch_i)
        if o is None or len(o) < 4:
            print(f"  {r['file']}  配不上点，丢")
            continue
        # **必须 float32。** calibrateCamera 要 Point3f/Point2f，喂 float64 直接抛
        # 「objectPoints should contain vector of vectors of points of type Point3f」。
        objp.append((o.astype(np.float32) * args.square))
        imgp.append(i.astype(np.float32))
        used.append(r["file"])

    print(f"\n用上 {len(objp)}/{len(recs)} 张")
    if len(objp) < 4:
        print("**少于 4 张解不出来。** 先回去多采几张。")
        return 1

    # 主点/长宽比钉不钉，是这条臂上**唯一还能压 fx 不确定性**的旋钮。
    # 道理：视图都挤在相近距离时，「fx 大 + 板子远」和「fx 小 + 板子近」在像素上
    # 几乎一样，解算器可以靠挪主点来偷这个自由度。把主点按住，这条偷法就断了。
    # 该不该钉，看下面那张表里**钉了之后均方根涨多少**：
    # 涨不动 = 原来那组解本来就没在偷，free 可信；涨很多 = free 的 fx 不可信。
    FLAG_SETS = {
        "free": 0,
        "pp": cv2.CALIB_FIX_PRINCIPAL_POINT,
        "aspect": cv2.CALIB_FIX_ASPECT_RATIO,
        "both": cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO,
    }

    def solve(objs, imgs, flags=0):
        rms, k, dist, _rv, _tv = cv2.calibrateCamera(
            objs, imgs, (CAM_W, CAM_H), None, None, flags=flags)
        return rms, k, dist

    def per_image(objs, imgs, k, dist):
        """每张图单独解一次板子位姿再投回去，看是哪几张把均方根抬起来的。

        位姿要现解（`solvePnP`）—— `calibrateCamera` 返回的那组是整批一起的，
        缺一张就对不上了。**`projectPoints` 的 `k` 和 `dist` 不能漏**，
        漏了它只报「missing required argument 'cameraMatrix'」。

        **两边都得 `reshape(-1, 2)`。** `matchImagePoints` 给的 imagePoints 是
        `(N,1,2)`，`projectPoints` 回来的是 `(N,1,2)`，而 `rv.reshape(-1,2)` 是
        `(N,2)` —— 直接相减会被 numpy 广播成 `(N,N,2)`，下面那个 `sum(axis=1)`
        就求错了轴，整列数字都是假的（曾经印出 1000+ px）。旧 cv2 的
        `matchImagePoints` 返回 `(N,2)`，正好躲过去；换 4.13 之后变成 `(N,1,2)`，
        这个潜伏的广播当场炸出来。
        """
        out = []
        for name, o, i in zip(used, objs, imgs):
            _ok, rvec, tvec = cv2.solvePnP(o, i, k, dist)
            rv, _jac = cv2.projectPoints(o, rvec, tvec, k, dist)
            d = i.reshape(-1, 2) - rv.reshape(-1, 2)
            e = float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
            out.append((name, e))
        return out

    rms, k, dist = solve(objp, imgp, FLAG_SETS[args.use])
    per = per_image(objp, imgp, k, dist)

    if args.drop > 0:
        keep = [(n, o, i) for (n, _e), o, i in zip(per, objp, imgp) if _e <= args.drop]
        if len(keep) < 4:
            print(f"**门槛 {args.drop}px 太紧，只剩 {len(keep)} 张。** 不提了，用全量解。")
        elif len(keep) < len(objp):
            print(f"\n剔掉 {len(objp) - len(keep)} 张（单张误差 > {args.drop}px），重解：")
            for n, e in per:
                if e > args.drop:
                    print(f"  {n}  {e:.3f} px")
            used = [n for n, _o, _i in keep]
            objp = [o for _n, o, _i in keep]
            imgp = [i for _n, _o, i in keep]
            rms, k, dist = solve(objp, imgp, FLAG_SETS[args.use])
            per = per_image(objp, imgp, k, dist)

    print(f"\n{'钉法':<8}{'fx':>10}{'fy':>10}{'cx':>10}{'cy':>10}{'均方根':>10}")
    for _name, _fl in FLAG_SETS.items():
        _r, _k, _d = solve(objp, imgp, _fl)
        mark = " <-用这个" if _name == args.use else ""
        print(f"{_name:<8}{_k[0, 0]:>10.2f}{_k[1, 1]:>10.2f}"
              f"{_k[0, 2]:>10.2f}{_k[1, 2]:>10.2f}{_r:>10.3f}{mark}")
    print("（fx 在各个钉法之间漂多少，就是这条内参的可信区间）")

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    d = np.asarray(dist, dtype=np.float64).reshape(-1)

    print(f"\n图像      {CAM_W}x{CAM_H}")
    print(f"张数      {len(objp)}")
    print(f"均方根    {rms:.3f} px   （<1px 好，1-2px 能用，>2px 别用）")
    print(f"fx fy     {fx:.2f}  {fy:.2f}   （差 {abs(fx - fy) / fx * 100:.2f}%）")
    print(f"cx cy     {cx:.2f}  {cy:.2f}   （画面正中是 {CAM_W / 2:.1f}  {CAM_H / 2:.1f}）")
    print(f"视场角    {2 * np.degrees(np.arctan(CAM_W / (2 * fx))):.2f}° 水平 x "
          f"{2 * np.degrees(np.arctan(CAM_H / (2 * fy))):.2f}° 垂直")
    print(f"畸变      {np.round(d, 4)}")
    print("\n逐张重投影误差（px）：")
    for name, e in sorted(per, key=lambda t: -t[1]):
        print(f"  {name}  {e:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "_note": "real/calib_views.py 采图 + calibrate_camera.py 解。"
                 "harness 的 mount() 只吃 k，畸变是另外记的（见文件头）。",
        "image_size": [CAM_W, CAM_H],
        "k": [[float(v) for v in row] for row in k],
        "dist": [float(v) for v in d],
        "rms_px": float(rms),
        "n_views": len(objp),
        "square_m": args.square,
        "views": used,
        "_calibrated_at": "2026-10-14",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写到 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
