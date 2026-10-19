"""实验：给仿真方块贴一张 ArUco 码，看俯视相机还认不认得出。

**要回答的只有一句话**：30mm 的方块贴上一张码以后，在 640x480 的俯视画面里
解码器还能不能把它读出来。读得出来，贴码这条路才谈得上往下做；读不出来，
就得先动相机分辨率、换更粗的码型、或者干脆换大方块 —— 不是写代码能绕过去的。

**为什么怀疑读不出来。** 之前量过：方块整个露出来的部分在俯视画面里约 1000 像素
（≈32x32）。码不能占满整个面，四周得留白边（**不留白一定解不出来**，我第一版
拿干净图试的时候就是这么全灭的），留完白、再被俯角压扁，码本身可能只剩十几
像素。合成图上 20 像素的 4x4 码能解，渲染出来那十几像素能不能解，只有拍一张看。

**做法三步**：
    1. `cv2.aruco` 生成一张 4x4 码图存成 PNG（白边留 12%）
    2. 在场景 XML 里注入 `<texture>` + `<material>`，把它贴到方块的 geom 上
    3. 俯视相机拍图 → `ArucoDetector` 解码 → 报「认出几个、码在画面里多大」

**走的是 harness 已有的那个口子**：`impls/arm_sim.py::_inject_markers` 就在包
`pick_env.build_object_scene_xml`，这里沿用它，不动只读的 nexus 包。

**一个已知的假象**：MuJoCo 的 2d 贴图会贴到 box 的**每个**面上，所以这个方块
六个面都是同一张码、同一个 id。真机上只贴一张。这里只要「俯视看得见的那一面」
解得出来就算数，检出多个不是 bug，是仿真贴图的必然结果 —— 后面接进定位时要
按面挑（或改成只贴顶面）。

跑：G:/anaconda/envs/sim2real/python.exe G:\\sim2real\\so101-real-harness\\exp_aruco_legibility.py
出图：G:\\sim2real\\so101-real-harness\\out\\aruco_legibility\\
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import impls.arm_sim                                  # noqa: E402,F401  导入即挂上牌子

from core.registry import make                        # noqa: E402

OUT = HERE / "out" / "aruco_legibility"
ASSETS = HERE / "assets"
PNG = ASSETS / "aruco_4x4_50_id7.png"

DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_ID = 7
MARKER_PX = 512          # 码图的边长（像素）。只是为了 PNG 画得干净，跟画面里多大无关
MARGIN_FRAC = 0.12       # 四周留白占多少。**不留白解码器直接罢工**
MODULES = 6              # 4x4 的码加一圈黑边框 = 6x6 个格子。可读性的分母是它
SEEDS = (0, 1, 2, 3, 4, 5)
RESOLUTIONS = ((640, 480), (1280, 960), (1920, 1080))   # 真机上两台相机都是 1080p

TEX_NAME = "aruco_tex"
MAT_NAME = "aruco_mat"


# ---------------------------------------------------------------- 造那张码

def make_marker_png(path: Path) -> None:
    """生成一张码图，四周留白，存成 PNG 给 MuJoCo 当贴图用。"""
    d = cv2.aruco.getPredefinedDictionary(DICT_ID)
    m = cv2.aruco.generateImageMarker(d, MARKER_ID, MARKER_PX)
    pad = int(MARKER_PX * MARGIN_FRAC)
    side = MARKER_PX + 2 * pad
    canvas = np.full((side, side), 255, np.uint8)
    canvas[pad:pad + MARKER_PX, pad:pad + MARKER_PX] = m
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    print(f"码图 {MARKER_PX}px + 白边 {pad}px → {path}")


# ---------------------------------------------------------------- 贴上去

def inject_aruco(png: Path, fb: tuple[int, int]) -> None:
    """把码作为材质贴到方块 geom 上，并把离屏缓冲放大到 `fb`。必须在造环境**之前**调用。

    四处改动都在生成的 XML 字符串上做，不去改 nexus：
        - 插一段 `<asset>`，里面是贴图和材质
        - 方块 geom（`type="box"`）挂上这个材质
        - 方块 geom 的 `rgba` 换成白 —— **留着原色会把整张码染红**，
          黑格变暗红、白格变粉，解码器就未必认得出
        - `<visual>` 里加 `<global offwidth/offheight>` —— **不加就渲染不了**：
          离屏缓冲默认 640x480 是 nexus 写死的，比它大的图 MuJoCo 直接拒绝渲染
    """
    import re

    import so101_nexus.mujoco.pick_env as pe

    orig = pe.build_object_scene_xml
    if getattr(orig, "_aruco_wrapped", False):
        return

    asset = (f'  <asset>\n'
             f'    <texture name="{TEX_NAME}" type="2d" file="{png.as_posix()}"/>\n'
             f'    <material name="{MAT_NAME}" texture="{TEX_NAME}" texuniform="false"/>\n'
             f'  </asset>\n\n')
    global_clause = f'  <visual>\n    <global offwidth="{fb[0]}" offheight="{fb[1]}"/>'
    box_re = re.compile(r'(<geom name="[^"]+" type="box" size="[^"]+")(\s*\n\s*)rgba="[^"]*"')

    def wrapped(*a, **kw):
        xml = orig(*a, **kw)
        if "  <asset>" in xml:
            # 现在的场景是纯方块，本来没有 asset 段。真出现了说明场景变了，
            # 这段插入得改写 —— 不能默默插第二个 asset 段，那会让 MuJoCo 解析出错。
            raise RuntimeError("场景里已经有 <asset> 段了，这段注入要改写")
        for anchor in ("  <visual>", "\n\n  <worldbody>"):
            if anchor not in xml:
                raise RuntimeError(f"没找到 {anchor!r}，XML 结构变了")
        xml = xml.replace("  <visual>", global_clause, 1)
        xml = xml.replace("\n\n  <worldbody>", "\n\n" + asset + "  <worldbody>", 1)
        xml, n = box_re.subn(
            lambda m: (f'{m.group(1)} material="{MAT_NAME}"\n'
                       f'{m.group(2)}rgba="1 1 1 1"'), xml)
        if n == 0:
            raise RuntimeError('XML 里没找到 type="box" 的方块 geom，材质没挂上')
        return xml

    wrapped._aruco_wrapped = True        # 自己防重复包；**别带 _harness_wrapped**，
    pe.build_object_scene_xml = wrapped  # 那个标记是给你接着往上包标识点的


# ---------------------------------------------------------------- 拍 + 解

def shot(arm) -> np.ndarray:
    """问渲染器要一张俯视图。不走 step —— step 会让方块掉、手臂动。"""
    rend = arm.u._overhead_obs_renderer
    rend.update_scene(arm.u.data, camera=arm.u._overhead_obs_cam)
    return np.asarray(rend.render())


def decode(rgb):
    """解出画面里所有的码。返回 [(id, 四角 (4,2) float, 平均边长 px)]，大的在前。"""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT_ID),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(rgb)
    out = []
    if ids is None:
        return out
    for q, i in zip(corners, ids.ravel()):
        q = np.asarray(q, dtype=np.float64).reshape(4, 2)
        edges = [float(np.linalg.norm(q[(k + 1) % 4] - q[k])) for k in range(4)]
        out.append((int(i), q, float(np.mean(edges))))
    out.sort(key=lambda t: -t[2])
    return out


def annotate(rgb, found) -> np.ndarray:
    """把解出来的码框出来，标上边长。"""
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR).copy()
    for i, q, side in found:
        pts = np.round(q).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(bgr, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
        u, v = int(round(q[:, 0].mean())), int(round(q[:, 1].mean()))
        cv2.putText(bgr, f"id{i} {side:.0f}px", (u - 30, v),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return bgr


def zoom(rgb, quad, mag: int = 8, pad: int = 10) -> np.ndarray:
    """把码那一小块裁出来放大 —— 「认不出来」到底长什么样，看这个。"""
    a = np.asarray(rgb)
    u0, v0 = quad.min(axis=0)
    u1, v1 = quad.max(axis=0)
    y0, y1 = max(0, int(v0) - pad), min(a.shape[0], int(v1) + pad)
    x0, x1 = max(0, int(u0) - pad), min(a.shape[1], int(u1) + pad)
    return cv2.resize(a[y0:y1, x0:x1], None, fx=mag, fy=mag, interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------- 主流程

def use_resolution(w: int, h: int) -> None:
    """把 nexus 俯视相机的默认分辨率换掉。

    `impls/arm_sim.py::_make_env` 里写的是 `OverheadCamera()`，吃的是默认值。
    这里在 `so101_nexus` 模块上把类换掉 —— 只改本进程的绑定，不动只读的包，
    也不动 harness 里的任何一行。
    """
    import so101_nexus

    class _Framed(so101_nexus.OverheadCamera):
        def __init__(self, width=w, height=h, fov_deg=45.0, **kw):
            super().__init__(width=width, height=height, fov_deg=fov_deg, **kw)

    so101_nexus.OverheadCamera = _Framed


def probe(width: int, height: int):
    """一种分辨率跑满所有 seed。返回 [(seed, 解出几个, 最大那一面的边长 px)]。"""
    use_resolution(width, height)
    arm = make("arm", "sim", cube_order=("red",))
    rows = []
    try:
        for seed in SEEDS:
            arm.reset(seed=seed)
            rgb = shot(arm)
            found = decode(rgb)
            rows.append((seed, len(found), found[0][2] if found else None))
            cv2.imwrite(str(OUT / f"{width}x{height}_{seed}.png"), annotate(rgb, found))
            if found:
                cv2.imwrite(str(OUT / f"{width}x{height}_{seed}_zoom.png"),
                            cv2.cvtColor(zoom(rgb, found[0][1]), cv2.COLOR_RGB2BGR))
    finally:
        try:
            arm.close()
        except Exception as exc:                # noqa: BLE001
            print(f"关环境时报错（不影响上面的结果）：{exc}")
    return rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if OUT.exists():
        shutil.rmtree(OUT)               # 上一次的图不留着
    OUT.mkdir(parents=True)

    make_marker_png(PNG)
    inject_aruco(PNG, max(RESOLUTIONS))
    cv2.imwrite(str(OUT / "0_marker.png"), cv2.imread(str(PNG)))

    print(f"码型 4x4_{DICT_ID} id={MARKER_ID}   方块 30mm   "
          f"码一共 {MODULES} 格，ArUco 常用门槛 3-4px/格\n")

    for width, height in RESOLUTIONS:
        rows = probe(width, height)
        n_ok = sum(1 for _s, n, _d in rows if n)
        print(f"俯视相机 {width}x{height}   解出 {n_ok}/{len(SEEDS)}")
        print(f"  {'seed':>5s}{'解出':>6s}{'最大那一面':>12s}{'每格':>9s}")
        for seed, n, side in rows:
            if side is None:
                print(f"  {seed:>5d}{n:>6d}{'—':>12s}{'—':>9s}")
            else:
                print(f"  {seed:>5d}{n:>6d}{side:>10.1f}px"
                      f"{side / MODULES:>7.2f}px")
        print()

    print(f"图在 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
