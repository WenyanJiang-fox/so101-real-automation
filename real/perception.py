"""真机感知：在腕上相机的画面里找那个**指定颜色的长方形**，并把它反推到基座系里。

**认什么颜色由配置决定**（`configs/real.yaml` 的 `objects:` 第一件的 `color`），
用的 HSV 区间是 `thresholds.json` 的 `color.hsv.<那个颜色>`。从前这里写死棕色 ——
那一版桌上只摆过一只牛皮纸盒。换目标只要改配置 + 标一次那个颜色，**不用改代码**。

两步，一步管「画面里哪块是它」，一步管「这一眼算不算数」：

    矩形度   从画面里挑出那块。同色系的杂物（当年那只牛皮纸盒旁边有一块卡其色布）
             颜色分不开；但方块是平的、布是皱的，`minAreaRect` 填充率分得很开（实测
             0.86-0.88 对 0.42-0.71）。长宽比**不用** —— 实测两者范围重叠（盒子
             1.68-1.70 落在布 1.05-3.11 里），卡了会误杀。
    重合度   把相机位姿 + 标定内参算出来的候选剪影投回画面，跟那块比重合多少
             （`core.vision.estimate_pose` 的 `min_iou`）。本地仿真实测认准的几把是
             0.72~0.90、认错的那把 0.02，中间空着一大段。够了才报世界坐标，不够报 None。

**反推链**（标定好的，不是现算的）：`RealWristCamera.mount()`（关节读数 → 正解 →
手眼变换 → 基座系里的相机位姿）→ `core.vision.estimate_pose`（八点剪影拟合，解水平
两维 + 绕竖轴转角）→ 基座系里的 [x, y, z] + yaw。畸变在相机出图那一步就抹掉了
（`real/cameras.py::_rectify_frame`），所以这里整条链是纯针孔的，`core/vision.py`
一个字不用改。

**三种结果要分清楚**（`facts` 里写明了是哪一种）：

    found=False               画面里压根没找到那个颜色的长方形（`facts["why"]` 说明为什么）
    found=True, 世界坐标 None  找到了，但这一眼不可信、或者相机没标定
                              （`facts["no_world_why"]` 说明为什么）
    found=True, 世界坐标有值   找到了、也量出来了，`facts["where"] == "world"`

下游别把前两种当成第三种。**要方块的形状和尺寸**：`estimate_pose` 就是拿那个形状的
剪影投回画面比重合度的，形状不对量出来的位置和角度都不对。

**形状和尺寸从这一眼里估，配置里写的那一行只是兜底。** 选中那块、拿到它自己的掩膜
之后就调 `core.vision.estimate_shape`（候选形状各投一遍剪影，谁对得上是谁），估出来的
那个拿去投位置。**估不出来才退回配置** —— 配置那个数是操作员申报的值，不是猜的，仿真
非盲测那一支用的也是它。退回这件事会明着打出来，不静默。`facts["est"]` 是估出来的那
一行（没估出来是 None），`facts["est_fallback"]` 说明用的是配置。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from color_check import load_thresholds                    # noqa: E402
from core import shapes, vision                            # noqa: E402
from core.interfaces import Perceptor                      # noqa: E402
from core.registry import register                         # noqa: E402

CAM_NAME = "real_wrist"

# 去噪用的形态学核。这**不是**标定出来的阈值，是处理参数：开运算去椒盐噪点，
# 闭运算把盒子表面被反光、印刷字打出来的空洞补上。跟判定阈值（在 thresholds.json
# 里）分开看，别混。改这两个不会改变「什么算棕色、什么算矩形」。
MORPH_K = np.ones((5, 5), np.uint8)
MORPH_OPEN_ITER = 2
MORPH_CLOSE_ITER = 2


def size_str(spec: dict) -> str:
    """清单一行里的尺寸，打成一行字：`40×30×25mm` 或者 `直径40×高25mm`。

    只给打印用。**「估出来的」和「配置里写的」要并排看** —— 两边都走这一个函数，
    格式一样才比得出差别。认不出来的形状如实说认不出来，不编一个数。
    """
    if spec.get("size_mm"):
        return "×".join(f"{float(v):.0f}" for v in spec["size_mm"]) + "mm"
    if spec.get("diameter_mm") is not None:
        return (f"直径{float(spec['diameter_mm']):.0f}"
                f"×高{float(spec.get('height_mm') or 0):.0f}mm")
    if spec.get("side_mm") is not None:
        return f"边长{float(spec['side_mm']):.0f}mm"
    return "尺寸没写"


def load_cal(color: str, path: Path | None = None) -> tuple[list, float, int]:
    """从 thresholds.json 拿三个数：(这个颜色的 HSV 段, 矩形度下限, 最小面积)。

    `color` 就是配置里 `objects:` 第一件的 `color` 字段（`red` / `brown` / …），
    也是 `color.hsv` 下面那个键名。**这里从前写死成 `brown`** —— 那一版桌上只摆过
    一只牛皮纸盒；换目标就得改代码，不对。

    矩形度下限和最小面积跟颜色无关（量的是「这块像不像一个矩形」），所以还是从
    `brown_rect` 那一节读 —— 那个键名是历史包袱，**别按字面理解成只管棕色**。

    **不给默认值。** 没标定就当场抛，不猜 —— 这条规矩跟 `color_check.color_cal`
    一致：拿一个没标定的阈值去认东西，出来的结论没法信。
    """
    cal = load_thresholds(path) if path else load_thresholds()
    ranges = ((cal.get("color") or {}).get("hsv") or {}).get(color)
    rr = cal.get("brown_rect") or {}
    fill, area = rr.get("rect_fill_min"), rr.get("min_area_px")
    if not ranges or fill is None or area is None:
        raise RuntimeError(
            f"thresholds.json 里的 {color!r} 还没标定"
            f"（color.hsv.{color}={ranges!r}, rect_fill_min={fill!r}, "
            f"min_area_px={area!r}）。标法见 thresholds.json 里 brown_rect 的 "
            "_note_* 字段：色相段用 calibrate_color.py 的 collect_pixels/hue_ranges "
            f"（`--color {color} --camera 1 --n 20`），"
            "矩形度下限拿同一批帧量候选的 rect_fill 分布取中间。")
    return ranges, float(fill), int(area)


def brown_mask(frame_bgr: np.ndarray, ranges: list) -> np.ndarray:
    """棕色掩膜 + 形态学去噪。多段 HSV 区间是或起来（红跨 0 点那种颜色会有两段）。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_K, iterations=MORPH_OPEN_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_K, iterations=MORPH_CLOSE_ITER)
    return mask


def candidates(mask: np.ndarray, min_area: int) -> list[dict]:
    """掩膜 -> 每个够大的连通域，量出它的矩形度指标。按面积从大到小排。

    `rect_fill` 是旋转不变的那个（轮廓面积 ÷ 最小外接旋转矩形面积），**判据用它**。
    `bbox_fill` 是正着的外框填充率，斜着的长方形会被它低估，只作对照打出来看。
    """
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a < min_area:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        rect_area = max(float(rw) * float(rh), 1e-6)
        area_c = float(cv2.contourArea(c))
        out.append({
            "label": i,
            "contour": c,
            "box": (x, y, w, h),
            "uv": (float(cent[i][0]), float(cent[i][1])),
            "area_px": a,
            "bbox_fill": a / float(max(w * h, 1)),
            "rect_fill": area_c / rect_area,
            "aspect": max(rw, rh) / max(min(rw, rh), 1e-6),
            "rect": ((float(cx), float(cy)), (float(rw), float(rh)), float(ang)),
        })
    out.sort(key=lambda d: -d["area_px"])
    return out


def find(frame_bgr: np.ndarray, ranges: list, fill_min: float, min_area: int,
         color: str = "目标色"):
    """一张图 -> (选中那块 or None, 事实字典)。这是纯函数，不吃相机、不碰阈值文件。

    `color` 只进日志文案（「画面里没有够大的红色块」），不参与判据 —— 判据就是
    `ranges` 那几段 HSV 区间，调用方已经照颜色取好了。

    选中 = **所有通过矩形度门槛的候选里面积最大的那个**。不是「最大的那个再看它像不像
    矩形」—— 那样布料一旦最大就必然出错。先筛再挑，顺序反了结论就反了。
    """
    mask = brown_mask(frame_bgr, ranges)
    cands = candidates(mask, min_area)
    passing = [d for d in cands if d["rect_fill"] >= fill_min]
    facts = {
        "color": color,
        "n_cand": len(cands),
        "n_passing": len(passing),
        "fill_min": float(fill_min),
        "cand": [{"area_px": d["area_px"], "rect_fill": round(d["rect_fill"], 3),
                  "bbox_fill": round(d["bbox_fill"], 3), "aspect": round(d["aspect"], 2)}
                 for d in cands[:5]],
    }
    if not passing:
        facts["why"] = (f"没有一块同时是{color}色、又够大（>={min_area}px）、"
                        f"又够方（矩形度>={fill_min}）的"
                        f"（有色候选 {len(cands)} 个，最高矩形度 "
                        f"{max((d['rect_fill'] for d in cands), default=float('nan')):.2f}）"
                        if cands else
                        f"画面里没有够大的{color}块（色相区间 {ranges}，"
                        f"最小面积 {min_area}px）")
        return None, facts
    return passing[0], facts


@register("perception", "brown_rect")
class BrownRectPerceptor(Perceptor):
    """接口 `Perceptor` 的真机实现，名字 `brown_rect`。

    构造签名跟仿真那几个对齐（`cams` / `ref` / `objects` / `palette`），这样配置里
    换一行名字就能换过来。`ref` 和 `objects` 这里用不上 —— 这一路不看空场景参考图
    （腕机跟着手走，参考图对它没意义），也不看摆件清单（只认 `objects` 第一件那一种色）。
    """

    name = "brown_rect"

    def __init__(self, cams=None, ref=None, objects=None, palette=None,
                 cam_name: str = CAM_NAME, min_iou: float = 0.5, **kw):
        self.cams = cams or {}
        self.ref = ref
        self.objects = objects
        self.cam_name = cam_name
        self.min_iou = float(min_iou)
        # **形状和尺寸是必须给的，而且必须说得出是哪一件。** `estimate_pose` 拿这个
        # 形状的剪影投回画面比重合度；形状不对，量出来的位置和角度都不对，还看不出来
        # （重合度会掉，但也有可能掉不下来）。
        # 来源是配置里 `objects:` 那一段，跟仿真**同一份**（`core.geom.object_specs()`）。
        spec = (list(objects) or [None])[0]
        if not spec:
            raise RuntimeError(
                "配置里没写桌上摆的是什么（`objects:` 是空的）—— 量世界坐标要拿那个"
                "形状的剪影投回画面比重合度。在 configs/real.yaml 里写一行，"
                "比如 `objects: [{shape: cube, color: red, side_mm: 20}]`"
                "（`cube` 用 `side_mm`、`box` 用 `size_mm` 三个全长，都是毫米，"
                "见 `core.shapes.from_spec`）。")
        # **颜色从这一行读，不在代码里写死。** 桌上摆什么色就认什么色，用的区间是
        # `thresholds.json` 的 `color.hsv.<这个颜色>` —— 没标定过 `load_cal` 会当场抛，
        # 并把该跑哪条标定命令打出来。从前这里写死 `brown`，换目标就得改代码。
        self.color = str(spec.get("color", "")).strip().lower()
        if not self.color:
            raise RuntimeError(
                "配置里 `objects:` 第一件没写 `color` —— 这一路的颜色判据（HSV 阈值）"
                "就是照这个字段取的，缺了它不知道该用哪一段区间。")
        self.ranges, self.fill_min, self.min_area = load_cal(self.color)
        self.spec = dict(spec)
        self.shape = shapes.from_spec(self.spec)
        self.last: dict = {}
        # **估出来的形状缓存**（`locate` 里第一次估成功就存这儿）。桌上这件东西的尺寸
        # 不变，而 `estimate_shape` 一次要 ~1.5 秒；每一眼都重估会把硬件线程占住，
        # 面板的按键和画面就一卡一卡的。见 `reset_shape`。
        self._est: dict | None = None

    def locate(self, instruction: str, *, guess=None) -> tuple[np.ndarray | None, dict]:
        """找画面里那块指定颜色的长方形，并把它反推到基座系里。

        第一个返回值：基座系里的 [x, y, z]（米，z 是方块中心的高度），量不出来是 None。
        量不出来分两种，`facts` 说得很清楚：画面里没找到（`found=False`），还是找到了
        但这一眼不可信、或相机没标定（`found=True` + `no_world_why`）。

        `facts["yaw"]` 是绕竖轴转过的弧度（`facts["where"] == "world"` 时才有）——
        合爪前把腕子转到开合轴对上某一面，靠的就是它。

        `guess` 起搜中心（米，基座系）。不给就用 `core.geom.REAL_PICK_XY` /
        `REAL_TABLE_Z`（配置里 `real:` 那一段）。**只当起点和 z 平面用，不是答案** ——
        真认出来之后位置由画面说了算。
        """
        cam = self.cams.get(self.cam_name)
        if cam is None:
            self.last = {"found": False, "why": f"没有 {self.cam_name} 这一路相机"}
            return None, self.last

        rgb = np.asarray(cam.shot())
        # 颜色判据要 BGR（cv2 的老规矩），相机给的是 RGB，这一处转。
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hit, facts = find(bgr, self.ranges, self.fill_min, self.min_area,
                          color=self.color)

        facts = {**facts, "found": hit is not None, "cam": self.cam_name}
        if hit is None:
            self.last = facts
            return None, facts

        facts.update({
            "uv": hit["uv"],
            "box": hit["box"],
            "area_px": hit["area_px"],
            "rect_fill": round(hit["rect_fill"], 3),
            "bbox_fill": round(hit["bbox_fill"], 3),
            "aspect": round(hit["aspect"], 2),
            "rect": hit["rect"],
            "color": self.color,
            "shape": self.spec.get("shape", "box"),
        })

        mount = cam.mount()
        if mount is None:
            facts["where"] = "image_only"
            facts["no_world_why"] = (
                f"{self.cam_name} 没做相机标定（`mount()` 返回 None），"
                "画面位置换算不到世界坐标")
            self.last = facts
            return None, facts

        # **掩膜只留选中的那一块。** `estimate_pose` 拿掩膜的质心当起搜起点 ——
        # 把画面里别的同色东西（当年那版是桌上一块卡其色布、暖色反光）一起塞进去，
        # 质心就被拽偏了。逐像素画出来而不是把整张掩膜递进去，图的就是这个。
        one = np.zeros(bgr.shape[:2], np.uint8)
        cv2.drawContours(one, [hit["contour"]], -1, 255, cv2.FILLED)

        if guess is None:
            from core import geom as _geom                 # noqa: PLC0415

            guess = np.array([*_geom.real_pick_xy(),
                              _geom.real_table_z() + float(self.shape.rest_z)])
        guess = np.asarray(guess, dtype=np.float64)

        # ---- 形状和尺寸**从这一眼里估**，不认配置里写死的那一行 ----
        # 仿真的盲测那一支就是这么干的（`impls/perception.py` 里同一套写法）：不给真值，
        # 拿一串候选形状各投一遍剪影，谁跟画面里这块对得上是谁。
        #
        # **估不出来就退回配置里那一行。** 配置那个数不是猜的，是操作员申报的值 ——
        # 仿真非盲测那一支用的也是它。但「退回去了」这件事必须打出来，不许静默。
        est, est_info = None, None
        _t_shape = time.perf_counter()
        # **形状只估一次**（2026-10-18 时延）：`estimate_shape` 一次 ~1.5 秒，而桌上
        # 这件东西的尺寸不变。第一次估成功就存下来，之后每一眼直接用；只有这一眼连
        # 姿态都拟合不出来（下面 `center is None`）时，才把缓存丢掉、下一眼重估。
        # 换了一块东西要重估，按面板的「重量一眼」（`reset_shape`）。
        if self._est is not None:
            est, est_info = self._est, {"cached": True}
        else:
            try:
                est, est_info = vision.estimate_shape(rgb, mount, one, guess)
            except Exception as exc:                # noqa: BLE001  估形状失败不该毁掉这一眼
                est_info = {"why": f"估形状的时候出错：{type(exc).__name__}: {exc}"}
        spec = self.spec if est is None else {**est, "color": self.color}
        shape = shapes.from_spec(spec)
        # **往下真正在用的那一份是哪个**，单存一个名字给下游问（`pick_real._box_half_m`
        # 拿它算「半个边长」）。`self.spec` 一直是配置里那一行、还得留着打「配置里写的
        # 是 …」，所以不能覆盖它。**2026-10-17 用户定：能用画面估出来的那份就用它。**
        self.spec_used = dict(spec)
        facts["est"] = est
        facts["est_info"] = est_info
        facts["shape"] = spec.get("shape", "box")
        facts["size"] = size_str(spec)
        cached = bool((est_info or {}).get("cached"))
        if est is not None:
            self._est = dict(est)
        if est is None:
            facts["est_fallback"] = True
            if not cached:
                print(f"    ！形状没估出来，退回配置里写的那一行"
                      f"（{(est_info or {}).get('why')}）")
                print(f"      配置里写的是 {size_str(self.spec)}")
        elif not cached:
            print(f"    形状从画面估出来：{est_info.get('estimate')}"
                  f"（重合度 {est_info.get('iou_est', float('nan')):.2f}，"
                  f"{'中空' if est_info.get('hollow') else '实心'}）")
            print(f"      估出来 {size_str(spec)}   ——   配置里写的 "
                  f"{size_str(self.spec)}   （这一次之后不再重估，"
                  f"要重估按「重量一眼」）")
            # 候选前三名（`estimate_shape` 一直在算，以前没打出来 —— 用户 2026-10-18
            # 问「为啥会估成圆柱」，日志里没有第二名是谁，答不了）。
            _top = (est_info or {}).get("top") or []
            if _top:
                print(f"      候选前三：{' / '.join(_top)}"
                      f"（外接矩形给的梯子 {est_info.get('size_hint_mm')}mm）")
            # 剪影本身是方是圆：方的（rect_fill≥0.85 或 圆度<0.88）时圆柱候选直接不上。
            print(f"      剪影：外接矩形填充 {est_info.get('rect_fill')}、圆度 "
                  f"{est_info.get('circularity')} —— "
                  + ("是方的，圆柱候选没上" if est_info.get("cyl_skipped")
                     else "没判成方的，圆柱候选照试"))

        # 剪影是按种子点的 z 投回画面的（`core/vision.py` 的 `z0`），换成**估出来的这个
        # 形状该躺多高**：斜俯视下 z 差 10mm 能把水平位置顶偏十几毫米。
        from core import geom as _geom                     # noqa: PLC0415

        if _geom.REAL_TABLE_Z is not None:
            guess = np.array([guess[0], guess[1],
                              _geom.REAL_TABLE_Z + float(shape.rest_z)],
                             dtype=np.float64)

        _t_pose = time.perf_counter()
        center, yaw, info = vision.estimate_pose(
            rgb, mount, None, guess, shape=shape, mask=one,
            min_iou=self.min_iou)
        if os.environ.get("PANEL_PROFILE"):
            print(f"[prof percep] estimate_shape {(_t_pose - _t_shape) * 1000:.0f}ms"
                  f" + estimate_pose {(time.perf_counter() - _t_pose) * 1000:.0f}ms",
                  flush=True)
        if center is None:
            # **量不出来就是量不出来。** 宁可不抓，也不要把一个自己都不信的数
            # 交给下游去转腕子、定落点。
            iou = info.get("iou")
            facts["where"] = "no_world"
            facts["no_world_why"] = (
                info.get("why") or f"这一眼没量出来（重合度 {iou}）")
            if iou is not None:
                facts["iou"] = round(float(iou), 3)
            # 缓存的那份形状连姿态都拟合不出来 —— 换了一件东西、或者那一份估歪了。
            # 丢掉它，下一眼重新估一遍。
            self._est = None
            self.last = facts
            return None, facts

        # ---- 扣掉现场那个固定偏移 ----
        # 腕相机这条链反推出来的 x 有个**常数**偏（2026-10-15 实测 35.8mm，三次记录一致）。
        # 不扣的话落点整个往下游走 36mm，爪子落在盒子后面合爪、嘴里是空的。
        # 偏移量在 `configs/real.yaml` 的 `real.percep_offset_mm`（跟 `table_z_mm` /
        # `pick_xy_mm` 一个性质的真机现场常数），没配就是 0。
        off = _geom.real_percep_offset()
        if float(np.any(off)):
            center = np.asarray(center, dtype=np.float64).copy()
            center[:2] = center[:2] - off
            facts["percep_offset_mm"] = [round(float(v) * 1000.0, 1) for v in off]

        facts.update({
            "where": "world",
            "world": [round(float(v), 5) for v in center],
            "yaw_deg": round(float(np.degrees(yaw)), 1),
            "iou": round(float(info["iou"]), 3),
            "seen_world": (None if info.get("seen_world") is None
                           else [round(float(v), 5) for v in info["seen_world"]]),
            "seen_c": [round(float(v), 1) for v in info["seen_c"]],
        })
        self.last = facts
        return center, facts

    def reset_shape(self) -> None:
        """丢掉缓存的那份形状，下一眼重新估。

        `locate` 现在只估一次形状（那一步 ~1.5 秒）。换了一块东西、或者觉得估歪了，
        按面板的「重量一眼」就会走到这里。
        """
        self._est = None

    def set_color(self, color: str) -> str:
        """换目标颜色。**颜色判据、形状缓存、上一眼的结论一起换掉。**

        区间还是从 `thresholds.json` 的 `color.hsv.<键>` 取，**没标定过就抛** ——
        跟构造时同一条路（`load_cal`），不在这里另写一套换算。抛出来的那句话本身就
        写着该怎么标，面板原样显示。

        `self.spec["color"]` 跟着换：`estimate_pose` 拿 spec 去投剪影，颜色是其中的
        身份字段。形状缓存（`_est`）清掉 —— 换颜色往往就是换了一块东西，尺寸不一定
        还是原来那件。
        """
        name = str(color or "").strip().lower()
        ranges, fill, area = load_cal(name)          # 没标定就当场抛
        self.color = name
        self.ranges, self.fill_min, self.min_area = ranges, fill, area
        self.spec = {**self.spec, "color": name}
        self.reset_shape()
        self.last = {}
        return name


def draw(frame_bgr: np.ndarray, facts: dict) -> np.ndarray:
    """把结论画在画面上，给人看。只画，不动 facts。"""
    img = frame_bgr.copy()
    if facts.get("found"):
        (cx, cy), (rw, rh), ang = facts["rect"]
        pts = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.int32)
        cv2.drawContours(img, [pts], 0, (0, 255, 0), 2)
        u, v = int(round(cx)), int(round(cy))
        cv2.drawMarker(img, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, f"{facts.get('color', '?')} rect  {facts['area_px']}px"
                          f"  fill {facts['rect_fill']:.2f}"
                          f"  aspect {facts['aspect']:.2f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if facts.get("where") == "world":
            wx, wy, wz = (v * 1000.0 for v in facts["world"])
            cv2.putText(img, f"base x={wx:.0f} y={wy:.0f} z={wz:.0f}mm"
                             f"  yaw {facts['yaw_deg']:+.1f}deg  iou {facts['iou']:.2f}",
                        (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            why = "no world coords: " + str(facts.get("no_world_why", ""))
            cv2.putText(img, why[:78], (8, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, f"no {facts.get('color', '')} rectangle", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(facts.get("why", ""))[:78], (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return img


def _config(path: Path) -> dict:
    """读配置。**跟主循环读的是同一份**（`configs/real.yaml`），所以自测和真跑量
    出来的数是同一套数，不会出现「自测准、跑起来偏」这种没法查的错。"""
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    # 直接跑 = 拍一张、跑一遍反推、把世界坐标打出来，**不动机械臂**（只读关节位姿）。
    # 这是给「一眼对账」用的：盒子摆在哪、它说在哪，两个数并排看。
    #
    #   G:/anaconda/envs/sim2real/python.exe real/perception.py
    #
    # **要开串口**（相机位姿得从关节读数正解出来），所以跑的时候不能同时在跑
    # `mcp_server.py` / `real/servo.py`。也不要在 `view_cam.py` 开着的时候跑 ——
    # 同一个相机开不了两次（`real/cameras.py` 文件头说了）。
    import time

    import numpy as np

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    argv = sys.argv[1:]
    cfg_path = Path(argv[0]) if argv and not argv[0].startswith("-") else _ROOT / "configs" / "real.yaml"
    cfg = _config(cfg_path)
    real = cfg.get("real") or {}

    from core import geom                                        # noqa: PLC0415
    from real.arm import RealArm                                 # noqa: PLC0415
    from real.cameras import RealWristCamera                     # noqa: PLC0415

    # 两个现场常数和摆件清单都从配置来 —— **跟 `run_sim.py` 建组件之前干的是同一件事**，
    # 一处不漏地照做，自测才验得到线上那一套。
    geom.OBJECTS = list(cfg.get("objects") or [])
    if real.get("table_z_mm") is None or real.get("pick_xy_mm") is None:
        raise SystemExit(f"{cfg_path} 里 `real:` 那一段要写全 `table_z_mm:` 和 "
                         f"`pick_xy_mm: [x, y]`（毫米）。现在读到 "
                         f"table_z_mm={real.get('table_z_mm')} "
                         f"pick_xy_mm={real.get('pick_xy_mm')}")
    geom.REAL_TABLE_Z = float(real["table_z_mm"]) / 1000.0
    geom.REAL_PICK_XY = np.array([float(v) for v in real["pick_xy_mm"]]) / 1000.0
    geom.REAL_PERCEP_OFFSET = np.array(
        [float(v) for v in real.get("percep_offset_mm") or (0.0, 0.0)]) / 1000.0

    arm = RealArm(port=str(real.get("port", "COM10")))
    cam = RealWristCamera(source=arm, index=int(real.get("cam_index", 1)))
    percep = BrownRectPerceptor(cams={"real_wrist": cam},
                                objects=geom.object_specs())

    w, h = cam.size()
    mount = cam.mount()
    print(f"相机 {cam.name} {w}x{h}   mount={'有' if mount is not None else '没有'}")
    if mount is not None:
        print(f"  相机在基座系 pos={np.round(mount['pos'], 4)}"
              f"  k={np.round(mount['k'], 1)}  fovy={mount['fovy']:.2f}°")
    print(f"起搜中心 {np.round(geom.REAL_PICK_XY, 4)}  "
          f"z 平面 {geom.REAL_TABLE_Z + float(percep.shape.rest_z):.4f}"
          f"（桌面 {geom.REAL_TABLE_Z:.4f} + 半高 {float(percep.shape.rest_z):.4f}）")
    print(f"TCP {np.round(arm.tcp(), 4)}   关节 {np.round(arm.angles(), 1)}")

    center, facts = percep.locate(
        f"pick up the {percep.color} {percep.spec.get('shape', 'object')}")
    if facts.get("found"):
        print(f"\n画面里找到了：{facts['area_px']}px  矩形度 {facts['rect_fill']:.3f}"
              f"  中心像素 {np.round(facts['uv'], 1)}")
    if center is not None:
        print(f"基座系世界坐标 x={center[0] * 1000:.0f}  y={center[1] * 1000:.0f}"
              f"  z={center[2] * 1000:.0f} mm   yaw {facts['yaw_deg']:+.1f}°"
              f"   重合度 {facts['iou']:.2f}")
        print(f"  拿画面质心直接打回水平面是 {facts['seen_world']}（**不是**答案，"
              f"只当起搜起点；质心不是方块中心）")
        d = np.array(facts["world"])[:2] - geom.REAL_PICK_XY
        print(f"  跟配置的摆放位置（{np.round(geom.REAL_PICK_XY * 1000, 0)}mm）差 "
              f"{np.round(d * 1000, 0)}mm —— **你对着看**：盒子实际摆在哪，"
              f"它说的跟实际差多少")
    else:
        print(f"\n没量出来：{facts.get('no_world_why') or facts.get('why')}")

    out = _ROOT / "out" / f"percep_world_{int(time.time())}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw(cv2.cvtColor(np.asarray(cam.shot()), cv2.COLOR_RGB2BGR),
                               facts))
    print(f"\n画面（带标注）-> {out}")
    cam.close()
    arm.close()
