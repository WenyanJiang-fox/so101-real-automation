"""算那一层：一帧画面 + 此刻关节 → 「差多少」和「下一步该干嘛」。**全是纯函数。**

不碰硬件（硬件在 `panel/rig.py`）、不画图（画的在 `panel/view.py`）、不看按键。

**判据和阈值一个都不在这儿定。** 缝的带宽、轴偏容差、以及最要紧的那个「画面偏差反解
成臂该走多少毫米」的算式，全部调 `skills/pick_real.py` 里现成的那一份：

    `_gap_ok` / `_gap_axis_px` / `_gap_need_px` / `_exit_px`   缝和轴偏的口径
    `_box_center_px`        方块**顶面**中心（不是整块重心，斜视下差 22 像素）
    `_solve_delta`          画面里的偏差 → 底座系里该走的毫米（带号 J 反解）
    `AXIS_TOL_PX` / `CLEAR_BAND_LO_PX` / `CLEAR_BAND_HI_PX` / `CLEAR_GAIN`
    `GRASP_Z_MARGIN_REAL`   第 1 截下探的底：方块中心 + 25mm
    `GRASP_Z_FINAL_REAL`    第 2 截（合爪前）的底：方块中心 + 2mm —— **到底看的是它**
    `DESCEND_Z_TOL_MM`      下探「到底了」的容差
    `NUDGE_STEP_M`          单步封顶：这一趟最多走 20mm（`pick_real.py:2171`）

那边改一个数，这里跟着变。**这里再抄一份的话，两边迟早不是一回事** —— 面板说的
「该往左 5mm」和机械臂真走的那 5mm 必须是同一个 5mm。

`jaw_metrics` 是 `pick_real._jaw_check` 里「从一帧算读数」的那一半，**区别只有一个：
它自己不去抓帧**。面板手上已经有一帧了（那一帧就是感知用的那一帧），再抓一帧量的是
另一个瞬间 —— 臂在动的时候这两个能差开一整个动作。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from real.perception import brown_mask, candidates              # noqa: E402
from skills.pick_real import knobs                              # noqa: E402
from skills.pick_real.rulers import (                           # noqa: E402
    _axis_seg, _box_center_px, _contour_pts, _exit_px, _gap_axis_px,
    _gap_leaving, _gap_need_px, _gap_ok, _gap_px, _want_from_dp)


# 相位名。**画面上的字必须是 ASCII**（`cv2.putText` 只认 ASCII），所以两套：
# `PHASE_*` 是给画面/日志看的英文，`TEXT` 里的说法是给控制台看的。
PHASE_LOST = "lost"          # 画面里没认出方块
PHASE_ALIGN = "align"        # 缝或轴偏没进带里，该横着挪
PHASE_DESCEND = "descend"    # 对准了，该竖直下探
PHASE_CLOSE = "close"        # 到底了，该合爪
PHASE_LIFT = "lift"          # 合上了，该抬
PHASE_SEE = "see"            # 量不了（没轮廓 / 没相机 / 方块被挡）

# 每个相位的两种说法，**只此一张表**：`view` 取 [0] 画画面，`ui` 取 [1] 写窗口。
PHASES = {
    PHASE_LOST: ("LOST", "没认出方块"),
    PHASE_SEE: ("SEE", "量不出来"),
    PHASE_ALIGN: ("ALIGN", "该横着挪"),
    PHASE_DESCEND: ("DESCEND", "该竖直下探"),
    PHASE_CLOSE: ("CLOSE", "该合爪"),
    PHASE_LIFT: ("LIFT", "该抬起来"),
}

# **差得远就一次走完的那条线**（用户 2026-10-17：「如果要动的超过 30 像素，就不走小步了，
# 该走多少走多少」）。画面里还差超过这么多像素：不乘 `CLEAR_GAIN`、不封 `NUDGE_STEP_M`，
# 反解出来多少就走多少；在这个以内照旧小步。
BIG_STEP_PX = 30.0


def jaw_metrics(bgr: np.ndarray, jaw: dict, ranges, fill_min: float,
                min_area: int) -> dict:
    """从**已经拿到的一帧**算缝和轴偏。`pick_real._jaw_check` 去掉抓帧那一步。

    返回的键跟 `_jaw_check` 一一对应（`ok` / `gap` / `gap_near` / `gap_exit` /
    `which` / `axis` / `along` / `box_uv` / `c_uv` / `band_uv` / `near_uv` /
    `axis_ref_px` / `axis_dir_px`），另外多三样只给画图用的：
    `contour`（红区轮廓）、`polys`（爪的几圈轮廓）、`pts`（重采样过的红区点列）。
    """
    mask = brown_mask(bgr, ranges)
    cands = [d for d in candidates(mask, min_area) if d["rect_fill"] >= fill_min]
    if not cands:
        return {"ok": False, "why": "画面里没有过得去的色块（掩膜为空或矩形度不够）"}

    cnt = cands[0]["contour"]
    pts = _contour_pts(cnt)
    polys = {k: np.asarray(jaw[k], dtype=np.float32).reshape(-1, 1, 2)
             for k in ("fixed_jaw", "moving_open", "moving_closed") if k in jaw}
    if "fixed_jaw" not in polys:
        return {"ok": False, "why": "描好的爪轮廓里没有 `fixed_jaw`"}

    raw, near, which = None, None, ""
    for name, poly in polys.items():
        d, q = _gap_px(poly, pts)
        if raw is None or d > raw:              # 哪一圈被探得最狠就报哪一圈
            raw, near, which = d, q, name
    gap = -float(raw)                           # 翻成「红区离爪轮廓外面多少像素」

    mom = cv2.moments(cnt)
    if abs(float(mom["m00"])) < 1e-9:
        return {"ok": False, "why": "色块退化成一个点，量不出中心"}
    c_uv = np.array([mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]], dtype=np.float64)

    box_uv = _box_center_px(cnt, bgr)
    if box_uv is None:
        return {"ok": False, "why": "色块里亮暗分不开（或亮块太小），量不出方块顶面中心"}

    band_uv = np.concatenate([polys[k].reshape(-1, 2) for k in polys], axis=0).mean(axis=0)
    ref = np.asarray(jaw["axis_ref_px"], dtype=np.float64)
    axis_dir = np.asarray(jaw["axis_dir_px"], dtype=np.float64)
    axis_dir = axis_dir / max(float(np.linalg.norm(axis_dir)), 1e-12)
    n_hat = np.array([-axis_dir[1], axis_dir[0]], dtype=np.float64)
    rel = box_uv - ref
    axis = float(np.dot(rel, n_hat))
    along = float(np.dot(rel, axis_dir))

    gap_exit = _exit_px(list(polys.values()), pts, axis_dir)
    gap_axis = _gap_axis_px(list(polys.values()), pts, axis_dir)
    ok = bool(_gap_ok(gap_axis) and abs(axis) <= knobs.AXIS_TOL_PX)

    return {"ok": True, "pass": ok,
            "gap": float(gap_axis), "gap_near": float(gap), "gap_exit": float(gap_exit),
            "which": which, "axis": axis, "along": along,
            "box_uv": box_uv, "c_uv": c_uv, "band_uv": band_uv, "near_uv": near,
            "axis_ref_px": ref, "axis_dir_px": axis_dir,
            "contour": cnt, "polys": polys, "pts": pts}


# `_axis_seg` / `_want_from_dp` 直接用 `pick_real.rulers` 里那一份，这里不再包一层。


def next_step(cam, m: dict | None, angles_deg, tcp, grip_deg, box: dict | None,
              *, gain: float | None = None) -> dict:
    """下一步该干嘛。**只算，不做。**

    返回：

        phase      `PHASE_*` 之一
        why        一句话，说清是拿哪条判据判出来的
        dxy_mm     该横着走**这一趟**多少（毫米，底座系 [dx, dy]），不用走就是 None
        dxy_need_mm 一路修正下来总共还差多少（不封顶的那个数），给画面上的字用
        capped     这一趟有没有被单步上限截短
        at_depth   爪子是不是已经趴在「该合爪的那个高度」上了
        dz_mm      该竖着走多少（毫米，负 = 往下），不用走就是 None
        close      要不要合爪
        px         支撑这个判断的画面读数（`dp_px`）

    **单步封顶 20mm。** 画面差多少就一次走多少的话，读数一抖就走飞；技能那边是**定步长**
    的 —— 方向用反解读出来，长度一律 `NUDGE_STEP_M`（`pick_real.py:2171-2172`，用户
    2026-10-15 定 20mm）。这里照办，但把「没封顶的差多少」也一起给出来，好在画面上看出
    还差多远。
    """
    g = float(knobs.CLEAR_GAIN if gain is None else gain)
    cap = float(knobs.NUDGE_STEP_M)
    out = {"phase": PHASE_SEE, "why": "", "dxy_mm": None, "dxy_need_mm": None,
           "capped": False, "capped_z": False, "at_depth": False,
           "dz_mm": None, "dz_need_mm": None,
           "close": False, "px": None, "gain": g, "cap_mm": cap * 1000.0,
           "big_step": False,
           "box": None, "tcp": None, "grip_deg": grip_deg, "angles_deg": angles_deg}

    if box is None or box.get("center") is None:
        out["phase"] = PHASE_LOST
        out["why"] = (box or {}).get("why") or "画面里还没认出方块"
        return out
    out["box"] = np.asarray(box["center"], dtype=np.float64)
    if tcp is not None:
        out["tcp"] = np.asarray(tcp, dtype=np.float64)
    if m is None or not m.get("ok"):
        out["why"] = (m or {}).get("why") or "还没量到缝/轴偏"
        return out

    # ---- 1 高度先算：它同时决定「拦门看什么」和「这一趟只修哪一段」----
    # `GRASP_Z_MARGIN_REAL` 是米、`DESCEND_Z_TOL_MM` 是毫米
    # （`pick_real.py:142` 写的就是 10.0 mm），**两个单位不一样**，别一起乘 1000。
    if tcp is not None:
        # **技能往下走的是两截，所以这里是两个高度。**
        #   第 1 截 `_descend`：到「方块中心 + `GRASP_Z_MARGIN_REAL`」（25mm）——
        #   爪尖落在上表面之上、横向让开时不会捅着方块走；
        #   第 2 截 `_descend_to_grasp`：微调对准之后竖直扎到「方块中心 +
        #   `GRASP_Z_FINAL_REAL`」（2mm），那才是**合爪**的高度。
        # 面板原来拿 25mm 这一个数既当「该往哪下」又当「到底了没有」：爪子才到 25mm 那一档，
        # 它就报 `at_depth` + 「该合爪」，而技能中间还夹着 23mm 下探 + 一趟只修轴偏
        # （用户 2026-10-17：「合爪这个微调和下探中间不该有的步骤」）。两个常数差 23mm，
        # 用错了这一行读数就是假的。
        z_hi = float(out["box"][2]) + float(knobs.GRASP_Z_MARGIN_REAL)
        z_lo = float(out["box"][2]) + float(knobs.GRASP_Z_FINAL_REAL)
        dz_hi = (z_hi - float(tcp[2])) * 1000.0
        dz_lo = (z_lo - float(tcp[2])) * 1000.0
        # 还没到第 1 截的底就先去那儿（`dz < 0` = 还得往下）；到了就报第 2 截还差多少。
        _two_stage = dz_hi < 0.0
        z_off = float(knobs.GRASP_Z_MARGIN_REAL) if _two_stage else float(knobs.GRASP_Z_FINAL_REAL)
        dz_mm = dz_hi if _two_stage else dz_lo
        # **到底 = 到了合爪那个高度**（第 2 截的底），不是到了第 1 截的底。
        out["at_depth"] = abs(dz_lo) <= float(knobs.DESCEND_Z_TOL_MM)
        # **竖直立刻封顶，不管这一眼是哪个相位。** 「该走多少」和「这一按走多少」是两件事：
        # `dz_need_mm` 是真差多少（给画面上的字），`dz_mm` 是这一下子给多少。面板上那个
        # `APPLY Z` 键读的是 `dz_mm`，不封的话对准之前一按就是整个高差、直接砸桌面。
        out["dz_need_mm"] = float(dz_mm)
        if abs(dz_mm) > cap * 1000.0:
            out["dz_mm"] = float(np.sign(dz_mm)) * cap * 1000.0
            out["capped_z"] = True
        else:
            out["dz_mm"] = float(dz_mm)

    # ---- 2 拦门 ----
    # **到底之后只看轴偏，不看缝。** 两个爪子这时正夹在方块两侧，缝本来就该是负的、
    # 或者几乎贴上 —— 拿它拦门的话一眼都过不了，永远走不到合爪那一步。技能里那一趟
    # 就是这个口径（`pick_real.py:2235` / `2298-2305` / `2408` 的 `axis_only=True`）。
    axis_ok = abs(float(m["axis"])) <= float(knobs.AXIS_TOL_PX)
    gate_ok = axis_ok if out["at_depth"] else (bool(_gap_ok(m["gap"])) and axis_ok)

    if gate_ok:
        if tcp is None:
            out["why"] = "对准了，但这一眼读不到 TCP 高度，判不了该不该下探"
            return out
        if not out["at_depth"]:
            out["phase"] = PHASE_DESCEND
            out["why"] = (f"对准了（gap {m['gap']:+.0f}px、axis {m['axis']:+.0f}px）；"
                          f"竖直走 {out['dz_mm']:+.1f}mm 到方块中心 +"
                          f"{z_off * 1000:.0f}mm")
            if out["capped_z"]:
                out["why"] += (f"（一共还差 {out['dz_need_mm']:+.1f}mm，"
                               f"这一趟封顶 {cap * 1000:.0f}mm）")
            return out
        out["phase"] = PHASE_CLOSE
        out["why"] = (f"对准了、也到底了（gap {m['gap']:+.0f}px、axis {m['axis']:+.0f}px、"
                      f"z 差 {out['dz_mm']:+.1f}mm）—— 合爪")
        out["close"] = True
        return out

    # ---- 3 还没对准：算这一趟该往哪挪 ----
    # 相位先按「没对准」记下。**反解失败也留在这个相位上**：量出来的读数明明没进带，
    # 说成 `SEE`（「看不出来」）是假话，看的人会以为该重量一眼。
    out["phase"] = PHASE_ALIGN
    dp_full = np.asarray(m["axis_ref_px"], dtype=np.float64) - np.asarray(
        m["box_uv"], dtype=np.float64)
    # 到底之后只修垂直合爪轴那一段。沿轴那段（方块在开口里前后偏）这时候不能再动 ——
    # 一挪就把已经伸进开口的爪子又拽出来（`pick_real.py:2403-2418` 拼 `seg_axis`）。
    if out["at_depth"]:
        dp = _axis_seg(dp_full, m["axis_dir_px"])
        need_gap = 0.0
    else:
        # **两条一起修**（`pick_real.py:2532-2542` 的 `seg_axis + seg_gap`）：
        #   · 沿合爪轴那段由**缝**给：`_gap_need_px`（带号，正 = 沿合爪轴推出去、
        #     负 = 把方块拉回来），长度就是还差多少 —— 不再拿 `ref − 顶面中心` 的沿轴
        #     投影当它（那个数在方块压进开口时和缝要的不是一回事）；
        #   · 垂直合爪轴那段由**轴偏**给：`−axis`，就是把方块顶面中心拉回合爪轴线。
        #     用户 2026-10-17：「你的对准都没有把爪子尖对到中心上去」—— 少了这一段，
        #     步子全花在沿轴上，尖端永远压不到中心。
        u = np.asarray(m["axis_dir_px"], dtype=np.float64)
        n = np.array([-u[1], u[0]], dtype=np.float64)
        need_gap = _gap_need_px(float(m["gap"]), float(m.get("gap_exit") or 0.0))
        seg_gap = need_gap * u if _gap_leaving(need_gap) else np.zeros(2)
        dp = seg_gap + (-float(m["axis"])) * n
    if float(np.linalg.norm(dp)) < 1e-6:
        out["why"] = "轴偏已经在容差里，这一眼没有别的可修；合爪那步归你自己按"
        return out
    # **两个分支同一条反解**（`−J⁻¹·dp`）；`at_depth` 那支和这一支原来各写一处，
    # 现在都走 `_want_from_dp`。
    want = _want_from_dp(cam, out["box"], dp)
    if want is None:
        out["why"] = "画面偏差反解不出来（没机位 / 矩阵退化），只能自己手动挪"
        return out

    # ---- 这一趟走多长：**差得远就一次走完** ----
    # 用户 2026-10-17：「如果要动的超过 30 像素，就不走小步了，该走多少走多少」。
    # 30 像素以内照旧收 `CLEAR_GAIN`、封顶 `NUDGE_STEP_M`（读数抖一下不会走飞）；
    # 超过就**原样走**：不乘增益、不封顶。
    _need_px = float(np.linalg.norm(dp))
    out["big_step"] = bool(_need_px > BIG_STEP_PX)
    want = np.asarray(want, dtype=np.float64)
    if out["big_step"]:
        pass                                    # 该走多少走多少：不乘增益、不封顶
    else:
        want = want * g
    out["dxy_need_mm"] = [float(want[0]) * 1000.0, float(want[1]) * 1000.0]
    if not out["big_step"]:
        length = float(np.linalg.norm(want))
        if length > cap:
            want = want * (cap / length)
            out["capped"] = True

    out["px"] = dp
    out["dxy_mm"] = [float(want[0]) * 1000.0, float(want[1]) * 1000.0]

    whichs = []
    if out["at_depth"]:
        whichs.append(f"axis {m['axis']:+.0f}px over tol {knobs.AXIS_TOL_PX:.0f}"
                      "（爪子已经贴上方块了，这一趟只修轴偏、不看缝）")
    else:
        if not _gap_ok(m["gap"]):
            whichs.append(f"gap {m['gap']:+.0f}px outside "
                          f"[{knobs.CLEAR_BAND_LO_PX:.0f}, {knobs.CLEAR_BAND_HI_PX:.0f}]"
                          f"（沿合爪轴 {need_gap:+.0f}px："
                          f"{'推出去' if need_gap > 0 else '拉回来'}）")
        if out["big_step"]:
            whichs.append(f"差 {_need_px:.0f}px > {BIG_STEP_PX:.0f}px"
                          "：这一趟不走小步，该走多少走多少")
        else:
            whichs.append(f"差 {_need_px:.0f}px：这一趟按 x{g:.2f} 小步走")
    tail = (f"；走底座系 ({out['dxy_mm'][0]:+.1f}, "
            f"{out['dxy_mm'][1]:+.1f}) mm")
    if out["capped"]:
        tail += (f"（一共还差 ({out['dxy_need_mm'][0]:+.1f}, "
                 f"{out['dxy_need_mm'][1]:+.1f}) mm，这一趟封顶 {cap * 1000:.0f}mm）")
    out["why"] = "；".join(whichs) + tail
    return out


def classify_grip(grip_deg, empty_deg, cal, open_deg) -> str:
    """6 号舵机角度落在行程的哪一段。**判据整个走 `grasp_check.classify_ref`**，
    不在这儿另定阈值 —— 它认的是「本次实测的两个端点 + 一条跟盒子无关的噪声地板」，
    换个盒子不用重标。

    三档：贴着小的一头（空嘴）-> empty；夹在两头中间 -> held；贴着大的一头（没在合）
    -> empty。上界不能省，少了它一只全张开的爪会被当成夹着东西（2026-10-14 踩过）。
    """
    try:
        import grasp_check                                      # noqa: PLC0415

        return grasp_check.classify_ref(grip_deg, empty_deg, cal, open_deg)
    except Exception as exc:                                    # noqa: BLE001
        return f"?（{type(exc).__name__}: {exc}）"
