"""离线定位：这一轮是哪一步开始不对的。**只看录像记录，不碰仿真。**

回流的第一段要回答的就一句话：第几轮、什么类型、第几帧。答案全在
`out/round_XX.json` 里 —— 那一轮的手臂关节角、末端指令、三个方块的位置、
指垫和目标方块之间的接触点数，是逐帧存下来的，不用再跑一遍仿真。

**判据只用「数出来的事实」，不用拍出来的阈值**：
    碰到过没有      接触点数，0 和非 0 是两回事
    方块动过没有    和摆件动画结束时比，逐位相等就是没动
    夹爪在开在合    动作第 7 维：负号是在合、正号是在松（`params.GENTLE` = 0.05、
                    `params.OPEN_STEP` = 0.04 就是那两条指令的步长）
    手臂还动不动    六个关节角逐位相等就是彻底停了
要拍阈值才能说的（比如「夹得够不够深」「放得准不准」）**不下结论，只把数摆出来**，
由你看。判官已经给了的那两个数（落点偏离、落地与否）直接引用，不重算。

帧号是**录像帧**，不是控制步：`run_sim.py --stride`（默认 4）帧才录一帧。
要换算控制步就乘 stride，脚本按 `meta` 里没有的东西不猜，用 `--stride` 给。

用法：
    G:/anaconda/envs/sim2real/python.exe G:/sim2real/so101-real-harness/failure_scan.py G:/sim2real/so101-real-harness/out/round_01.json
    G:/anaconda/envs/sim2real/python.exe G:/sim2real/so101-real-harness/failure_scan.py G:/sim2real/so101-real-harness/out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.geom import CUBE_ORDER, PICK_XY, PLACE_XY   # noqa: E402

MOVE_TOL = 5e-4      # 「方块动过没有」的分界：和摆件结束时比，位移小于它就算没动。
                     # 这是个**分界**不是**精度**：动过的那几块都是几十毫米量级，
                     # 没动的那几块是 0（摆完就没人碰过），中间空得很开。
BUMP_TOL = 0.01      # 「被搬走」还是「只被蹭了一下」的分界。蹭到是几毫米（实测 4~6mm），
                     # 搬走是几十毫米。同样只在打印时给个读法，数照样摆出来。
STILL_TOL = 1e-3     # 「关节还动不动」的分界（度）。同理，冻住的那些是逐位相等。
IDLE_TOL = 5.0       # 「整轮没动过手」的分界（度）。**不能拿 STILL_TOL 当这个用**：
                     # 机械臂站着不动时关节也会自己沉一下就位，实测摆件那 15 帧里肘关节
                     # 走了 1.9°；真动过手的那几轮是几十到一百多度，5° 卡在中间空处。
CUBE_HALF = 0.015    # 方块半高（`skills/pick.py` 的 `CUBE_HALF`，30mm 方块）。
                     # 只用来把接触高度翻译成「上半截／下半截」，不做判定。


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runs(flag) -> list[tuple[int, int]]:
    """一串 True/False 收成 [(起, 止)]，闭区间，帧号。"""
    out, s = [], None
    for i, v in enumerate(bool(x) for x in flag):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(flag) - 1))
    return out


def park_end(cube: np.ndarray) -> int:
    """摆件动画到哪一帧结束：开局几帧里方块被一个个挪到工位上，之后才轮到手臂动。"""
    step = np.linalg.norm(np.diff(cube, axis=0), axis=2).max(axis=1)
    k = 0
    while k < len(step) and step[k] > MOVE_TOL:
        k += 1
    return int(k)      # 第 k 帧起没人再动方块 —— 开局位置取这一帧


def nearest_mark(xy) -> tuple[str, float]:
    """这个点离哪个工位标识点最近。标识点表在 `core/geom.py`（拾取位/放置位）。"""
    best = ("—", float("inf"))
    for i, p in enumerate(PICK_XY):
        d = float(np.linalg.norm(np.asarray(xy, dtype=float) - p))
        if d < best[1]:
            best = (f"拾取位{i}", d)
    for i, p in enumerate(PLACE_XY):
        d = float(np.linalg.norm(np.asarray(xy, dtype=float) - p))
        if d < best[1]:
            best = (f"放置位{i}", d)
    return best


def mm(v: float) -> str:
    return f"{v * 1000:.1f}mm"


def xy_mm(p) -> str:
    return f"[{p[0] * 1000:.0f},{p[1] * 1000:.0f}]"


def seg_of(chunks, frame: int) -> str:
    """这一帧落在哪一段（`meta.chunks`，说明这一步在干什么）。老记录没有这一段。"""
    if not chunks:
        return ""
    k = 0
    for c in chunks:
        k += c["n_frames"]
        if frame < k:
            return c["seg"]
    return chunks[-1]["seg"]


def scan(d: dict, stride: int = 4) -> dict:
    """读一轮记录，把事实摆出来。返回给下一段用的摘要（第几轮 / 类型 / 哪一帧）。"""
    extra = d.get("extra") or []
    if not extra:
        return {"index": d.get("index"), "kind": "记录里没有逐帧证据", "frame": None}
    n = len(extra)
    cube = np.asarray([f["cube"] for f in extra], dtype=float)
    arm = np.asarray(d["arm"], dtype=float)
    act = np.asarray(d["action"], dtype=float)
    ev = d.get("evidence") or {}
    v = d.get("verdict") or {}
    det = v.get("detail") or {}
    chunks = (d.get("meta") or {}).get("chunks")
    idx = ev.get("target_index")
    name = CUBE_ORDER[idx] if isinstance(idx, int) and idx < len(CUBE_ORDER) else "?"

    k0 = park_end(cube)
    base, final = cube[k0], cube[-1]

    # ---- 接触（是**和目标方块之间**的接触，别人被撞它不算）----
    nf = np.array([f["touch"][0] for f in extra])
    nm = np.array([f["touch"][1] for f in extra])
    tz = np.array([np.nan if f["touch"][2] is None else float(f["touch"][2])
                   for f in extra])
    hit = (nf + nm) > 0
    both = (nf > 0) & (nm > 0)
    spans = runs(hit)

    # ---- 夹爪：动作第 7 维负号在合、正号在松 ----
    close = runs(act[:, 6] < 0)
    open_ = runs(act[:, 6] > 0)

    # ---- 手臂还动不动：末尾连着多少帧六个关节逐位相等 ----
    tail = 0
    for i in range(n - 1, 0, -1):
        if float(np.max(np.abs(arm[i] - arm[i - 1]))) < STILL_TOL:
            tail += 1
        else:
            break
    frozen_from = (n - tail) if tail >= 2 else None

    out = {"index": d.get("index"), "target": name, "seed": d.get("seed"),
           "n_frames": n, "park_end": k0, "spans": spans, "close": close,
           "open": open_, "frozen_from": frozen_from,
           "success": bool(v.get("success")), "reason": v.get("reason"),
           "kind": None, "frame": None, "why": []}

    print(f"\n{'=' * 78}")
    print(f"第 {d.get('index')} 轮  {Path(d.get('_path', '')).name}   "
          f"目标 {name}（索引 {idx}）  seed {d.get('seed')}  "
          f"{n} 帧 / {(d.get('meta') or {}).get('seconds')}s")
    print(f"  判官  {'成功' if v.get('success') else '失败'} — {v.get('reason')}"
          + (f"  |  落点偏离 {mm(det['offset_m'])}"
             f"  落地 {'是' if det['on_ground'] else '否'}" if "offset_m" in det else ""))
    if det.get("goal_xy"):
        print(f"        要去 {xy_mm(det['goal_xy'])}"
              f"  最后停在 {xy_mm(det['final_xyz'])}（高 {det['final_xyz'][2] * 1000:.1f}mm）")
    if chunks:
        print("  分段  " + "  ".join(f"{c['seg']}({c['n_frames']})" for c in chunks))
    meta = d.get("meta") or {}
    if meta.get("timeout"):
        # 判官只按最后的现场判，它不知道这一轮是被掐掉的。不点出来的话，
        # 「还挂在天上」会被读成抓取本身出错。
        print(f"  ⏱ 这一轮是被时间上限掐掉的（{meta.get('timeout_why') or '没写理由'}），"
              "不是自己走完的")
    elif meta.get("stop_why"):
        print(f"  ⏹ 收工理由：{meta['stop_why']}")
    if (meta.get("attempts") or 0) > 1:
        print(f"  重试  这一轮试了 {meta['attempts']} 次（一次尝试 = 抓一次+放一次）")

    # ---- 三个方块各自：开局（摆件结束）→ 收场 ----
    print(f"  方块  开局（第 {k0} 帧摆件完）→ 收场")
    for i in range(len(CUBE_ORDER)):
        dxy = float(np.linalg.norm(final[i][:2] - base[i][:2]))
        tag = "  ← 这一轮的目标" if i == idx else ""
        # 「动没动」只是给个读法，数就在旁边，分不清就看数
        if dxy < MOVE_TOL:
            tag += "   没动过"
        elif dxy < BUMP_TOL:
            tag += "   只蹭了一下（不到 10mm，没被搬走）"
        lab, dd = nearest_mark(final[i][:2])
        print(f"        {CUBE_ORDER[i]:<5} {xy_mm(base[i])} → {xy_mm(final[i])}"
              f"  动了 {mm(dxy)}   最近标识点 {lab} 差 {mm(dd)}{tag}")

    # ---- 接触时间线 ----
    if not spans:
        print("  接触  对目标方块**全程 0 点接触** —— 从来没碰到过它")
    else:
        txt = []
        for a, b in spans:
            zs = tz[a:b + 1]
            zs = zs[~np.isnan(zs)]
            txt.append(f"f{a}–{b}" + (f"（高度 {zs[0] * 1000:.1f}mm）" if len(zs) else ""))
        print("  接触  " + " ／ ".join(txt))
        firsts = [a for a, _ in runs(both)]
        if firsts:
            a = firsts[0]
            z = tz[a]
            # **只摆数，不给「深/浅」的结论** —— 那个分界要拍，拍了就是我在替你判。
            print(f"        两指同时贴上：f{a}  接触高度 "
                  f"{'—' if z != z else f'{z * 1000:.1f}mm'}"
                  f"（方块底面 0、中心 15.0mm、顶面 30.0mm）")

    # ---- 夹爪动作 ----
    if close:
        print("  夹爪指令  合 f" + " ／ f".join(f"{a}–{b}" for a, b in close)
              + ("    开 f" + " ／ f".join(f"{a}–{b}" for a, b in open_) if open_ else ""))
    else:
        print("  夹爪指令  整轮没合过（动作第 7 维一直非负）")
    if frozen_from is not None:
        print(f"  冻结  从 f{frozen_from} 到收场（{n - frozen_from} 帧）六个关节一动不动")

    # ---- 定类型 ----
    kind, frame, why = None, None, []
    # 手臂整轮一步没动：技能在出发前就退出了（实测那两轮是「这个点够不着，不去」，
    # 接着腕部那一眼又没量出方块）。**这种失败的原因不在记录里** —— 技能说了什么只打
    # 在日志上，没进 json。所以这里只报「没动过手」，不猜它是抓空还是抓错。
    # 两种判法：有分段就按分段（只有摆件那一段＝技能一步没跑），老记录没分段才退回关节角。
    idle = (len(chunks) <= 1 if chunks else
            float(np.max(np.abs(arm - arm[0]))) < IDLE_TOL)
    if idle and not v.get("success"):
        kind = "技能没动过手就退出了（记录里没有原因，看那一轮的日志）"
        why.append(f"整轮 {n} 帧六个关节一直停在原地")
        if idx is not None and float(np.linalg.norm(final[idx][:2] - base[idx][:2])) < MOVE_TOL:
            why.append("目标方块一直摆在那儿，没人碰过它")
    elif not hit.any():
        kind = "抓空／抓错：整轮没碰到目标方块"
        # 出事帧取**夹爪合到底**那一帧：那是「这一合到底夹到了什么」见分晓的时刻。
        # 不取合爪指令的收尾 —— 专家合完爪还会一路给着一份恒定的合爪量（`params.GRIP_BIAS`），
        # 那条指令一直到收场都是负的，取它的尾巴会指到轮末去。
        # 也不从头找最小：开局夹爪就是合的（−10°），从头找必然指到第 0 帧。从**合爪指令
        # 开始那一帧**往后找才对。
        a0 = close[0][0] if close else 0
        frame = int(a0 + np.argmin(arm[a0:, 5]))
        why.append("接触点数全程为 0")
        if idx is not None and float(np.linalg.norm(final[idx][:2] - base[idx][:2])) < MOVE_TOL:
            why.append("目标方块的位置从摆件结束到收场没变过")
    elif not hit[-1] and spans:
        # 接触断了。**要分两种断**：主动松开（放置那一下，指令是正的）不算故障；
        # 指令还在合、接触却没了，那是滑落。判据在动作第 7 维的符号上，不用拍阈值。
        last = spans[-1][1]
        if float(act[last + 1, 6]) <= 0:
            kind = "滑落：夹到过又松脱了"
            frame = last + 1
            why.append(f"接触 f{spans[-1][0]}–{last}，f{last + 1} 起断掉"
                       f"（那一帧夹爪指令 {act[last + 1, 6]:+.3f}，还在合）")
            if len(spans) > 1:
                why.append(f"中途断过 {len(spans) - 1} 次、又接回来 {len(spans) - 1} 次")
            if v.get("success"):
                kind = "中途掉过，但最后接回来了（判官判成功）"
    if kind is None and not v.get("success"):
        if hit[-1]:
            kind = "没松手：收场时还夹着目标方块"
            frame = n - 1
            why.append("最后一帧接触点数仍非 0")
        elif det.get("on_ground"):
            kind = "放偏：放下去了，但没落到目标点"
            frame = open_[-1][1] if open_ else None
            why.append(f"落点偏离 {mm(det['offset_m'])}（判官阈值 20mm）")
        elif frozen_from is not None:
            kind = "失速：关节冻住，指令还在发"
            frame = frozen_from
            why.append(f"f{frozen_from} 起六个关节逐位相等")

    if kind is None and v.get("success"):
        kind = "没找到故障（判官判成功）"
    out["kind"], out["frame"] = kind, frame
    out["why"] = why

    if kind:
        mark = "✓" if "没找到" in kind or "接回来" in kind else "✗"
        where = seg_of(chunks, frame) if frame is not None else ""
        print(f"  {mark} 类型  {kind}")
        if frame is not None:
            print(f"        出事帧  f{frame}"
                  + (f"（{where} 段）" if where else "")
                  + f"  = 第 {frame * stride} 个控制步")
        for w in why:
            print(f"        · {w}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="离线定位一轮录像里哪一步开始不对")
    ap.add_argument("paths", nargs="+", help="round_XX.json 或装着它们的目录")
    ap.add_argument("--stride", type=int, default=4,
                    help="几帧录一帧（run_sim.py --stride 的值），只用来把帧号换算成控制步")
    ap.add_argument("--json", type=Path, default=None,
                    help="把结果写成 json（给回流的下一段用）")
    args = ap.parse_args()

    files: list[Path] = []
    for p in args.paths:
        p = Path(p)
        files.extend(sorted(p.glob("round_*.json")) if p.is_dir() else [p])

    out = []
    for f in files:
        d = load(f)
        d["_path"] = str(f)
        r = scan(d, stride=args.stride)
        out.append({k: v for k, v in r.items() if not k.startswith("_")})
    if args.json is not None:
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\n写到 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
