# -*- coding: utf-8 -*-
"""下探之后、合爪之前，让 VLM 看一眼「方块还在两爪之间吗」。

2026-10-23 用户定：

- VLM 只能**否决这一次合爪**，不能指挥臂往哪走、不改任何旋钮；
- 否决的代价是多下一次（不是错误动作），一集仍然只合一趟爪；
- 超时、断网、看不清、两次回答不一致 —— 一律放行，照旧合爪；
- 没装 `knobs.VLM_JUDGE`（面板、一次性脚本、单测）时，行为一个字不差。

每一集落一行 `out/vlm_judge.jsonl`：程序现量的离轴、VLM 的判定、否决后重下再量的
离轴、最终 verdict。`_vlm_judge_tally.py` 出表。

**每次尝试另落一份**（2026-10-24）：一对原图存进 `out/vlm_attempts/`、一行记录进
`out/vlm_attempts.jsonl`，带目标编号、第几次尝试、合爪那一刻的 TCP、方块中心。
`vlm_gallery` 每集只留最后一眼，同一个方块重试几趟就只剩最后一次；这里按次留，
才能拿 `_vlm_attempt_pairs.py` 算出「失败那几趟差多少偏移」。**不装 VLM 也照留。**
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import knobs as K

#: 这一集的记录（`begin_episode` 开、`end_episode` 落盘）。没采集时一直是 None。
_PENDING: dict | None = None
#: 这一集开始的时刻。`stash_frames` 用它把**上一集残留的图**挡掉：没走到下探的那一集，
#: 盘上的 `jaw_raw.jpg` 还是上一集的，直接拷过去就等于给样本贴错标签。
_EP_T0: float = 0.0


def begin_episode(ep: int, tier: str, *, target: int = 0, attempt: int = 1,
                  perturb: dict | None = None) -> None:
    """开一集的记录。

    `target` / `attempt` 是**同一只方块的第几次尝试**（采集脚本跨集累计）——
    失败几趟之后才成的那条链，全靠这两个数把图和结果串起来；`perturb` 记这一趟
    相对上一趟改了哪个量（下扎偏置 / 是否推过），算偏移量时要靠它做归因。
    """
    global _PENDING, _EP_T0
    _EP_T0 = time.time()
    _PENDING = {"ep": int(ep), "tier": str(tier), "checks": [],
                "target": int(target), "attempt": int(attempt),
                "perturb": dict(perturb or {})}


def end_episode(verdict, code: str = "", ok: bool = False,
                extra: dict | None = None) -> None:
    """把这一集的记录写成一行。没有检查过（比如没走到下探）就不写。

    **两份日志**：`vlm_judge.jsonl` 保持原样（一问一答的评估用）；
    `vlm_attempts.jsonl` 是每次尝试一行，带图目录和位置，给「差多少偏移」当标签。
    """
    global _PENDING
    p, _PENDING = _PENDING, None
    if not p or not p.get("checks"):
        return
    p["verdict"] = None if verdict is None else str(verdict)
    p["ok"] = bool(ok)
    p["code"] = str(code or "")
    p["t"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        K.VLM_JUDGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with K.VLM_JUDGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"  [VLM·记] 写不进 {K.VLM_JUDGE_LOG}：{exc}", flush=True)
    _write_attempt(p, extra or {})


def _write_attempt(p: dict, extra: dict) -> None:
    """一次尝试一行。`checks[0]` 是下探到底那一眼（`ask=True`），后面只量的那眼不带图。"""
    c0 = (p.get("checks") or [{}])[0]
    row = {
        "ep": p.get("ep"), "target": p.get("target"),
        "attempt": p.get("attempt"), "tier": p.get("tier"),
        "perturb": p.get("perturb") or {},
        "dir": c0.get("dir"),
        "tcp_mm": c0.get("tcp_mm"), "center_mm": c0.get("center_mm"),
        "gap_px": c0.get("gap_px"), "axis_px": c0.get("axis_px"),
        "p_mm": c0.get("p_mm"), "crosses": c0.get("crosses"),
        "vlm": c0.get("vlm"), "veto": c0.get("veto"),
        "n_checks": len(p.get("checks") or []),
        "verdict": p.get("verdict"), "ok": p.get("ok"),
        "code": p.get("code"), "why": extra.get("why") or "",
        # 合爪那一刻的位置：和 `tcp_mm`（下探到底、修轴偏之前）分开记 ——
        # 算偏移要用的是**真正合下去的那个位姿**。
        "tcp_close_mm": extra.get("tcp_close_mm"),
        # 合爪之后方块被带到了哪（感知认到的）；认不到就是 None，那条当标签时丢掉。
        "center_after_mm": extra.get("center_after_mm"),
        "box_shift_mm": extra.get("box_shift_mm"),
        "t": p.get("t"),
    }
    try:
        K.VLM_ATTEMPT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with K.VLM_ATTEMPT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"  [VLM·记] 写不进 {K.VLM_ATTEMPT_LOG}：{exc}", flush=True)


def _stash_attempt(imgs: list[str]) -> str | None:
    """把这一眼的原图按「目标 / 第几次 / 第几集」归档，返回目录。

    固定文件名会被下一趟覆盖，所以**在这里当场拷走**，不能等到整集结束。
    只收比开集时刻新的图（和 `stash_frames` 同一条时间闸）。
    """
    if _PENDING is None:
        return None
    dst = (K.VLM_ATTEMPT_DIR
           / f"t{int(_PENDING['target']):03d}_a{int(_PENDING['attempt']):02d}"
             f"_ep{int(_PENDING['ep']):04d}")
    moved = 0
    try:
        dst.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(imgs[:2]):
            p = Path(src)
            if p.is_file() and p.stat().st_mtime >= _EP_T0 - 1.0:
                shutil.copy2(p, dst / ("now.jpg" if i == 0 else "pre.jpg"))
                moved += 1
    except OSError as exc:
        print(f"  [VLM·记] 尝试图没归档（{exc}）", flush=True)
        return None
    if not moved:
        try:
            dst.rmdir()
        except OSError:
            pass
        return None
    return str(dst)


def stash_frames(ep: int, verdict) -> str | None:
    """把这一集的【此刻】和【参考】两张原始帧按成败归档，给以后当标注样本。

    采集时**每集都留**，成败都要 —— 失败集的图像在 `rec.discard()` 时会被删掉，
    不留这一份，磁盘上就永远只有成功的图。

    返回归档目录；没图（没装 VLM_JUDGE、没走到下探）返回 None。
    """
    # 2026-10-23：**不装 VLM 也归档**（`--no-vlm-judge` 时支路关着，但样本照攒）。
    # 上一集残留的图由下面 `_EP_T0` 那道时间闸挡掉，不靠这个开关。
    tagged = str(verdict or "unknown")
    dst = K.VLM_GALLERY / f"{time.strftime('%m%d_%H%M%S')}_ep{int(ep):04d}_{tagged}"
    moved: list[str] = []
    skipped: list[str] = []
    # 合爪后那一张按**合爪判据**命名（`held` / `partial` / `empty`），不跟整集成败：
    # 摆件阶段出事的集，抓取其实是成的，用整集标签会把它训成反例。
    after_name = "after.jpg"
    try:
        _j = json.loads(Path(K.JAW_AFTER_JSON).read_text(encoding="utf-8"))
        after_name = f"after_{str(_j.get('verdict') or 'unknown')}.jpg"
    except (OSError, ValueError):
        pass
    try:
        dst.mkdir(parents=True, exist_ok=True)
        for src, name in ((Path(K.JAW_RAW_IMG), "now.jpg"),
                          (Path(K.VLM_PRE_IMG), "pre.jpg"),
                          (Path(K.JAW_AFTER_IMG), after_name)):
            # 只收**这一集内**写出来的图。比开集时刻还早的，是上一集留下来的。
            if src.is_file() and src.stat().st_mtime >= _EP_T0 - 1.0:
                shutil.copy2(src, dst / name)
                moved.append(name)
            elif src.is_file():
                skipped.append(name)
    except OSError as exc:
        print(f"  [VLM·记] 归档没写成 {dst}：{exc}", flush=True)
        return None
    if not moved:
        try:
            dst.rmdir()
        except OSError:
            pass
        if skipped:
            print(f"  [VLM·记] 这一集没走到下探，盘上那两张是上一集的，不收"
                  f"（{', '.join(skipped)}）", flush=True)
        return None
    print(f"  [VLM·记] 归档 {tagged}：{dst}", flush=True)
    return str(dst)


def save_pre_descend(cams) -> str | None:
    """下探之前留一张**原始帧**（一个像素都不画），给 VLM 当参照。"""
    from .pose import _wrist                                      # noqa: PLC0415

    # 2026-10-23：**没装 VLM 也照写**。这张图现在有第二个用途 —— 存进
    # `out/vlm_gallery/` 当以后的标注样本；`stash_frames` 靠它配 `now.jpg`。
    cam = _wrist(cams)
    if cam is None:
        return None
    try:
        import cv2                                                 # noqa: PLC0415

        from real.safety_report import _save_jpg                   # noqa: PLC0415
        # `shot()` 给的是 RGB，`cv2.imwrite` 按 BGR 存 —— 不转这一下，红蓝对调，
        # 橙色的臂会变成蓝的。`_jaw_check` 写 `JAW_RAW_IMG` 时转的就是这一下。
        frame = np.asarray(cam.shot())
        if frame.ndim == 3 and frame.shape[2] >= 3:
            frame = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        ok = _save_jpg(frame, K.VLM_PRE_IMG)
    except Exception as exc:                                       # noqa: BLE001
        print(f"  [VLM·记] 下探前那一帧没存下：{exc}", flush=True)
        return None
    return str(K.VLM_PRE_IMG) if ok else None


def check(arm, cams, percep, *, tag: str, pre_img: str | None = None,
          ask: bool = True, center=None) -> dict:
    """量一眼（现算离轴）+ 问一次 VLM。**只读，不动臂。**

    返回这一行的记录；`veto=True` 表示方块被判「不在两爪之间」。
    `ask=False` 时只量不问（重下之后那一趟用它记 `p_after`）。

    `center` 是此刻看到的方块中心（米）。给了就记进这一行 —— 后面算「失败那几趟
    差多少偏移」要用它把 TCP 换成**相对方块**的位姿，方块中途被推走也不影响。
    """
    from .pose import _wrist                                      # noqa: PLC0415
    from .rulers import _jaw_check                                # noqa: PLC0415

    look = _jaw_check(_wrist(cams), percep, arm, tag, axis_only=True)
    fp = look.get("footprint") or {}
    row = {"tag": tag, "ok": bool(look.get("ok")),
           "p_mm": fp.get("p_mm"), "crosses": fp.get("crosses"),
           "gap_px": look.get("gap"), "axis_px": look.get("axis"),
           "why": look.get("why") or "", "vlm": "off", "veto": False, "secs": 0.0}
    try:
        row["tcp_mm"] = [round(float(v) * 1000.0, 1)
                         for v in np.asarray(arm.tcp(), dtype=np.float64).reshape(-1)[:3]]
    except Exception:                                              # noqa: BLE001
        row["tcp_mm"] = None
    row["center_mm"] = (None if center is None else
                        [round(float(v) * 1000.0, 1)
                         for v in np.asarray(center, dtype=np.float64).reshape(-1)[:3]])
    if ask:
        imgs: list[str] = []
        caps: list[str] = []
        raw = Path(K.JAW_RAW_IMG)
        if raw.is_file():
            imgs.append(str(raw))
            caps.append("【此刻】下探之后、爪子张开、还没合拢（原始帧，没有标注）。")
        if pre_img and Path(pre_img).is_file():
            imgs.append(str(pre_img))
            caps.append("【参考】这一趟下探之前的同一位置，还没下扎。")
        if imgs:
            # **图先当场拷走**：固定文件名下一趟就被覆盖，等整集结束再存就只剩最后一次。
            # 不看 `VLM_JUDGE` 装没装 —— `--no-vlm-judge` 时支路关着，训练样本照攒。
            row["dir"] = _stash_attempt(imgs)
        if imgs and K.VLM_JUDGE is not None:
            t0 = time.time()
            d = K.VLM_JUDGE(imgs, caps)
            row["secs"] = round(time.time() - t0, 1)
            if d is None:
                row["vlm"], row["vlm_why"] = "unsure", "调用失败或没解析出来"
            else:
                row["vlm"] = str(d.get("between") or "unsure")
                row["vlm_why"] = str(d.get("why") or "")[:120]
                row["conf"] = d.get("confidence")
            row["veto"] = bool(row["vlm"] == "no")
            print(f"    👁 VLM 看一眼：{row['vlm']}（{row['vlm_why']}；"
                  f"{row['secs']:.1f}s）", flush=True)
        elif imgs:
            row["vlm"], row["vlm_why"] = "off", "VLM 支路关着（只留图）"
        else:
            row["vlm"], row["vlm_why"] = "unsure", "没有可发的图"
    if _PENDING is not None:
        _PENDING["checks"].append(row)
    return row
