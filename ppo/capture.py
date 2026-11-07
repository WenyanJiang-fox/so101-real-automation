# -*- coding: utf-8 -*-
"""真机上把 VLM 诊断链路跑一遍，看它输出到底有没有效。

每一趟：
  1. 回初始位置（不录）
  2. 按当前动作档跑一趟 `pick_real`（一趟只合一次爪）
  3. 没成 → 收这一趟写过的图 + 技能交回来的实测数 → 打包 → 问 `deepseek-flash` → 校验
  4. 把模型说的 `cause` 和技能自己归的失败码**并排**打出来 —— 这就是「有没有效」的证据
  5. 下一趟换成模型指的那一档（模型不给 / 校验不过 → 退回 `RetryPolicy` 的规则表）

盘上落在 `ppo/runs/<时间戳>/`：
    attempt_XX.json      packet（含真值）、模型原始输出、校验结果
    attempt_XX_*.jpg     这一趟写出来的标注图
    summary.txt          逐趟对照 + 汇总

**数值参数现在是真执行的**：下探偏置、进场高度走 `apply_action`；目标点平移
（`nudge_dx_mm` / `nudge_dy_mm`）直接改 `geom.REAL_PERCEP_OFFSET`，和下探前每趟现读的
定位走同一条路 —— 模型说「往 +x 挪 10mm」，下一趟整个定位目标就偏过去 10mm。

用法（真机，用户在场）：
    G:/anaconda/envs/sim2real/python.exe -m ppo.capture --attempts 3
    ... --start no_measure --no-vlm      只跑规则表，不调 API
    ... --start no_measure --offset-mm 55,0   第一趟故意抓空，把 VLM 分支逼出来
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# **stdout 必须是 UTF-8**：技能里那句 `↻ 初始位置…` 是 U+21BB，GBK 编不了，
# 编到一半就抛 UnicodeEncodeError 把整趟打断（第一版就是这么崩在第一趟的）。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass

# 闸口一律关掉：不停下来等人（和 `_oneshot_grasp.py` 同口径）。
for _k in ("PICK_CONFIRM_STEP", "PICK_CONFIRM", "PICK_CONFIRM_DESCEND",
           "PICK_RESUME", "PICK_START_AT"):
    os.environ.pop(_k, None)

import numpy as np                                               # noqa: E402

from core import geom                                            # noqa: E402
from core.registry import make                                   # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg            # noqa: E402

from ppo import vlm                                              # noqa: E402
from ppo.live_view import LiveView                               # noqa: E402
from ppo.policy import (ACTIONS, HOVER_DEFAULT_M, RetryPolicy,    # noqa: E402
                        _diff, apply_action)
from skills.pick_real import strategy as STRAT                   # noqa: E402


def _jpgs() -> dict[str, float]:
    out = ROOT / "out"
    return {str(p): p.stat().st_mtime for p in out.glob("*.jpg") if p.is_file()}


def _img_rank(path: str) -> int:
    """发图顺序：**失败那一帧排第一**。以前按修改时间排，下探前的画面排在前面。"""
    n = Path(path).name
    if "jaw_check" in n:
        return 0
    if "jaw_raw" in n:
        return 1
    if "look_see" in n:
        return 2
    return 3


def _collect_images(before: dict[str, float], run_dir: Path, tag: str) -> list[str]:
    """这一趟里被重写过的图。先按「是不是失败那一刻」排，同一档再按修改时间。"""
    changed = [(p, mt) for p, mt in _jpgs().items()
               if mt > before.get(p, 0.0) + 1e-6]
    saved: list[str] = []
    for p, _mt in sorted(changed, key=lambda kv: (_img_rank(kv[0]), kv[1])):
        src = Path(p)
        dst = run_dir / f"{tag}_{src.name}"
        dst.write_bytes(src.read_bytes())
        saved.append(str(dst))
    return saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=3, help="最多跑几趟（抓成就停）")
    ap.add_argument("--start", default="baseline", choices=[a.name for a in ACTIONS])
    ap.add_argument("--port", default=None)
    ap.add_argument("--no-vlm", action="store_true", help="不调 API，只跑规则表")
    ap.add_argument("--keep-going", action="store_true", help="抓成了也继续跑")
    ap.add_argument("--max-same", type=int, default=2,
                    help="同一档连选这么多次就本轮禁用（默认 2）")
    ap.add_argument("--offset-mm", default="0,0",
                    help="**只对第一趟**把目标点挪开这么多毫米（写成 dx,dy）："
                         "感知坐标整体平移，爪子落在方块旁边合空 —— 用来把失败分支逼出来")
    args = ap.parse_args()
    off_mm = [float(v) for v in str(args.offset_mm).replace(" ", "").split(",")[:2]]
    if len(off_mm) != 2:
        raise SystemExit(f"--offset-mm 要写成 'dx,dy'，例如 55,0；现在读到 {args.offset_mm!r}")

    cfg = load_cfg(DEFAULT_CONFIG)
    rig = cfg["real"]
    geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
    geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                                 dtype=np.float64) / 1000.0
    geom.REAL_PERCEP_OFFSET = np.array(
        [float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
        dtype=np.float64) / 1000.0
    #: 配置里那个标定偏移的备份。`--offset-mm` 只在第 1 趟加上去，之后要还原成它。
    base_off = np.array(geom.REAL_PERCEP_OFFSET, dtype=np.float64)
    geom.OBJECTS = list(cfg.get("objects") or [])

    import impls.judge_held                                      # noqa: F401
    import impls.policy_real                                     # noqa: F401
    import real.arm                                              # noqa: F401
    import real.cameras                                          # noqa: F401
    import real.perception                                       # noqa: F401
    import skills.home_real                                      # noqa: F401
    import skills.pick_real                                      # noqa: F401
    from core.round_state import RoundState                      # noqa: PLC0415

    run_dir = ROOT / "ppo" / "runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    say(f"[VLM 链路] 模型 {vlm.MODEL}；最多 {args.attempts} 趟；"
        f"{'不调 API' if args.no_vlm else '调 API'}；输出 {run_dir}")

    port = args.port or str(rig.get("port", "COM10"))
    arm = make("arm", "real", port=port, speed=int(rig.get("speed", 0)),
               fast=bool(rig.get("fast", False)))
    arm.terminated = False
    cams = {"real_wrist": make("camera", "real_wrist", source=arm,
                               index=int(rig.get("cam_index", 0)))}
    percep_kw = ({"cam_name": str(cfg["perception_cam"])}
                 if cfg.get("perception_cam") else {})
    percep = make("perception", "brown_rect", cams=cams, ref=None,
                  objects=geom.object_specs(), **percep_kw)
    # **实时画面从这里出**：相机在这个进程手里，面板开不了第二份，所以由它按 10 帧/秒
    # 把最新一帧画好标记写成 `out/live.jpg`。起不来不影响抓取，照实打印。
    live = None
    try:
        live = LiveView(cams["real_wrist"], percep).start()
        say(f"  实时画面：{live.path}（约 {1.0 / live.period:.0f} 帧/秒）")
    except Exception as exc:                                      # noqa: BLE001
        say(f"  实时画面没起来：{type(exc).__name__}: {exc}")

    pol = RetryPolicy(max_attempts=None)
    cur = next(a for a in ACTIONS if a.name == args.start)
    prev = None
    n_ok = 0
    #: 这一趟要用的数值参数（上一趟模型给的；还没给过就用默认）
    next_bias, next_hover = 0.0, None
    #: 下一趟的目标点平移（毫米，基座系 +x/+y）。模型没给就是 0,0 = 用标定值。
    next_nudge = [0.0, 0.0]
    same = 0

    try:
        home = make("skill", "home_real")
        skill = make("skill", "pick_real")

        for i in range(1, max(1, args.attempts) + 1):
            tag = f"attempt_{i:02d}"
            change = _diff(prev, cur)
            say("")
            say(f"===== 第 {i} 趟「{cur.label}」（{change}）=====")
            say(f"  参数：下探偏置 {next_bias:+.1f}mm，悬停高度 "
                f"{((HOVER_DEFAULT_M if next_hover is None else next_hover)) * 1000:.0f}mm，"
                f"目标点平移 ({next_nudge[0]:+.1f}, {next_nudge[1]:+.1f})mm")

            # **第一趟故意把目标点挪开**：`perception.py:398` 每次定位都现读
            # `geom.real_percep_offset()` 再扣掉，所以这里改大一点，整趟算出来的落点就整体偏出去，
            # 爪子落在方块旁边合空。后面各趟还原成配置里那个标定值 —— 只有这样才看得见
            # 「第 1 趟失败 → VLM 给档 → 第 2 趟按它走」这一整条。
            if i == 1 and (off_mm[0] or off_mm[1]):
                geom.REAL_PERCEP_OFFSET = base_off + np.array(off_mm) / 1000.0
                say(f"  ！人为把目标点挪开 ({off_mm[0]:+.1f}, {off_mm[1]:+.1f})mm"
                    f"（只这一趟）—— 这一趟注定抓空")
            elif next_nudge[0] or next_nudge[1]:
                # **减号不能少**：`real/perception.py:398` 拿画面反推的坐标**减去**这个偏移，
                # 所以偏移加多少、爪子就朝反方向走多少。模型说的是「让爪子往 +x/+y 挪」，
                # 这里就得从标定值里减掉它。
                geom.REAL_PERCEP_OFFSET = base_off - np.array(next_nudge) / 1000.0
                say(f"  模型给的修正量：目标点挪 ({next_nudge[0]:+.1f}, "
                    f"{next_nudge[1]:+.1f})mm（基座 +x/+y）")
            else:
                geom.REAL_PERCEP_OFFSET = base_off

            apply_action(cur, bias_mm=next_bias, hover_m=next_hover)
            # **上一趟的「停手」会把 `arm.terminated` 立起来**，而那一位是给「走位中途
            # 叫停」用的 —— `goto_path` 第一步看到它就收工。不清掉的话下一趟的回家和进场
            # 都会当场返回（2026-10-20 那轮：第 2、3 趟全部「回初始位置没到位 35°」）。
            arm.terminated = False
            h = home.run(arm, cams)
            say(f"  回初始位置：{h}")

            before = _jpgs()
            state = RoundState()
            out = skill.run(arm, cams, state=state, index=0, percep=percep,
                            instruction="pick up the brown box")
            out = dict(out or {})
            # 这一趟实际用了多少平移 —— 发给模型，它才看得出「自己上一次要的偏了多少」
            out["nudge_mm"] = [round(float(next_nudge[0]), 1),
                               round(float(next_nudge[1]), 1)]
            ok = bool(out.get("ok"))
            truth = STRAT.code(str(out.get("why") or ""))
            say(f"  结果：{'抓成' if ok else '没成'}；技能自己的失败码 = {truth}"
                f"；why = {out.get('why') or ''}")

            images = _collect_images(before, run_dir, tag)
            packet = vlm.build_packet(out, attempt=i, strategy=cur.strategy,
                                      history=pol.history, images=images,
                                      truth_code=truth)

            decision = None
            raw = None
            if not ok and not args.no_vlm:
                t0 = time.time()
                raw = vlm.diagnose(packet)
                good, why = vlm.check(raw, packet.get("state"))
                decision = good
                say(f"  VLM {time.time() - t0:.0f}s：cause = "
                    f"{(raw or {}).get('cause')}；action = {(raw or {}).get('action')}"
                    f"；closeness = {(raw or {}).get('closeness')}"
                    f"；校验 {'通过' if good else '不过（' + why + '）'}")
                if good:
                    say(f"    判{'✓' if good['cause'] == truth else '✗'}"
                        f"（真值 {truth}）；证据：{good['evidence']}"
                        f"；下一手：{good['next_hint']}")
                    _ck = good.get("contour_ok")
                    say("    轮廓：" + ("对得上" if _ck is True else
                                       ("**对不上** —— " + (good.get("contour_why") or "")
                                        if _ck is False else "模型没给")))
                    # 模型没给这两个字段就退回 0 —— 它们是绝对量，0 就是标定值
                    next_nudge = [float(good["params"].get("nudge_dx_mm", 0.0)),
                                  float(good["params"].get("nudge_dy_mm", 0.0))]
                    say(f"    → 下一趟目标点平移 ({next_nudge[0]:+.1f}, "
                        f"{next_nudge[1]:+.1f})mm")

            rec = {
                "attempt": i, "action": cur.name, "ok": ok, "truth_code": truth,
                "packet": packet,
                "vlm_raw": raw,
                "vlm_last": dict(vlm.LAST),   # finish_reason / usage / content / reasoning
                "vlm": decision,
                "images": [Path(p).name for p in images],
            }
            (run_dir / f"{tag}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

            say("  " + pol.observe(out, act=cur))
            # 这一趟实际用的参数和模型判断记进账本 —— 下一趟的提示词里靠它知道「试过什么」
            pol.history[-1].update({
                "cause": None if decision is None else decision["cause"],
                "contour_ok": None if decision is None else decision.get("contour_ok"),
                "params": {"descend_offset_mm": next_bias,
                           "approach_height_mm": round(
                               (HOVER_DEFAULT_M if next_hover is None else next_hover)
                               * 1000.0, 1),
                           "nudge_dx_mm": round(float(next_nudge[0]), 1),
                           "nudge_dy_mm": round(float(next_nudge[1]), 1)},
            })

            if ok:
                n_ok += 1
                if not args.keep_going:
                    say("  抓成了 —— 这一轮到此为止")
                    break

            # 下一趟：模型说了算；说不了就退回规则表。
            prev = cur
            next_bias, next_hover = 0.0, None
            if decision is not None:
                nxt = next(a for a in ACTIONS if a.name == decision["action"])
                p = decision.get("params") or {}
                next_bias = float(p.get("descend_offset_mm", 0.0))
                if "approach_height_mm" in p:
                    next_hover = float(p["approach_height_mm"]) / 1000.0
                # **同一档连选 `--max-same` 次就本轮禁用**：这台臂上「完整微调」
                # 历史上就是这样连选三趟、每趟都卡在同一个轴偏不收敛。
                same = same + 1 if nxt.name == cur.name else 1
                if same > max(1, args.max_same):
                    alt, _c = pol.choose()
                    say(f"  ！「{cur.label}」已经连选 {same} 次 —— 本轮禁用，"
                        f"改走「{alt.label}」")
                    nxt, same = alt, 1
                    next_bias, next_hover = 0.0, None
                cur = nxt
            else:
                cur, _c = pol.choose()
    finally:
        # **先停出图线程再关相机**：反过来的话它正在 `shot()` 里拿帧，`close()` 之后会报错。
        if live is not None:
            live.stop()
        try:
            cams["real_wrist"].close()
        except Exception as exc:                                  # noqa: BLE001
            say(f"关相机：{exc}")
        try:
            arm.close()
        except Exception as exc:                                  # noqa: BLE001
            say(f"关串口：{exc}")
        say("")
        say(pol.summary())
        (run_dir / "summary.txt").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")
        print(f"落盘：{run_dir}", flush=True)

    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
