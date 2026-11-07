# -*- coding: utf-8 -*-
"""把 `out/vlm_judge.jsonl` 整理成表：VLM 那一问到底有没有用。

口径（2026-10-23 用户定）：

  判对率   = VLM 判「不在」的集里，程序现量的 |离轴| > 5mm 的比例
  改善量   = 否决重下前后 |离轴| 的逐集配对差（负数 = 变好），取均值 / 中位数
  对照组   = VLM 判「在」那组的 |离轴| 分布（同 run、同一档）
  成功率   = 附在最后，不当主指标

    G:/anaconda/envs/sim2real/python.exe -u _vlm_judge_tally.py
    G:/anaconda/envs/sim2real/python.exe -u _vlm_judge_tally.py out/vlm_judge.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
DEFAULT = ROOT / "out" / "vlm_judge.jsonl"
#: 离轴进带的门槛（和技能里 `CLEAR_FP_GAP` 那套同一个数：`_footprint` 判「射线穿过」）。
OFF_TOL_MM = 5.0


def _before(row: dict) -> dict:
    for c in row.get("checks") or []:
        if c.get("vlm") != "off" or c.get("p_mm") is not None:
            return c
    return {}


def _after(row: dict) -> dict:
    ch = row.get("checks") or []
    for c in ch[1:]:
        if "只量" in str(c.get("tag") or ""):
            return c
    return ch[1] if len(ch) > 1 else {}


def _f(v):
    return None if v is None else float(v)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.is_file():
        print(f"没有 {path} —— 还没跑过带 VLM 那一问的采集")
        return 1
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    print(f"\n=== {path.name} ===  集数 {len(rows)}")
    print(f"{'集':>4} {'档位':<12} {'离轴前':>7} {'VLM':<7} {'否决':>4} "
          f"{'离轴后':>7} {'结果':<8} {'VLM 的说法'}")
    for r in rows:
        b, a = _before(r), _after(r)
        p0, p1 = _f(b.get("p_mm")), _f(a.get("p_mm"))
        print(f"{r.get('ep')!s:>4} {str(r.get('tier'))[:12]:<12} "
              f"{'—' if p0 is None else f'{p0:+7.1f}'} {str(b.get('vlm')):<7} "
              f"{'是' if b.get('veto') else '':>4} "
              f"{'—' if p1 is None else f'{p1:+7.1f}'} "
              f"{str(r.get('verdict'))[:8]:<8} {str(b.get('vlm_why') or '')[:40]}")

    asked = [r for r in rows if _before(r).get("vlm") not in (None, "off")]
    yes = [r for r in asked if _before(r).get("vlm") == "yes"]
    no = [r for r in asked if _before(r).get("vlm") == "no"]
    uns = [r for r in asked if _before(r).get("vlm") == "unsure"]
    n = max(len(asked), 1)
    print(f"\n问过 {len(asked)} 集：yes {len(yes)} / no {len(no)} / unsure {len(uns)}"
          f"  → 有效回答率 {(len(yes) + len(no)) / n * 100:.0f}%")

    def _share_over(rs, tol=OFF_TOL_MM):
        vals = [abs(_f(_before(r).get("p_mm"))) for r in rs
                if _f(_before(r).get("p_mm")) is not None]
        over = sum(1 for v in vals if v > tol)
        return (over, len(vals),
                (sum(vals) / len(vals) if vals else None))

    for name, rs in (("判「不在」", no), ("判「在」", yes), ("看不清", uns)):
        over, tot, mean = _share_over(rs)
        if not tot:
            print(f"  {name}：没有可用的离轴读数")
            continue
        print(f"  {name}：{tot} 集有读数，|离轴| > {OFF_TOL_MM:.0f}mm 的 {over}/{tot}"
              f" = {over / tot * 100:.0f}%，均值 {mean:+.1f}mm")

    pairs, in_band = [], 0
    for r in no:
        p0 = _f(_before(r).get("p_mm"))
        p1 = _f(_after(r).get("p_mm"))
        if p0 is None or p1 is None:
            continue
        pairs.append(abs(p1) - abs(p0))
        if abs(p1) <= OFF_TOL_MM:
            in_band += 1
    if pairs:
        mid = sorted(pairs)[len(pairs) // 2]
        print(f"\n否决后重下：{len(pairs)} 对读数，|离轴| 变化 "
              f"均值 {sum(pairs) / len(pairs):+.1f}mm、中位 {mid:+.1f}mm"
              f"（负数 = 离轴变小）；重下后 |离轴| ≤ {OFF_TOL_MM:.0f}mm 的 "
              f"{in_band}/{len(pairs)}")
    else:
        print("\n否决后重下：还没有成对的读数")

    def _ok(r):
        return bool(r.get("ok"))

    print(f"\n成功率（附）：全 run {sum(1 for r in rows if _ok(r))}/{len(rows)}"
          f"；判「在」{sum(1 for r in yes if _ok(r))}/{len(yes)}"
          f"；判「不在」{sum(1 for r in no if _ok(r))}/{len(no)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
