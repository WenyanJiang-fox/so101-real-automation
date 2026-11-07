# -*- coding: utf-8 -*-
"""把「同一只方块失败几次、最后抓成了」的尝试链，算成一条条偏移标签。

    G:/anaconda/envs/sim2real/python.exe -u _vlm_attempt_pairs.py

输入 `out/vlm_attempts.jsonl`（每次尝试一行，采集时由 `vlm_gate.end_episode` 写）。
输出 `datasets/vlm_attempts/pairs.jsonl`，一行 = 一次失败尝试 + 它那只方块最后成功
的那一趟：

    rel = 合爪那一刻的 TCP − 当时认到的方块中心     （相对位姿，米 → 毫米）
    need_mm = rel(成功) − rel(失败)                 （失败那趟差多少才成）

**为什么是相对位姿**：方块在重试之间会被下探的爪面推走、被合爪带偏，存绝对 TCP
的话，`need_mm` 里混着「方块自己挪了多少」。减掉每趟自己的方块中心，剩下的才是
「臂相对方块要挪多少」。

**归因**：一行只在「失败那趟和成功那趟只隔了一级扰动」时才算干净样本 ——
`perturb.changed` 记的就是这一级改了什么（`bias_mm` 下扎偏置 / `pushes` 推了几次）。
两条之间改了两样东西的，`clean=False`，训练时按需筛掉。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
SRC = ROOT / "out" / "vlm_attempts.jsonl"
DST = ROOT / "datasets" / "vlm_attempts"


def _rel(row: dict) -> list[float] | None:
    """合爪位姿相对方块中心。缺任何一项就返回 None（这条不进标签）。"""
    tcp, ctr = row.get("tcp_close_mm"), row.get("center_mm")
    if not tcp or not ctr:
        return None
    if any(v is None for v in tcp[:2]) or any(v is None for v in ctr[:2]):
        return None
    # z 也带上：下扎深浅是这一轮唯一开环的变量，z 差是主标签之一。
    return [round(float(t) - float(c), 1) for t, c in zip(tcp[:3], ctr[:3])]


def _changed(a: dict, b: dict) -> tuple[str, int]:
    """两趟之间改了哪个量。返回 `(人看的一句, 改了几样)`。

    `改了几样 == 1` 才是干净样本：一次只动一个量，`need_mm` 才能归到那个量上；
    一样没改（只靠实测精进）也留下，但不算「干净」—— 它解释不了为什么这次成了。
    """
    pa, pb = (a.get("perturb") or {}), (b.get("perturb") or {})
    parts = []
    if float(pa.get("bias_mm") or 0.0) != float(pb.get("bias_mm") or 0.0):
        parts.append(f"bias {float(pa.get('bias_mm') or 0.0):+.0f}→"
                     f"{float(pb.get('bias_mm') or 0.0):+.0f}mm")
    if int(pa.get("pushes") or 0) != int(pb.get("pushes") or 0):
        parts.append(f"推 {pa.get('pushes')}→{pb.get('pushes')}")
    if a.get("tier") != b.get("tier"):
        parts.append(f"方法 {a.get('tier')}→{b.get('tier')}")
    return ("；".join(parts) or "没改量，只靠实测精进"), len(parts)


def main() -> int:
    if not SRC.is_file():
        print(f"没有 {SRC} —— 还没跑过带尝试记录的采集")
        return 1
    rows = [json.loads(x) for x in SRC.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_target: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_target[int(r.get("target") or 0)].append(r)

    DST.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    n_targets_with_success = 0
    for tgt, seq in sorted(by_target.items()):
        seq.sort(key=lambda r: int(r.get("attempt") or 0))
        succ = next((r for r in seq if r.get("ok") and r.get("verdict") == "held"), None)
        if succ is None:
            continue
        n_targets_with_success += 1
        rel_s = _rel(succ)
        if rel_s is None:
            print(f"  [跳过目标 {tgt}] 成功那趟缺位姿（tcp_close_mm / center_mm）")
            continue
        for r in seq:
            if r is succ or int(r.get("attempt") or 0) >= int(succ.get("attempt") or 0):
                continue
            rel_f = _rel(r)
            if rel_f is None:
                continue
            need = [round(s - f, 1) for s, f in zip(rel_s, rel_f)]
            chg, n_chg = _changed(r, succ)
            out.append({
                "target": tgt,
                "failed_attempt": r.get("attempt"), "success_attempt": succ.get("attempt"),
                "failed_ep": r.get("ep"), "success_ep": succ.get("ep"),
                "failed_dir": r.get("dir"), "success_dir": succ.get("dir"),
                "failed_verdict": r.get("verdict"), "success_verdict": succ.get("verdict"),
                "rel_failed_mm": rel_f, "rel_success_mm": rel_s,
                "need_mm": need, "changed": chg,
                # 干净 = 只改了一个量；两样都动的还是留（统计用），但训练时筛掉。
                "clean": n_chg == 1,
                "failed_perturb": r.get("perturb") or {},
                "success_perturb": succ.get("perturb") or {},
            })

    path = DST / "pairs.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_clean = sum(1 for r in out if r["clean"])
    print(f"目标 {len(by_target)} 个，其中抓成过的 {n_targets_with_success} 个")
    print(f"失败→成功 配对 {len(out)} 条（只改一个量的 {n_clean} 条）")
    if out:
        import statistics as st

        for i, name in enumerate(("Δx", "Δy", "Δz")):
            vals = [r["need_mm"][i] for r in out]
            print(f"  {name}: 中位 {st.median(vals):+.1f}mm  "
                  f"范围 {min(vals):+.1f} ~ {max(vals):+.1f}mm")
    print(f"写到 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
