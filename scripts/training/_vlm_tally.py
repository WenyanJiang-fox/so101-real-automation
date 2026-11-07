# -*- coding: utf-8 -*-
"""把采集日志里的 VLM 干预整理成一张表，算「干预前 / 干预后」的成功率。

口径（每行一次 VLM 应用）：

  干预前 3 集   = 触发那一集往前数 3 集的成功数（含触发集本身）
  干预后 3 集   = 触发之后紧接着的 3 集成功数
  撤销          = 干预后出现 `[VLM·撤销]`

单个 run 的 VLM 次数只有 0~4 次，样本量靠累积多个 run 才有意义；
脚本默认把 `out/collect_*.log` 全部读进来分别统计，再给一张合计。

    G:/anaconda/envs/sim2real/python.exe -u _vlm_tally.py
    G:/anaconda/envs/sim2real/python.exe -u _vlm_tally.py out/collect_xxx.log
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")

RE_EP = re.compile(r"^\[第 (\d+) 集\] 结果 ok=(\w+) verdict=(\S*) dropped=(\S*) "
                   r"帧数=(\d+) 失败码=(\S+)")
RE_ASK = re.compile(r"^\s*\[VLM\] (\d+)s\s+action=(\S+)\s+move=(\S+)\s+cause=(\S+)")
# 2026-10-23 起日志里没有 `档=`（档位不归 VLM 定），两个写法都要认。
RE_APPLY = re.compile(r"^\s*\[VLM·(.+?)\] 应用：(?:档=(.*?)\s+)?动作=(\S*)\s+参数=(.*?)"
                      r"(?:\s+（只作用下一集）)?$")
RE_SAFE = re.compile(r"\[安全记录\] VLM：safety / 变量：(\{.*\})")


def parse(path: Path) -> tuple[list[dict], list[dict], int]:
    """返回 (每集结果, VLM 干预, 撤销次数)。"""
    eps: list[dict] = []
    evs: list[dict] = []
    reverts = 0
    pending: dict | None = None
    cur_ep: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_EP.match(line)
        if m:
            cur_ep = int(m.group(1))
            eps.append({"ep": cur_ep, "ok": m.group(2) == "True",
                        "verdict": m.group(3), "code": m.group(6)})
            continue
        m = RE_ASK.match(line)
        if m:
            pending = {"secs": int(m.group(1)), "action": m.group(2),
                       "move": m.group(3), "cause": m.group(4)}
            continue
        m = RE_APPLY.match(line)
        if m and pending is not None:
            evs.append({"ep": cur_ep, "tag": m.group(1), "label": m.group(2),
                        "action": m.group(3), "params": m.group(4),
                        "cause": pending["cause"], "move": pending["move"],
                        "secs": pending["secs"]})
            pending = None
            continue
        if "[VLM·撤销]" in line:
            reverts += 1
            if evs:
                evs[-1]["reverted"] = True
    return eps, evs, reverts


def window(eps: list[dict], ep: int | None, n: int, *, before: bool) -> tuple[int, int]:
    """以 ep 为界取窗口，返回 (成功数, 总集数)。

    **前后都不含触发集本身**：触发集按定义是失败集，算进「干预前」会把基线压低。
    """
    if ep is None:
        return 0, 0
    if before:
        seg = [e for e in eps if ep - n <= e["ep"] <= ep - 1]
    else:
        seg = [e for e in eps if ep < e["ep"] <= ep + n]
    return sum(1 for e in seg if e["ok"]), len(seg)


def mark_effective(evs: list[dict]) -> None:
    """标出**真的改了东西**的干预：参数变了。

    2026-10-23 起档位不归 VLM 定，`action` 字段只剩回填，不能再拿来判「有效」；
    只认参数是否真的和之前不同。`参数={}` = 什么都没动的空转。
    """
    cur: dict = {}
    for e in evs:
        p = {}
        for k, v in re.findall(r"'(\w+)':\s*(-?[\d.]+)", e["params"]):
            p[k] = float(v)
        e["effective"] = any(cur.get(k) != v for k, v in p.items())
        cur.update(p)


def report(path: Path) -> dict:
    eps, evs, reverts = parse(path)
    mark_effective(evs)
    ok = sum(1 for e in eps if e["ok"])
    print(f"\n=== {path.name} ===")
    print(f"集数 {len(eps)}  成功 {ok}  失败 {len(eps) - ok}  "
          f"成功率 {ok / max(len(eps), 1) * 100:.0f}%  "
          f"VLM 干预 {len(evs)}  撤销 {reverts}")
    if not evs:
        return {"eps": len(eps), "ok": ok, "ev": 0, "eff": 0, "pre": (0, 0),
                "post": (0, 0), "post1": (0, 0), "revert": 0}
    print(f"{'触发集':>5} {'原因':<14} {'move':<10} {'参数':<40} "
          f"{'后1':>4} {'后3':>4} {'有效':>4} {'撤销'}")
    pre_ok = pre_n = post_ok = post_n = 0
    post1_ok = post1_n = 0
    for e in evs:
        a, b = window(eps, e["ep"], 3, before=True)
        c, d = window(eps, e["ep"], 3, before=False)
        c1, d1 = window(eps, e["ep"], 1, before=False)
        if e["effective"]:
            pre_ok += a
            pre_n += b
            post_ok += c
            post_n += d
            post1_ok += c1
            post1_n += d1
        print(f"{e['ep']!s:>5} {e['cause']:<14} {e['move']:<10} "
              f"{e['params'][:38]:<40} "
              f"{c1}/{d1:<3} {c}/{d:<3} "
              f"{'是' if e['effective'] else '否':>4} "
              f"{'是' if e.get('reverted') else '':>4}")
    n_eff = sum(1 for e in evs if e["effective"])
    print(f"  有效 {n_eff}/{len(evs)} 次（其余是参数没变的空转）；"
          f"下一集 {post1_ok}/{post1_n} = {post1_ok / max(post1_n, 1) * 100:.0f}%"
          f"（全 run 基线 {ok}/{len(eps)} = {ok / max(len(eps), 1) * 100:.0f}%）；"
          f"后 3 集 {post_ok}/{post_n} = {post_ok / max(post_n, 1) * 100:.0f}%")
    return {"eps": len(eps), "ok": ok, "ev": len(evs), "eff": n_eff,
            "pre": (pre_ok, pre_n), "post": (post_ok, post_n),
            "post1": (post1_ok, post1_n), "revert": reverts}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    files = ([Path(a) for a in args] if args
             else sorted(ROOT.glob("out/collect_*.log")))
    tot = {"eps": 0, "ok": 0, "ev": 0, "eff": 0, "pre": (0, 0),
           "post": (0, 0), "post1": (0, 0), "revert": 0}
    for f in files:
        r = report(f)
        if r["ev"] == 0:
            continue
        tot["eps"] += r["eps"]
        tot["ok"] += r["ok"]
        tot["ev"] += r["ev"]
        tot["eff"] += r["eff"]
        tot["revert"] += r["revert"]
        tot["pre"] = (tot["pre"][0] + r["pre"][0], tot["pre"][1] + r["pre"][1])
        tot["post"] = (tot["post"][0] + r["post"][0], tot["post"][1] + r["post"][1])
        tot["post1"] = (tot["post1"][0] + r["post1"][0],
                        tot["post1"][1] + r["post1"][1])
    if tot["ev"]:
        print("\n=== 合计（只含发生过 VLM 干预的 run） ===")
        print(f"VLM 干预 {tot['ev']} 次（有效 {tot['eff']}），撤销 {tot['revert']} 次")
        print(f"有效干预的下一集 {tot['post1'][0]}/{tot['post1'][1]} = "
              f"{tot['post1'][0] / max(tot['post1'][1], 1) * 100:.0f}%  ；"
              f"这些 run 的整体 {tot['ok']}/{tot['eps']} = "
              f"{tot['ok'] / max(tot['eps'], 1) * 100:.0f}%")
        print(f"干预前 3 集 {tot['pre'][0]}/{tot['pre'][1]} = "
              f"{tot['pre'][0] / max(tot['pre'][1], 1) * 100:.0f}%  →  "
              f"干预后 3 集 {tot['post'][0]}/{tot['post'][1]} = "
              f"{tot['post'][0] / max(tot['post'][1], 1) * 100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
