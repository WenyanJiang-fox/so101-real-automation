# -*- coding: utf-8 -*-
"""验「下探之后问 VLM 一眼」这条线，**不动臂**：

1. 没装 `knobs.VLM_JUDGE` 时 `vlm_gate.check` 只量不问、永不否决（面板/一次性脚本的路径）；
2. 装了之后判 `no` 就 `veto=True`，判 `yes`/`unsure` 不否决；
3. 调用失败（返回 None）按 `unsure` 办，不否决；
4. `ppo.vlm.judge` 在请求出错时返回 None，不抛；
5. 一集的记录能落成一行 JSON。

    G:/anaconda/envs/sim2real/python.exe -u _probe_vlm_judge.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

from skills.pick_real import knobs as K                            # noqa: E402
from skills.pick_real import pose, rulers, vlm_gate               # noqa: E402
import ppo.vlm as vlm                                              # noqa: E402

FP = {"crosses": False, "p_mm": -13.4, "hn_mm": 11.9,
      "s_mm": 31.9, "hu_mm": 11.4, "gap_mm": 20.5}


def fake_jaw_check(cam, percep, arm, tag, axis_only=False):
    return {"ok": False, "why": "探针", "gap": 113.0, "axis": 40.0,
            "footprint": FP}


def fake_wrist(cams):
    return object()


def run() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pose._wrist = fake_wrist
    rulers._jaw_check = fake_jaw_check

    # 探针写到临时文件，不污染真正的 vlm_judge.jsonl
    tmp = ROOT / "out" / "_probe_vlm_judge.jsonl"
    K.VLM_JUDGE_LOG = tmp
    if tmp.exists():
        tmp.unlink()

    ok_all = True

    print("[1] 没装钩子：只量不问")
    K.VLM_JUDGE = None
    vlm_gate.begin_episode(1, "no_measure")
    r = vlm_gate.check(None, {}, None, tag="探针", pre_img=None)
    print(f"    vlm={r['vlm']}  veto={r['veto']}  p_mm={r['p_mm']}")
    ok_all &= (r["vlm"] == "off" and r["veto"] is False)
    vlm_gate.end_episode("empty", "slip", False)

    print("\n[2] 判 no → 否决；判 yes / 调用失败 → 放行")
    for answer, want_veto in (({"between": "no", "why": "在爪外", "confidence": 0.8}, True),
                              ({"between": "yes", "why": "在两爪之间", "confidence": 0.7}, False),
                              (None, False)):
        K.VLM_JUDGE = lambda imgs, caps, a=answer: a
        vlm_gate.begin_episode(2, "no_measure")
        r = vlm_gate.check(None, {}, None, tag="探针", pre_img=None)
        print(f"    回答={answer and answer['between']}  → vlm={r['vlm']}"
              f"  veto={r['veto']}")
        ok_all &= (r["veto"] == want_veto)
        vlm_gate.end_episode("empty", "slip", False)

    print("\n[3] 落盘那一行")
    rows = [json.loads(x) for x in tmp.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"    {len(rows)} 行；最后一行的 vlm={rows[-1]['checks'][0]['vlm']}"
          f"  verdict={rows[-1]['verdict']}")
    ok_all &= len(rows) == 4 and rows[-1]["verdict"] == "empty"

    print("\n[4] 网络/解析失败不抛")
    class Boom(Exception):
        pass

    def bad_urlopen(*a, **k):
        raise Boom("断网")

    img = ROOT / "out" / "_probe_vlm_judge.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")          # 只要是个文件就够，不发真请求
    old = vlm.urllib.request.urlopen
    vlm.urllib.request.urlopen = bad_urlopen
    try:
        ans = vlm.judge([str(img)])
    finally:
        vlm.urllib.request.urlopen = old
        img.unlink(missing_ok=True)
    print(f"    judge(...) = {ans}")
    ok_all &= ans is None

    tmp.unlink(missing_ok=True)
    print(f"\n全部通过：{bool(ok_all)}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(run())
