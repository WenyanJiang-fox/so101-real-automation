# -*- coding: utf-8 -*-
"""离线验一遍 2026-10-22 新加的两条 VLM 规则，**不动臂**。

用的是上一轮真机真实跑出来的那份包（`out/vlm_adjust/20261022_201347/ep020`）：
那一趟下探前后的方块中心都量到了，正好是「候选表非空」的场面。

跑法：
    G:/anaconda/envs/sim2real/python.exe -u _probe_vlm_rules.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

import numpy as np                                                 # noqa: E402

from core import geom                                              # noqa: E402
from ppo import vlm                                                # noqa: E402
from skills.pick_real import knobs as KNOBS                        # noqa: E402

SRC = ROOT / "out" / "vlm_adjust" / "20261022_201347" / "ep020" / "decision.json"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = json.loads(SRC.read_text(encoding="utf-8"))
    st = dict(d["packet"]["state"])
    print(f"取的真实包：{SRC}")
    print(f"  center_before_mm   {st.get('center_before_mm')}")
    print(f"  box_center_after_mm {st.get('box_center_after_mm')}")
    print(f"  当时的 nudge_mm     {st.get('nudge_mm')}")

    cand = vlm.nudge_candidates(st)
    st["nudge_candidates"] = cand
    print(f"\n[1] 程序算出的候选：{cand}")

    base = {"cause": "off_axis", "evidence": "探针", "action": "baseline",
            "move": "x_plus", "closeness": 0.3, "confidence": 0.5,
            "contour_ok": None}

    print("\n[2] 校验：模型给的方向在不在候选里")
    for mv in ("x_plus", "x_minus", "y_plus", "y_minus"):
        good, why = vlm.check(dict(base, move=mv), st)
        print(f"  move={mv:8s} -> " + (f"通过 {good['params']}" if good else f"拒收（{why}）"))

    print("\n[3] 应用 + 复发撤销")
    import collect_real_dataset as C                              # noqa: PLC0415

    geom.REAL_PERCEP_OFFSET = np.array([0.0358, 0.0], dtype=np.float64)
    KNOBS.JOINT1_STEP_DEG, KNOBS.JOINT_STEP_DEG = 30.0, None
    params_out: dict = {}
    snap = C._knob_snapshot(params_out)
    print(f"  改之前：offset={np.round(geom.REAL_PERCEP_OFFSET, 4)} "
          f"joint1={KNOBS.JOINT1_STEP_DEG} step={KNOBS.JOINT_STEP_DEG}")
    params_out.update({"nudge_dx_mm": -10.0, "joint1_step_deg": 20.0})
    KNOBS.JOINT1_STEP_DEG = 20.0
    geom.REAL_PERCEP_OFFSET = np.array([0.0458, 0.0], dtype=np.float64)
    C._VLM_PENDING = {"code": "slip", "snap": snap}
    print(f"  改之后：offset={np.round(geom.REAL_PERCEP_OFFSET, 4)} "
          f"joint1={KNOBS.JOINT1_STEP_DEG} params={params_out}")

    C._revert_if_repeated("slip", params_out)
    print(f"  复发后：offset={np.round(geom.REAL_PERCEP_OFFSET, 4)} "
          f"joint1={KNOBS.JOINT1_STEP_DEG} step={KNOBS.JOINT_STEP_DEG} "
          f"params={params_out}")
    back = (abs(geom.REAL_PERCEP_OFFSET[0] - 0.0358) < 1e-9
            and KNOBS.JOINT1_STEP_DEG == 30.0 and params_out == {})
    print(f"  还原成功：{back}")

    print("\n[4] 换了失败码就不撤")
    C._VLM_PENDING = {"code": "slip", "snap": C._knob_snapshot({})}
    C._revert_if_repeated("off_axis", {})
    print(f"  pending 已清：{C._VLM_PENDING is None}")
    return 0 if back else 1


if __name__ == "__main__":
    sys.exit(main())
