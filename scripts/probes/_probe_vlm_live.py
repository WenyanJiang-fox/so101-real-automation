# -*- coding: utf-8 -*-
"""**在真机上造一次「下探之后才失败、且方块被推走」**，好让新的 VLM 规则真的跑起来。

做法只有一个旋钮：把合爪高度从「方块中心下方 1mm」抬到「中心上方 5mm」。
方块 18mm 高，顶面在 +9mm，所以爪子会合在方块上缘附近 —— 夹不住、但要蹭到方块。
蹭到了，`center_before` 和 `box_center_after` 就拉开差距，`nudge_candidates` 才非空。

不写数据集（`--dry-run`），只跑几集看日志。

    G:/anaconda/envs/sim2real/python.exe -u _probe_vlm_live.py 8
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))

def main() -> int:
    # `collect_real_dataset.py` 只在 `__main__` 那一支里重设编码；这里是 import 进来的，
    # 不补这一句，日志里的「↻ / ✗」在 GBK 下直接抛 UnicodeEncodeError。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    n = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].isdigit() else "8"
    from skills.pick_real import knobs as KNOBS                    # noqa: PLC0415

    print(f"[造] 合爪高度 {KNOBS.GRASP_Z_FINAL_REAL * 1000:+.0f}mm → +5mm（方块中心上方）",
          flush=True)
    KNOBS.GRASP_Z_FINAL_REAL = 0.005

    import collect_real_dataset as C                              # noqa: PLC0415

    sys.argv = ["collect_real_dataset.py", "--episodes", n, "--dry-run"]
    return int(C.main())


if __name__ == "__main__":
    sys.exit(main())
