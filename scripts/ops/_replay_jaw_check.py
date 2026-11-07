"""拿一张已经存下来的画面，跑一遍合爪前那两道判据（`skills/pick_real.py::_jaw_check`）。

**为什么要有这个。** 那道检查只有真机上才跑得到，跑一次要动臂、要合爪。想改描出来的轮廓
或者改阈值的时候，用上一趟存下来的那一帧重放一遍就够 —— 判据代码一份，重放跟真跑走的是
同一个 `_jaw_check`，不是另写一套。

用法（默认拿 `out/real_37_preclose_f82.jpg`）：

    G:/anaconda/envs/sim2real/python.exe _replay_jaw_check.py
    G:/anaconda/envs/sim2real/python.exe _replay_jaw_check.py --img out/别的帧.jpg

叠出来的图写 `out/jaw_check_replay.jpg`（**不覆盖** `out/jaw_check.jpg`，那个是真跑留下的）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(_ROOT))
DEFAULT_IMG = _ROOT / "out" / "real_37_preclose_f82.jpg"
REPLAY_IMG = _ROOT / "out" / "jaw_check_replay.jpg"


class _Cam:
    """只喂 `_jaw_check` 要的那几样：一张图、尺寸。不碰真相机。"""

    def __init__(self, rgb):
        self._rgb = rgb

    def shot(self):
        return self._rgb

    def size(self):
        return self._rgb.shape[1], self._rgb.shape[0]

    def mount(self):                                            # pragma: no cover
        raise RuntimeError("重放不动臂，不该走到这儿")


class _Arm:
    def tcp(self):                                              # pragma: no cover
        raise RuntimeError("重放不动臂，不该走到这儿")


def main() -> int:
    import cv2
    import skills.pick_real as P
    from real.perception import BrownRectPerceptor

    args = list(sys.argv[1:])
    img_path = Path(args[args.index("--img") + 1]) if "--img" in args else DEFAULT_IMG
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"读不到 {img_path}")
        return 1
    cfg = yaml.safe_load((_ROOT / "configs" / "real.yaml").read_text(encoding="utf-8"))
    percep = BrownRectPerceptor(cams={}, objects=cfg["objects"])
    P.knobs.JAW_CHECK_IMG = REPLAY_IMG                                 # 别覆盖真跑那张
    print(f"重放 {img_path}（{bgr.shape[1]}x{bgr.shape[0]}）  轮廓 {P.JAW_SHAPE_JSON.name}")
    out = P._jaw_check(_Cam(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), percep, _Arm(), "重放")
    print(f"  gap {out.get('gap')}  axis {out.get('axis')}  ok {out.get('ok')}"
          f"  {out.get('why', '')}")

    # 「方块中心」到底取画面里哪个点？把几个候选点各自离轴多远并排打出来。
    from real.perception import brown_mask, candidates
    mask = brown_mask(bgr, percep.ranges)
    cands = [d for d in candidates(mask, percep.min_area)
             if d["rect_fill"] >= percep.fill_min]
    if cands and out.get("c_uv") is not None:
        cnt = cands[0]["contour"]
        ref = np.asarray(out["axis_ref_px"], dtype=np.float64)
        d = np.asarray(out["axis_dir_px"], dtype=np.float64)
        n = np.array([-d[1], d[0]])
        p = np.asarray(cnt, dtype=np.float64).reshape(-1, 2)
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        ap = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True).reshape(-1, 2)
        top4 = ap[np.argsort(ap[:, 1])[:4]].mean(axis=0) if len(ap) >= 4 else np.full(2, np.nan)
        cand_pts = {"整块外接矩形中心": box.mean(axis=0),
                    "轮廓四个最上面的角平均": top4,
                    "最高那个点": p[np.argmin(p[:, 1])],
                    "剪影外框中心": np.array([(p[:, 0].min() + p[:, 0].max()) / 2,
                                              (p[:, 1].min() + p[:, 1].max()) / 2])}
        print(f"  红区 {len(cnt)} 点，外框 x{p[:, 0].min():.0f}-{p[:, 0].max():.0f}"
              f" y{p[:, 1].min():.0f}-{p[:, 1].max():.0f}"
              f"（宽 {p[:, 0].max() - p[:, 0].min():.0f} 高 {p[:, 1].max() - p[:, 1].min():.0f} 像素）")
        for k, q in cand_pts.items():
            print(f"    {k:<12} ({q[0]:.0f},{q[1]:.0f})  离轴 {float(np.dot(q - ref, n)):+.0f} 像素")
    print(f"  图 {REPLAY_IMG}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
