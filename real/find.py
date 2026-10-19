"""真机入口 A：摄像头 + 感知，认出画面里的棕色长方形。**不碰机械臂。**

这一版只做「认出来」，所以世界坐标那格是空的（没做相机标定，见 `real/cameras.py`
文件头）。它回答的是「画面里有没有、在哪一块像素上」，不是「桌上哪个坐标」。

两个部件都**从注册表拿**，不是直接 import 类：

    make("camera",     "real_wrist")
    make("perception", "brown_rect")

这样它们跟仿真那套（`sim_wrist` / 那几个感知器）走的是同一套接口和同一条装配路径，
以后要把这条链接到机械臂上，换的是名字，不是结构。

用法（PowerShell 一行）：
    G:/anaconda/envs/sim2real/python.exe G:/sim2real/so101-real-harness/real/find.py --seconds 8
结果：一段带标注的 mp4，落在 G:\\sim2real\\so101-real-harness\\out\\
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.registry import available, make                        # noqa: E402
from real import cameras as _cameras                             # noqa: F401,E402
from real import perception as _perception                       # noqa: F401,E402
from real.video import write_mp4                                 # noqa: E402

OUT = _ROOT / "out"
DEFAULT_INSTRUCTION = "拿起前面的棕色长方形"


def main() -> int:
    ap = argparse.ArgumentParser(description="真机：认出画面里的棕色长方形，录一段视频")
    ap.add_argument("--seconds", type=float, default=8.0, help="录多久")
    ap.add_argument("--fps", type=float, default=15.0, help="目标帧率（受相机实际速度限制）")
    ap.add_argument("--index", type=int, default=1, help="摄像头序号，屏幕上是 camera 1")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="要它找什么")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"注册表里有  camera: {available('camera')}")
    print(f"            perception: {available('perception')}")
    print(f"指令：{args.instruction}")

    cam = make("camera", "real_wrist", index=args.index)
    percep = make("perception", "brown_rect", cams={"real_wrist": cam})
    print(f"相机：{cam.size()[0]}x{cam.size()[1]}   mount()={cam.mount()}")
    print(f"判据：矩形度 >= {percep.fill_min}   面积 >= {percep.min_area}px")

    frames, n_found = [], 0
    fills, areas, uvs = [], [], []
    try:
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            rgb = cam.shot()
            # 颜色判据按 BGR 算 —— 标定时就是这么读的（cv2 的默认通道序），
            # 这一处不能省，换成 RGB 直接喂 HSV 会让色相整体偏掉。
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            _, facts = percep.locate(args.instruction)
            img = _perception.draw(bgr, facts)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if facts.get("found"):
                n_found += 1
                fills.append(facts["rect_fill"])
                areas.append(facts["area_px"])
                uvs.append(facts["uv"])
            time.sleep(1.0 / args.fps)
        elapsed = time.time() - t0
    finally:
        cam.close()

    if not frames:
        print("一帧都没录到，不写文件。")
        return 1

    real_fps = len(frames) / elapsed if elapsed > 0 else args.fps
    path = OUT / f"brown_rect_{int(time.time())}.mp4"
    write_mp4(path, frames, real_fps)

    print(f"\n录了 {len(frames)} 帧 / {elapsed:.1f} 秒（实际 {real_fps:.1f} fps）")
    if n_found:
        u, v = np.mean(uvs, axis=0)
        print(f"认出 {n_found}/{len(frames)} 帧（{n_found / len(frames):.0%}）")
        print(f"  矩形度 {min(fills):.2f}-{max(fills):.2f}（门槛 {percep.fill_min}）")
        print(f"  面积   {min(areas)}-{max(areas)}px")
        print(f"  中心   平均 u={u:.0f} v={v:.0f}（画面 {cam.size()[0]}x{cam.size()[1]}）")
        print("  世界坐标：无 —— 相机没标定，只有画面位置。")
    else:
        last = percep.last
        print(f"整段没认出。原因：{last.get('why')}")
        for c in last.get("cand", []):
            print(f"  候选 {c}")
    print(f"视频：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
