"""实验：CLIP 能不能当「指令 → 哪个方块」那一层。

**要回答的只有一句话**：把画面里每个候选块单独裁出来、连同几句指令一起喂给 CLIP，
它排得对不对 —— 尤其是**深绿和浅绿**。这一条过不了，接进 harness 就没意义。

**为什么挑深绿浅绿当考题。** 现在 `core/vision.py::color_mask` 是三条写死的数值门限
（绿 = G>120 且 R、B<100），里面没有「深浅」这个概念，深绿浅绿要么并成一条、
要么深绿整个挑不出来。换成 CLIP 以后判据变成「跟这句话最像的是哪一块」，
理论上深浅就只是措辞问题 —— 理论上。CLIP 对细分色差偏弱是出了名的，所以得实测。

**候选块从哪来。** 用仓库里已经在的那条不看颜色的路：拍一张空工位的参考图，
跟当前图比（`vision.foreground_mask`），多出来的连通域就是候选（`vision.blobs_of`）。
这样「找候选」和「认是哪个」两件事分开，测的纯粹是 CLIP 后半截。

跑：G:/anaconda/envs/sim2real/python.exe G:\\sim2real\\so101-real-harness\\exp_clip_lang.py
出图：G:\\sim2real\\so101-real-harness\\out\\clip_lang\\
（要先下权重，本机 HF 官方站不通，走镜像：
  HF_ENDPOINT=https://hf-mirror.com  …）
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import impls.arm_sim                                  # noqa: E402,F401  导入即挂上牌子

from core import vision                               # noqa: E402
from core.geom import CUBE_Z, PLACE_XY, PICK_XY       # noqa: E402
from core.registry import make                        # noqa: E402

OUT_ROOT = HERE / "out" / "lang_models"
# 同一件事、两代模型。跑同一个 `--model` 就是同一个目录，两次的结果不互相覆盖。
MODELS = {
    "clip":    "openai/clip-vit-base-patch32",       # 2021，老基线
    "siglip2": "google/siglip2-base-patch16-224",    # 2025，SigLIP 的直系后代
}
IMG_SIZE = 224                     # 两个模型都吃 224
PARK_XY = (3.0, 3.0)               # 把方块临时挪到画面外，拍参考图用

# 把三个方块改成三块绿，深浅不同 —— 这就是给 CLIP 出的题。
# 蓝色那个改成深绿 (0,0.45,0)，绿色那个改成浅绿 (0,1,0)，红色的不动当对照。
CUBE_PAINT = {
    0: (1.00, 0.00, 0.00, 1.0),    # 红（对照）
    1: (0.00, 1.00, 0.00, 1.0),    # 浅绿
    2: (0.00, 0.45, 0.00, 1.0),    # 深绿
}
# 涂成什么样，只用来解释结果。**不拿它给候选块贴标签** —— 哪个候选是哪个方块，
# 靠读那一块的平均颜色认，不靠位置对。
PAINTED = {0: "红", 1: "浅绿(0,1,0)", 2: "深绿(0,0.45,0)"}

# 光照条件。改的是所有灯的亮度/色温乘一个系数 —— 跟 harness 里 `--randomize`
# 抽的是同一套量。**除光以外什么都不动**，方块位置、朝向、涂色全场一致。
CONDS = [
    ("1_normal",  "正常",       1.00, (1.00, 1.00, 1.00)),
    ("2_dim",     "变暗到 35%", 0.35, (1.00, 1.00, 1.00)),
    ("3_warm",    "暖光",       1.00, (1.00, 0.55, 0.25)),
    ("4_dimwarm", "又暗又暖",   0.50, (1.00, 0.55, 0.25)),
    ("5_bright",  "变亮到 125%", 1.25, (1.00, 1.00, 1.00)),
    ("6_cool",    "冷光",       1.00, (0.80, 0.92, 1.00)),
]

SENT_LIGHT = "a light green cube"
SENT_DARK = "a dark green cube"

# 喂给 CLIP 的话。前三句是正题，最后一句是陪跑（看它会不会盖过浅绿）。
TEXTS = [
    "a red cube",
    "a light green cube",
    "a dark green cube",
    "a green cube",
]


# ---------------------------------------------------------------- 拍/摆

def shot(arm) -> np.ndarray:
    """现渲一张俯视图。不走 step —— step 会让方块掉、手臂动。"""
    rend = arm.u._overhead_obs_renderer
    rend.update_scene(arm.u.data, camera=arm.u._overhead_obs_cam)
    return np.asarray(rend.render())


def park_cubes(arm, park: bool, keep: dict) -> None:
    """把方块临时挪到画面外（拍参考图），用原样放回。

    `place_cube` 按 yaw 重算四元数会抹掉原姿态，所以存 qpos 原值、放回时写回去。
    改完 qpos 必须 `mj_forward` —— 渲染器读的是 `geom_xpos`。
    """
    import mujoco

    u = arm.u
    for i in range(arm.n_cubes):
        a, d = arm.cube_addr[i], arm.cube_dof[i]
        if park:
            keep[i] = (u.data.qpos[a:a + 3].copy(), u.data.qpos[a + 3:a + 7].copy())
            u.data.qpos[a:a + 3] = [PARK_XY[0], PARK_XY[1], CUBE_Z]
            u.data.qvel[d:d + 6] = 0.0
        else:
            p, q = keep[i]
            u.data.qpos[a:a + 3] = p
            u.data.qpos[a + 3:a + 7] = q
            u.data.qvel[d:d + 6] = 0.0
    mujoco.mj_forward(u.model, u.data)


def paint_cubes(arm) -> None:
    """把三个方块涂成 CUBE_PAINT 里那三块绿/红。改运行时模型，不动只读包里那份 xml。"""
    for i, rgba in CUBE_PAINT.items():
        for g in arm.cube_geom[i]:
            arm.u.model.geom_rgba[g] = np.asarray(rgba, dtype=np.float64)


def set_lights(model, base, scale: float = 1.0, tint=(1.0, 1.0, 1.0)) -> None:
    """所有灯一起改亮度/色温。`base` 是场景原样，每次从它算，不叠着上一次改。

    从原样算而不是就地乘，跟 `arm_sim.py::_randomize_look` 一条道理：
    就地乘的话跑上几档全场就黑了。
    """
    tint = np.asarray(tint, dtype=np.float64)
    for i in range(int(model.nlight)):
        model.light_diffuse[i] = np.clip(
            np.asarray(base[i], dtype=np.float64) * float(scale) * tint, 0.0, 1.0)


def crops_of(rgb, blobs, mask, pad: int = 6):
    """每个候选块裁一小块出来，放大到 CLIP 要的 224。返回 [(标签号, 平均RGB, 裁好的图, 外框)]。

    裁的是**外框**不是掩膜里那几个像素 —— CLIP 看的是整块东西长什么样，
    抠成异形反而少了上下文。外框外多留一点边，别贴着方块切。
    """
    import cv2

    a = np.asarray(rgb)
    h, w = a.shape[:2]
    m = np.asarray(mask)
    n, labels = cv2.connectedComponents((m > 0).astype(np.uint8) * 255, 8)
    out = []
    for _area, _c, (x, y, bw, bh), lab in blobs:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        cut = a[y0:y1, x0:x1]
        if cut.size == 0:
            continue
        # 这一块的平均颜色：只统计属于它自己的那些像素
        sel = (labels[y0:y1, x0:x1] == int(lab))
        mean = np.asarray(cut, dtype=np.float64)[sel].mean(axis=0) if sel.any() else np.zeros(3)
        big = cv2.resize(cut, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        out.append((int(lab), tuple(float(v) for v in mean[:3]), big, (x0, y0, x1, y1)))
    del n
    return out


# ---------------------------------------------------------------- CLIP

def load_model(kind: str):
    """按名字把模型和预处理器拉起来。

    **别照仓库名挑类。** `google/siglip2-base-patch16-224` 的 config 里
    `model_type` 写的是 `siglip`（固定分辨率那一支），拿 `Siglip2Model` 去装会撞
    shape mismatch —— 那个类对应的是 `-naflex` 变体，patch embedding 结构不一样。
    所以走 `AutoModel`，让 config 自己决定用哪个类。
    """
    import torch
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

    repo = MODELS[kind]
    if kind == "clip":
        model = CLIPModel.from_pretrained(repo).eval()
        proc = CLIPProcessor.from_pretrained(repo)
    else:
        model = AutoModel.from_pretrained(repo).eval()
        proc = AutoProcessor.from_pretrained(repo)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev), proc, dev


def score(model, proc, dev, imgs, texts, kind: str):
    """每张图对每句话的分数。返回 (n_img, n_text) 的数组，大的表示更像。

    **两代模型的归一方式不一样，这是它们的训练方式决定的**，不是随便挑的：
        CLIP   用 softmax 对比损失 —— 「这几句话里哪句最配这张图」，所以对**句子那一维**
               做 softmax，一行加起来是 1。
        SigLIP 用 sigmoid 成对损失 —— 「这张图跟这句话是不是一对」，每对各自独立，
               所以逐元素 sigmoid，一行加起来不是 1（可能好几句都高，也可能全都低）。
    两种都只影响显示出来的数；**谁排第一**是同一件事，看 `logits_per_image` 的最大值。
    """
    import torch

    # 图文分开预处理再合起来：SigLIP 的文本塔要按 max_length 补齐，CLIP 不用。
    t = proc(text=texts, padding="max_length", return_tensors="pt")
    v = proc(images=imgs, return_tensors="pt")
    inputs = {**t, **v}
    inputs = {k: (val.to(dev) if hasattr(val, "to") else val) for k, val in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits_per_image
    return (logits.softmax(dim=1) if kind == "clip" else torch.sigmoid(logits)).cpu().numpy()


# ---------------------------------------------------------------- 主流程

def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="拿哪一代模型判「这块是哪个」")
    ap.add_argument("--model", choices=sorted(MODELS), default="clip",
                    help="clip = 2021 的老基线；siglip2 = 2025 的 SigLIP 2。"
                         "两次跑的条件完全一样，数可以直接对着看。")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = OUT_ROOT / args.model
    if out.exists():
        shutil.rmtree(out)                 # 只清自己这一次的图，不动另一代模型的
    out.mkdir(parents=True)
    OUT = out                              # 下面各处沿用它

    import cv2

    arm = make("arm", "sim")               # 三个方块：red / green / blue
    try:
        arm.reset(seed=0)
        paint_cubes(arm)
        base_lights = arm.u.model.light_diffuse.copy()

        # 参考图拍一次就够。`foreground_mask` 先做光照归一（减中位数、除以 MAD），
        # 比的是「这一片比全场典型值亮多少」—— 本来就设计成跟光照无关。
        # 真机上参考图也是拍一次就一直用，不会每拍一张重拍一张。
        keep: dict = {}
        park_cubes(arm, True, keep)
        ref = shot(arm)
        park_cubes(arm, False, keep)
        cv2.imwrite(str(OUT / "0_ref_empty.png"), cv2.cvtColor(np.asarray(ref), cv2.COLOR_RGB2BGR))

        model, proc, dev = load_model(args.model)
        print(f"模型 {args.model} = {MODELS[args.model]}  跑在 {dev}"
              f"   喂进去的是 {IMG_SIZE}x{IMG_SIZE} 的裁块")
        print(f"涂成：{PAINTED}")
        print(f"指令：{TEXTS}")
        print(f"除光以外什么都不动：同一局、同一批位置朝向、同一张参考图\n")

        from PIL import Image

        i_light, i_dark = TEXTS.index(SENT_LIGHT), TEXTS.index(SENT_DARK)
        rows = []
        for key, label, scale, tint in CONDS:
            set_lights(arm.u.model, base_lights, scale, tint)
            rgb = shot(arm)
            mask = vision.foreground_mask(rgb, ref)
            cuts = crops_of(rgb, vision.blobs_of(mask), mask)
            cv2.imwrite(str(OUT / f"{key}_overhead.png"),
                        cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
            for lab, _m, big, _box in cuts:
                cv2.imwrite(str(OUT / f"{key}_crop_label{lab}.png"),
                            cv2.cvtColor(big, cv2.COLOR_RGB2BGR))

            print(f"【{label}】灯 ×{scale:.2f}  色温 ({tint[0]:.2f},{tint[1]:.2f},{tint[2]:.2f})")
            if not cuts:
                print("    一个候选都没挑出来 —— 这一档连方块都找不到了\n")
                rows.append((label, None, None, None))
                continue

            name = {lab: f"标签{lab}({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f})"
                    for lab, mean, _b, _x in cuts}
            p = score(model, proc, dev,
                      [Image.fromarray(big) for _l, _m, big, _b in cuts], TEXTS,
                      args.model)

            w0 = 26
            print(f"  {'候选(自己那块的平均RGB)':<{w0}s}"
                  + "".join(f"{t[:18]:>20s}" for t in TEXTS))
            for r, (lab, _m, _b, _x) in enumerate(cuts):
                print(f"  {name[lab]:<{w0}s}"
                      + "".join(f"{p[r, c]:>20.3f}" for c in range(len(TEXTS)))
                      + f"   ← 最像「{TEXTS[int(np.argmax(p[r]))]}」")

            # 谁该是浅绿、谁该是深绿，**只看候选块自己的平均颜色**：绿通道占上风的那几块
            # 按 G 从大到小排，最亮的那块是浅绿、最暗的那块是深绿。
            greens = sorted([c for c in cuts
                             if c[1][1] > c[1][0] and c[1][1] > c[1][2]],
                            key=lambda c: -c[1][1])
            if len(greens) < 2:
                print(f"    绿候选只有 {len(greens)} 块，深浅没得比\n")
                rows.append((label, None, None, None))
                continue
            want_light, want_dark = greens[0][0], greens[-1][0]
            got_light = cuts[int(np.argmax(p[:, i_light]))][0]
            got_dark = cuts[int(np.argmax(p[:, i_dark]))][0]
            ok_light, ok_dark = got_light == want_light, got_dark == want_dark
            print(f"    「{SENT_LIGHT}」→ {name[got_light]}({np.max(p[:, i_light]):.3f})"
                  f"   该给 {name[want_light]}   {'✓' if ok_light else '✗'}")
            print(f"    「{SENT_DARK}」→ {name[got_dark]}({np.max(p[:, i_dark]):.3f})"
                  f"   该给 {name[want_dark]}   {'✓' if ok_dark else '✗'}\n")
            rows.append((label, ok_light, ok_dark,
                         (name[want_light], name[want_dark],
                          name[got_light], name[got_dark])))

        set_lights(arm.u.model, base_lights)     # 还回原样

        print("结果：")
        print(f"  {'条件':<12s}{'「浅绿」给对了':>16s}{'「深绿」给对了':>16s}")
        for label, ok_l, ok_d, _ in rows:
            if ok_l is None:
                print(f"  {label:<12s}{'——':>16s}{'——':>16s}")
            else:
                print(f"  {label:<12s}{'✓' if ok_l else '✗':>16s}{'✓' if ok_d else '✗':>16s}")
        print(f"\n图在 {OUT}")
    finally:
        try:
            arm.close()
        except Exception as exc:               # noqa: BLE001
            print(f"关环境时报错（不影响上面的结果）：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
