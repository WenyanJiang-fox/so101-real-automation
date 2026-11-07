"""实验：场上摆好几块**不同颜色**，给一句话，看它挑的是不是那一块。

**要回答的只有一句话**：不再是「这三块里哪块是浅绿」，而是真的一句话下去 ——
「the blue cube」—— 它挑中的那块，是不是蓝的那块。

**和上一版（`exp_clip_lang.py`）的两处不同**：
1. 场上从 3 块变成 5 块，颜色改成五个互相分得开的（红/橙/蓝/绿/紫），
   不再是深浅两个绿那种刁难题。这一版测的是「多色同场能不能选对」。
2. **候选块跟方块怎么对上，改用 MuJoCo 的分割渲染**，不靠位置、也不靠平均颜色猜。
   上一版是拿「这一块的平均 RGB」认它是什么色，五色同场时容易认错；
   分割图直接把每个像素属于哪个 geom 标出来了，`cube_geom[i]` 一查就知道是第几块。

走的是 `impls/arm_sim.py` 那个口子：`_make_env(cube_order)` 传几个颜色就建几块，
`markers=False` 不注标识盘（这版不需要标识点）。

跑：G:/anaconda/envs/sim2real/python.exe G:\\sim2real\\so101-real-harness\\exp_lang_pick.py --model siglip2
出图：G:\\sim2real\\so101-real-harness\\out\\lang_pick\\<模型>\\
（下权重走镜像：HF_ENDPOINT=https://hf-mirror.com）
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

import impls.arm_sim                                  # noqa: E402,F401  导入即挂上牌子

from core import vision                               # noqa: E402
from core.geom import CUBE_Z                          # noqa: E402
from core.registry import make                        # noqa: E402

OUT_ROOT = HERE / "out" / "lang_pick"
MODELS = {
    "clip":    "openai/clip-vit-base-patch32",
    "siglip2": "google/siglip2-base-patch16-224",
}
IMG_SIZE = 224
PARK_XY = (3.0, 3.0)               # 把方块临时挪到画面外，拍参考图用

# 五块同场。挑的都是互相离得远的颜色 —— 这版要测的是「多色能不能选对」，
# 不是「长得像的能不能分开」（那个上一版测过了）。
COLORS = ("red", "orange", "blue", "green", "purple")
TEXTS = [f"the {c} cube" for c in COLORS]

CONDS = [
    ("1_normal",  "正常",       1.00, (1.00, 1.00, 1.00)),
    ("2_dim",     "变暗到 35%", 0.35, (1.00, 1.00, 1.00)),
    ("3_warm",    "暖光",       1.00, (1.00, 0.55, 0.25)),
    ("4_dimwarm", "又暗又暖",   0.50, (1.00, 0.55, 0.25)),
    ("5_bright",  "变亮到 125%", 1.25, (1.00, 1.00, 1.00)),
    ("6_cool",    "冷光",       1.00, (0.80, 0.92, 1.00)),
]


# ---------------------------------------------------------------- 拍/摆

def shot(arm) -> np.ndarray:
    """现渲一张俯视图。不走 step —— step 会让方块掉、手臂动。"""
    rend = arm.u._overhead_obs_renderer
    rend.update_scene(arm.u.data, camera=arm.u._overhead_obs_cam)
    return np.asarray(rend.render())


def seg_of(arm) -> np.ndarray:
    """现渲一张分割图：每个像素上写的是它属于哪个 geom 的编号。

    **拿它把候选块跟方块对上，不靠位置也不靠颜色猜。** 分割图和上面那张彩图是
    同一个相机、同一个仿真状态渲的，像素一一对应，所以候选块落在哪几个像素，
    去分割图上读那几个像素的 geom 编号，就知道这一块是谁。
    """
    rend = arm.u._overhead_obs_renderer
    rend.enable_segmentation_rendering()
    try:
        rend.update_scene(arm.u.data, camera=arm.u._overhead_obs_cam)
        s = np.asarray(rend.render())
    finally:
        rend.disable_segmentation_rendering()
    # MuJoCo 的分割图是 (H,W,2)：第一层是编号。个别版本给 (H,W)，一并兜住。
    return s[..., 0] if s.ndim == 3 else s


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


def set_lights(model, base, scale: float = 1.0, tint=(1.0, 1.0, 1.0)) -> None:
    """所有灯一起改亮度/色温。每次从 `base` 算，不叠着上一次改。"""
    tint = np.asarray(tint, dtype=np.float64)
    for i in range(int(model.nlight)):
        model.light_diffuse[i] = np.clip(
            np.asarray(base[i], dtype=np.float64) * float(scale) * tint, 0.0, 1.0)


# ---------------------------------------------------------------- 候选块

def crops_of(rgb, blobs, mask, seg, geom_owner, pad: int = 6):
    """裁候选块 + 用分割图查出每块的归属。

    返回 [(标签号, 属于第几块方块(查不出来给 None), 平均RGB, 224图, 外框)]。
    **归属只看那一块自己盖住的那几个像素在分割图上写的是哪个 geom**，
    跟它在画面里哪儿、离谁近无关。
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
        sel = (labels[y0:y1, x0:x1] == int(lab))
        mean = (np.asarray(cut, dtype=np.float64)[sel].mean(axis=0) if sel.any()
                else np.zeros(4))
        ids = np.asarray(seg)[y0:y1, x0:x1][sel]
        vals, counts = (np.unique(ids, return_counts=True) if ids.size
                        else (np.array([]), np.array([])))
        top = int(vals[int(np.argmax(counts))]) if vals.size else -1
        big = cv2.resize(cut, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        out.append((int(lab), geom_owner.get(top), tuple(float(v) for v in mean[:3]),
                    big, (x0, y0, x1, y1)))
    del n
    return out


# ---------------------------------------------------------------- 模型

def load_model(kind: str):
    """**别照仓库名挑类。** `google/siglip2-base-patch16-224` 的 config 里
    `model_type` 写的是 `siglip`，拿 `Siglip2Model` 去装会撞 shape mismatch
    （那个类对应 `-naflex` 变体）。走 `AutoModel`，让 config 自己决定。
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
    """每张图对每句话的分数。归一两代不一样（CLIP softmax 对句子那一维、
    SigLIP 逐对 sigmoid），但**谁排第一**是同一件事，看 `logits_per_image`。"""
    import torch

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

    ap = argparse.ArgumentParser(description="五色同场，一句话下去挑哪块")
    ap.add_argument("--model", choices=sorted(MODELS), default="siglip2")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT = OUT_ROOT / args.model
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    import cv2

    arm = make("arm", "sim", cube_order=COLORS, markers=False, ring=False)
    try:
        info = arm.reset(seed=0)
        base_lights = arm.u.model.light_diffuse.copy()
        geom_owner = {int(g): i for i in range(arm.n_cubes) for g in arm.cube_geom[i]}

        keep: dict = {}
        park_cubes(arm, True, keep)
        ref = shot(arm)
        park_cubes(arm, False, keep)
        cv2.imwrite(str(OUT / "0_ref_empty.png"), cv2.cvtColor(np.asarray(ref), cv2.COLOR_RGB2BGR))

        model, proc, dev = load_model(args.model)
        print(f"场上 {arm.n_cubes} 块：{COLORS}   这一局要抓第 {info['target_index']} 块"
              f"（{info['task']}）")
        print(f"模型 {args.model} = {MODELS[args.model]}  跑在 {dev}")
        print(f"指令：{TEXTS}\n")

        from PIL import Image

        all_rows = []
        for key, label, scale, tint in CONDS:
            set_lights(arm.u.model, base_lights, scale, tint)
            rgb = shot(arm)
            seg = seg_of(arm)
            mask = vision.foreground_mask(rgb, ref)
            blobs = vision.blobs_of(mask)
            cuts = crops_of(rgb, blobs, mask, seg, geom_owner)
            cv2.imwrite(str(OUT / f"{key}_overhead.png"),
                        cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))

            print(f"【{label}】灯 ×{scale:.2f}  色温 ({tint[0]:.2f},{tint[1]:.2f},{tint[2]:.2f})"
                  f"   候选 {len(cuts)} 个")
            if not cuts:
                print("    一个候选都没挑出来\n")
                all_rows.append((label, {}))
                continue
            for lab, owner, mean, big, _box in cuts:
                who = COLORS[owner] if owner is not None else "不是方块(机械臂?)"
                cv2.imwrite(str(OUT / f"{key}_l{lab}_{who}.png"),
                            cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
                print(f"    候选 标签{lab}  {who:<18s}平均RGB "
                      f"({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f})")

            p = score(model, proc, dev,
                      [Image.fromarray(big) for _l, _o, _m, big, _b in cuts], TEXTS,
                      args.model)

            hit = {}
            for s, text in enumerate(TEXTS):
                w = int(np.argmax(p[:, s]))
                lab, owner, _mean, _big, _box = cuts[w]
                got = COLORS[owner] if owner is not None else "不是方块"
                ok = (owner is not None and COLORS[owner] == COLORS[s])
                hit[COLORS[s]] = ok
                print(f"    指令「{text}」→ {got:<18s}(分 {p[w, s]:.3f})"
                      f"   该给 {COLORS[s]:<7s}{'✓' if ok else '✗'}")
            n_ok = sum(hit.values())
            print(f"    这句场景：{n_ok}/{len(TEXTS)} 对\n")
            all_rows.append((label, hit))

        set_lights(arm.u.model, base_lights)

        print("结果（每档光下，五句指令各挑对几块）：")
        head = "".join(f"{c:>10s}" for c in COLORS)
        print(f"  {'条件':<12s}{head}{'合计':>8s}")
        for label, hit in all_rows:
            cells = "".join(f"{'✓' if hit.get(c) else ('✗' if hit else '—'):>10s}"
                            for c in COLORS)
            print(f"  {label:<12s}{cells}{sum(hit.values()):>6d}/{len(TEXTS)}")
        print(f"\n图在 {OUT}")
    finally:
        try:
            arm.close()
        except Exception as exc:               # noqa: BLE001
            print(f"关环境时报错（不影响上面的结果）：{exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
