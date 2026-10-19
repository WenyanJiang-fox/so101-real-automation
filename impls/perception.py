"""感知：看着画面和一句话，说清楚「这话指的是哪一个、它在哪」。

**它只管一件小事**：从固定的俯视画面里按指令挑出目标方块，量出它在地面上的位置。
后面转腕子、下探、合爪那一整套还是 `skills/pick.py` 的事。

**为什么单独立一个接口。** 主循环里那几个接口都是「换一样、改配置一行」，感知也得
能换 —— `nearest`（不看指令、按工位表取最近的那块，当基线）、`clip` / `siglip2`
（真拿一句话去挑）。策略、机械臂、判官一个字不用动。

**候选从哪来：不看颜色。** 跟一张空工位的参考图比（`vision.foreground_mask`），比
出来多出来的连通域就是候选 —— 场上摆了五块不同颜色的方块、换灯、换桌面颜色都一样
成立，因为它比的是明暗结构，不是色值（`vision.lit_normalized` 先把光照除掉了）。
颜色只在最后「挑哪一块」那一步用得上。

**挑法**：每个候选裁一小块出来，跟「a red cube」「a white cylinder」这一组句子逐对
打分，取**指令点名那一件**里分最高的那块。指令点的是哪一件，是从指令文字里读出来的
（先要求颜色词和形状词都对上，一个都没有才退一步只对一样），不是模型读的 ——
**模型干的事是「画面里哪块是白的、哪块是方的」**。场上摆什么就认什么：词的来源是
`core/geom.py` 的摆件清单，不是写死的三色三方块。指令点不出来就当场报「没认出来」，不猜。

**两代模型都留着**，配置里一行切换，号都一样的东西好对照：
    clip     2021 年的老基线，softmax 对比损失
    siglip2  2025 年的 SigLIP 2，sigmoid 成对损失
两者的归一方式不同（一行加起来是不是 1），但**谁排第一**是同一件事。

**权重下不来会静静地挂住**：本机 huggingface.co 直连是超时的，裸 `from_pretrained`
不报错、一直重试。所以这里先按「只用本地缓存」试一把，不在本地就当场说清楚该加什么
环境变量，不让你对着一个十分钟没动静的窗口猜。
"""

from __future__ import annotations

import numpy as np

from core import blind
from core import shapes as _shapes
from core import vision
from core.geom import CUBE_ORDER, MARKER_CENTER, object_rest_z
from core.interfaces import Perceptor
from core.registry import register

IMG_SIZE = 224            # 两个模型的视觉塔都吃 224

MODELS = {
    "clip":    "openai/clip-vit-base-patch32",
    "siglip2": "google/siglip2-base-patch16-224",
}


def _intr(mount, size):
    """这台相机的内参。标定过（`mount["k"]`）就用标定那份，没有就按视场角算。

    **不能一律按「视场角 + 画面中心」算**：俯视相机的主点不在画面正中（实测偏 25 个
    像素），按居中那套算，打回地面的点会被顶偏几十毫米。
    """
    k = mount.get("k")
    if k is not None:
        return np.asarray(k, dtype=np.float64)
    return vision.intrinsics(mount["fovy"], size[0], size[1])


def _crop(rgb, box, pad: int = 6) -> np.ndarray:
    """把一个候选的外框裁出来、放大到 224。

    裁的是**外框**不是掩膜里那几个像素 —— 模型看的是整块东西长什么样，抠成异形反而
    少了上下文。外框外多留一点边，别贴着方块切。
    """
    import cv2

    a = np.asarray(rgb)
    h, w = a.shape[:2]
    x, y, bw, bh = box
    cut = a[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
    if cut.size == 0:
        return None
    return cv2.resize(cut, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)


def _load(kind: str):
    """把模型和预处理器拉起来。**先只认本地缓存**，没有再教怎么下。

    **别照仓库名挑类。** `google/siglip2-base-patch16-224` 的 config 里 `model_type`
    写的是 `siglip`（固定分辨率那一支），拿 `Siglip2Model` 去装会撞 shape mismatch ——
    那个类对应的是 `-naflex` 变体。走 `AutoModel`，让 config 自己决定。
    """
    import os

    import torch
    from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

    repo = MODELS[kind]
    try:
        if kind == "clip":
            model = CLIPModel.from_pretrained(repo, local_files_only=True).eval()
            proc = CLIPProcessor.from_pretrained(repo, local_files_only=True)
        else:
            model = AutoModel.from_pretrained(repo, local_files_only=True).eval()
            proc = AutoProcessor.from_pretrained(repo, local_files_only=True)
    except Exception as exc:                                  # noqa: BLE001
        raise RuntimeError(
            f"{repo} 的权重不在本地缓存里（{type(exc).__name__}）。\n"
            f"本机 huggingface.co 直连是超时的，裸下载会静静地挂住十几分钟不报错 ——\n"
            f"要下就带上镜像重跑一次：\n"
            f"    HF_ENDPOINT=https://hf-mirror.com "
            f"G:/anaconda/envs/sim2real/python.exe run_sim.py\n"
            f"（缓存落在 C 盘 {os.path.expanduser('~')}/.cache/huggingface/hub）") from exc
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(dev), proc, dev


def _score(model, proc, dev, imgs, texts, kind: str):
    """每张图对每句话的分数，返回 (n_img, n_text) 的数组，大的表示更像。

    **两代模型的归一方式不一样，这是它们的训练方式决定的**，不是随便挑的：
        CLIP    softmax 对比损失 —— 「这几句话里哪句最配这张图」，对**句子那一维**
                做 softmax，一行加起来是 1。
        SigLIP  sigmoid 成对损失 —— 「这张图跟这句话是不是一对」，每对各自独立，
                逐元素 sigmoid，一行加起来不是 1。
    两种都只影响显示出来的数；**谁排第一**是同一件事，看 `logits_per_image`。
    """
    import torch
    from PIL import Image

    t = proc(text=texts, padding="max_length", return_tensors="pt")
    v = proc(images=[Image.fromarray(im) for im in imgs], return_tensors="pt")
    inputs = {**t, **v}
    inputs = {k: (val.to(dev) if hasattr(val, "to") else val) for k, val in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
    return (logits.softmax(dim=1) if kind == "clip" else torch.sigmoid(logits)).cpu().numpy()


class _OverheadPerceptor(Perceptor):
    """从俯视画面里找目标：候选不看颜色，挑哪一个交给子类的 `_choose`。

    公共的那一段（找候选、裁块、拿选中的那块拟合位置）写在这里，子类只决定「挑哪块」
    —— `nearest` 和拿模型挑，差别就只有这一件事。

    **词表按场上摆的东西来，不是写死的三色三方块。** 桌上有什么，它就认得什么：
    颜色词（白/黄/黑…）和形状词（方块/圆柱/球）都算，两个都对上才算点名。
    """

    cam_name = "sim_overhead"

    def __init__(self, cams, ref=None, palette=None, objects=None):
        self.cams = cams
        self.ref = ref
        # 场上这几件「颜色 + 形状」。不传 objects 就按老口径：palette 里每个颜色一块方块。
        self.objects = _specs(objects) if objects else [
            {"color": str(c), "shape": "cube"} for c in (palette or CUBE_ORDER)]
        self.palette = tuple(o["color"] for o in self.objects)
        self.last: dict = {}

    def set_reference(self, ref) -> None:
        """换一张空工位参考图。桌面颜色被换掉之后（域随机化）要重拍一张。"""
        self.ref = ref

    def _choose(self, cuts, instruction, guide_uv):
        """在这些候选里挑一个。返回它在 `cuts` 里的位置，挑不出来给 None。

        **选中了就得把「是哪一件」记进 `self.last["obj"]`** —— 位置拟合要拿它的形状
        去投剪影（方块投八个角、圆柱投两个圆），这一件不知道是哪种就没法投。
        """
        raise NotImplementedError

    def _want_index(self, instruction) -> int | None:
        """指令点的是 `self.objects` 里的第几件。读不出来给 None。

        先要求**颜色词和形状词都对上**（「白色的圆柱」不会挑成「白色的方块」），
        一个都没有再退一步：只对颜色、或只对形状。都读不出来就是没点名，不猜。
        """
        low = str(instruction or "").lower()
        for want_both in (True, False):
            for i, o in enumerate(self.objects):
                c_ok = o["color"].lower() in low
                s_ok = o["shape"].lower() in low
                if (c_ok and s_ok) if want_both else (c_ok or s_ok):
                    return i
        return None

    def _want_color(self, instruction) -> str | None:
        """指令点的是哪个颜色词（只给日志用，挑哪一件由 `_want_index` 说了算）。"""
        i = self._want_index(instruction)
        return None if i is None else self.objects[i]["color"]

    def locate(self, instruction: str, *, guess=None):
        cam = (self.cams or {}).get(self.cam_name)
        if cam is None:
            return None, {"why": f"没有 {self.cam_name} 这一路相机"}
        rgb = np.asarray(cam.shot())
        mount = cam.mount()
        if mount is None:
            return None, {"why": f"{self.cam_name} 报不出位姿，反推不出世界坐标"}
        if self.ref is None:
            return None, {"why": "没有空工位的参考图，挑不出候选"}

        mask = vision.foreground_mask(rgb, self.ref)
        cands = vision.blobs_of(mask)
        if not cands:
            return None, {"why": "跟空工位参考图比完，一个像样的候选都没有"}

        import cv2

        size = (rgb.shape[1], rgb.shape[0])
        k = _intr(mount, size)
        # 带路点落在多高。**盲测下不能拿真值**：那等于把「这东西多高」提前告诉感知。
        z0 = blind.PLACEHOLDER_Z if blind.ON else object_rest_z(0)
        base = np.array([float(MARKER_CENTER[0]), float(MARKER_CENTER[1]), z0])
        g = np.asarray(guess, dtype=np.float64) if guess is not None else base

        # 带路点投到画面上，只给 `nearest` 那种按坐标挑的用
        gu = vision.project(mount["pos"], mount["mat"], k, [float(g[0]), float(g[1]), z0])
        _n, lab = cv2.connectedComponents(np.asarray(mask, np.uint8), 8)
        cuts = []
        for area, uv, box, _label in cands:
            big = _crop(rgb, box)
            if big is not None:
                cuts.append((_label, box, uv, big, area))
        if not cuts:
            return None, {"why": "候选都没裁出来"}

        j = self._choose(cuts, instruction, gu)
        if j is None:
            return None, {**self.last, "n_cand": len(cuts)}

        label, box, uv, _big, area = cuts[j]
        one = (lab == label).astype(np.uint8) * 255
        # **按这一件自己的形状投剪影。** 挑出来是什么形状就按什么形状量位置 ——
        # 拿方块的八个角去套一根圆柱，重合度上不去，位置也就跟着错。
        oi = int(self.last.get("obj") or 0)
        spec = self.objects[min(oi, len(self.objects) - 1)]
        est, est_info = None, None
        if str(spec.get("shape", "")).lower() in ("unknown", "?"):
            # **盲测：清单里没写这东西是什么形状。** 形状和尺寸只能从这一眼里猜
            # （`vision.estimate_shape`：拿一串候选形状各投一遍，谁的重合度高是谁）。
            # 猜不出来就当场停 —— 拿一个自己都不信的形状去算落点，比不抓更糟。
            est, est_info = vision.estimate_shape(rgb, mount, one, g)
            if est is None:
                return None, {**self.last, **(est_info or {}), **{
                    "why": (est_info or {}).get("why", "猜不出形状"),
                    "color": spec["color"], "shape": "unknown", "obj": oi,
                    "n_cand": len(cuts), "uv": uv, "box": box, "area": area}}
            spec = {**est, "color": spec["color"]}
        sh = _shapes.from_spec(spec)
        # 剪影是按 `guess` 的 z 投回画面的，这一个传的是**这东西自己有多高**（中心
        # 离地 = `rest_z`）。斜俯视下 z 差 10mm 就能把水平位置顶偏十几毫米。
        center, _yaw, info = vision.estimate_pose(
            rgb, mount, spec["color"], [float(g[0]), float(g[1]), float(sh.rest_z)],
            shape=sh, mask=one)
        info = {**self.last, **info, "est": est, "est_info": est_info,
                "color": spec["color"], "shape": spec["shape"], "obj": oi,
                "n_cand": len(cuts), "uv": uv, "box": box, "area": area}
        return (None if center is None else np.asarray(center, dtype=np.float64)), info


def _specs(objects) -> list[dict]:
    """把清单一行行补成 `{"color","shape"}`。`core.shapes.from_spec` 认这个格式。"""
    out = []
    for o in objects:
        if isinstance(o, str):
            out.append({"color": o, "shape": "cube"})
        else:
            out.append({"color": str(o.get("color", "red")),
                        "shape": str(o.get("shape", "cube")),
                        **{k: v for k, v in o.items() if k not in ("color", "shape")}})
    return out


@register("perception", "nearest")
class NearestPerceptor(_OverheadPerceptor):
    """不看指令，取离带路点最近的那个候选。

    **这是基线，不是主路。** 它代表「操作员把方块摆到工位上、我知道那个坐标」这套
    口径 —— 场景里方块就摆在工位上时它够用，摆偏了、或者场上有别的东西像方块，它
    就没有依据了。拿它当对照，才看得出「按指令挑」到底多值钱。
    """

    name = "nearest"

    def _choose(self, cuts, instruction, guide_uv):
        if guide_uv is None:
            self.last = {"why": "带路点投不进画面，没法按坐标挑"}
            return None
        # **它不看指令，所以也不知道挑中的是哪一件** —— 场上几种形状混着摆的时候，
        # 这条基线拿第一件的形状去量位置。要按形状区分就得走拿模型挑的那一路。
        self.last = {"how": "离带路点最近", "obj": 0,
                     "guide_uv": tuple(np.round(guide_uv, 1))}
        return int(min(range(len(cuts)),
                       key=lambda i: float(np.hypot(cuts[i][2][0] - guide_uv[0],
                                                    cuts[i][2][1] - guide_uv[1]))))


class _LangPerceptor(_OverheadPerceptor):
    """拿一句话去挑：每个候选跟一组颜色句子打分，取指令点名那个颜色里分最高的。"""

    kind = "siglip2"

    def __init__(self, cams, ref=None, palette=None, objects=None):
        super().__init__(cams, ref, palette, objects)
        # 一句话一张图打分，句子就是**场上这几件各自的名字**：颜色 + 形状。
        # 所以「白色的圆柱」和「白色的方块」是两个不同的词条，模型分得开。
        # 盲测下清单里没有形状词 —— 句子就只说颜色（不说形状，正是这一支要的口径）。
        self.texts = [f"a {o['color']} {o['shape']}"
                      if str(o["shape"]).lower() not in ("unknown", "?")
                      else f"a {o['color']} object" for o in self.objects]
        self.model, self.proc, self.dev = _load(self.kind)

    def _choose(self, cuts, instruction, guide_uv):
        k = self._want_index(instruction)
        if k is None:
            self.last = {"why": f"指令里点不出场上这几件里的任何一件"
                                f"（场上：{self.texts}；指令：{instruction!r}）"}
            return None
        p = _score(self.model, self.proc, self.dev, [c[3] for c in cuts],
                   self.texts, self.kind)
        j = int(np.argmax(p[:, k]))
        order = np.argsort(-p[:, k])
        self.last = {
            "how": f"{self.kind}：跟「{self.texts[k]}」逐块打分",
            "want": self.texts[k], "obj": k,
            # 赢的那块分多少、第二名多少 —— 两个数挨着，就知道这一挑是压过去的还是险胜
            "score": [round(float(p[i, k]), 3) for i in order[:3]],
            "cand": [f"{cuts[i][4]}px" for i in order[:3]],
        }
        return j


@register("perception", "clip")
class ClipPerceptor(_LangPerceptor):
    """CLIP（2021）那一支。留着是为了跟 SigLIP 2 对着看。"""

    name = "clip"
    kind = "clip"


@register("perception", "siglip2")
class Siglip2Perceptor(_LangPerceptor):
    """SigLIP 2（2025）。"""

    name = "siglip2"
    kind = "siglip2"
