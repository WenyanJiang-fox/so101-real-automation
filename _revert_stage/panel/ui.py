"""按那一层：**一个原生 Windows 窗口**（tkinter），左边画面、右边按钮和读数。

**为什么从 cv2 换成 tkinter。** 原来那版是 `cv2.namedWindow` + 自己画按钮，两个硬伤：
按钮是用 `cv2.rectangle` 摆出来的、`cv2.putText` **只认 ASCII**（中文画出来是一排 `?`，
这套 cv2 里也没有 `freetype`，没有别的写字路径），所以整块面板被迫全英文。
tkinter 是标准库、在 Windows 上用的就是系统原生控件，**中文正常**，按钮能按、能高亮、
能禁用。不需要装任何东西（画面那一路用 Pillow 的 `ImageTk`，Pillow 本来就有）。

**这一层不动作、不算数、不碰硬件。** 它只做三件事：把 `rig.snap()` 抄成画面和读数、
把按钮变成 `rig.submit(...)`、显示 `rig` 交出来的中文日志。动作全在 `panel/rig.py`
那个线程里排队执行（技能那一段在 `panel/skill_driver.py`）。

**画面上的字仍然是英文**：那一层是 `cv2.putText` 画到帧上的（`panel/view.py`），
换窗口救不了它。中文一律走这个窗口的「读数」「日志」两块和终端。
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from panel import measure, view                                     # noqa: E402

TITLE = "SO-101 真机面板"
DEG_STEPS = (0.2, 0.5, 1.0, 2.0, 5.0)
MM_STEPS = (1.0, 2.0, 5.0, 10.0)
# 界面刷新周期。画面那一路每帧要过一遍「画标注 + 转 RGB + 缩放」，比 cv2 那次贵，
# 所以按 12Hz 走；快照本身是 10Hz（`rig.TICK_S`），再快也是白转。
UI_S = 0.08
LOG_KEEP = 300
PIC_H = 620                       # 画面默认显示高度（等比缩放，原帧不裁）

JOINTS_CN = ("1 肩转", "2 肩抬", "3 肘弯", "4 腕俯", "5 腕转", "6 夹爪")
PHASE_CN = {
    measure.PHASE_LOST: "没认出方块",
    measure.PHASE_SEE: "量不出来",
    measure.PHASE_ALIGN: "该横着挪",
    measure.PHASE_DESCEND: "该竖直下探",
    measure.PHASE_CLOSE: "该合爪",
    measure.PHASE_LIFT: "该抬起来",
}

BG = "#23262b"
FG = "#e8e8e8"
DIM = "#9aa0a8"
ACC = "#2f6fd0"
OKC = "#2f9e44"
BAD = "#d94a3d"

FONT = ("Microsoft YaHei UI", 9)
FONT_B = ("Microsoft YaHei UI", 9, "bold")
FONT_MONO = ("Consolas", 9)


def _n(v, fmt="{:+.1f}") -> str:
    """一个数 → 一行里的字。**`None` 就写 `--`** —— 断点那几份读数里键是缺的比是空的多
    （量不出来的时候技能自己就不写那一项），照 `None` 格式会当场抛。"""
    return "--" if v is None else fmt.format(float(v))


def gate_lines(gate: dict) -> list[str]:
    """断点自己写的那份读数（`out/clear_step.json` 那几份）→ 中文几行。

    **只照它给的键念，缺哪个就不印那一行。** 这一层不重算、不补数 —— 断点上的判据是
    技能算的，面板照抄一遍就成了第二个真相。
    """
    rec = gate.get("rec") or {}
    kind = gate.get("kind")
    out: list[str] = []
    if kind == "微调":
        out.append(f"        第 {_n(rec.get('tries'), '{:.0f}')} 次微调"
                   f"   高度钉在 {_n(rec.get('z0_mm'))}mm")
        if rec.get("off_axis_mm") is not None:
            out.append(f"        离轴 {_n(rec['off_axis_mm'])}mm   沿轴 "
                       f"{_n(rec.get('along_axis_mm'))}mm   （容 ≤{_n(rec.get('tol_mm'), '{:.0f}')}mm）")
        band = rec.get("band_px")
        want = (f"落到 {band[0]:.0f}~{band[1]:.0f}px" if band
                else f"≥{_n(rec.get('want_gap'), '{:.0f}')}px")
        out.append(f"        缝 {_n(rec.get('gap'))}px（要 {want}）   轴偏 "
                   f"{_n(rec.get('axis'))}px（容 {_n(rec.get('tol_axis'), '{:.0f}')}px）")
        if rec.get("step_mm"):
            out.append(f"        技能想走 横 {_n(rec['step_mm'][0])} {_n(rec['step_mm'][1])}mm"
                       f"   到 {rec.get('tgt_mm')}")
    elif kind == "判过":
        out.append(f"        判过了（拦门的是 {rec.get('ruler') or '--'}）—— "
                   f"「允许」接着走：竖直下探 → 再修一次轴偏 → 才合爪")
        out.append(f"        离轴 {_n(rec.get('off_axis_mm'))}mm   沿轴 "
                   f"{_n(rec.get('along_axis_mm'))}mm   （容 ≤{_n(rec.get('tol_mm'), '{:.0f}')}mm）")
        out.append(f"        缝 {_n(rec.get('gap_px'))}px   轴偏 {_n(rec.get('axis_px'))}px")
    elif kind == "下探":
        out.append(f"        现在 z {_n(rec.get('z_now_mm'))}mm → 下到 "
                   f"{_n(rec.get('z_target_mm'))}mm（下 {_n(rec.get('drop_mm'), '{:.1f}')}mm，"
                   f"x/y 不动）")
        out.append(f"        方块中心 z {_n(rec.get('center_z_mm'))}mm + "
                   f"{_n(rec.get('final_margin_mm'), '{:.0f}')}mm")
    elif kind == "抬高度":
        out.append(f"        现在 z {_n(rec.get('z_now_mm'))}mm，比 "
                   f"{_n(rec.get('z_floor_mm'))}mm 低了 {_n(rec.get('drop_mm'), '{:.1f}')}mm"
                   f" —— 「允许」就竖直抬回去")
    return out


class Panel:
    """一块窗口。构造完调 `run()` —— 它进 `mainloop`，不会再回来。"""

    def __init__(self, rig, pic_h: int = PIC_H):
        self.rig = rig
        self.pic_h = int(pic_h)
        self.deg_step = 1.0
        self.mm_step = 5.0
        self.log: deque[str] = deque(maxlen=LOG_KEEP)
        self.running = True
        self._last_t = None
        self._last_frame_id = None
        self._pic_size = (0, 0)
        self._render_size = (0, 0)
        self._sized = False                   # 头一帧到了才定窗口尺寸，见 `_show`
        self._photo: ImageTk.PhotoImage | None = None
        self._last_snap: dict = {}
        # `rig` 的中文日志接到这块窗口上（原来只 print 到终端）。**在构造之后接**，
        # 装配那几行（串口、相机）还是直接打到终端 —— 那时候窗口还没起来。
        self.rig.set_log(self.log.append)

    # ---- 造窗口 ----

    def run(self) -> None:
        self.root = tk.Tk()
        self.root.title(TITLE)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        left = tk.Frame(self.root, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        # **画面这块必须是 Canvas，不能是 Label。** Label 的「想要多大」= 它当前那张图的
        # 大小，而那张图的大小又是照 Label 当时的分配尺寸算的 —— 这是个闭环：图一大，
        # Label 就想要更大，布局给它更大，下一帧按更大的尺寸重算，图又更大。窗口最大化
        # 之后这个环就转起来了，看着就是「画面自己在变大」。Canvas 的大小是构造时定死的
        # （下面那两个 `width` / `height`），往里塞多大的图都不改它的诉求，环断了。
        self.pic = tk.Canvas(left, bg="#181a1d", highlightthickness=0, bd=0,
                             width=1100, height=self.pic_h)
        self.pic.pack(fill="both", expand=True)
        self.pic.bind("<Configure>", self._on_resize)

        right = tk.Frame(self.root, bg=BG, padx=8, pady=6)
        right.pack(side="right", fill="y")
        self._build_skill(right)
        self._build_joints(right)
        self._build_misc(right)
        self._build_readout(right)

        self._bind_keys()
        self.root.after(50, self._loop)
        self.root.mainloop()

    def _sec(self, parent, title: str) -> tk.Frame:
        tk.Label(parent, text=title, bg=BG, fg=DIM, font=FONT_B,
                 anchor="w").pack(fill="x", pady=(8, 2))
        box = tk.Frame(parent, bg=BG)
        box.pack(fill="x")
        return box

    def _btn(self, parent, text, cmd, *, width=8, fg=FG, bg="#39404a") -> tk.Button:
        b = tk.Button(parent, text=text, command=cmd, width=width, font=FONT,
                      bg=bg, fg=fg, activebackground=ACC, activeforeground="#ffffff",
                      relief="raised", bd=1, highlightthickness=0)
        return b

    def _build_skill(self, parent) -> None:
        box = self._sec(parent, "技能（跑到断点上自己停）")
        self.run_btn = self._btn(box, "▶ 跑一趟 / F1",
                                 lambda: self.rig.submit(kind="pick_run"),
                                 width=26, fg="#ffffff", bg=ACC)
        self.run_btn.pack(fill="x", pady=1)
        self.gate_lbl = tk.Label(box, text="没在跑", bg=BG, fg=DIM, font=FONT,
                                 anchor="w", wraplength=360, justify="left")
        self.gate_lbl.pack(fill="x")

        box = self._sec(parent, "断点上：允许下一步 / 即时调整")
        self.gate_btns: dict[str, tk.Button] = {}
        specs = (("go", "允许，照算的走 / 空格", 26),
                 ("relook", "原地重量一眼 / F2", 26),
                 ("descend", "微调收工，直接下探 / F3", 26))
        for key, text, w in specs:
            b = self._btn(box, text, lambda k=key: self._gate(k), width=w)
            b.pack(fill="x", pady=1)
            self.gate_btns[key] = b
        r = tk.Frame(box, bg=BG)
        r.pack(fill="x", pady=1)
        tk.Label(r, text="改高度 mm（负=再往下）", bg=BG, fg=DIM, font=FONT).pack(
            side="left")
        self.dz_var = tk.StringVar(value="-3")
        tk.Entry(r, textvariable=self.dz_var, width=5, font=FONT_MONO, bg="#181a1d",
                 fg=FG, insertbackground=FG).pack(side="left", padx=3)
        self.gate_btns["dz"] = self._btn(r, "走", lambda: self._gate("dz"), width=3)
        self.gate_btns["dz"].pack(side="left")
        self.gate_btns["abort"] = self._btn(box, "放弃这一趟 / F4", lambda: self._gate("abort"),
                                            width=26, fg="#ffd0c8", bg="#7a2f28")
        self.gate_btns["abort"].pack(fill="x", pady=1)
        tk.Label(box, text="停在断点上时，手动那一整块（关节 / 夹爪 / TCP 点动）照样能用。",
                 bg=BG, fg=DIM, font=FONT, anchor="w", wraplength=360,
                 justify="left").pack(fill="x", pady=(4, 0))

    def _build_joints(self, parent) -> None:
        box = self._sec(parent, "手动：关节（度）")
        row = tk.Frame(box, bg=BG)
        row.pack(fill="x")
        tk.Label(row, text="步长", bg=BG, fg=DIM, font=FONT).pack(side="left")
        self.deg_var = tk.StringVar(value=str(self.deg_step))
        om = tk.OptionMenu(row, self.deg_var, *[str(v) for v in DEG_STEPS],
                           command=lambda v: setattr(self, "deg_step", float(v)))
        om.configure(font=FONT, bg="#39404a", fg=FG, highlightthickness=0, width=4)
        om.pack(side="left", padx=4)

        self.joint_lbl: list[tk.Label] = []
        for i, nm in enumerate(JOINTS_CN):
            r = tk.Frame(box, bg=BG)
            r.pack(fill="x", pady=1)
            lb = tk.Label(r, text=f"{nm}   --", bg=BG, fg=FG, font=FONT_MONO,
                          width=13, anchor="w")
            lb.pack(side="left")
            self.joint_lbl.append(lb)
            self._btn(r, "−", lambda j=i: self._jog(j, -1), width=3).pack(side="left")
            self._btn(r, "＋", lambda j=i: self._jog(j, +1), width=3).pack(side="left")

    def _build_misc(self, parent) -> None:
        box = self._sec(parent, "手动：夹爪 / 臂")
        r = tk.Frame(box, bg=BG)
        r.pack(fill="x")
        self._btn(r, "张开", lambda: self.rig.submit(kind="grip", to="open")).pack(
            side="left", padx=1)
        self._btn(r, "合上", lambda: self.rig.submit(kind="grip", to="close")).pack(
            side="left", padx=1)
        self._btn(r, "量行程", lambda: self.rig.submit(kind="grip_cal"),
                  width=7).pack(side="left", padx=1)

        r = tk.Frame(box, bg=BG)
        r.pack(fill="x", pady=1)
        self._btn(r, "回初始位", lambda: self.rig.submit(kind="home"),
                  width=7).pack(side="left", padx=1)
        self._btn(r, "力矩开", lambda: self.rig.submit(kind="torque"),
                  width=7).pack(side="left", padx=1)
        self._btn(r, "重量一眼", lambda: self.rig.submit(kind="see"),
                  width=8).pack(side="left", padx=1)
        self._btn(r, "停手", lambda: self.rig.submit(kind="stop"), fg="#ffd0c8",
                  bg="#7a2f28", width=5).pack(side="left", padx=1)

        box = self._sec(parent, "手动：TCP 点动（底座系）")
        r = tk.Frame(box, bg=BG)
        r.pack(fill="x")
        tk.Label(r, text="步长", bg=BG, fg=DIM, font=FONT).pack(side="left")
        self.mm_var = tk.StringVar(value=str(self.mm_step))
        om = tk.OptionMenu(r, self.mm_var, *[str(v) for v in MM_STEPS],
                           command=lambda v: setattr(self, "mm_step", float(v)))
        om.configure(font=FONT, bg="#39404a", fg=FG, highlightthickness=0, width=4)
        om.pack(side="left", padx=4)
        for ax in "XYZ":
            rr = tk.Frame(box, bg=BG)
            rr.pack(fill="x", pady=1)
            tk.Label(rr, text=f"{ax}", bg=BG, fg=FG, font=FONT_MONO,
                     width=3).pack(side="left")
            self._btn(rr, "−", lambda a=ax: self._nudge(a, -1), width=3).pack(
                side="left")
            self._btn(rr, "＋", lambda a=ax: self._nudge(a, +1), width=3).pack(
                side="left")

    def _build_readout(self, parent) -> None:
        box = self._sec(parent, "读数")
        self.readout = tk.Text(box, height=14, width=44, font=FONT_MONO, bg="#181a1d",
                               fg=FG, bd=0, highlightthickness=0, wrap="none")
        self.readout.pack(fill="x")
        box = self._sec(parent, "日志（中文都在这儿）")
        self.logbox = tk.Text(box, height=12, width=44, font=FONT, bg="#181a1d",
                              fg=DIM, bd=0, highlightthickness=0, wrap="char")
        self.logbox.pack(fill="x")
        self.hint = tk.Label(parent, text="", bg=BG, fg=DIM, font=FONT, anchor="w",
                             wraplength=360, justify="left")
        self.hint.pack(fill="x")

    # ---- 按钮 / 按键的去处。**只发命令，不做动作。** ----

    def _gate(self, what: str) -> None:
        """在断点上给技能一个答复。**只写文件，不动臂** —— 技能自己读。"""
        dz = None
        if what == "dz":
            try:
                dz = float(self.dz_var.get())
            except ValueError:
                self._sink("    改高度得给个数（毫米，负 = 再往下）")
                return
        self.rig.submit(kind="gate", what=what, dz=dz)

    def _jog(self, joint: int, sign: int) -> None:
        self.rig.submit(kind="jog", joint=joint, delta_deg=sign * self.deg_step)

    def _nudge(self, axis: str, sign: int) -> None:
        """TCP 走一小步。**发的是毫米** —— `rig._nudge` 自己除 1000（`rig.py:344`）。"""
        v = sign * self.mm_step
        d = {"X": (v, 0.0, 0.0), "Y": (0.0, v, 0.0), "Z": (0.0, 0.0, v)}[axis]
        self.rig.submit(kind="nudge", dx=d[0], dy=d[1], dz=d[2])

    def _hint(self, s: str) -> None:
        self.hint.configure(text=s)

    def _sink(self, s: str) -> None:
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {s}")

    def _quit(self) -> None:
        self.running = False
        self.root.destroy()

    def _bind_keys(self) -> None:
        r = self.root
        r.bind("<F1>", lambda e: self.rig.submit(kind="pick_run"))
        r.bind("<space>", lambda e: self._gate("go"))
        r.bind("<F2>", lambda e: self._gate("relook"))
        r.bind("<F3>", lambda e: self._gate("descend"))
        r.bind("<F4>", lambda e: self._gate("abort"))
        r.bind("<Escape>", lambda e: self._quit())
        r.bind("q", lambda e: self._quit())
        r.bind("h", lambda e: self.rig.submit(kind="home"))
        r.bind("o", lambda e: self.rig.submit(kind="grip", to="open"))
        r.bind("c", lambda e: self.rig.submit(kind="grip", to="close"))
        r.bind("m", lambda e: self.rig.submit(kind="grip_cal"))
        r.bind("t", lambda e: self.rig.submit(kind="torque"))
        r.bind("r", lambda e: self.rig.submit(kind="see"))
        r.bind("s", lambda e: self.rig.submit(kind="stop"))
        for i in range(6):
            r.bind(str(i + 1), lambda e, j=i: self._jog(j, +1))
            sym = "!@#$%^"[i]
            r.bind(sym, lambda e, j=i: self._jog(j, -1))

    # ---- 主循环 ----

    def _on_resize(self, ev) -> None:
        self._pic_size = (max(1, ev.width), max(1, ev.height))

    def _loop(self) -> None:
        if not self.running:
            return
        try:
            snap = self.rig.snap()
            self._last_snap = snap
            bgr = snap.get("bgr")
            # **`t` 是每一拍现打的时间戳**（`rig.py:499`），拿它当「这一帧换没换」的凭据。
            # 不用 `id(bgr)`：那个数组从快照里丢掉之后 id 会被下一帧复用，看起来像没换。
            key = (snap.get("t"), bool(snap.get("moving")))
            if bgr is not None and key != self._last_frame_id:
                self._last_frame_id = key
                self._show(bgr, snap)
            elif bgr is not None and self._pic_size != self._render_size:
                self._show(bgr, snap)          # 窗口被拉过，按新尺寸重画
            self._gates(snap)
            self._readout(snap)
            self._logs()
        except Exception as exc:                                    # noqa: BLE001
            self._sink(f"！界面刷新出错：{type(exc).__name__}: {exc}")
        self.root.after(int(UI_S * 1000), self._loop)

    def _show(self, bgr, snap: dict) -> None:
        """一帧 → 标好 → 缩到窗口大小 → 上去。**原帧不裁**，只等比缩。"""
        w, h = self._pic_size if self._pic_size[0] > 2 else (1100, self.pic_h)
        img = view.draw(bgr, snap)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if not self._sized:
            # 头一帧到了才知道画面多大。**这时候才定窗口初始尺寸**：照原帧的宽高比给
            # 左边留够，右边按控件实际要的宽度撑出去。定死一个 1600x900 的话，相机换个
            # 画幅就会两边都空一大块、或者右边按钮被挤掉。
            self._sized = True
            bh = min(980, max(360, self.pic_h))
            bw = int(pil.width * bh / max(1, pil.height))
            self.root.geometry(f"{min(1900, bw + 470)}x{bh + 90}")
        k = min(w / pil.width, h / pil.height)
        size = (max(1, int(pil.width * k)), max(1, int(pil.height * k)))
        if size != pil.size:
            pil = pil.resize(size, Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(pil)       # 必须留引用，否则被回收成白框
        self.pic.delete("all")
        self.pic.create_image(w // 2, h // 2, image=self._photo, anchor="center")
        self._render_size = self._pic_size

    def _gates(self, snap: dict) -> None:
        """把「技能那一头现在是什么状态」画到按钮上。

        **按钮的能按不能按是按技能那一头说的算的**（`skill_driver.ALLOWED`）：在「判过」
        那个断点上给技能一个它不认的 `descend`，它读不到 `go` 就当成「外面叫停」，
        整趟白跑。所以不能只按「有没有停在断点上」来开。
        """
        gate = snap.get("gate")
        busy = bool(snap.get("busy"))
        self.run_btn.configure(state="disabled" if busy else "normal",
                               bg=("#4a5058" if busy else ACC))
        if gate:
            self.gate_lbl.configure(
                text=f"⏸ 停在断点：{gate['kind']}\n{gate.get('note') or ''}",
                fg="#ffd479")
        elif busy:
            self.gate_lbl.configure(text="技能在跑……（到断点会自己停）", fg=DIM)
        else:
            self.gate_lbl.configure(text="没在跑", fg=DIM)
        allow = tuple(gate["allow"]) if gate else ()
        for key, b in self.gate_btns.items():
            ok = key in allow
            b.configure(state="normal" if ok else "disabled")

    def _readout(self, snap: dict) -> None:
        step = snap.get("step") or {}
        box = snap.get("box") or {}
        m = snap.get("m") or {}
        lines: list[str] = []

        if snap.get("moving") or snap.get("busy"):
            lines.append("【状态】动作进行中 —— 画面读数先冻着")
        c = box.get("center")
        c = None if c is None else np.asarray(c) * 1000.0
        if c is None:
            lines.append(f"【盒子】没认出来：{box.get('why') or '--'}")
        else:
            yaw = box.get("yaw_deg")
            lines.append(f"【盒子】x {c[0]:6.1f} y {c[1]:6.1f} z {c[2]:6.1f} mm"
                         f"   转 {0.0 if yaw is None else yaw:+5.1f}°"
                         f"   {box.get('size') or ''}")

        if m.get("ok"):
            lines.append(f"【缝/轴】gap {m['gap']:+6.0f}px   axis {m['axis']:+6.0f}px"
                         f"   容忍 {float(measure.PR.AXIS_TOL_PX):.0f}px")
        else:
            lines.append(f"【缝/轴】没量出来：{m.get('why') or '--'}")

        if step:
            ph = PHASE_CN.get(step.get("phase"), step.get("phase") or "--")
            lines.append(f"【画面建议】{ph}")
            if step.get("dxy_mm"):
                dx, dy = step["dxy_mm"]
                extra = ""
                if step.get("capped") and step.get("dxy_need_mm"):
                    n = step["dxy_need_mm"]
                    extra = f"（还差 {n[0]:+.0f},{n[1]:+.0f}）"
                lines.append(f"            横挪 {dx:+6.1f} {dy:+6.1f} mm {extra}")
            if step.get("dz_mm") is not None:
                lines.append(f"            竖直 {step['dz_mm']:+6.1f} mm"
                             f"   到底了 {'是' if step.get('at_depth') else '不是'}")

        a = snap.get("angles_deg")
        if a is not None:
            a = np.asarray(a)
            src = "实测" if snap.get("angles_src") == "measured" else "指令"
            lines.append("【关节】" + " ".join(f"{v:+6.1f}" for v in a[:6]) + f"  ({src})")
        t = snap.get("tcp")
        if t is not None:
            t = np.asarray(t) * 1000.0
            lines.append(f"【TCP】x {t[0]:6.1f} y {t[1]:6.1f} z {t[2]:6.1f} mm")
        g = snap.get("grip_deg")
        e, o = getattr(self.rig.arm, "empty_deg", None), getattr(
            self.rig.arm, "open_deg", None)
        v = snap.get("verdict") or "--"
        vt = {"held": "夹住了", "empty": "空抓", "UNCLEAR": "说不准"}.get(v, v)
        lines.append(f"【夹爪】{'--' if g is None else f'{g:6.1f}'}°  判定 {vt}"
                     f"   （空抓基准 {'--' if e is None else f'{e:.1f}'}°"
                     f" / 全张 {'--' if o is None else f'{o:.1f}'}°）")

        gate = snap.get("gate")
        if gate:
            lines.append("")
            lines.append(f"⏸ 停在断点【{gate['kind']}】—— 等你在右边给一个答复")
            lines.extend(gate_lines(gate))

        ls = snap.get("last_skill")
        if ls:
            name = "一趟"
            head = "✓" if ls.get("ok") else ("⌛" if ls.get("give_up") else "✗")
            lines.append(f"【技能】{head} 停在「{name}」  {ls.get('seconds', 0):.1f}s")
            lines.append(f"        {ls.get('why') or ''}")
        if not self.rig.jaw:
            lines.append(f"【爪轮廓】没有：{self.rig.jaw_why}")

        self._fill(self.readout, lines)

    def _logs(self) -> None:
        if not self.log:
            return
        cur = self.logbox.get("1.0", "end-1c").count("\n")
        if cur == len(self.log):
            return
        self._fill(self.logbox, list(self.log))
        self.logbox.see("end")

    @staticmethod
    def _fill(widget: tk.Text, lines: list[str]) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", "\n".join(lines))
        widget.configure(state="disabled")
