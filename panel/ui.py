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

import os
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

from panel import measure, targets, view                            # noqa: E402
from skills.pick_real import knobs                                  # noqa: E402

TITLE = "SO-101 真机面板"
DEG_STEPS = (0.2, 0.5, 1.0, 2.0, 5.0)
MM_STEPS = (1.0, 2.0, 5.0, 10.0)
# 界面刷新周期。画面那一路每帧要过一遍「画标注 + 转 RGB + 缩放」，比 cv2 那次贵，
# 所以按 12Hz 走；快照本身是 10Hz（`rig.TICK_S`），再快也是白转。
UI_S = 0.08
# 右边「读数 / 日志」那两块的重抄周期。画面按 `UI_S` 走，这两块没必要时时重写。
READOUT_S = 0.40
LOG_KEEP = 300
PIC_H = 620                       # 画面默认显示高度（等比缩放，原帧不裁）

JOINTS_CN = ("1 肩转", "2 肩抬", "3 肘弯", "4 腕俯", "5 腕转", "6 夹爪")
# 相位的说法只维护一张表：`measure.PHASES`（画面用英文那列，这里用中文那列）。

BG = "#23262b"
FG = "#e8e8e8"
DIM = "#9aa0a8"
ACC = "#2f6fd0"
OKC = "#2f9e44"
BAD = "#d94a3d"

FONT = ("Microsoft YaHei UI", 9)
FONT_B = ("Microsoft YaHei UI", 9, "bold")
FONT_MONO = ("Consolas", 9)

# ---- 临时计时（`PANEL_PROFILE=1` 才打）----
_PROF = bool(os.environ.get("PANEL_PROFILE"))
_prof_acc: dict[str, float] = {}
_prof_n = 0


def _prof_add(tag: str, dt: float) -> None:
    if not _PROF:
        return
    _prof_acc[tag] = _prof_acc.get(tag, 0.0) + dt


def _prof_flush() -> None:
    """每 20 帧打一次平均，看的是稳态成本。"""
    global _prof_n
    if not _PROF:
        return
    _prof_n += 1
    if _prof_n % 20:
        return
    parts = "  ".join(f"{k} {v / 20 * 1000:.0f}ms" for k, v in _prof_acc.items())
    print(f"[prof ui {_prof_n} 帧] 每帧平均：{parts}", flush=True)
    for k in _prof_acc:
        _prof_acc[k] = 0.0


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
        self._t_readout = 0.0
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
        self._build_estop(right)
        self._build_target(right)
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

    def _build_estop(self, parent) -> None:
        """**急停**：按一下 = 立刻松爪 + 回初始位置。

        这一下**不进命令队列**（`rig.request_estop()` 直接在界面线程置事件）：技能正在
        跑的时候队列里那一条要等它跑完才轮得到，急停不能等。真正的动作由硬件线程在
        手里这一小步做完之后执行，串口始终只有一个写手。
        """
        b = tk.Button(parent, text="🛑 急停：开爪 + 回初始位置   Esc",
                      command=self.rig.request_estop, font=FONT_B,
                      bg="#a4231c", fg="#ffffff", activebackground="#d03a2f",
                      activeforeground="#ffffff", relief="raised", bd=1,
                      highlightthickness=0, pady=6)
        b.pack(fill="x", pady=(0, 6))

    def _build_target(self, parent) -> None:
        """**目标那一栏**：人直接写一句话，认里面的颜色。

        颜色不是按钮选出来的，是因为桌上摆的东西是用人话说的（用户 2026-10-18：
        「我能输入目标描述」）。这一栏只换感知认的颜色，不动臂；换完以后「跑一趟」、
        「重量一眼」、自动连跑全都按新颜色走 —— 它们背后是同一个 `Perceptor`。
        """
        box = self._sec(parent, "目标：写一句话，认里面的颜色")
        # 预填当前颜色：这一栏常常只是「改成另一个颜色」，有个现成的字好改。
        cur = str(getattr(self.rig, "color", "") or "")
        self.target_var = tk.StringVar(value=cur)
        ent = tk.Entry(box, textvariable=self.target_var, font=FONT,
                       bg="#181a1d", fg=FG, insertbackground=FG)
        ent.pack(fill="x", pady=(0, 2))
        # **回车等于按「设定」** —— 手不用离开键盘。
        ent.bind("<Return>", lambda _e: self._set_target())
        r = tk.Frame(box, bg=BG)
        r.pack(fill="x")
        self._btn(r, "设定目标", self._set_target, width=10,
                  fg="#ffffff", bg=ACC).pack(side="left", padx=1)
        self.target_lbl = tk.Label(r, text=f"当前：{cur or '--'}", bg=BG, fg=DIM,
                                   font=FONT, anchor="w")
        self.target_lbl.pack(side="left", padx=4)
        tk.Label(box, text=f"认得：{targets.known_words()}；没标定过的颜色会提示怎么标。",
                 bg=BG, fg=DIM, font=FONT, anchor="w", wraplength=360,
                 justify="left").pack(fill="x")

    def _set_target(self) -> None:
        self.rig.submit(kind="color", text=self.target_var.get())

    def _build_skill(self, parent) -> None:
        box = self._sec(parent, "技能（跑到断点上自己停）")
        self.run_btn = self._btn(box, "▶ 跑一趟 / F1",
                                 lambda: self.rig.submit(kind="pick_run"),
                                 width=26, fg="#ffffff", bg=ACC)
        self.run_btn.pack(fill="x", pady=1)
        # **关节回放**（用户 2026-10-19）：把上一次抓成的那条轨迹原样走一遍；
        # 抬升完没夹住会自己回初始位置、用正常逻辑重跑这一趟。
        self.replay_btn = self._btn(box, "⟲ 回放上一趟",
                                    lambda: self.rig.submit(kind="replay_run"),
                                    width=26, fg="#d6e8ff", bg="#243a50")
        self.replay_btn.pack(fill="x", pady=1)
        # **自动连跑**（用户 2026-10-18）：闸口全关，跑完回初始位置再接下一样。
        # 再按一下（或 F5）就关；「停手」「放弃」也会把它关掉。
        self.auto_btn = self._btn(box, "自动连跑：关 / F5", self._toggle_auto,
                                  width=26, fg="#ffe9b0", bg="#5a4a1f")
        self.auto_btn.pack(fill="x", pady=1)
        # **从断点继续**（用户 2026-10-18）：停手 / 关掉自动之后，手动动过臂再按这个，
        # 从技能停下的那一段接着走。按哪个口由 `rig.resume_stage()` 定（对准 / 微调 /
        # 下探一截并合爪）；手里抓着东西时它给 None，按钮灰着。
        self.resume_btn = self._btn(box, "↻ 从断点继续", self._resume,
                                    width=26, fg="#d6ffe0", bg="#24503a")
        self.resume_btn.pack(fill="x", pady=1)
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
        """在断点上给技能一个答复。**只写文件，不动臂** —— 技能自己读。

        **技能不在断点上就不发**（用户 2026-10-18：「点一下指令就走下一步，而不是某些
        步骤要点两次」）。按钮本身会被 `_gates()` 关掉，但空格/F2~F4 那几条键绑定不看
        按钮状态 —— 臂正在走的时候按下去，批示会排到下一个断点，把那一关直接跳过去；
        而按得太晚又会被技能下一个断点 `unlink` 掉。两种都让人以为「按了没反应」。
        所以在这里当场拦：不在断点上，一行日志，不发。
        """
        if (self._last_snap or {}).get("gate") is None:
            self._sink("技能没停在断点上 —— 这一次没发（先按 F1 跑一趟）")
            return
        dz = None
        if what == "dz":
            try:
                dz = float(self.dz_var.get())
            except ValueError:
                self._sink("    改高度得给个数（毫米，负 = 再往下）")
                return
        self.rig.submit(kind="gate", what=what, dz=dz)

    def _toggle_auto(self) -> None:
        """翻「自动连跑」开关。**状态以 `rig` 那一头为准**（它会因为停手/放弃自己关）。"""
        on = not bool((self._last_snap or {}).get("auto"))
        self.rig.submit(kind="auto", on=on)

    def _resume(self) -> None:
        """从断点继续。**入口由 `rig.resume_stage()` 定**，这里只把命令递进去：
        `align` 先抬到安全高度再从头走，`fine` 续微调，`close` 直接下探一截并合爪。"""
        stage = self.rig.resume_stage()
        if stage is None:
            return
        # **当场灰掉**：下一拍刷新之前再按一次会再排一趟（用户明确要求不许新老并存）。
        self.resume_btn.configure(state="disabled", bg="#4a5058")
        self.rig.submit(kind="pick_run", start_at=stage)

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
        r.bind("<F5>", lambda e: self._toggle_auto())
        r.bind("<space>", lambda e: self._gate("go"))
        r.bind("<F2>", lambda e: self._gate("relook"))
        r.bind("<F3>", lambda e: self._gate("descend"))
        r.bind("<F4>", lambda e: self._gate("abort"))
        # **Esc 改成急停**（用户 2026-10-18：「没有急停按钮…一按下就能立刻开爪子回到
        # 初始位置」）。原来 Esc 是退出面板 —— 手忙的时候按错一次就是关窗，
        # 关窗还会和技能抢舵机。退出面板现在只认 `q`。
        r.bind("<Escape>", lambda e: self.rig.request_estop())
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
        _t_loop = time.perf_counter()
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
            _t = time.perf_counter()
            self._gates(snap)
            # **读数那块按自己的节拍照抄，不跟画面同频**（2026-10-18 时延）：它一次
            # 重写要 ~78ms（数字一直在抖，缓存挡不住），占着主线程时按键就排后面。
            # 画面 12Hz、读数 2.5Hz —— 人看数字不需要 12Hz。
            if time.perf_counter() - self._t_readout >= READOUT_S:
                self._t_readout = time.perf_counter()
                self._readout(snap)
                self._logs()
            _prof_add("读数区", time.perf_counter() - _t)
        except Exception as exc:                                    # noqa: BLE001
            self._sink(f"！界面刷新出错：{type(exc).__name__}: {exc}")
        # **按「开始时刻」排下一拍**，不是「干完再等 80ms」：以前这一拍要是花了 70ms，
        # 下一拍就变成 150ms 之后，画面越跟越慢（2026-10-18 时延）。
        wait = UI_S - (time.perf_counter() - _t_loop)
        self.root.after(max(1, int(wait * 1000)), self._loop)

    def _show(self, bgr, snap: dict) -> None:
        """一帧 → 标好 → 缩到窗口大小 → 上去。**原帧不裁**，只等比缩。"""
        w, h = self._pic_size if self._pic_size[0] > 2 else (1100, self.pic_h)
        _t0 = time.perf_counter()
        img = view.draw(bgr, snap)
        _t1 = time.perf_counter()
        _prof_add("标注", _t1 - _t0)
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
        _t2 = time.perf_counter()
        if size != pil.size:
            pil = pil.resize(size, Image.BILINEAR)
        _t3 = time.perf_counter()
        self._photo = ImageTk.PhotoImage(pil)       # 必须留引用，否则被回收成白框
        self.pic.delete("all")
        self.pic.create_image(w // 2, h // 2, image=self._photo, anchor="center")
        self._render_size = self._pic_size
        _prof_add("缩放", _t3 - _t2)
        _prof_add("上屏", time.perf_counter() - _t3)
        _prof_flush()

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
        auto = bool(snap.get("auto"))
        if auto:
            n = int(snap.get("round") or 0) + 1
            self.auto_btn.configure(text=f"自动连跑：开（第 {n} 轮）/ F5", bg=OKC)
        else:
            self.auto_btn.configure(text="自动连跑：关 / F5", bg="#5a4a1f")
        # **当前段**（`state.seg` 的原文）：停手之后按这个决定从哪儿接。
        seg = str(snap.get("stage") or "")
        seg_txt = f"\n当前段：{seg}" if seg else ""
        if gate:
            self.gate_lbl.configure(
                text=f"⏸ 停在断点：{gate['kind']}\n{gate.get('note') or ''}{seg_txt}",
                fg="#ffd479")
        elif busy:
            self.gate_lbl.configure(
                text=("自动模式：这一轮在跑，不停" if auto
                      else "技能在跑……（到断点会自己停）") + seg_txt, fg=DIM)
        else:
            self.gate_lbl.configure(
                text=("自动模式：等下一轮开跑" if auto else "没在跑") + seg_txt, fg=DIM)
        # **续跑按钮**：入口由 `rig.resume_stage()` 定，跑着的时候不给按。
        stage = self.rig.resume_stage()
        names = {"align": "对准", "fine": "微调", "close": "下探一截并合爪"}
        if stage is not None and not busy and not auto:
            self.resume_btn.configure(text=f"↻ 继续：{names.get(stage, stage)}",
                                      state="normal", bg="#24503a")
        else:
            why = self.rig.resume_blocked()
            self.resume_btn.configure(
                text=("↻ " + why) if (why and not busy) else "↻ 从断点继续",
                state="disabled", bg="#4a5058")
        allow = tuple(gate["allow"]) if gate else ()
        for key, b in self.gate_btns.items():
            ok = key in allow
            b.configure(state="normal" if ok else "disabled")

    def _readout(self, snap: dict) -> None:
        tgt = snap.get("target")
        if tgt:
            self.target_lbl.config(text=f"当前：{tgt}")
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
                         f"   容忍 {float(knobs.AXIS_TOL_PX):.0f}px")
        else:
            lines.append(f"【缝/轴】没量出来：{m.get('why') or '--'}")

        if step:
            ph = measure.PHASES.get(step.get("phase"), ("", step.get("phase") or "--"))[1]
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
        """把几行字灌进那块 Text。**内容没变就一个字都不动。**

        原来每 80ms 都 `delete` + `insert` 整块重来：实测这块要 ~78ms，主线程一半的
        时间花在重写没变的字上，按键就排在这后面（2026-10-18 时延）。
        """
        text = "\n".join(lines)
        if getattr(widget, "_last_text", None) == text:
            return
        widget._last_text = text
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")
