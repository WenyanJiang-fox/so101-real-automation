# -*- coding: utf-8 -*-
"""抓取那一轮的监督窗：实时看图 + 暂停 + 急停。

**为什么不是 `python -m panel`**：`ppo/capture.py` 自己占着 COM10，同一根串口开不了第二份。
所以这个窗口平时**不碰串口**，只做三件事：

  1. 把 `ppo.capture` 拉成子进程，输出同时进这个窗口和日志文件；
  2. **实时显示 `out/live.jpg`** —— 抓取子进程手里的那台相机按约 10 帧/秒重写这张图，
     上面画好了爪子轮廓、合爪轴线、方块红区轮廓（窗口里按比例缩放，完整画幅，不裁）。
     只有这张图不存在时才退回 `out/jaw_check.jpg`（每一眼一张的判据图，带 gap/axis 读数）；
  3. **暂停 / 继续**：直接把子进程挂起（`psutil.suspend`），舵机保持最后一条位置指令，
     臂停在原地；继续之后从那一步接着走。急停（按钮 / Esc）仍然是**先杀子进程**（它
     一死，最后一条位置指令就停在那儿，舵机自己保持住不下滑）→ 等串口放开 → **开爪 +
     回初始位置**，跟面板那个急停同一套动作（`panel/rig_motion.py::_estop_run` 的顺序）。

窗口关掉 = 这一轮结束（子进程一起杀掉），不留一条没人看着的臂在动。

用法：
    G:/anaconda/envs/sim2real/python.exe -m ppo.estop_panel -- --attempts 3 --start no_measure --offset-mm=-55,0

`--` 后面的参数原样交给 `ppo.capture`。**负数的参数要写成 `--offset-mm=-55,0`**（等号），
否则 argparse 会把它当成另一个选项。
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                              # noqa: BLE001
    pass

import tkinter as tk                                            # noqa: E402
from tkinter import scrolledtext                                # noqa: E402

import psutil                                                   # noqa: E402
from PIL import Image, ImageTk                                  # noqa: E402

SWEEP_MS = 50           # 子进程输出 / 实时画面多久刷一次窗口
LIVE_IMG = ROOT / "out" / "live.jpg"        # 抓取进程按 ~10 帧/秒重写（带爪轮廓/合爪轴/方块轮廓）
JAW_IMG = ROOT / "out" / "jaw_check.jpg"    # `_jaw_check` 每量一眼重写一次（判据图，实时画面不在时用）


class Supervisor:
    """子进程 + 一个急停按钮。**界面线程只发信号，动作全在后台线程。**"""

    def __init__(self, tkroot: tk.Tk, child_args: list[str],
                 port: str, fast: bool, log_path: Path) -> None:
        self.root = tkroot
        self.port = port
        self.fast = fast
        self.log_path = log_path
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.child: subprocess.Popen | None = None
        self.stop_lock = threading.Lock()
        self.estopped = False
        self.paused = False
        self._tkimg = None
        self._img_key: tuple | None = None

        self._build()
        self.child = subprocess.Popen(
            [sys.executable, "-u", "-m", "ppo.capture", *child_args],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        threading.Thread(target=self._reader, name="reader", daemon=True).start()
        self.say(f"抓取进程 PID {self.child.pid}；日志 {self.log_path}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(SWEEP_MS, self._pump)

    # ---- 界面 ----

    def _build(self) -> None:
        self.root.title("抓取监督 · 实时判据图 / 暂停 / 急停")
        self.root.geometry("1180x900")
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=(10, 4))
        self.status = tk.Label(top, text="运行中", anchor="w",
                               font=("Microsoft YaHei", 12, "bold"))
        self.status.pack(fill="x")
        tk.Label(top, text=f"串口 {self.port}（急停时才打开）；"
                           f"上图 = 腕相机实时画面约 10 帧/秒（整幅按比例缩放），"
                           f"已画：爪轮廓 / 合爪轴（品红）/ 方块轮廓（红）",
                 anchor="w", fg="#555").pack(fill="x")
        # ---- 实时判据图 ----
        # **用 Canvas，不用 Label**：Label 会按图片的尺寸重新要空间，图一换窗口就被撑大
        # 一圈（2026-10-20 实测：框子一直放大）。Canvas 的尺寸只由 pack 决定，图是画上去的
        # 一个 item，不影响布局。
        self.canvas = tk.Canvas(self.root, bg="#101010", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=4)
        self.img_item = self.canvas.create_image(0, 0, anchor="nw")
        self.img_info = None
        self.img_note = self.canvas.create_text(
            14, 14, anchor="nw", fill="#dddddd", font=("Microsoft YaHei", 12),
            text="等相机出第一帧…（实时画面；暂停时停在最后一帧）")
        self.canvas.bind("<Configure>", lambda _e: self._invalidate_image())
        # ---- 暂停 / 急停 ----
        row = tk.Frame(self.root)
        row.pack(fill="x", padx=10, pady=4)
        self.pause_btn = tk.Button(row, text="暂停（Space）—— 臂停在原地，不动也不松",
                                   command=self.toggle_pause, bg="#f9a825",
                                   activebackground="#f57f17",
                                   font=("Microsoft YaHei", 13, "bold"), height=2)
        self.pause_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.btn = tk.Button(row, text="🛑 急停：杀进程 + 开爪 + 回初始位置   Esc",
                             command=self.request_estop, bg="#c62828", fg="white",
                             activebackground="#8e0000", activeforeground="white",
                             font=("Microsoft YaHei", 13, "bold"), height=2)
        self.btn.pack(side="left", fill="x", expand=True)
        self.text = scrolledtext.ScrolledText(self.root, font=("Consolas", 9),
                                              state="disabled", wrap="none", height=12)
        self.text.pack(fill="x", expand=False, padx=10, pady=(0, 10))
        self.root.bind("<Escape>", lambda _e: self.request_estop())
        self.root.bind("<F12>", lambda _e: self.request_estop())
        self.root.bind("<space>", lambda _e: self.toggle_pause())

    def say(self, msg: str) -> None:
        self.lines.put(msg)

    # ---- 实时判据图 ----

    def _invalidate_image(self) -> None:
        """窗口尺寸变了：下一次 pump 按新尺寸重画一张（50ms 内）。"""
        self._img_key = None

    def _source_image(self):
        """优先读**实时画面**（抓取进程按 ~10 帧/秒重写）。没有它才退到判据图。

        **实时画面不按时间淘汰**：暂停（子进程被挂起）时它本来就不再更新，那时停在
        最后一帧才是对的 —— 画面上的时间戳会如实写出「多少秒没更新」。
        """
        for path, tag in ((LIVE_IMG, "LIVE"), (JAW_IMG, "eye")):
            try:
                return path, path.stat().st_mtime_ns, tag
            except OSError:
                continue
        return None

    def _refresh_image(self) -> None:
        """图一变就换一张。**整幅按比例缩到画布里，不裁。**"""
        src = self._source_image()
        if src is None:
            return
        path, mt, tag = src
        if (path, mt) == self._img_key:
            return
        try:
            im = Image.open(path)
            im.load()
        except Exception:                                       # noqa: BLE001
            return                      # 对方正在写这一个文件，下一拍再读
        self._img_key = (path, mt)
        w = max(self.canvas.winfo_width(), 320)
        h = max(self.canvas.winfo_height(), 200)
        im = im.copy()
        im.thumbnail((w, h), Image.LANCZOS)
        self._tkimg = ImageTk.PhotoImage(im)
        self.canvas.itemconfigure(self.img_item, image=self._tkimg)
        if self.img_note is not None:
            self.canvas.delete(self.img_note)
            self.img_note = None
        age = time.time() - mt / 1e9
        txt = f"{tag} " + time.strftime("%H:%M:%S", time.localtime(mt / 1e9))
        if tag == "LIVE" and age > 2.0:
            txt += f"  (not updating for {age:.0f}s)"
        if self.img_info is None:
            self.img_info = self.canvas.create_text(
                12, h - 12, anchor="sw", fill="#00ff00",
                font=("Consolas", 12), text=txt)
        else:
            self.canvas.itemconfigure(
                self.img_info, text=txt,
                fill="#00ff00" if tag == "LIVE" else "#ffd54f")
            self.canvas.coords(self.img_info, 12, h - 12)

    # ---- 暂停 / 继续 ----

    def toggle_pause(self) -> None:
        """挂起 / 恢复抓取子进程。**舵机是位置控制，挂起期间保持最后一条指令。**"""
        if (self.child is None or self.child.poll() is not None
                or self.estopped):
            return
        try:
            proc = psutil.Process(self.child.pid)
            if self.paused:
                proc.resume()
                self.paused = False
                self.pause_btn.configure(
                    text="暂停（Space）—— 臂停在原地，不动也不松",
                    bg="#f9a825", fg="black")
                self.status.configure(text="运行中（已继续）")
                self.say("▶ 继续（子进程 resume）")
            else:
                proc.suspend()
                self.paused = True
                self.pause_btn.configure(
                    text="继续（Space）—— 现在停着，可以对话",
                    bg="#43a047", fg="white")
                self.status.configure(text="⏸ 已暂停：臂停住、舵机保持，等你操作")
                self.say("⏸ 暂停（子进程 suspend）—— 臂不动了")
        except Exception as exc:                                # noqa: BLE001
            self.say(f"暂停/继续失败：{type(exc).__name__}: {exc}")

    def _pump(self) -> None:
        self._refresh_image()
        got = []
        while True:
            try:
                got.append(self.lines.get_nowait())
            except queue.Empty:
                break
        if got:
            self.text.configure(state="normal")
            for line in got:
                self.text.insert("end", line.rstrip() + "\n")
            self.text.see("end")
            self.text.configure(state="disabled")
        if self.child is not None and self.child.poll() is not None:
            if not self.estopped:
                # **进程结束了，急停按钮照样留着**：臂还站在那儿，按下去仍然走
                # 「开爪 + 回初始位置」那两下（这一路不依赖子进程）。
                self.status.configure(
                    text=f"抓取进程已结束（退出码 {self.child.returncode}）"
                         " —— 急停按钮仍可开爪 + 回初始位置")
                self.btn.configure(text="🛑 开爪 + 回初始位置   Esc")
                self.pause_btn.configure(state="disabled", bg="#9e9e9e")
        else:
            self.root.after(SWEEP_MS, self._pump)

    # ---- 子进程输出 ----

    def _reader(self) -> None:
        assert self.child is not None and self.child.stdout is not None
        with open(self.log_path, "a", encoding="utf-8") as fh:
            for line in self.child.stdout:
                fh.write(line)
                fh.flush()
                self.say(line)

    # ---- 急停 ----

    def request_estop(self) -> None:
        """**界面线程里只做这两件事**：置标志、按钮变灰。杀进程和动臂交给后台线程。"""
        if self.estopped:
            return
        self.estopped = True
        self.btn.configure(state="disabled", bg="#9e9e9e")
        self.status.configure(text="🛑 急停中…")
        threading.Thread(target=self._estop, name="estop", daemon=True).start()

    def _estop(self) -> None:
        self.say("🛑 急停：杀抓取进程")
        self._kill_child()
        self.say("🛑 等串口放开，然后开爪 + 回初始位置")
        try:
            self._recover_arm()
        except Exception as exc:                                # noqa: BLE001
            self.say(f"✗ 急停动作失败：{type(exc).__name__}: {exc}")
            self.status.configure(text="急停：机械臂动作失败（进程已停）")
            return
        self.status.configure(text="已急停：开爪、回初始位置")

    def _kill_child(self) -> None:
        with self.stop_lock:
            if self.child is None or self.child.poll() is not None:
                return
            if self.paused:
                # 挂起状态先恢复再杀。`kill()` 是 TerminateProcess，`finally` 本来也不会跑；
                # 这一步只是不让「杀一个挂着的进程」这种状态留在系统里。
                try:
                    psutil.Process(self.child.pid).resume()
                except Exception:                               # noqa: BLE001
                    pass
                self.paused = False
            self.child.kill()
            try:
                self.child.wait(timeout=5)
            except Exception:                                   # noqa: BLE001
                pass
        self.say("🛑 抓取进程已停")

    def _recover_arm(self) -> None:
        """开串口 → 开爪 → 回初始位置。**重试到串口放开为止**（子进程刚死，句柄可能还
        没释放）。最后一定关串口，舵机保持住最后一个位置。"""
        import numpy as np                                      # noqa: PLC0415

        from real.arm import RealArm                            # noqa: PLC0415
        from real.joint_move import goto_path, home_q           # noqa: PLC0415

        arm = None
        last: Exception | None = None
        for _ in range(30):
            try:
                arm = RealArm(port=self.port, fast=self.fast)
                break
            except Exception as exc:                            # noqa: BLE001
                last = exc
                time.sleep(0.3)
        if arm is None:
            raise RuntimeError(f"串口 {self.port} 打不开：{last}")
        try:
            arm.terminated = False
            q0 = arm._ticks_to_qpos(arm.read_ticks())
            q1 = np.array(q0, dtype=np.float64, copy=True)
            q1[5] = float(getattr(arm, "grip_open", arm.grip_hi))
            self.say(f"🛑 开爪：{np.degrees(q0[5]):.1f}° → {np.degrees(q1[5]):.1f}°")
            goto_path(arm, [q0, q1])
            q0 = arm._ticks_to_qpos(arm.read_ticks())
            self.say("🛑 回初始位置")
            goto_path(arm, [q0, home_q(arm)])
            self.say("🛑 已回初始位置")
        finally:
            try:
                arm.close()
            except Exception:                                   # noqa: BLE001
                pass

    # ---- 关窗 ----

    def on_close(self) -> None:
        if self.child is not None and self.child.poll() is None:
            self.say("窗口关闭 —— 抓取进程一起停")
            self._kill_child()
        self.root.destroy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ppo.estop_panel")
    ap.add_argument("child", nargs=argparse.REMAINDER,
                    help="`--` 后面的参数原样交给 ppo.capture")
    a = ap.parse_args(argv)
    child_args = list(a.child)
    if child_args and child_args[0] == "--":
        child_args = child_args[1:]
    if not child_args:
        child_args = ["--attempts", "3", "--start", "no_measure"]

    port = "COM10"
    fast = False
    try:
        from panel.rig_common import DEFAULT_CONFIG, load_cfg   # noqa: PLC0415
        rig = (load_cfg(DEFAULT_CONFIG) or {}).get("real") or {}
        port = str(rig.get("port", port))
        fast = bool(rig.get("fast", fast))
    except Exception:                                           # noqa: BLE001
        pass

    log_path = ROOT / "out" / time.strftime("estop_child_%Y%m%d_%H%M%S.log")
    tkroot = tk.Tk()
    Supervisor(tkroot, child_args, port=port, fast=fast, log_path=log_path)
    tkroot.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
