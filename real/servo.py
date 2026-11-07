"""真机视觉伺服：腕相机盯着棕色盒子，把爪子挪到它正上方、下探、合爪、抬起、换个地方放。

**这一路不用世界坐标。** 相机没标定（`real/cameras.py` 的 `mount()` 返回 None），
整条链只看画面像素：

    盒子的像素位置  -  抓取点的像素位置  =  误差  →  挪手臂把这个误差压到零

挪多少由**现场量的图像雅可比**给：抓取点往世界 +x 走 10mm，看盒子像素动几个 px，
同理 +y；两列凑成 2x2 的 J，解 `J·d = -误差` 得到这一步该走的位移。J 是量出来的，
不是标定出来的，相机装歪、内参不知道都不影响。

**控制点用 `grasp_point`，不用 `gripperframe`。** 官方 MJCF 里三个 site：
`baseframe`（底座）、`gripperframe`（nexus 的 TCP）、`grasp_point`。按几何量：
指垫在 gripper body 的 z∈[-0.06, 0] 那一段，`grasp_point` 落在 z=-0.04（两片指垫
正中），而 `gripperframe` 在 z=-0.0981 —— **在两指尖再往外 38mm 的空处**。
按 `gripperframe` 下探，指头会比你想的多沉 58mm。**抓取点是爪子上的量，
用 `GRASP_LOCAL` 从模型里量出来（工具系常数，实测三个姿态都一模一样）。**

**抓取点像素从哪来。** 优先级：`--grasp-uv` > `thresholds.json` 的 `grasp_pixel`
（点选标定过）> `jaw_cal` 自标。自标那个在这个画面上不可靠：两根指头是白塑料压在
浅灰桌面上，开合两帧灰度差不到 30，差分里最大的一块反而是**指头投在桌面上的影子**
（实测 13521px，最近点对落在影子边界 (281,38)，跟指头没关系）。所以标这条路改成
**人点一次**：`--teach` 开窗，左键点两个指尖相碰的那个点，按 q 写进 `thresholds.json`。
腕子朝向不变时这个像素是常数（纯平移时相机和爪子一起动），标一次能一直用。

**姿态必须是「手指朝下」。** 手指沿 **+工具x** 伸出（`grasp_point` 在 `gripperframe`
的 -工具x 方向 58.1mm）。所以下探 = 解 `+工具x 指向世界 -z`。**位置和朝向一起解**
（`solve_dir`，scipy 最小二乘，5 个约束对 5 个关节），别用 `arm._ik`：它姿态权重只有
0.05（nexus 的 `EE_ORIENTATION_WEIGHT`），位置一紧腕子就自己转，转完爪尖在画面里的
像素就变，上面那条「抓取点像素是常数」的假设当场失效。

窗口：**实时画面由本进程开**（相机同一个只能开一次）。别同时开 `view_cam.py`。

用法（PowerShell 一行）：
    G:/anaconda/envs/sim2real/python.exe G:/sim2real/so101-real-harness/real/servo.py --port COM10
    # 标抓取点像素（不动臂，只开窗点选，按 q 存盘）
    G:/anaconda/envs/sim2real/python.exe G:/sim2real/so101-real-harness/real/servo.py --port COM10 --teach
    G:/anaconda/envs/sim2real/python.exe G:/sim2real/so101-real-harness/real/servo.py --port COM10 --go
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.trajectory import build                               # noqa: E402
from real import perception as Pc                               # noqa: E402
from real.arm import JOINT_ORDER, RealArm                       # noqa: E402
from real.cameras import RealWristCamera                        # noqa: E402
from servo_bridge import open_controller                        # noqa: E402

OUT = _ROOT / "out"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 位置误差 1mm 一步、朝向误差 5mrad 一步算进代价。朝上松了腕子会自己转，位置松了
# 手臂就走偏。5mrad = 0.29°，一整趟下来朝向漂不到 1° 都还能干活。
SIG_POS = 1e-3
SIG_ORI = 5e-3
# 正则项：把解往**上一次的解**拽。解里绕手指轴的滚转是自由的（5 个关节对 5 个约束，
# 没有多余自由度，但零空间里还有一点），不拽的话两次解的滚转可能差很多，腕机就跟着
# 转，爪尖像素跟着漂。软约束，位置和朝向优先。
SIG_REG = 0.15

PREGRASP_Z = 0.12           # 转成手指朝下、且先降到这个高度再做其它事
DESCEND_STEP_FAR = 0.010    # 高处一步 10mm
DESCEND_STEP_NEAR = 0.004   # 贴到盒子附近换 4mm
DESCEND_SWITCH_Z = 0.060
GRASP_Z_FLOOR = 0.012       # 抓取点的硬下限，兜底用
LIFT_UP = 0.08
PLACE_Z = 0.16

# 判「夹住了没有」：合爪后读回的爪角比**空夹**时大多少。盒子 40mm 比空夹间隙宽，
# 爪子合不到底，两数差得很远，20° 留了很大余量。
HELD_MARGIN_DEG = 20.0

# 自标爪尖时，取扫过那片**最低的几条行**求中心。单取最底部一个像素太吃噪声：
# 那片的下边缘是斜的，一个像素的抖动就能把横坐标带偏几十个像素。
JAW_TIP_BAND = 7

# 量图像雅可比之前，先把盒子整个推进画面。**盒子只要有一边压在画面边界上，质心就被
# 那条边拖住**，量出来的灵敏度当场假掉（实测贴着右边缘时 J[0,0] = -22 px/米，等于横
# 着走一米盒子才动 0.022 像素）。把它推离边界留出 FRAME_MARGIN 像素再量。
FRAME_STEP = 0.015
FRAME_MARGIN = 12


# ---------------------------------------------------------------- 实时画面 ----
class LiveCam(threading.Thread):
    """后台线程不停抓帧，主线程随时取最新那张。窗口才能一直动。

    主线程每步只花 21ms 写舵机，自己再去抓一帧要 60ms 以上，画面就一顿一顿。
    抓帧单独一根线程，主线程零成本拿最新帧。取不到帧就沿用上一张，接着跑 ——
    停下的是观看，不是控制。
    """

    def __init__(self, cam):
        super().__init__(daemon=True)
        self.cam = cam
        self.lock = threading.Lock()
        self._bgr = None
        self._stop = False
        self._seq = 0

    def run(self) -> None:
        while not self._stop:
            try:
                rgb = self.cam.shot()
            except Exception:
                time.sleep(0.05)
                continue
            with self.lock:
                self._bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self._seq += 1

    def latest(self):
        with self.lock:
            return None if self._bgr is None else self._bgr.copy()

    def fresh(self, after: int = -1, timeout: float = 3.0):
        """等一张比 `after` 更新的帧。帧差要用，帧太旧会把结论带偏。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self._seq > after and self._bgr is not None:
                    return self._bgr.copy(), self._seq
            time.sleep(0.02)
        with self.lock:
            return (None if self._bgr is None else self._bgr.copy()), self._seq

    def stop(self) -> None:
        self._stop = True


class Window:
    """状态窗口。**屏幕上只能写 ASCII** —— cv2.putText 画不出汉字。

    `pick=True` 时鼠标左键点哪儿就记哪儿（1:1 显示，窗口坐标 = 画面坐标）。
    用来**人眼点一次抓取点像素**：`jaw_cal` 那套开合差分在这个画面上不成立
    （白爪子压在浅灰桌面上，两帧灰度差不到 30，差分里最大的一块是指头的影子），
    点一下是唯一稳的办法。点选是「教」不是「修」：像素在**腕子朝向不变**时是常数，
    所以标一次能一直用（见 `thresholds.json` 的 `grasp_pixel`）。
    """

    TITLE = "so101 real harness   [q]=quit   [left click]=pick"

    def __init__(self, record: bool = True, pick: bool = False) -> None:
        # **窗口必须整个塞得进屏幕。** 帧是 1920x1080，屏幕也正好是 1920x1080 ——
        # 用 WINDOW_AUTOSIZE 时窗口比屏幕还高一个标题栏，下沿被切在屏幕外，又没有
        # 滚动条，看上去就是「实时画面只有一部分」。改 WINDOW_NORMAL 缩到一半，
        # 整幅才看得全。鼠标回调给的坐标仍是**原图**坐标，点选标定不受影响。
        cv2.namedWindow(self.TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.TITLE, 960, 540)
        self.quit = False
        self._frames = [] if record else None
        self._n = 0
        self.pick = None                      # 最近一次点选，画面像素
        self.picking = bool(pick)
        if self.picking:
            cv2.setMouseCallback(self.TITLE, self._on_mouse)

    def _on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pick = (float(x), float(y))

    def tick(self, frame_bgr, phase: str, lines, marks=()):
        img = np.zeros((480, 640, 3), np.uint8) if frame_bgr is None else frame_bgr.copy()
        if self.picking and self.pick is not None:
            u, v = int(round(self.pick[0])), int(round(self.pick[1]))
            cv2.drawMarker(img, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
            cv2.putText(img, f"picked ({u},{v})", (u + 16, v + 8), FONT, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
        for uv, color, tag in marks:
            if uv is None:
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 26, 2)
            cv2.putText(img, tag, (u + 14, v - 8), FONT, 0.5, color, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 22 + 18 * (len(lines) + 1)), (0, 0, 0), -1)
        cv2.putText(img, phase, (10, 26), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        for i, ln in enumerate(lines):
            cv2.putText(img, ln, (10, 46 + i * 18), FONT, 0.45, (225, 225, 225), 1, cv2.LINE_AA)
        cv2.imshow(self.TITLE, img)
        # 录进视频的每 3 帧取 1。一帧 0.9MB，全程几千帧全留会吃掉几个 G。
        self._n += 1
        if self._frames is not None and self._n % 3 == 1:
            self._frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            self.quit = True

    def save_video(self, fps: float = 20.0) -> Path | None:
        if not self._frames:
            return None
        from real.video import write_mp4
        path = OUT / f"real_servo_{int(time.time())}.mp4"
        write_mp4(path, self._frames, fps)
        return path


# ------------------------------------------------------- 推不动就退（防障碍）----
# **「推不动」只有总线读得出来**：指令让它去 A，它没到 A，就是顶住了。不读就不知道
# 臂在哪，所以「即时」全靠把单次读的成本压下去。`sync_read_positions` 是每个 ID
# 单独收一包、每包固定 sleep 0.01s（mcp_robot/sources/servo_control_standalone.py:412），
# 所以**读 1 个舵机约 11ms、读 6 个约 66ms**。做法：只盯这一段真会动的那几个，轮着读。
#
# 判据两条，中任意一条就**当场断**，绝不把这段轨迹走完：
#   * 硬顶：指令走了它没跟上，差超过 STALL_DEG 度。
#   * 慢顶：同一个关节两次采样之间实测没动（< STALL_DEAD_TICKS 或不到指令行程的
#     STALL_IDLE_FRAC），而指令已经走了 STALL_MIN_CMD_TICKS。**这条是「即时」的那条** ——
#     不用等误差攒大，两个采样周期就出结论；硬顶那条得等 STALL_DEG / 速度 秒。
#
# **6 号夹爪永远排除。** CLOSE 那一步就是故意顶到堵转的（见下面 "closing until stall"），
# 把它卷进来会把夹取判废。
STALL_DEG = 5.0             # 硬顶阈值（度）
STALL_DEAD_TICKS = 2.0      # 慢顶：实测挪不到这么多 tick 就算「没动」（噪声地板）
STALL_IDLE_FRAC = 0.25      # 慢顶：实测行程不到指令行程这么多倍，也算「没动」
STALL_MIN_CMD_TICKS = 6.0   # 慢顶：指令先得走够这么多 tick 才有资格判
STALL_BACKOFF = 0.5         # 退回比例：从实测位置退回本段起点的百分之多少
STALL_MIN_MOVE_DEG = 1.0    # 这一段里位移不到这个度数的关节不盯（没动就无从判）
STALL_TICKS_PER_DEG = 4096.0 / 360.0


# ------------------------------------------------------------------ 运动 ----
class Mover:
    """关节空间匀速走。**每一步读一个正在动的关节**，推不动就当场停、退一点。

    读一个舵机约 11ms，30Hz 的一步是 33ms，所以每步读一个还留得住节奏；读六个
    （66ms）就把这一步吃掉了。所以是**轮着读**，不是每步读全部。
    """

    def __init__(self, arm, ui: Window, lc: LiveCam, hz=30.0, jvel=0.35, jacc=1.5,
                 settle=8, stall_deg=STALL_DEG, backoff=STALL_BACKOFF):
        self.arm, self.ui, self.lc = arm, ui, lc
        self.hz, self.jvel, self.jacc, self.settle = hz, jvel, jacc, settle
        self.stall_deg = stall_deg
        self.backoff = backoff

    def _read_subset(self, sids):
        """只读这几个舵机，并进 `arm.last_ticks`，再换算成整条 q。

        并进 `last_ticks` 是必须的：`_ticks_to_qpos` 要六个关节齐全才换得出整条 q。
        **不动 `_dirty`** —— 那是 arm 自己「自上次写之后真读过没有」的账，不归这里管。
        """
        arm = self.arm
        got = arm.ctrl.sync_read_positions(list(sids))
        for sid, v in got.items():
            if v is not None:
                arm.last_ticks[int(sid)] = int(v)
        return arm._ticks_to_qpos(arm.last_ticks)

    def _stream(self, qs, q_goal, dt, phase, lines, watch=()):
        """把 qs 发完，再在 q_goal 上压 settle 步。watch 非空就每步读一个。

        返回顶住的关节号，正常走完返回 None。
        """
        arm, ui = self.arm, self.ui
        t0 = time.time()
        prev = {}                                   # 关节号 -> 上次采样到的 (指令, 实测)
        for k, q in enumerate(qs):
            arm.write_ticks(arm._qpos_to_ticks(q))
            if watch:
                j = watch[k % len(watch)]
                qa = self._read_subset([j + 1])     # 关节序号 j <-> 舵机号 j+1
                err = abs(float(q[j] - qa[j]))
                hit = err > np.radians(self.stall_deg)
                if not hit and j in prev:
                    cmd_moved = abs(float(q[j] - prev[j][0])) * STALL_TICKS_PER_DEG
                    act_moved = abs(float(qa[j] - prev[j][1])) * STALL_TICKS_PER_DEG
                    hit = (cmd_moved > STALL_MIN_CMD_TICKS
                           and act_moved < max(STALL_DEAD_TICKS,
                                               STALL_IDLE_FRAC * cmd_moved))
                prev[j] = (float(q[j]), float(qa[j]))
                if hit:
                    return j
                # 读花了 11ms，把时基往后挪，别让后面几步挤成一串补时间。
                t0 = time.time() - (k + 1) * dt
            if k % 3 == 0:
                ui.tick(self.lc.latest(), phase, list(lines))
                if ui.quit:
                    raise KeyboardInterrupt
            nap = (k + 1) * dt - (time.time() - t0)
            if nap > 0:
                time.sleep(nap)
        for _ in range(self.settle):
            arm.write_ticks(arm._qpos_to_ticks(q_goal))
            ui.tick(self.lc.latest(), phase, list(lines))
            time.sleep(dt)
        return None

    def to(self, q_goal, phase: str, lines=(), guard: bool = True) -> str:
        """走到 q_goal。返回 "ok" 或 "blocked"。

        guard 开着时每步读一个正在动的关节；判到推不动就当场断，然后**沿原路往回退**
        一点 —— 来路上那几个位置刚才走通过，是已知安全的 —— 再返回 "blocked"。
        **绝不把这段轨迹走完**，也绝不继续往目标压。
        """
        arm = self.arm
        q_now = arm._ticks_to_qpos(arm.read_ticks())
        q_goal = np.asarray(q_goal, dtype=np.float64)
        if float(np.max(np.abs(q_goal - q_now))) < 1e-4:
            return "ok"
        traj = build([q_now, q_goal], vmax=self.jvel, amax=self.jacc, density=0.01)
        _ts, qs, _ = traj.sample(1.0 / self.hz)
        dt = 1.0 / self.hz

        span_deg = np.degrees(np.abs(q_goal - q_now))
        watch = [j for j in range(5) if span_deg[j] > STALL_MIN_MOVE_DEG]
        if not guard or self.stall_deg <= 0.0:
            watch = []

        jam = self._stream(qs, q_goal, dt, phase, lines, watch)
        if jam is None:
            return "ok"

        q_act = arm._ticks_to_qpos(arm.read_ticks())
        q_back = q_act + float(self.backoff) * (q_now - q_act)
        print(f"**推不动** {phase}：{JOINT_ORDER[jam]} 顶住了，这一步到此为止，"
              f"退回本段行程的 {self.backoff:.0%}。", flush=True)
        if self.backoff > 0.0 and float(np.max(np.abs(q_back - q_act))) > 1e-4:
            back = build([q_act, q_back], vmax=self.jvel * 0.5, amax=self.jacc,
                         density=0.01)
            _ts, qs, _ = back.sample(dt)
            self._stream(qs, q_back, dt, phase + " [退]", list(lines))
        return "blocked"


# ---------------------------------------------------------------- 位姿求解 ----
class Kine:
    """抓取点在哪、怎么把抓取点弄到指定位置。全部围着 `grasp_point` 转。"""

    def __init__(self, arm: RealArm):
        self.arm = arm
        sid = mujoco.mj_name2id(arm.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_point")
        if sid < 0:
            raise RuntimeError("MJCF 里没有 grasp_point 这个 site")
        self.site = sid
        q = arm._ticks_to_qpos(arm.read_ticks())
        R = self.rot(arm._fk(q)[1])
        # 抓取点在工具系里的偏移。两个 site 都是 gripper body 的子节点，这个偏移
        # **不随姿态变**（实测三个姿态都是 (-0.0581, 0.0002, 0.0079)，58.7mm）。
        self.local = R.T @ (arm.data.site_xpos[sid] - arm.data.site_xpos[arm.site])

    @staticmethod
    def rot(quat) -> np.ndarray:
        return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

    def at(self, q):
        """抓取点的世界位置 + 手指方向（= +工具x）。"""
        q = np.asarray(q, dtype=np.float64)
        pos, quat = self.arm._fk(q)
        R = self.rot(quat)
        return pos + R @ self.local, R[:, 0]

    def solve(self, q_prev, gp_target, x_des=np.array([0.0, 0.0, -1.0]), q_guide=None):
        """解一组关节角：抓取点落到 gp_target，手指（+工具x）指向 x_des。

        5 个约束对 5 个关节，位置和朝向一起解。解出来的滚转用 `q_guide` 那一组
        拽住（见 SIG_REG）。够不着时返回的残差会很大，调用方要查。
        """
        arm = self.arm
        q_prev = np.asarray(q_prev, dtype=np.float64)
        guide = q_prev[:5] if q_guide is None else np.asarray(q_guide, dtype=np.float64)[:5]
        target = np.asarray(gp_target, dtype=np.float64)
        want = np.asarray(x_des, dtype=np.float64)

        def resid(x):
            q = q_prev.copy()
            q[:5] = x
            gp, fx = self.at(q)
            return np.concatenate([(gp - target) / SIG_POS,
                                   (fx - want) / SIG_ORI,
                                   (x - guide) / SIG_REG])

        r = least_squares(resid, q_prev[:5], bounds=(arm.q_lo[:5], arm.q_hi[:5]),
                          method="trf", max_nfev=400)
        q = q_prev.copy()
        q[:5] = r.x
        gp, fx = self.at(q)
        return q, {"gp_mm": float(np.linalg.norm(gp - target) * 1000),
                   "ori_deg": float(np.degrees(np.arccos(np.clip(np.dot(fx, want), -1, 1)))),
                   "gp": gp}


# ---------------------------------------------------------------- 感知 ----
def box_blob(bgr, ranges, min_area):
    """画面里最大的那块棕色的中心像素。

    **不卡矩形度**（`perception.find` 那道 0.78 的门）。近处从上方看盒子，同时看得见
    顶面和侧面，轮廓本来就不是矩形 —— 实测同一只盒子从 0.84 掉到 0.52。卡它就等于
    「越靠近越认不出来」，而下探恰恰最需要认出来。面积仍然卡（阈值还是 thresholds.json
    里那一个），它才是「够不够大」的量。
    """
    mask = Pc.brown_mask(bgr, ranges)
    n, lab, st, ce = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, {"area": 0, "n": 0}
    i = 1 + int(np.argmax(st[1:, 4]))
    area = int(st[i, 4])
    info = {"area": area, "n": n - 1, "box": tuple(int(v) for v in st[i, :4])}
    if area < min_area:
        return None, info
    return (float(ce[i][0]), float(ce[i][1])), info


def load_grasp_pixel():
    """从 thresholds.json 取点选标定过的抓取点像素 + 标它时的关节角。

    没标过返回 (None, None)。**关节角要一起带回来**：腕子一转，这个像素就作废。
    """
    from grasp_check import load_thresholds
    rec = load_thresholds().get("grasp_pixel") or {}
    uv = rec.get("uv")
    if not uv:
        return None, None
    pose = rec.get("_pose_deg")
    return (np.array([float(uv[0]), float(uv[1])], dtype=np.float64),
            None if pose is None else np.array([float(v) for v in pose], dtype=np.float64))


def save_grasp_pixel(uv, q_deg) -> None:
    """把点选的抓取点像素写进 thresholds.json。"""
    from grasp_check import load_thresholds, save_thresholds
    cal = load_thresholds()
    cal["grasp_pixel"] = {
        "_note": "抓取点（两个指尖相碰处）在画面里的像素。**腕子朝向不变时它是常数** —— "
                 "纯平移时相机和爪子一起动，爪子在画面里的位置不动；腕子一转就作废，得重点。"
                 "标法：servo.py --teach，鼠标左键点两个指尖相碰的那个点，按 q 存盘。"
                 "**不要用 jaw_cal 的自标**：白爪子压在浅灰桌面上，开合两帧灰度差不到 30，"
                 "差分里最大的一块是爪子投在桌面上的影子，最近点对落在影子边界上。",
        "uv": [round(float(uv[0]), 1), round(float(uv[1]), 1)],
        "_calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "_pose_deg": [round(float(v), 1) for v in np.degrees(q_deg)],
    }
    save_thresholds(cal)


def empty_grip_deg(arm, mv, q_home) -> float:
    """空中合爪读回的角度 = 判「夹住没有」的空夹基准。读完把爪子放回原样。"""
    q_open = np.asarray(q_home, dtype=np.float64).copy()
    q = q_open.copy()
    q[5] = float(arm.grip_lo)
    mv.to(q, "JAW CAL  empty gripper (in air)")
    time.sleep(0.4)
    deg = float(np.degrees(arm.grip_q()))
    mv.to(q_open, "JAW CAL  open again")
    return deg


def jaw_cal(arm, mv, lc, q_home, seq0, ranges):
    """自标两个数：抓取点像素、空夹时爪子读回的角度。**只当兜底，优先用点选的。**

    张开拍一张、合上拍一张，两帧之差就是指头扫过的那一片。指头朝下，张开时两个指尖
    往上、往两边分，合上时它们收到**最低的那一点** —— 所以取那一片的**最低点**。

    （原来取「面积最大的两块 + 各自最近的一对点」，在这个画面上不成立：两根指头是白
    塑料压在浅灰桌面上，两帧灰度差不到 30，右边那根压根没被差出来；最大的一块反而是
    左边那根连同它投在桌面上的影子，最近点对落在两块碎片的接缝上，实测 (281,38)。）

    盒子那一片（棕色掩膜的位置）要从差分里挖掉：盒沿受曝光抖动，会在画面右边留一条
    竖直的亮线，把「最低点」往右下带。
    """
    lo, hi = float(arm.grip_lo), float(arm.grip_hi)
    q = q_home.copy()
    q[5] = hi
    mv.to(q, "JAW CAL 1/3  open gripper")
    A, sA = lc.fresh(after=seq0 + 3)
    q[5] = lo
    mv.to(q, "JAW CAL 2/3  close gripper (in air)")
    empty_deg = float(np.degrees(arm.grip_q()))
    B, sB = lc.fresh(after=sA + 3)
    q[5] = q_home[5]
    mv.to(q, "JAW CAL 3/3  open again")
    if A is None or B is None:
        raise RuntimeError(f"标爪尖：对照帧没拿到（open={A is not None} closed={B is not None}）")
    cv2.imwrite(str(OUT / "servo_jawcal_open.png"), A)
    cv2.imwrite(str(OUT / "servo_jawcal_closed.png"), B)

    d = cv2.absdiff(A, B).max(axis=2)
    m = (d > 30).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _uv_box, info_box = box_blob(B, ranges, 0)
    if _uv_box is not None:
        bx, by, bw, bh = info_box["box"]
        m[by:by + bh, bx:bx + bw] = 0
    cv2.imwrite(str(OUT / "servo_jawcal_diff.png"), 255 - d)
    n, lab, st, _ce = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        raise RuntimeError("标爪尖：开合两帧差分里一块都没有，看不出指头扫过哪")
    i = 1 + int(np.argmax(st[1:, 4]))
    ys, xs = np.where(lab == i)
    vmax = int(ys.max())
    sel = ys >= vmax - JAW_TIP_BAND
    grasp_uv = np.array([float(xs[sel].mean()), float(ys[sel].mean())])
    print(f"  抓取点像素 ({grasp_uv[0]:.0f}, {grasp_uv[1]:.0f})   "
          f"扫过的那片 {int(st[i, 4])}px（最低点 y={vmax}）   空夹爪角 {empty_deg:.1f}°",
          flush=True)
    return grasp_uv, empty_deg


def main() -> int:                                            # noqa: C901
    ap = argparse.ArgumentParser(description="真机视觉伺服：对齐 → 下探 → 夹 → 抬 → 挪 → 放")
    ap.add_argument("--port", default="COM10", help="从臂串口，先跑 identify_arms.py 认")
    ap.add_argument("--go", action="store_true", help="真的动。不给就只开窗看画面")
    ap.add_argument("--home", action="store_true",
                    help="开始前先回 real_home.RAW_HOME（手指朝下、相机正对桌面）那个点位")
    ap.add_argument("--index", type=int, default=1, help="相机序号")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--jvel", type=float, default=0.35, help="关节速度上限 弧度/秒")
    ap.add_argument("--jacc", type=float, default=1.5)
    ap.add_argument("--stall-deg", type=float, default=STALL_DEG,
                    help="推不动判据：指令和实测的差超过这么多度就当场停+退。0=关掉")
    ap.add_argument("--backoff", type=float, default=STALL_BACKOFF,
                    help="推不动之后退回本段行程的比例（0=原地不动）")
    ap.add_argument("--gain", type=float, default=0.8, help="图像误差每步吃掉多少")
    ap.add_argument("--step-max", type=float, default=0.015, help="单步位移上限（米）")
    ap.add_argument("--tol-px", type=float, default=10.0, help="对齐到多少像素算够")
    ap.add_argument("--budget", type=int, default=12, help="每个对齐阶段最多迭代几次")
    ap.add_argument("--probe", type=float, default=0.010, help="量雅可比时探多远（米）")
    ap.add_argument("--seed", type=int, default=None, help="随机落点，想复现就给定它")
    ap.add_argument("--grasp-uv", default=None,
                    help="直接给抓取点像素 \"u,v\"，跳过自标（优先级最高）")
    ap.add_argument("--teach", action="store_true",
                    help="只开窗点选抓取点像素：左键点两个指尖相碰处，按 q 存盘退出")
    ap.add_argument("--upto", default="open",
                    choices=("reorient", "jaw", "align", "descend", "close", "lift", "place", "open"),
                    help="走到哪一步为止。第一次调姿态用 reorient，它完了就停")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = random.Random(args.seed)
    ranges, _fill, min_area = Pc.load_cal()
    log: list[str] = []

    def say(s: str) -> None:
        print(s, flush=True)
        log.append(s)

    ctrl = open_controller(args.port)
    arm = RealArm(ctrl=ctrl, speed=0)
    cam = RealWristCamera(index=args.index)
    lc = LiveCam(cam)
    lc.start()
    ui = Window(pick=args.teach)
    mv = Mover(arm, ui, lc, hz=args.hz, jvel=args.jvel, jacc=args.jacc,
               stall_deg=args.stall_deg, backoff=args.backoff)
    kin = Kine(arm)

    def status(phase, lines, marks=()):
        ui.tick(lc.latest(), phase, lines, marks)
        if ui.quit:
            raise KeyboardInterrupt

    def look():
        bgr = lc.latest()
        uv, info = box_blob(bgr, ranges, min_area) if bgr is not None else (None, {})
        return bgr, uv, info

    def stop_here(name: str) -> None:
        """`--upto` 到了就停在这。臂留在当前姿态，力矩不松。"""
        if args.upto == name:
            say(f"--upto {name}：到这停。臂留在当前姿态，力矩不松。")
            raise SystemExit(0)

    try:
        t0 = time.time()
        while lc.latest() is None and time.time() - t0 < 8:
            status("WAITING FOR CAMERA", [f"camera {args.index}"])
        if lc.latest() is None:
            raise RuntimeError("相机开不出画面")

        if args.home:
            from real_home import RAW_HOME                      # noqa: PLC0415
            q_pose = arm._ticks_to_qpos(RAW_HOME)
            say(f"先回初始点位（手指朝下、相机正对桌面）")
            mv.to(q_pose, "HOME  go to initial pose", ["raw ticks from real_home.RAW_HOME"])
            time.sleep(0.4)
            say(f"     到了 关节(度) {np.round(np.degrees(arm._ticks_to_qpos(arm.read_ticks())), 1)}")

        q_now = arm._ticks_to_qpos(arm.read_ticks())
        gp0, fx0 = kin.at(q_now)
        say(f"起点 抓取点 {np.round(gp0, 4)}  手指朝 {np.round(fx0, 3)}"
            f"（离正下 {np.degrees(np.arccos(np.clip(-fx0[2], -1, 1))):.0f}°）")
        say(f"      TCP {np.round(arm.tcp(), 4)}   关节(度) {np.round(np.degrees(q_now), 1)}")
        say(f"抓取点在 TCP 外 {np.linalg.norm(kin.local) * 1000:.1f}mm"
            f"（工具系 {np.round(kin.local, 4)}）")

        if args.teach:
            say("点选抓取点：鼠标左键点**两个指尖相碰的那个点**，按 q 存盘退出。")
            say(f"     现在关节(度) {np.round(np.degrees(q_now), 1)}（这个姿态会被一起记下）")
            # 先把自标那套跑一遍，把结果**预填**成十字。它只开合爪子，不动臂。
            # 对就按 q 收；不对就点一下盖掉。
            try:
                auto_uv, auto_empty = jaw_cal(arm, mv, lc, q_now, lc._seq, ranges)
                ui.pick = (float(auto_uv[0]), float(auto_uv[1]))
                say(f"     预填了自标的结果 ({auto_uv[0]:.0f}, {auto_uv[1]:.0f})"
                    f"（空夹爪角 {auto_empty:.1f}°）—— 对就按 q，不对就点一下")
            except Exception as exc:                            # noqa: BLE001
                say(f"     自标没跑出来（{type(exc).__name__}: {exc}），直接点吧")
            last = None
            while not ui.quit:
                lines = ["LEFT CLICK on the point where the two fingertips meet",
                         "then press q to save & quit"]
                if ui.pick is not None:
                    lines.append(f"picked ({ui.pick[0]:.0f}, {ui.pick[1]:.0f})")
                    if ui.pick != last:
                        last = ui.pick
                        say(f"     记下 ({ui.pick[0]:.0f}, {ui.pick[1]:.0f})")
                status("TEACH  grasp pixel", lines)
                time.sleep(1.0 / 30)
            if ui.pick is None:
                say("一次都没点，什么都没存。")
                return 0
            save_grasp_pixel(np.array(ui.pick, dtype=np.float64), q_now)
            say(f"存进 thresholds.json 的 grasp_pixel：{ui.pick[0]:.0f}, {ui.pick[1]:.0f}")
            return 0

        if not args.go:
            say("没给 --go：只开窗看画面，一个字节都不写舵机。按 q 退出。")
            t0 = time.time()
            while time.time() - t0 < 900 and not ui.quit:
                bgr, uv, info = look()
                ui.tick(bgr, "DRY RUN  (no --go, arm will not move)",
                        [f"box area {info.get('area', 0)}px  n={info.get('n', 0)}",
                         f"uv {None if uv is None else tuple(int(v) for v in uv)}",
                         "add --go to run for real"],
                        marks=[(uv, (0, 255, 0), "box")])
                time.sleep(0.03)
            return 0

        # ---------- 1. 转成手指朝下，并把抓取点降到 PREGRASP_Z ----------
        q = arm._ticks_to_qpos(arm.read_ticks())
        tgt = gp0.copy()
        tgt[2] = PREGRASP_Z
        q_rot, diag = kin.solve(q, tgt, q_guide=q)
        say(f"转手指朝下：抓取点到 {np.round(diag['gp'], 4)}"
            f"（目标 {np.round(tgt, 4)}，差 {diag['gp_mm']:.1f}mm）"
            f"  手指离正下 {diag['ori_deg']:.2f}°")
        if diag["gp_mm"] > 8 or diag["ori_deg"] > 8:
            raise RuntimeError(f"转不成手指朝下（差 {diag['gp_mm']:.0f}mm / "
                               f"{diag['ori_deg']:.1f}°），不硬来")
        say(f"目标关节(度) {np.round(np.degrees(q_rot), 1)}")
        say(f"目标原始值   {arm._qpos_to_ticks(q_rot)}")
        mv.to(q_rot, "REORIENT  fingers down", [f"-> grasp z {PREGRASP_Z * 1000:.0f}mm"])
        time.sleep(0.4)
        q_land = arm._ticks_to_qpos(arm.read_ticks())
        gp_land, fx_land = kin.at(q_land)
        say(f"走完 关节(度) {np.round(np.degrees(q_land), 1)}")
        say(f"实读原始值   {arm.read_ticks()}")
        say(f"落点 抓取点 {np.round(gp_land, 4)}（目标 {np.round(tgt, 4)}，"
            f"差 {np.linalg.norm(gp_land - tgt) * 1000:.1f}mm）  "
            f"手指离正下 {np.degrees(np.arccos(np.clip(-fx_land[2], -1, 1))):.2f}°")
        if args.upto == "reorient":
            ui.tick(lc.latest(), "HOLD  fingers down",
                    ["check the live view: camera should face the table",
                     "then update real_home.RAW_HOME with the raw ticks above"])
            time.sleep(3.0)
            cv2.imwrite(str(OUT / "servo_reorient.png"), lc.latest())
            stop_here("reorient")

        # ---------- 2. 抓取点像素 ----------
        # 优先级：--grasp-uv > thresholds.json 里点选过的 grasp_pixel > 自标 jaw_cal。
        # 自标是兜底：它在这个画面上不可靠（见 jaw_cal 的 docstring）。
        q_now = arm._ticks_to_qpos(arm.read_ticks())
        grasp_uv, src, empty_deg = None, "", None
        if args.grasp_uv:
            grasp_uv = np.array([float(v) for v in args.grasp_uv.split(",")], dtype=np.float64)
            src = "--grasp-uv"
        else:
            gp_cal, gp_pose = load_grasp_pixel()
            if gp_cal is not None:
                grasp_uv, src = gp_cal, "thresholds.json 的 grasp_pixel（点选）"
                if gp_pose is not None:
                    # 只看**手臂**那 5 个关节。6 号是夹爪，它开合不改变爪尖在画面里的位置。
                    dq = float(np.max(np.abs(np.degrees(q_now) - gp_pose)[:5]))
                    say(f"     点选时的手臂角跟现在差最多 {dq:.1f}°"
                        f"（腕子转了超过几度这个像素就不准）")
        if grasp_uv is None:
            grasp_uv, empty_deg = jaw_cal(arm, mv, lc, q_now, lc._seq, ranges)
            src = "自标 jaw_cal（兜底，不如点选准）"
        if empty_deg is None:
            empty_deg = empty_grip_deg(arm, mv, q_now)
        q_ref = arm._ticks_to_qpos(arm.read_ticks())
        say(f"抓取点像素 ({grasp_uv[0]:.0f}, {grasp_uv[1]:.0f})   来源 {src}"
            f"   空夹爪角 {empty_deg:.1f}°")
        stop_here("jaw")

        # ---------- 3 之前：先把爪子张开 ----------
        # **下探必须以张开姿态下去。** `real_home` 记的那个姿态里 6 号舵机原始值是
        # 1450，按 lerobot 标定（`little_little_pig.json` 的 range_min=1425 /
        # range_max=2944）算，它只比全闭大 1.8°，其实就是闭着的。`jaw_cal` 收尾又把
        # 爪子放回「原来那样」——「原来那样」也是闭的。合着爪往下扎只会把盒子推走，
        # 所以这里显式张开，不指望上游。
        q_open = arm._ticks_to_qpos(arm.read_ticks())
        q_open[5] = float(getattr(arm, "grip_open", arm.grip_hi))
        mv.to(q_open, "OPEN  gripper before descending", ["wide open, then align / descend"])
        q_ref = arm._ticks_to_qpos(arm.read_ticks())

        # **对齐和上一步/下一步的基准是「指令位置」，不是「量到的位置」。**
        # 舵机每走一趟，量到的抓取点都比指令低约 5mm（重力静差，实测 reorient：
        # 目标 z=0.12、量到 0.1153）。如果拿量到的位置当下一步的起点，这 5mm 就一步
        # 一点地攒起来 —— 实测对齐 12 步把抓取点从 z=0.12 拖到 z=0.051，本来它该一直
        # 待在上方（对齐阶段 z 根本不动）。攒的是指令，静差就固定成一个常数偏移，
        # 不影响 z 平面。
        gp_cmd = tgt.copy()

        def move_gp(dxyz, phase, lines=()):
            """把抓取点从**上一条指令的位置**挪 dxyz，手指保持朝下。"""
            nonlocal gp_cmd
            gp_cmd = gp_cmd + np.asarray(dxyz, dtype=np.float64)
            q_cur = arm._ticks_to_qpos(arm.read_ticks())
            q_new, diag = kin.solve(q_cur, gp_cmd, q_guide=q_ref)
            mv.to(q_new, phase, lines)
            return diag

        def align(phase, budget=None, tol=None, step_max=None):
            """把盒子像素怼到抓取点像素上。J 是 2x2（px/米）。"""
            budget = budget or args.budget
            tol = args.tol_px if tol is None else tol
            step_max = args.step_max if step_max is None else step_max
            for it in range(budget):
                bgr, uv, info = look()
                status(f"{phase}  {it + 1}/{budget}",
                       [f"box {info.get('area', 0)}px  n={info.get('n', 0)}",
                        f"grasp pixel ({grasp_uv[0]:.0f},{grasp_uv[1]:.0f})"],
                       marks=[(grasp_uv, (255, 0, 255), "grasp"),
                              (uv, (0, 255, 0), "box")])
                if uv is None:
                    return None, f"盒子看不见（{info}）"
                err = np.array(uv) - grasp_uv
                if np.linalg.norm(err) < tol:
                    return uv, f"到位，差 {np.linalg.norm(err):.0f}px"
                d = -args.gain * np.linalg.solve(J, err)
                n = float(np.linalg.norm(d))
                if n > step_max:
                    d = d * (step_max / n)
                say(f"    {phase} 误差 {np.round(err, 0)} px → 走 {np.round(d * 1000, 1)} mm")
                diag = move_gp([d[0], d[1], 0.0], f"{phase} moving")
                if diag["gp_mm"] > 5:
                    return None, f"逆解到不了（差 {diag['gp_mm']:.1f}mm）"
            return uv, "步数用光"

        # ---------- 2.5 先把盒子整个推进画面 ----------
        # 量雅可比的前提是盒子**完整**在画面里。初始点位上盒子贴着画面右边缘（实测
        # bbox 右边 = 640 = 图宽），质心被那条边拖住：量出来的横向灵敏度 J[0,0] =
        # -22 px/米，真值至少高一个量级。用这个 J 对齐，每一步都被 step_max 顶死成
        # 同一个 15mm，走 10 步撞出工作空间（实测报「逆解到不了（差 10.5mm）」）。
        #
        # **方向靠相位相关量，不靠盒子质心。** 盒子被切掉大半时质心**两个轴都是废的**：
        # 实测它停在画面**下**边缘、只露 166x44 一条，按横坐标猜方向反而把剩下那条也推
        # 出去了。相位相关量的是**整幅画面**的平移 —— 相机一动，桌子、海报、盒子一起移，
        # 那个位移就是手臂位移在画面上的投影，跟盒子看不看得全没关系。
        # 符号约定和下面第 3 步一样：`phaseCorrelate(之前, 之后)` 就是画面内容移了多少。
        bgr0, _, _ = look()
        fh, fw = (bgr0.shape[:2] if bgr0 is not None else (480, 640))
        center = np.array([fw / 2.0, fh / 2.0])

        def clear_of_edges(info):
            """盒子四边都离画面边界有 FRAME_MARGIN 以上。"""
            bx, by, bw, bh = info["box"]
            return (bx > FRAME_MARGIN and by > FRAME_MARGIN
                    and bx + bw < fw - FRAME_MARGIN and by + bh < fh - FRAME_MARGIN)

        def gray():
            fr = look()[0]
            return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        def coarse_J():
            """整幅画面的平移量除以探多远。盒子被切也照样能量，所以它比质心那条路稳。"""
            g0 = gray()
            cols = []
            for ax, axn in ((0, "+x"), (1, "+y")):
                d = np.zeros(3)
                d[ax] = FRAME_STEP
                move_gp(d, f"FRAME  probe {axn}")
                time.sleep(0.5)
                g1 = gray()
                move_gp(-d, "FRAME  probe back")
                time.sleep(0.5)
                sh, resp = cv2.phaseCorrelate(g0, g1)
                say(f"    探 {axn}：整幅画面移 ({sh[0]:+.1f}, {sh[1]:+.1f}) px，"
                    f"响应 {resp:.2f}")
                cols.append(np.array(sh, dtype=np.float64) / FRAME_STEP)
            return np.array(cols).T

        _, uv_b, info_b = look()
        if uv_b is None:
            raise RuntimeError(f"画面里认不出棕色盒子：{info_b}")
        if not clear_of_edges(info_b):
            say(f"盒子压着画面边（bbox {info_b['box']}，画面 {fw}x{fh}），先把它挪进去")
            Jc = coarse_J()
            if abs(float(np.linalg.det(Jc))) < 1e4:
                raise RuntimeError(f"粗雅可比太差，定不了方向：{np.round(Jc, 0).tolist()} px/米")
            say(f"    粗雅可比 {np.round(Jc, 0).tolist()} px/米")
            for k in range(10):
                _, uv, info = look()
                if uv is None:
                    raise RuntimeError(f"挪盒子：第 {k + 1} 步认不出盒子了 {info}")
                if clear_of_edges(info):
                    break
                step = 0.8 * np.linalg.solve(Jc, center - np.array(uv))
                n = float(np.linalg.norm(step))
                if n > FRAME_STEP:
                    step = step * (FRAME_STEP / n)
                diag = move_gp([step[0], step[1], 0.0], f"FRAME  {k + 1} 把盒子挪进画面")
                if diag["gp_mm"] > 5:
                    raise RuntimeError(f"挪盒子：逆解到不了（差 {diag['gp_mm']:.1f}mm）")
                time.sleep(0.4)
            else:
                raise RuntimeError("挪了 10 步盒子还压着画面边，量不了雅可比")
        _, uv, info = look()
        say(f"盒子在画面里：质心 ({uv[0]:.0f},{uv[1]:.0f})  "
            f"bbox {info['box']}  面积 {info['area']}px")

        # ---------- 3. 量图像雅可比 ----------
        bgr, uv0, info0 = look()
        if uv0 is None:
            status("ALIGN  no box", [str(info0)])
            raise RuntimeError(f"画面里认不出棕色盒子：{info0}")
        say(f"对齐前 盒子像素 ({uv0[0]:.0f},{uv0[1]:.0f})  面积 {info0['area']}px")
        cols = []
        for ax, axn in ((0, "+x"), (1, "+y")):
            d = np.zeros(3)
            d[ax] = args.probe
            move_gp(d, f"JACOBIAN probe {axn}")
            time.sleep(0.4)
            _, uv1, _ = look()
            if uv1 is None:
                raise RuntimeError(f"量雅可比时（{axn}）盒子看不见了")
            cols.append((np.array(uv1) - np.array(uv0)) / args.probe)
            move_gp(-d, "JACOBIAN back")
            time.sleep(0.4)
        J = np.array(cols).T
        det = float(np.linalg.det(J))
        say(f"图像雅可比 J = {np.round(J, 0)} px/米   |det|^0.5 = {abs(det) ** 0.5:.0f} px/米")
        if abs(det) < 2e6:
            say("  ⚠ 行列式偏小：两个方向在画面里几乎同向，横着收敛会很慢")

        # ---------- 4. 对齐 ----------
        uv, msg = align("ALIGN")
        say(f"对齐：{msg}")
        if uv is None:
            raise RuntimeError(msg)
        say(f"对齐后 抓取点 {np.round(kin.at(arm._ticks_to_qpos(arm.read_ticks()))[0], 4)}"
            f"  残余 {np.linalg.norm(np.array(uv) - grasp_uv):.0f}px")
        time.sleep(0.5)
        stop_here("align")

        # ---------- 5. 下探 ----------
        # 停下来的判据只有一个：**盒子在画面里的面积掉头**。爪子贴上去会挡住盒子上沿，
        # 面积先长后落。它跟「离桌面多少毫米」无关，所以桌子多高、盒子多高都不影响。
        # 硬下限（GRASP_Z_FLOOR）只当兜底，防它一路扎下去。
        peak = info0["area"]
        for k in range(200):
            gz = float(kin.at(arm._ticks_to_qpos(arm.read_ticks()))[0][2])
            step = DESCEND_STEP_FAR if gz > DESCEND_SWITCH_Z else DESCEND_STEP_NEAR
            if gz - step < GRASP_Z_FLOOR:
                say(f"下探：抓取点到下限 z={(gz - step) * 1000:.0f}mm，停")
                break
            diag = move_gp([0.0, 0.0, -step], f"DESCEND {k + 1}  grasp z={(gz - step) * 1000:.0f}mm")
            if diag["gp_mm"] > 5:
                say(f"下探：解不动了（差 {diag['gp_mm']:.1f}mm），停")
                break
            uv, msg = align(f"DESCEND {k + 1} align", budget=6, tol=8.0, step_max=0.010)
            if uv is None:
                say(f"下探：{msg}，停")
                break
            _, _, info = look()
            area = info.get("area", 0)
            gz = float(kin.at(arm._ticks_to_qpos(arm.read_ticks()))[0][2])
            say(f"  下探 {k + 1} 步  抓取点 z={gz * 1000:.0f}mm  盒子 {area}px（峰值 {peak}）")
            if area > peak:
                peak = area
            elif area < 0.85 * peak:
                say("  盒子面积掉头 —— 爪子已经贴上去挡住一块，停止下探")
                break
            status("DESCEND", [f"grasp z={gz * 1000:.0f}mm  box {area}px  peak {peak}"])
        gz = float(kin.at(arm._ticks_to_qpos(arm.read_ticks()))[0][2])
        say(f"下探结束：抓取点 z={gz * 1000:.0f}mm")
        stop_here("descend")

        # ---------- 6. 合爪 + 判夹住 ----------
        status("CLOSE", ["closing gripper"])
        q = arm._ticks_to_qpos(arm.read_ticks())
        q[5] = float(arm.grip_lo)
        mv.to(q, "CLOSE  gripper", ["closing until stall"])
        time.sleep(0.6)
        grip_deg = float(np.degrees(arm.grip_q()))
        held = (grip_deg - empty_deg) > HELD_MARGIN_DEG
        say(f"合爪：读回 {grip_deg:.1f}°，空夹基准 {empty_deg:.1f}°，差 "
            f"{grip_deg - empty_deg:.1f}°（界 {HELD_MARGIN_DEG:.0f}°）→ "
            f"{'夹住了' if held else '没夹住'}")
        stop_here("close")
        if not held:
            say("没夹住就没有后面了：抬起来也是空的。停在原地。")
            raise SystemExit(0)

        # ---------- 7. 抬起 ----------
        move_gp([0.0, 0.0, LIFT_UP], f"LIFT  +{LIFT_UP * 1000:.0f}mm")
        gz = float(kin.at(arm._ticks_to_qpos(arm.read_ticks()))[0][2])
        g_deg = float(np.degrees(arm.grip_q()))
        say(f"抬起：抓取点 z={gz * 1000:.0f}mm  爪角 {g_deg:.1f}°（比空夹高 {g_deg - empty_deg:.1f}°）")
        if (g_deg - empty_deg) <= HELD_MARGIN_DEG:
            say("抬起来的过程中掉了。停在原地。")
            raise SystemExit(0)
        stop_here("lift")

        # ---------- 8. 挪到随机落点 ----------
        q_cur = arm._ticks_to_qpos(arm.read_ticks())
        gp, _ = kin.at(q_cur)
        tgt = gp.copy()
        tgt[0] = rng.uniform(0.16, 0.24)
        tgt[1] = rng.uniform(-0.06, 0.06)
        tgt[2] = PLACE_Z
        q_place, diag = kin.solve(q_cur, tgt, q_guide=q_ref)
        say(f"落点 {np.round(tgt, 3)}（随机，seed={args.seed}）解出来差 {diag['gp_mm']:.1f}mm")
        if diag["gp_mm"] > 8:
            raise RuntimeError(f"落点解不动（差 {diag['gp_mm']:.0f}mm）")
        mv.to(q_place, "PLACE", [f"-> {np.round(tgt, 3)}"])
        gz = float(kin.at(arm._ticks_to_qpos(arm.read_ticks()))[0][2])
        g_deg = float(np.degrees(arm.grip_q()))
        say(f"到位：抓取点 z={gz * 1000:.0f}mm  爪角 {g_deg:.1f}°")
        stop_here("place")

        # ---------- 9. 松爪 ----------
        say(f"松爪前再确认还夹着：比空夹高 {g_deg - empty_deg:.1f}°")
        q = arm._ticks_to_qpos(arm.read_ticks())
        q[5] = float(getattr(arm, "grip_open", arm.grip_hi))
        mv.to(q, "OPEN", ["releasing"])
        time.sleep(0.5)
        say(f"松爪：读回 {np.degrees(arm.grip_q()):.1f}°")
        status("DONE", ["grasp -> lift -> move -> release"])
        time.sleep(1.5)

    except KeyboardInterrupt:
        say("\n打断。停在当前位置，不松力矩。")
    except SystemExit:
        raise
    except Exception as exc:                                    # noqa: BLE001
        say(f"\n出错：{type(exc).__name__}: {exc}")
    finally:
        lc.stop()
        time.sleep(0.3)
        cam.close()
        ctrl.close()
        cv2.destroyAllWindows()
        path = ui.save_video()
        LOG = OUT / "servo_log.txt"
        LOG.write_text("\n".join(log) + f"\n\n视频：{path}\n", encoding="utf-8")
        print(f"\n日志 {LOG}\n视频 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
