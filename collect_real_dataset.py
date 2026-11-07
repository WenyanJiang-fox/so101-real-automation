# -*- coding: utf-8 -*-
"""真机单路相机抓取数据集：脚本专家自动跑，只录抓取那一段。

    关掉旧面板，然后：
    G:/anaconda/envs/sim2real/python.exe -u collect_real_dataset.py --episodes 10

一集 = 回家（不录）→ `pick_real` 从头走一趟（录）→ 判 held 才存。
抓成的那些，技能自己会把方块摆到随机点（170~210mm）再回家 —— 那一段照跑不录，
它只负责给下一集换一个位置。

**格式**（LeRobot v3.0，`lerobot 0.6.1`）：
    observation.state              float32 (6,)   度
    action                         float32 (6,)   度（发出去的目标位置）
    observation.images.camera1     video (540,960,3)   腕相机，原生一半分辨率、不变形
    task                           每帧一句（configs/real.yaml 里那个颜色）
    默认 fps=4（0.2s 闸口隔一步录一帧），写进
    `G:\\sim2real\\so101-real-harness\\datasets\\so101_real_grasp_960_4hz`。

**每次启动先清 `out/`**：删掉修改时间超过 `--keep-out-days`（默认 3 天）的文件。
不想清就 `--keep-out-days 0`；只想清不跑臂就 `--clean-out-only`。

**抓到没抓到只看技能的判据**（`verdict == "held"` 且没掉），不另设阈值。

**同一只方块的失败链**（2026-10-24 用户定）：前三次换方法，三次都不成就把方块推回
机械臂正前方（`skills/pick_real/push.py`），再从头来。**没有下扎深度阶梯**
（用户 2026-10-24：「偏了怎么扎都没用」—— 深度改不了横向偏，所以不再拿它当一套
独立策略）。每次尝试都留一对原图和位置到
`out/vlm_attempts/`、一行记录到 `out/vlm_attempts.jsonl`；`_vlm_attempt_pairs.py`
把这些串成「失败那趟差多少偏移」的标签。

**安全闸触发不停整轮**（2026-10-22 用户定）：这一集作废，现场（原因 + 关节角 +
抓取点 + 相机画面）交给 VLM 分析，原因和整改方案写到
`out/safety_reports/<时间>_ep<集号>/`，**只存档不执行**，随后回初始位置继续下一集。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 闸口一律关掉：不停下来等人（和 `_oneshot_grasp.py` 一样）。
for _k in ("PICK_CONFIRM_STEP", "PICK_CONFIRM", "PICK_CONFIRM_DESCEND",
           "PICK_RESUME", "PICK_START_AT"):
    os.environ.pop(_k, None)

import numpy as np                                               # noqa: E402

from core import geom                                            # noqa: E402
from core.round_state import RoundState                          # noqa: E402
from core.registry import make                                   # noqa: E402
from panel.rig_common import DEFAULT_CONFIG, load_cfg            # noqa: E402

#: **2026-10-20 用户定：数据放新文件夹。** 原来写在 `G:\sim2real\data\`（那里放的是
#: 仿真训练集），现在收在 harness 自己下面，和仿真数据分开。
#:
#: **后缀写的是格式。** `_960` = 540×960 的图，`_4hz` = 0.2s 定频。LeRobot 把 feature 形状和
#: `fps` 都写死在 `meta/info.json` 里，格式一变就往同一个目录续写会报错，所以按格式分目录。
DEFAULT_ROOT = r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_960_4hz"
DEFAULT_REPO = "so101_real/pick_grasp"

#: `out/` 里放的是每次实验的图、视频、日志。**每次采集启动时按天数清一次**，
#: 不再挂系统的定时任务（2026-09-21 用户要求：无人看管的定时任务不许有）。
OUT_DIR = ROOT / "out"
#: **手动停机的开关**（2026-10-22）：放到这个路径就停。硬杀进程会把正在写的 parquet
#: 分片截断（`data/file-00N.parquet` 没有尾 → 整个分片读不出来），所以停下来一律走这里：
#: 当前这一集走完 → 退出循环 → `finally` 里 `rec.close()` 把分片收尾。
STOP_FILE = OUT_DIR / "STOP"
STOP_REASON = "收到停止文件"
#: 成败样本都攒够了。和 `STOP_REASON` 一样**不重开新一轮**。
GALLERY_REASON = "图够了"


def _gallery_counts() -> tuple[int, int]:
    """`out/vlm_gallery/` 里已经存了多少张 (held, empty)。

    目录名以 `_held` / `_empty` 结尾（`vlm_gate.stash_frames` 按 verdict 起名）。
    """
    from skills.pick_real import knobs as _knobs                  # noqa: PLC0415

    root = Path(_knobs.VLM_GALLERY)
    if not root.is_dir():
        return 0, 0
    held = empty = 0
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.endswith("_held"):
            held += 1
        elif p.name.endswith("_empty"):
            empty += 1
    return held, empty
#: 安全闸触发时的 VLM 存档（只记录，不改参数、不重试这一集）。
SAFETY_DIR = OUT_DIR / "safety_reports"

#: 一集少于这么多帧就不存（正常一集 600~900 帧；2 秒都不到的必然是中途异常）。
MIN_FRAMES = 20
#: **连续这么多集抛异常就停手**（2026-10-20 用户定：普通抓取失败不停，只有硬件出问题
#: 才停）。串口掉了 / 相机掉了会每集都抛同一条，继续跑只是空转。
MAX_CONSEC_EXC = 3
#: 每集「回初始位置」读回来的 TCP 相对第一集允许漂多少（毫米）。超了说明回位本身在漂，
#: 下一集的起点和这一集不一样，数据不能再往下攒。
HOME_DRIFT_MM = 10.0
#: 整集墙钟上限（秒）。**覆盖 `skill.run` 全程**，不只是录制那一段 —— `real/recorder.py`
#: 的看门狗在「抓成、录制停下」之后就不再看了，摆件那一段卡住没人管。
EPISODE_WALL_S = 420.0
#: 被叫停之后最多重开几轮（2026-10-20 用户定：**叫停一次就重开一轮，再叫停就彻底停**）。
MAX_ROUNDS = 2
#: 重开前等多久（秒）：串口和相机句柄刚关，让驱动放开再重开。
RESTART_WAIT_S = 5.0

#: **推是最后一道策略**（2026-10-24 用户定）：VLM 支路（下探那一问、三档失败那一问）
#: 先走，三档都试完才推；VLM 现在没启动，所以它就是「其他策略都失败之后」的那一步。
#: **没推到允许的位置就继续推，不限次数**。
#:
#: 原来这里有一套「下扎深度阶梯」（−10/+10/−5/+5mm），2026-10-24 用户否掉：
#: 「扎有时候并不能改变，因为你偏了，怎么扎都是没用的」—— 横向偏了，深度再变也
#: 夹不中，所以它不算一套独立策略，直接删掉。（要改横向，得在落点 `aim` 上加
#: xy 偏置，那个量现在还没标定，等 `out/vlm_attempts.jsonl` 攒出 `need_mm` 再说。）
#:
#: 唯一会停手的条件是**推不动**：连着 `MAX_PUSH_STALLS` 次推完，方块挪动都不到
#: `PUSH_MOVED_MIN_MM`（或者干脆认不到方块）。这是「再推也没用」，不是「还没推到位」。
MAX_PUSH_STALLS = 3
PUSH_MOVED_MIN_MM = 5.0


def _setup_geom(cfg: dict) -> None:
    rig = cfg["real"]
    geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
    geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                                 dtype=np.float64) / 1000.0
    geom.REAL_PERCEP_OFFSET = np.array(
        [float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
        dtype=np.float64) / 1000.0
    geom.OBJECTS = list(cfg.get("objects") or [])


def clean_out(days: float) -> None:
    """删掉 `out/` 里修改时间早于 `days` 天的文件。**只按时间，不按文件名。**

    清的是每次实验留下的截图、视频、日志；`days <= 0` 就不清。
    """
    if days <= 0 or not OUT_DIR.is_dir():
        return
    cut = time.time() - float(days) * 86400.0
    n = 0
    nbytes = 0
    failed = 0
    for p in OUT_DIR.rglob("*"):
        if not p.is_file():
            continue
        try:
            st = p.stat()
            if st.st_mtime >= cut:
                continue
            p.unlink()
        except OSError as exc:
            failed += 1
            print(f"[清理] 删不掉 {p}：{exc}", flush=True)
            continue
        n += 1
        nbytes += st.st_size
    print(f"[清理] out/ 删掉 {n} 个文件，释放 {nbytes / 1e6:.1f}MB"
          f"（保留最近 {days:g} 天；删不掉 {failed} 个）", flush=True)


def _fail(msg: str):
    """参数不对就**在动臂之前**停下。所有检查都不猜、不纠正，直接报出读到的是什么。"""
    raise SystemExit(f"[参数检查] ✗ {msg}")


def _size_wh(text) -> tuple[int, int]:
    """`"960x540"` / `"512"` → `(宽, 高)`。非法就抛，不退回默认值。"""
    t = str(text).lower().replace("×", "x").strip()
    try:
        if "x" in t:
            w, h = (int(v) for v in t.split("x", 1))
        else:
            w = h = int(t)
    except ValueError:
        _fail(f"--size 读不懂：{text!r}（要写「宽x高」，例如 960x540）")
    return w, h


def _mm(name: str, v, lo: float, hi: float) -> float:
    """毫米量的范围检查。**量级不对时明确说是单位问题**：写成米的值会落到 1 以下。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        _fail(f"{name} 不是数：{v!r}")
    if not (lo <= x <= hi):
        hint = ""
        if abs(x) < 5.0 and lo >= 5.0:
            hint = "（这个量级像是按「米」写的，这里一律用毫米）"
        _fail(f"{name} = {x:g}mm 超出允许范围 {lo:g}~{hi:g}mm{hint}")
    return x


def validate(cfg: dict, args) -> tuple[int, int]:
    """**动臂之前**把这一轮会用到的每个数过一遍：单位、范围、数据集格式。

    这里挡的是「参数传错一行，臂跑出去」——所以宁可误报也不放过：
      · 毫米量的量级；`pick_xy` 是否落进安全包络；端口 / 相机序号格式；
      · `--episodes / --fps / --size` 的取值范围；
      · 续写已有数据集时，`meta/info.json` 里的 fps 和相机形状必须和这一次一致
        （LeRobot 不会替你改格式，写进去才发现就晚了）。
    """
    n_ep = int(args.episodes)
    # `0` = 不设上限（2026-10-24 用户定）。上限那一侧仍然挡：写错一位数就当成
    # 「跑一万集」那种事，必须在动臂之前拦下来。
    if not (0 <= n_ep <= 10000):
        _fail(f"--episodes = {n_ep}，允许 0（不设上限）或 1~10000")
    if not (1 <= int(args.fps) <= 60):
        _fail(f"--fps = {args.fps}，允许 1~60")
    w, h = _size_wh(args.size)
    if not (64 <= w <= 1920 and 64 <= h <= 1080):
        _fail(f"--size = {w}x{h}，每边允许 64~1920（宽） / 64~1080（高）")
    if w * h > 1920 * 1080:
        _fail(f"--size = {w}x{h} 超过原生 1920x1080")

    rig = (cfg or {}).get("real") or {}
    for k in ("table_z_mm", "pick_xy_mm", "percep_offset_mm"):
        if k not in rig:
            _fail(f"configs/real.yaml 缺 real.{k}（不拿猜的值驱动机械臂）")
    _mm("real.table_z_mm", rig["table_z_mm"], -150.0, 50.0)

    px = list(rig["pick_xy_mm"])
    if len(px) != 2:
        _fail(f"real.pick_xy_mm 要两个数，现在读到 {rig['pick_xy_mm']!r}")
    x = _mm("real.pick_xy_mm[0]", px[0], 60.0, 320.0)
    y = _mm("real.pick_xy_mm[1]", px[1], -320.0, 320.0)
    if not (60.0 <= float(np.hypot(x, y)) <= 300.0):
        _fail(f"real.pick_xy_mm 的水平半径 {float(np.hypot(x, y)):.0f}mm 不在 "
              f"60~300mm 之间（300 是安全闸 `real/safety.py` 的上限）")

    off = list(rig.get("percep_offset_mm") or (0.0, 0.0))
    if len(off) != 2:
        _fail(f"real.percep_offset_mm 要两个数，现在读到 {rig['percep_offset_mm']!r}")
    for i, v in enumerate(off):
        _mm(f"real.percep_offset_mm[{i}]", v, -100.0, 100.0)

    port = str(rig.get("port", ""))
    if not (port.upper().startswith("COM") and port[3:].isdigit()):
        _fail(f"real.port = {port!r}，要像「COM10」这样的串口名")
    ci = int(rig.get("cam_index", -1))
    if not (0 <= ci <= 10):
        _fail(f"real.cam_index = {ci}，允许 0~10")
    if not (cfg.get("objects") or []):
        _fail("configs/real.yaml 的 objects 是空的 —— 没有要抓的东西")
    for i, o in enumerate(cfg.get("objects") or []):
        if not o.get("color"):
            _fail(f"objects[{i}] 缺 color")
        sz = o.get("size_mm") or o.get("side_mm")
        if not sz:
            _fail(f"objects[{i}] 缺 size_mm / side_mm")

    root = Path(args.root)
    if not root.is_absolute():
        _fail(f"--root 要绝对路径，现在读到 {args.root!r}")
    info = root / "meta" / "info.json"
    if info.exists():
        import json                                              # noqa: PLC0415
        meta = json.loads(info.read_text(encoding="utf-8"))
        if int(meta.get("fps", -1)) != int(args.fps):
            _fail(f"{root} 已有数据集的 fps = {meta.get('fps')}，这一次是 {args.fps} —— "
                  f"换一个 --root，别把两种时间轴写进同一份")
        cam = (meta.get("features") or {}).get("observation.images.camera1") or {}
        shp = tuple(cam.get("shape") or ())
        if shp and (shp[0], shp[1]) != (h, w):
            _fail(f"{root} 已有数据集的 camera1 是 {shp[1]}x{shp[0]}，这一次是 {w}x{h} —— "
                  f"换一个 --root")
    return w, h


@contextlib.contextmanager
def episode_watchdog(arm, seconds: float):
    """整集墙钟上限：到点就立 `arm.terminated`，让技能每个检查点自己退出来。

    **交回一个 `Event` 就是「是不是看门狗掐的」。** 不能事后看 `arm.terminated` ——
    普通失败收工时技能自己也会把它立起来（`skills/pick_real/skill.py` 的停手路径）。
    """
    done = threading.Event()
    hit = threading.Event()

    def _run():
        if done.wait(max(0.0, seconds)):
            return
        hit.set()
        print(f"[看门狗] ⏱ 这一集超过 {seconds:.0f}s —— 立 terminated 收工", flush=True)
        arm.terminated = True

    threading.Thread(target=_run, name="episode-watchdog", daemon=True).start()
    try:
        yield hit
    finally:
        done.set()


def recover_home(arm, why: str) -> None:
    """收工动作：**先开爪，再回初始位置**（和面板急停同一顺序）。

    只在这条采集脚本自己退出前调一次。失败也只打日志，不把原始异常盖掉。
    """
    from real.joint_move import goto_path, home_q                    # noqa: PLC0415

    print(f"[收工] {why} —— 开爪 + 回初始位置", flush=True)
    try:
        arm.terminated = False
        q0 = arm._ticks_to_qpos(arm.read_ticks())
        q1 = np.array(q0, dtype=np.float64, copy=True)
        q1[5] = float(getattr(arm, "grip_open", arm.grip_hi))
        goto_path(arm, [q0, q1])
        q0 = arm._ticks_to_qpos(arm.read_ticks())
        goto_path(arm, [q0, home_q(arm)])
        print("[收工] 已回初始位置", flush=True)
    except Exception as exc:                                         # noqa: BLE001
        print(f"[收工] ✗ 收工动作失败：{type(exc).__name__}: {exc}", flush=True)


def stop_reason(out: dict, truth: str, arm, hit, exc_n: int) -> str | None:
    """这一集之后要不要停整轮。**普通没抓住不停**（2026-10-20 用户定）。

    **安全闸不停整轮**（2026-10-22 用户定）：这一集作废、存档、清标志，下一集照常。
    另外三种情况停：目标认不出来了 / 这一集被看门狗掐了 / 异常连着来。
    """
    if truth == "no_detect":
        return "目标消失（连续扫视没认出来）"
    why = str(out.get("why") or "")
    if "找不着" in why or "没认出来" in why:
        return "目标消失（连续扫视没认出来）"
    if hit is not None and hit.is_set():
        return f"这一集被看门狗掐了（超过 {EPISODE_WALL_S:.0f}s）"
    if exc_n >= MAX_CONSEC_EXC:
        return f"连续 {exc_n} 集抛异常（多半是串口 / 相机掉了）"
    return None


#: 每轮开始时的标定偏移（`geom.REAL_PERCEP_OFFSET`）。VLM 的 nudge 以它为基准。
_BASE_OFF: np.ndarray | None = None

#: 发给 VLM 的工作空间事实（毫米）。`main` 里按配置填一次。
#: 只给「臂能到哪」这类算好的常数，不给「该去哪」——后者仍由策略算。
_WORKSPACE: dict = {}


def _workspace(cfg: dict) -> dict:
    """发给 VLM 的**静态**空间事实。会变的那两项（起搜中心、当前平移）另算。"""
    from core import geom                                              # noqa: PLC0415
    from skills.pick_real import knobs as K                            # noqa: PLC0415

    rig = (cfg or {}).get("real") or {}
    rc = np.asarray(geom.MARKER_CENTER, dtype=np.float64).reshape(-1)[:2]
    return {
        # 可达环是**绕工作台中心**量的：半径 130~260mm，环心 `MARKER_CENTER`。
        # 只给半径不给环心，模型判不出「该往哪边挪才是靠近环心」。
        "reach_ring_mm": [130, 260],
        "ring_center_mm": [round(float(rc[0]) * 1000, 1), round(float(rc[1]) * 1000, 1)],
        "table_z_mm": rig.get("table_z_mm"),
        # 标定值，**不是**这一趟的起搜中心（那个由 `_start_xy_mm()` 现算）
        "calib_pick_xy_mm": rig.get("pick_xy_mm"),
        "percep_offset_mm": rig.get("percep_offset_mm"),
        "place_radius_mm": [round(float(K.PLACE_R_MIN_M) * 1000),
                            round(float(K.PLACE_R_MAX_M) * 1000)],
    }


def _start_xy_mm() -> list[float] | None:
    """这一趟进场用的**起搜中心**（毫米）。优先级和 `skills/pick_real/skill.py` 一致：
    `knobs.REPLAY` → `knobs.GUESS_XY`（上一趟放下的位置）→ 配置里的 `pick_xy`。"""
    from core import geom                                              # noqa: PLC0415
    from skills.pick_real import knobs as K                            # noqa: PLC0415

    if K.REPLAY is not None and K.REPLAY.get("guess") is not None:
        g = K.REPLAY["guess"]
    elif K.GUESS_XY is not None:
        g = K.GUESS_XY
    else:
        try:
            g = geom.real_pick_xy()
        except RuntimeError:
            return None
    return [round(float(g[0]) * 1000, 1), round(float(g[1]) * 1000, 1)]


def _nudge_now() -> list[float] | None:
    """当前生效的目标点平移（毫米）。`_apply_vlm` 写的是
    `REAL_PERCEP_OFFSET = _BASE_OFF − nudge/1000`，这里反解回来。"""
    from core import geom                                              # noqa: PLC0415

    if _BASE_OFF is None:
        return None
    cur = np.asarray(geom.REAL_PERCEP_OFFSET, dtype=np.float64).reshape(-1)[:2]
    base = np.asarray(_BASE_OFF, dtype=np.float64).reshape(-1)[:2]
    d = (base - cur) * 1000.0
    return [round(float(d[0]), 1), round(float(d[1]), 1)]


#: VLM 上一次改完留下的现场，用来在「同一个失败码又来了」时整条撤回。
#: `{"code": 触发时的失败码, "snap": 改之前的旋钮快照}`。None = 没有待验的改动。
_VLM_PENDING: dict | None = None


def _knob_snapshot(params_out: dict) -> dict:
    """把 VLM 会碰的旋钮全部拍一份，撤销时原样还回去。"""
    from core import geom                                              # noqa: PLC0415
    from skills.pick_real import knobs as KNOBS                        # noqa: PLC0415

    return {
        "params": dict(params_out),
        "joint1": getattr(KNOBS, "JOINT1_STEP_DEG", None),
        "step": getattr(KNOBS, "JOINT_STEP_DEG", None),
        "off": np.array(geom.REAL_PERCEP_OFFSET, dtype=np.float64, copy=True),
    }


def _knob_restore(snap: dict, params_out: dict) -> None:
    from core import geom                                              # noqa: PLC0415
    from skills.pick_real import knobs as KNOBS                        # noqa: PLC0415

    params_out.clear()
    params_out.update(snap.get("params") or {})
    KNOBS.JOINT1_STEP_DEG = snap.get("joint1")
    KNOBS.JOINT_STEP_DEG = snap.get("step")
    if snap.get("off") is not None:
        geom.REAL_PERCEP_OFFSET = np.array(snap["off"], dtype=np.float64, copy=True)


def _revert_if_repeated(code: str, params_out: dict) -> None:
    """VLM 改完之后这一集**还是同一个失败码** → 提前撤销那次改动。

    `_vlm_episode_begin` 会在下一集开头无条件还原；这里只是提前一集把
    「改错了」的变量收回，少跑一集。两条路都是还原同一份快照。
    """
    global _VLM_PENDING
    if not _VLM_PENDING:
        return
    p, _VLM_PENDING = _VLM_PENDING, None
    if code != p.get("code"):
        return
    _knob_restore(p["snap"], params_out)
    print(f"  [VLM·撤销] 改完还是同一个失败码（{code}）—— 旋钮全部回到改之前",
          flush=True)


def _vlm_episode_begin(ep: int, params_out: dict) -> None:
    """VLM 的变量**只作用一集**，用完了在下一集开头无条件还原。

    2026-10-23 用户定：VLM 只提少量变量，绝大部分仍由三档策略自己决定，
    且变量不能影响之后的策略。所以这里不靠失败码判断，到点就还原。
    """
    global _VLM_PENDING
    p = _VLM_PENDING
    if not p:
        return
    if p.get("used_ep") is None:
        p["used_ep"] = ep                    # 这一集就是带变量跑的那一集
        return
    _knob_restore(p["snap"], params_out)
    _VLM_PENDING = None
    print("  [VLM·复位] 变量只作用于上一集 —— 旋钮回到改之前", flush=True)


def _apply_kw(params: dict) -> dict:
    """VLM 的参数 → `apply_action` 的关键字。"""
    hover = params.get("approach_height_mm")
    return {"bias_mm": float(params.get("descend_offset_mm", 0.0)),
            "hover_m": None if hover is None else float(hover) / 1000.0}


def _apply_vlm(decision: dict | None, pol, stats: dict, params_out: dict,
               *, tag: str, code: str | None = None):
    """把 VLM 的 `params` 写进旋钮。**永远返回 None** —— 档位不归 VLM 定。

    2026-10-23 用户定：VLM 只提少量变量（感知偏移 / 安全限幅），绝大部分仍由
    三档策略自己决定，且变量不影响之后的策略。所以这里不碰 `apply_action`，
    也不清 `pol.tried`；变量只作用下一集，`_vlm_episode_begin` 负责到点还原。

    `code` 是触发这次改动的失败码，只用来让 `_revert_if_repeated` 提前撤销。
    """
    global _VLM_PENDING
    if not isinstance(decision, dict):
        return None
    from core import geom                                            # noqa: PLC0415
    from skills.pick_real import knobs as KNOBS                      # noqa: PLC0415

    snap = _knob_snapshot(params_out)
    p = dict(decision.get("params") or {})
    params_out.update(p)
    if "nudge_dx_mm" in p or "nudge_dy_mm" in p:
        # 感知是「画面反推坐标 − 偏移」，所以让爪子往 +x/+y 走就要把偏移减掉。
        # **取累积值**：这一趟只改 dx 时，上一趟的 dy 还得留着。
        if _BASE_OFF is not None:
            # 偏移量的维度随配置变（有的配置只有 x/y），按 `_BASE_OFF` 的长度补零。
            _d = np.zeros_like(_BASE_OFF, dtype=np.float64)
            _d[:2] = [float(params_out.get("nudge_dx_mm", 0.0)),
                      float(params_out.get("nudge_dy_mm", 0.0))]
            geom.REAL_PERCEP_OFFSET = _BASE_OFF - _d / 1000.0
    if "joint1_step_deg" in p:
        KNOBS.JOINT1_STEP_DEG = float(p["joint1_step_deg"])
    if "step_deg" in p:
        # 其它关节的单步上限（默认 `None` = 不限）。只有 VLM 给了才生效。
        KNOBS.JOINT_STEP_DEG = float(p["step_deg"])
    stats["vlm"] = int(stats.get("vlm", 0)) + 1
    _VLM_PENDING = {"code": code, "snap": snap, "used_ep": None}
    print(f"  [VLM·{tag}] 应用：动作={decision.get('move') or '—'}  参数={p}"
          "  （只作用下一集）", flush=True)
    return None


def _ask_vlm_failure(arm, wrist, side, *, ep: int, round_no: int, cur, out: dict,
                     pol, params_out: dict, stats: dict,
                     since: float | None = None) -> dict | None:
    """三档都试过还没成：拍一眼现场问 VLM，返回校验过的决策（没有就 None）。

    `since` 是这一集开始的时刻。技能每量一眼都会重写 `out/jaw_check.jpg` 和
    `out/jaw_raw.jpg`，但那两个路径是**固定的**——这一集要是一眼都没量（`no_measure`
    或者开头就没认出方块），文件里留的是上一集/上一轮的旧画面。拿 mtime 卡住，
    过期的图不发。
    """
    import ppo.vlm as vlm                                            # noqa: PLC0415
    from skills.pick_real import knobs as KNOBS                       # noqa: PLC0415
    from skills.pick_real import strategy as STRAT                    # noqa: PLC0415

    d = OUT_DIR / "vlm_adjust" / time.strftime("%Y%m%d_%H%M%S") / f"ep{ep:03d}"
    images: list[str] = []
    try:
        d.mkdir(parents=True, exist_ok=True)
        from real.safety_report import _save_jpg                      # noqa: PLC0415
        # **失败那一刻的标注图排最前**（`ppo/vlm.py::_caption` 按文件名认它）。
        # 这一张上画着合爪轴、方块轮廓和轴偏/缝的读数 —— 以前根本没发，只在提示词里
        # 说「图上画好了」，模型只能照着 `clear` 编。
        for src in (KNOBS.JAW_CHECK_IMG, KNOBS.JAW_RAW_IMG):
            try:
                if since is not None and src.stat().st_mtime < since - 0.5:
                    continue
                dst = d / src.name
                dst.write_bytes(src.read_bytes())
                images.append(str(dst))
            except OSError:
                continue
        if wrist is not None and _save_jpg(wrist.shot(), d / "grasp_end.jpg"):
            images.append(str(d / "grasp_end.jpg"))
        if side is not None and _save_jpg(side.shot(), d / "side_view.jpg"):
            images.append(str(d / "side_view.jpg"))
    except Exception as exc:                                          # noqa: BLE001
        print(f"  [VLM] 存图失败：{exc}", flush=True)

    truth = STRAT.code(str(out.get("why") or ""))
    nud = _nudge_now()
    # `nudge_mm` 以前只有 `ppo/capture.py` 那条路会填，采集这条一直是 None —— 模型
    # 看不到自己上一次要的平移有没有生效。这里按当前实际偏移反解出来填上。
    packet = vlm.build_packet({**out, "nudge_mm": nud}, attempt=ep,
                              strategy=cur.strategy,
                              history=pol.history, images=images, truth_code=truth,
                              workspace={**_WORKSPACE,
                                         "start_xy_mm": _start_xy_mm()})
    packet["state"]["trigger"] = "all_tiers_failed"
    packet["state"]["vlm_params_now"] = dict(params_out)
    # **位移候选取自这一趟的实测**：方块被推走的位移方向就是下一趟该跟过去的方向。
    # 量不到前后两个中心时是空表，模型这时不许选任何位移类（`vlm.check` 会拒）。
    packet["state"]["nudge_candidates"] = vlm.nudge_candidates(packet["state"])
    # 有没有发标注图决定模型能不能判 `contour_ok`（见 `vlm._PROMPT`）
    packet["state"]["has_jaw_image"] = any("jaw_check" in Path(p).name
                                           for p in images)
    t0 = time.time()
    raw = vlm.diagnose(packet)
    good, why = vlm.check(raw, packet.get("state"))
    print(f"  [VLM] {time.time() - t0:.0f}s  action={(raw or {}).get('action')}"
          f"  move={(raw or {}).get('move')}  cause={(raw or {}).get('cause')}  校验"
          f"{'通过' if good else '不过：' + why}", flush=True)
    # **把这次的判断记进账本**：下一次调它时 `_history_lines` 会把「上次推了哪儿、
    # 结果如何」一起给它 —— 没有这段反馈，它只能在同一根轴上来回翻。
    if good and pol.history:
        pol.history[-1].update({
            "cause": good.get("cause"), "params": good.get("params") or {},
            "contour_ok": good.get("contour_ok"),
            "failure_code": truth, "nudge": nud,
        })
    try:
        (d / "decision.json").write_text(json.dumps(
            {"packet": packet, "raw": raw, "decision": good,
             "response": dict(vlm.LAST)}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception as exc:                                          # noqa: BLE001
        print(f"  [VLM] 写记录失败：{exc}", flush=True)
    return good


def _safety_episode(arm, wrist, side, *, ep: int, round_no: int, strategy,
                    out: dict, stats: dict, pol=None, params_out=None,
                    use_vlm: bool = True) -> bool:
    """这一集是不是被安全闸拦下的。

    是 → 把现场交给 VLM 存档（只写记录，不改任何参数），清掉停手标志，
    返回 True 让采集继续下一集；否 → 返回 False，调用方按原逻辑处理。
    """
    if not getattr(arm, "safety_why", None):
        return False
    dec = None
    try:
        from real.safety_report import record                        # noqa: PLC0415

        _d, dec = record(arm, wrist, side, ep=ep, round_no=round_no,
                         strategy=strategy, out=out or {}, root=SAFETY_DIR,
                         ask=bool(use_vlm))
    except Exception as exc:                                        # noqa: BLE001
        print(f"  [安全记录] 存档失败：{type(exc).__name__}: {exc}", flush=True)
    # **VLM 的限制/变量直接写进策略旋钮**，三档带着它继续跑；没有合法输出就退回原样。
    if dec and pol is not None and params_out is not None:
        _apply_vlm(dec, pol, stats, params_out, tag="安全闸", code="safety")
    stats["safety"] = int(stats.get("safety", 0)) + 1
    print(f"  [安全闸] 这一集作废，继续下一集（累计 {stats['safety']} 次）", flush=True)
    arm.safety_why = None
    arm.terminated = False
    return True


def _run_round(cfg: dict, args, stats: dict, round_no: int) -> str | None:
    """跑一轮采集：建连接 → 集循环 → 收工。

    跑满 `--episodes` 返回 `None`；被安全闸 / 目标消失 / 看门狗 / 异常叫停返回原因。
    **集数、成功数是跨轮累计的**（`stats`），重开一轮不会从头再数。
    """
    import impls.judge_held                                      # noqa: E402,F401
    import impls.policy_real                                     # noqa: E402,F401
    import real.arm                                              # noqa: E402,F401
    import real.cameras                                          # noqa: E402,F401
    import real.perception                                       # noqa: E402,F401
    import real.recorder                                         # noqa: E402,F401
    import skills.home_real                                      # noqa: E402,F401
    import skills.pick_real                                      # noqa: E402,F401
    from ppo.policy import ACTIONS, RetryPolicy, apply_action     # noqa: E402
    from real.recorder import RealRecorder                       # noqa: E402
    from skills.pick_real import strategy as STRAT               # noqa: E402

    rig = cfg["real"]
    color = (cfg.get("objects") or [{"color": "green"}])[0].get("color", "green")
    task = f"pick up the {color} box"

    # ---- 失败后换方法（2026-10-20 用户定）----
    # 第一趟用「完美版」：`no_measure` = 一步到位、合爪前一眼都不量。
    # 没抓住就按 `RetryPolicy` 的顺序换下一档（先量一眼 → 完整微调），
    # 全试过之后回到实测开口最大的那一档。抓成了一趟就复位，下一个目标重新从完美版开始。
    pol = RetryPolicy(max_attempts=None)
    cur = next(a for a in ACTIONS if a.name == "no_measure")

    arm = make("arm", "real", port=str(rig.get("port", "COM10")),
               speed=int(rig.get("speed", 0)),
               fast=bool(rig.get("fast", False)))
    arm.terminated = False
    cams = {"real_wrist": make("camera", "real_wrist", source=arm,
                               index=int(rig.get("cam_index", 1)))}
    percep_kw = ({"cam_name": str(cfg["perception_cam"])}
                 if cfg.get("perception_cam") else {})
    percep = make("perception", "brown_rect", cams=cams, ref=None,
                  objects=geom.object_specs(), **percep_kw)

    # **侧相机只进数据、不帮抓取**（和面板 `panel/rig_assembly.py:90-105` 同一口径）：
    # 它固定在三脚架上，没有标定，感知那条链根本不看它。
    side = None
    side_cfg = (cfg.get("collect") or {}).get("side_cam") or {}
    if side_cfg:
        from real.side_cam import SideCamera                      # noqa: E402,PLC0415
        side = SideCamera(index=int(side_cfg.get("index", 0)),
                          crop=side_cfg.get("crop"),
                          rotate=float(side_cfg.get("rotate", 0.0)))
        sw, sh = side.size()
        print(f"[侧相机] index {side_cfg.get('index', 0)}：{sw}x{sh}"
              f"（裁 {side_cfg.get('crop')}，转 {side_cfg.get('rotate', 0)}°）",
              flush=True)
        # `PICK_VLA_END=1` 时技能要从 `cams` 里拿这一路（末端那一段的第二个视角）
        cams["side"] = side

    rec = None
    if not args.dry_run:
        rec = RealRecorder(args.root, args.repo_id, task, fps=args.fps,
                           size=args.size, side=side)
        arm.on_step = rec.tick

    home = make("skill", "home_real")
    skill = make("skill", "pick_real")

    # **把 VLM 那一问接上**（2026-10-23 用户定）：下探之后、合爪之前，让它看一眼
    # 「方块还在两爪之间吗」，判「不在」就把这一趟抬回去重下一次。**只有采集这一条
    # 路装** —— 面板、一次性脚本不装，`knobs.VLM_JUDGE is None` 时技能行为一个字不差。
    from skills.pick_real import knobs as _K                         # noqa: PLC0415
    from skills.pick_real import vlm_gate as _VG                     # noqa: PLC0415
    if not args.no_vlm_judge:
        import ppo.vlm as _vlm                                       # noqa: PLC0415
        _K.VLM_JUDGE = lambda imgs, caps: _vlm.judge(imgs, caps)      # noqa: E731
        print("[VLM·记] 下探之后那一问已接上（判「不在」→ 抬回去重下一次）",
              flush=True)
    else:
        print("[VLM·记] 下探之后那一问关着（--no-vlm-judge）", flush=True)

    n_exc = 0
    stop_why: str | None = None
    #: 这一轮第一集「回初始位置」读回来的 TCP。之后每集和它比，看起点有没有在漂。
    home_tcp0: np.ndarray | None = None
    #: VLM 当前给的变量（下探偏移 / 悬停高度 / nudge）。每集开头按它重设，否则
    #: `apply_action(cur)` 会把偏置和悬停高度冲回默认值。
    vlm_params: dict = {}
    #: **同一只方块的失败链**（跨轮累计，放 `stats` 里）：`target` 抓成一次 +1、
    #: `attempt` 这只方块的第几次尝试、`pushes` 推过几次。
    #: `out/vlm_attempts.jsonl` 靠这几个数把「失败几趟 → 终于成了」串成一条链。
    #: `push_stall` 是「连续几次推完方块几乎没动」——推不动才停推，推不到位不停。
    for _k in ("target", "attempt", "pushes", "push_stall"):
        stats.setdefault(_k, 0)
    allow_recover = True
    t0 = float(stats["t0"])
    try:
        target_ok = int(getattr(args, "target_ok", 0) or 0)
        while (stats["done"] < int(args.episodes)
               and (target_ok <= 0 or stats["ok"] < target_ok)):
            if STOP_FILE.exists():
                STOP_FILE.unlink()
                stop_why = STOP_REASON
                print(f"\n[停手] {STOP_REASON}：这一集已经走完，收尾后退出", flush=True)
                break
            want_held = int(getattr(args, "target_held", 0) or 0)
            want_empty = int(getattr(args, "target_empty", 0) or 0)
            if want_held > 0 and want_empty > 0:
                n_held, n_empty = _gallery_counts()
                if n_held >= want_held and n_empty >= want_empty:
                    stop_why = GALLERY_REASON
                    print(f"\n[停手] {GALLERY_REASON}：成功 {n_held} 张 / "
                          f"失败 {n_empty} 张，收尾后退出", flush=True)
                    break
            ep = stats["done"] + 1
            stats["done"] = ep
            stats["attempt"] = int(stats["attempt"]) + 1
            # VLM 的变量只作用一集：上一集用完了，这一集开头就还原。
            _vlm_episode_begin(ep, vlm_params)
            #: 这一集开始的时刻。`out/jaw_*.jpg` 是全轮共用的固定路径，
            #: 问 VLM 时按它卡住「这张图是不是这一集的」。
            ep_t0 = time.time()
            print("\n" + "-" * 66)
            print(f"[第 {ep}/{args.episodes} 集·第 {round_no} 轮] 回初始位置（不录）",
                  flush=True)
            arm.terminated = False
            arm.safety_why = None
            state = RoundState()
            out: dict = {}
            exc: Exception | None = None
            rec_started = False
            safety_hit = False
            # **看门狗罩住整集**（回位 + 抓取 + 摆件），不只是录制那一段。
            with episode_watchdog(arm, EPISODE_WALL_S) as hit:
                h = home.run(arm, cams)
                if not h.get("ok"):
                    safety_hit = _safety_episode(arm, cams["real_wrist"], side,
                                                 ep=ep, round_no=round_no,
                                                 strategy=cur, out=h, stats=stats,
                                                 pol=pol, params_out=vlm_params,
                                                 use_vlm=not args.no_vlm_judge)
                    if not safety_hit:
                        stop_why = f"回初始位置没到位：{h.get('why')}"
                        break
                if h.get("ok"):
                    tcp_home = np.asarray(arm.tcp(), dtype=np.float64).reshape(-1) * 1000.0
                    if home_tcp0 is None:
                        home_tcp0 = tcp_home
                        print(f"[起点] 记下本轮的初始 TCP：{np.round(tcp_home, 1)}mm",
                              flush=True)
                    else:
                        drift = float(np.linalg.norm(tcp_home - home_tcp0))
                        print(f"[起点] 和第一集的初始 TCP 差 {drift:.1f}mm"
                              f"（上限 {HOME_DRIFT_MM:.0f}）", flush=True)
                        if drift > HOME_DRIFT_MM:
                            stop_why = (f"回初始位置的落点在漂：比第一集差 {drift:.1f}mm"
                                        f"（上限 {HOME_DRIFT_MM:.0f}mm）")
                            break

                    arm.terminated = False
                    apply_action(cur, **_apply_kw(vlm_params))
                    print(f"[方法] 这一集用「{cur.label}」（{cur.name}）", flush=True)
                    # 这一集的 VLM 记录从这里开始；跑完落 `out/vlm_judge.jsonl` +
                    # `out/vlm_attempts.jsonl`（每次尝试一行，带图目录和位置）。
                    _VG.begin_episode(
                        ep, cur.name, target=int(stats["target"]),
                        attempt=int(stats["attempt"]),
                        perturb={"bias_mm": float(vlm_params.get("descend_offset_mm", 0.0)),
                                 "pushes": int(stats["pushes"])})
                    if rec is not None:
                        rec.start(arm, cams["real_wrist"], state)
                        rec_started = True
                    try:
                        out = skill.run(arm, cams, state=state, index=0, percep=percep,
                                        instruction=task)
                    except Exception as exc_:                      # noqa: BLE001
                        exc = exc_
                        print(f"[第 {ep} 集] ✗ 技能抛异常：{exc_}", flush=True)
                        traceback.print_exc()
                    # **安全闸只作废这一集**：存档之后清标志，下一集照常回位。
                    safety_hit = _safety_episode(arm, cams["real_wrist"], side,
                                                 ep=ep, round_no=round_no,
                                                 strategy=cur, out=out, stats=stats,
                                                 pol=pol, params_out=vlm_params,
                                                 use_vlm=not args.no_vlm_judge)

            n_frames = rec.stop() if (rec is not None and rec_started) else 0

            if safety_hit:
                if rec is not None and rec_started:
                    rec.discard("安全闸")
                stats["fail"] += 1
                n_exc = 0
                print(f"[进度] 成功 {stats['ok']} / 失败 {stats['fail']} / 安全闸 "
                      f"{stats.get('safety', 0)} / 已用 {time.time() - t0:.0f}s",
                      flush=True)
                continue

            if exc is not None:
                if rec is not None:
                    rec.discard(f"异常 {exc}")
                stats["fail"] += 1
                n_exc += 1
                stop_why = stop_reason({}, "exception", arm, hit, n_exc)
                if stop_why:
                    print(f"[停手] {stop_why}", flush=True)
                    break
                print(f"[进度] 成功 {stats['ok']} / 失败 {stats['fail']} / 已用 "
                      f"{time.time() - t0:.0f}s", flush=True)
                continue
            n_exc = 0

            ok = (bool(out.get("ok")) and out.get("verdict") == "held"
                  and not out.get("dropped"))
            truth = STRAT.code(str(out.get("why") or ""))
            # 上一集被 VLM 改过旋钮？这一集还是同一个失败码就把那次改动撤掉。
            # 放在判 ok 之前：成功的时候 code 是 other，自然不等于待验的码，直接就清了。
            _revert_if_repeated(truth, vlm_params)
            print(f"[第 {ep} 集] 结果 ok={out.get('ok')} verdict={out.get('verdict')} "
                  f"dropped={out.get('dropped')} 帧数={n_frames} 失败码={truth}",
                  flush=True)
            # 这一集的 VLM 记录落盘：程序量的离轴、VLM 的判定、否决后重下的离轴、结果。
            # `extra` 是算「差多少偏移」要用的位置事实：合爪那一刻的 TCP、合爪之后的
            # 方块中心、下探把方块推走了多少。
            _VG.end_episode(out.get("verdict"), truth, ok, extra={
                "why": out.get("why") or "",
                "tcp_close_mm": out.get("tcp_close_mm"),
                "center_after_mm": out.get("box_center_after"),
                "box_shift_mm": out.get("box_shift_mm"),
            })
            # 成败都留两张原始帧（`out/vlm_gallery/`）—— 失败集的图随后会被
            # `rec.discard()` 删掉，这里不先留一份，以后就没有失败样本可喂 VLM。
            _VG.stash_frames(ep, out.get("verdict"))

            if ok and n_frames >= MIN_FRAMES:
                if rec is not None:
                    rec.save()
                stats["ok"] += 1
                # **这一趟放下的位置 = 下一趟的起搜中心**（和面板 `panel/rig.py:680` 同一口径）。
                # 只当起搜中心，进场之后照旧看一眼、画面量出来的说了算。
                place = out.get("place") or {}
                if place.get("ok") and place.get("at"):
                    from skills.pick_real import knobs as K
                    K.GUESS_XY = (float(place["at"][0]), float(place["at"][1]))
                    print(f"[下一趟] 起搜中心 = 这一趟放下的位置 "
                          f"({K.GUESS_XY[0] * 1000:.0f}, {K.GUESS_XY[1] * 1000:.0f})mm",
                          flush=True)
                # 抓成了 = 这个目标收工；下一个目标重新从「完美版」开始。
                pol.begin_target()
                cur = next(a for a in ACTIONS if a.name == "no_measure")
                stats["target"] = int(stats["target"]) + 1
                stats["attempt"] = 0
                stats["pushes"] = 0
                stats["push_stall"] = 0
                # 新目标从干净的下扎终点开始（阶梯的偏置不带到下一只方块）。
                vlm_params.pop("descend_offset_mm", None)
            else:
                why = out.get("why") or (
                    f"只录到 {n_frames} 帧（少于 {MIN_FRAMES}）" if ok else "没抓住")
                print(f"[第 {ep} 集] ✗ {why}", flush=True)
                if rec is not None:
                    rec.discard(str(why))
                stats["fail"] += 1
                # 没抓住 → 记下实测，下一集换一档方法（同一目标继续）。
                print("  " + pol.observe(out, act=cur), flush=True)
                # **三档都试过还没成 → 问一次 VLM 要限制/变量**，然后三档重新跑。
                # 没接上或输出不合法就退回原来的规则表选档。
                chosen = None
                # `--no-vlm-judge` 关的是**整条 VLM 支路**：下探后那一问、三档都试过
                # 之后那一问，都不发；档位只按规则表换，变量也不再变。
                if not getattr(args, "no_vlm_judge", False) and len(pol.tried) >= len(ACTIONS):
                    dec = _ask_vlm_failure(arm, cams["real_wrist"], side, ep=ep,
                                           round_no=round_no, cur=cur, out=out,
                                           pol=pol, params_out=vlm_params,
                                           stats=stats, since=ep_t0)
                    _apply_vlm(dec, pol, stats, vlm_params,
                               tag="三档失败", code=truth)
                # ---- 推方块（2026-10-24 用户定）----
                # 前三趟换方法；三档都试完还不成，就把方块往机械臂正前方推 ——
                # **推是最后一道策略**（VLM 支路先走；VLM 关着时，它就是「其他策略
                # 都失败之后」的那一步）。没推到允许的位置就一直推，不限次数。
                # 下扎深度阶梯已删（用户：「偏了怎么扎都没用」）。
                if len(pol.tried) >= len(ACTIONS):
                    if int(stats.get("push_stall", 0)) < MAX_PUSH_STALLS:
                        from skills.pick_real.push import (                # noqa: PLC0415
                            front_target, push_toward)

                        # 目标 = 正前方那个点。`push_toward` 自己在「离目标不到
                        # PUSH_NEAR_M」时直接返回「已经在目标附近」。
                        tgt = front_target()
                        c_after = out.get("box_center_after") or out.get("center_before")
                        if c_after is None:
                            print("[推] 没认出方块中心 —— 这一趟不推", flush=True)
                        else:
                            # **推到允许区为止**：方块没到目标附近就继续推。
                            # 每一步推完都重新认方块（推的实际位移只认画面，不认指令）。
                            while not getattr(arm, "terminated", False):
                                # 推之前再认一眼：上一趟合爪/抬升可能把方块带偏了。
                                c_now, _f = percep.locate(task, guess=c_after)
                                if c_now is not None:
                                    c_after = c_now
                                pr = push_toward(arm, cams, percep,
                                                 np.asarray(c_after, dtype=np.float64),
                                                 tgt, text=task)
                                stats["pushes"] = int(stats["pushes"]) + 1
                                moved = pr.get("moved_mm")
                                why_pr = str(pr.get("why") or "")
                                print(f"[推] 第 {stats['pushes']} 次："
                                      f"{'成了' if pr.get('ok') else '没成'}"
                                      f"（{why_pr or '推了一段'}，"
                                      f"方块实际挪 {moved}mm）", flush=True)
                                if why_pr.startswith("已经在目标附近"):
                                    # 已进允许区：不再推，回去试抓。
                                    stats["push_stall"] = 0
                                    break
                                if moved is None or float(moved) < PUSH_MOVED_MIN_MM:
                                    stats["push_stall"] = int(
                                        stats.get("push_stall", 0)) + 1
                                    print(f"[推] 这一下没推动 "
                                          f"（连续第 {stats['push_stall']}/"
                                          f"{MAX_PUSH_STALLS} 次）", flush=True)
                                    if int(stats["push_stall"]) >= MAX_PUSH_STALLS:
                                        print("[推] 推不动了 —— 这个目标不再推，"
                                              "接着按三档方法试抓", flush=True)
                                        break
                                else:
                                    stats["push_stall"] = 0
                                # 更新起点：用推完重认的中心，下一推从这里算。
                                if pr.get("center_after"):
                                    c_after = np.asarray(pr["center_after"],
                                                         dtype=np.float64) / 1000.0
                            # 方块已经挪了位置：策略账本清空、从第一档重新试，
                            # 但 `attempt` 继续往上数 —— 整条链仍属于同一个目标。
                            pol.begin_target()
                            vlm_params["descend_offset_mm"] = 0.0
                # 档位永远由策略自己选；VLM 只留下变量（下一集用完就还原）。
                cur, change = pol.choose()
                print(f"[方法] 下一集换「{cur.label}」：{change}", flush=True)

            # **普通没抓住不停**；只有目标消失 / 安全闸 / 看门狗 / 异常连着来才停。
            stop_why = stop_reason(out, truth, arm, hit, n_exc)
            if stop_why:
                print(f"[停手] {stop_why}", flush=True)
                break

            print(f"[进度] 成功 {stats['ok']} / 失败 {stats['fail']} / 已用 "
                  f"{time.time() - t0:.0f}s", flush=True)
    except KeyboardInterrupt:
        # 人手按 Ctrl-C：**原地停住，不替你动臂**。
        allow_recover = False
        raise
    finally:
        if allow_recover:
            recover_home(arm, stop_why or "这一轮跑完了")
        if rec is not None:
            rec.close()
        if side is not None:
            try:
                side.close()
            except Exception as exc:                              # noqa: BLE001
                print("关侧相机：", exc, flush=True)
        try:
            cams["real_wrist"].close()
        except Exception as exc:                                  # noqa: BLE001
            print("关相机：", exc, flush=True)
        try:
            arm.close()
        except Exception as exc:                                  # noqa: BLE001
            print("关串口：", exc, flush=True)
    return stop_why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10,
                    help="最多跑多少集；**0 = 不设上限**（跑到停止文件 / 停手判据）")
    ap.add_argument("--target-ok", type=int, default=0,
                    help="存够这么多「抓成」的集数就停（0 = 只看 --episodes）")
    ap.add_argument("--target-held", type=int, default=0,
                    help="`out/vlm_gallery/` 里成功样本攒够这么多张就停（要配 --target-empty）")
    ap.add_argument("--target-empty", type=int, default=0,
                    help="`out/vlm_gallery/` 里失败样本攒够这么多张就停（要配 --target-held）")
    ap.add_argument("--keep-out-days", type=float, default=3.0,
                    help="out/ 里超过这么多天的文件清掉；0 = 不清")
    ap.add_argument("--clean-out-only", action="store_true",
                    help="只清 out/，不碰机械臂")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="数据集目录")
    ap.add_argument("--repo-id", default=DEFAULT_REPO)
    ap.add_argument("--fps", type=int, default=4,
                    help="写进元数据的时间轴；0.2s 闸口隔一步录一帧 ≈ 4Hz")
    ap.add_argument("--size", default="960x540",
                    help="图像存多大，写「宽x高」；只写一个数就是正方形")
    ap.add_argument("--dry-run", action="store_true", help="跑但不写数据集")
    ap.add_argument("--no-vlm-judge", action="store_true",
                    help="关掉「下探之后问 VLM 一眼」这一步（默认开）")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args()

    if args.clean_out_only:
        # 只清 out/，不碰机械臂，也不读配置。
        clean_out(args.keep_out_days)
        return 0

    # **先过参数闸，再碰 out/ 和串口。** 单位 / 范围 / 数据集格式有一处不对就当场退出，
    # 不拿错参数去驱动机械臂。
    cfg = load_cfg(args.config)
    validate(cfg, args)
    # 参数闸过了再归一：`0` = 不设上限。下游只读 `args.episodes`，统一成一个极大值，
    # 停不停交给停止文件 / 停手判据。
    if int(args.episodes) <= 0:
        args.episodes = 10 ** 9

    # 参数过了才清旧的实验产物。
    clean_out(args.keep_out_days)
    _setup_geom(cfg)
    # VLM 的 nudge 以标定值为基准；这里存一份，之后每轮都拿它重算。
    global _BASE_OFF, _WORKSPACE
    _BASE_OFF = np.array(geom.REAL_PERCEP_OFFSET, dtype=np.float64, copy=True)
    _WORKSPACE = _workspace(cfg)

    print("=" * 66)
    print(f"真机数据集采集：{'不设上限' if int(args.episodes) >= 10 ** 9 else str(args.episodes)} 集，"
          f"单路腕相机，fps {args.fps}，{args.size}")
    print(f"数据集：{args.root}（{'dry-run，不写' if args.dry_run else args.repo_id}）")
    print(f"停手策略：被叫停就重开一轮（最多 {MAX_ROUNDS} 轮），再被叫停就彻底停",
          flush=True)
    print("=" * 66, flush=True)

    stats = {"done": 0, "ok": 0, "fail": 0, "t0": time.time()}
    stop_why: str | None = None
    for round_no in range(1, MAX_ROUNDS + 1):
        stop_why = _run_round(cfg, args, stats, round_no)
        if stop_why is None or stop_why in (STOP_REASON, GALLERY_REASON):
            break
        # 集数已经跑到头，最后这一下就别再重开了。
        if stats["done"] >= int(args.episodes) or (
                int(getattr(args, "target_ok", 0) or 0) > 0
                and stats["ok"] >= int(args.target_ok)):
            print(f"\n[完成] 集数已跑满；最后这一集被叫停（{stop_why}），不再重开",
                  flush=True)
            break
        if round_no < MAX_ROUNDS:
            print(f"\n[重开] 第 {round_no} 轮被叫停：{stop_why}", flush=True)
            print(f"[重开] 等 {RESTART_WAIT_S:.0f}s 让串口/相机放开，然后重新开始一轮",
                  flush=True)
            time.sleep(RESTART_WAIT_S)
            continue
        print(f"\n[硬停] 第 {round_no} 轮又被叫停：{stop_why} —— 彻底停住，不再重开",
              flush=True)

    print("\n" + "=" * 66)
    print(f"结束：成功 {stats['ok']} 集 / 失败 {stats['fail']} 集 / 安全闸 "
          f"{stats.get('safety', 0)} 次 / 用时 {time.time() - stats['t0']:.0f}s")
    if stop_why:
        print(f"停手原因：{stop_why}")
    if not args.dry_run:
        print(f"数据集：{args.root}")
    else:
        print("dry-run，没有写数据集。")
    print("=" * 66, flush=True)
    return 0 if stats["ok"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
