"""新主循环（阶段 1）：在 so101_nexus 那套仿真上跑 pick-and-place。

    G:/anaconda/envs/sim2real/python.exe run_sim.py --rounds 2 --seed 0

和旧的 `harness.py` 是两条线：那条跑在 `so101-smolvla-sim/sim_env.py`（单方块、
按颜色判），这条跑在 `so101_nexus` 的 `MuJoCoPickLift-v1`（三个方块、按落点判）。
旧的文件一个字没动。

**主循环只认五个接口。** 它不知道机械臂是仿的还是真的、策略是写死的还是大模型、
判官看真值还是看画面 —— 这些都写在 `configs/default.yaml` 里，换一样改一行。
所以这个文件里没有一句 MuJoCo、没有一句串口。

一轮做的事：
    开一局（拿到这一局要抓哪个方块）→ 策略走一遍 → 判官下结论 → 存视频 + 存记录
"""

from __future__ import annotations

import argparse
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import impls.arm_sim        # noqa: E402,F401  导入即挂上牌子
import impls.cam_sim        # noqa: E402,F401
import impls.judge_grasp    # noqa: E402,F401
import impls.judge_truth    # noqa: E402,F401
import impls.perception     # noqa: E402,F401
import impls.policy_expert  # noqa: E402,F401
import skills.home          # noqa: E402,F401
import skills.pick          # noqa: E402,F401
import skills.place         # noqa: E402,F401
import skills.sweep         # noqa: E402,F401

from core import blind                             # noqa: E402
from core import geom                               # noqa: E402
from core import shapes                             # noqa: E402
from core.registry import available, make          # noqa: E402
from core.round_state import RoundState, evidence  # noqa: E402
from core.types import Round                        # noqa: E402


def load_config(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resume_key(*, seed, yaw, off, place, cfg, steps, single, randomize) -> str:
    """这一轮「是按什么参数跑的」的指纹，断点续跑就靠它认领旧记录。

    **光看文件名不够。** 换个 seed、换个偏移、改了配置再跑，轮次号照样是 01，
    旧记录会冒充新结果 —— 那比从头重跑糟得多。所以记录里存指纹，对不上就当没跑过。
    """
    import hashlib
    import json

    payload = {
        "seed": seed,
        "yaw": yaw,
        "off": None if off is None else [float(v) for v in off],
        "place": None if place is None else [float(v) for v in place],
        "steps": steps,
        "single": bool(single),
        "randomize": bool(randomize),
        "cfg": cfg,          # 配置整份进去：改了一行配置，之前那批记录就不算数了
    }
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def read_round(path: Path) -> dict | None:
    """读一份旧记录。读不出来的当没跑过 —— 宁可重跑一轮，也别拿半份记录去判。"""
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:                   # 文件写坏 / 被截断 / 不是 json
        print(f"  ! {path} 读不出来（{type(exc).__name__}），当这轮没跑过")
        return None


# 回放速度按仿真的实际时间算。nexus 一步控制 = timestep 0.005 × 4 个子步 = 0.02 秒，
# 所以控制是 50Hz；每 stride 个控制步留一帧，实速就是 50/stride 帧每秒。
CONTROL_HZ = 50


# 整轮视频里叠上机械爪轮廓和合爪方向（用户 2026-10-15 要求：「视频里面也要带上轮廓和方向」）。
# 那一份是 `_trace_jaw.py` 描的、`skills/pick_real.py` 合爪前判据吃的东西 —— 同一份文件，
# 视频里看到的就是判据看到的。**画幅对不上就不画**（仿真那几路不是这个尺寸）。
JAW_SHAPE_JSON = HERE / "assets" / "camera_calibration" / "real_wrist_jaw_shape.json"
_JAW_OVERLAY: list = []                # 惰性读一次：[image_size, {名字: 轮廓}, 轴起点, 轴方向]


def overlay_jaw(img: np.ndarray) -> np.ndarray:
    """把描好的爪子轮廓 + 合爪轴（带方向箭头）画到画面上；读不到、画幅对不上就原样返回。

    颜色跟 `skills/pick_real.py::_jaw_check` 写的那张 `out/jaw_check.jpg` 一套：固定爪青、
    动爪张到底绿、动爪合到底橙，合爪轴洋红、箭头指 `axis_dir`（第 1 点 → 第 2 点）。
    **这里的图是 RGB**（`write_video` 拿到的是相机那一路的 RGB），所以三元组按 RGB 给。

    方向箭头**只画 350 像素长**（不是画到画面外）—— 轴的方向常常是朝画面外那一边，
    画满整条线的话箭头整个跑到画外，看不出哪头是前方。轴上另外画一条贯穿整幅的细线，
    那条只表示「轴在哪」，不带方向。
    """
    import cv2
    import json

    if not JAW_SHAPE_JSON.exists():
        return img
    if not _JAW_OVERLAY:
        d = json.loads(JAW_SHAPE_JSON.read_text(encoding="utf-8"))
        polys = {k: np.asarray(d[k], dtype=np.int32) for k in
                 ("fixed_jaw", "moving_open", "moving_closed") if k in d}
        _JAW_OVERLAY.append([list(d.get("image_size", ())), polys,
                             np.asarray(d.get("axis_ref_px", (0, 0)), dtype=np.float64),
                             np.asarray(d.get("axis_dir_px", (0, 0)), dtype=np.float64)])
    size, polys, ref, adir = _JAW_OVERLAY[0]
    h, w = img.shape[:2]
    if [w, h] != [int(v) for v in size]:
        return img
    n = float(np.linalg.norm(adir))
    if n < 1e-9 or not polys:
        return img
    adir = adir / n
    out = img.copy()
    col = {"fixed_jaw": (0, 255, 255), "moving_open": (0, 255, 0),
           "moving_closed": (255, 165, 0)}              # RGB
    for k, p in polys.items():
        cv2.polylines(out, [p.reshape(-1, 1, 2)], True, col[k], 3, cv2.LINE_AA)
    far = 0.8 * float(w)
    a = tuple(np.round(ref).astype(int))
    cv2.line(out, tuple(np.round(ref - far * adir).astype(int)),
             tuple(np.round(ref + far * adir).astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(out, a, tuple(np.round(ref + 350.0 * adir).astype(int)),
                    (255, 0, 255), 4, cv2.LINE_AA, tipLength=0.12)
    cv2.putText(out, "jaw outline + closing axis (arrow = direction)", (28, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, "jaw outline + closing axis (arrow = direction)", (28, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_video(path: Path, frames: list[dict], cams: list[str], fps: float) -> None:
    """把每帧的几路画面横着拼起来写成一个 mp4。

    码率固定、按物理步等间隔取帧 —— 按画面差异丢帧会让回放速度忽快忽慢。
    **每一路都先过一遍 `overlay_jaw`**（画幅对不上它自己会跳过）。
    """
    import av

    stack = [np.hstack([overlay_jaw(np.asarray(f[c]).astype(np.uint8)) for c in cams])
             for f in frames]
    h, w = stack[0].shape[:2]
    with av.open(str(path), "w", format="mp4") as c:
        # 帧率会是 12.5 这种小数（50 ÷ 4）；写成 12 或 13 都比实速差一截，用分数存。
        s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "23", "preset": "veryfast"}
        for img in stack:
            for pk in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img),
                                                          format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)


# 视频怎么切。日志里按 `state.seg` 分得很细，那是给排查用的；看的时候想连着看的
# 是「抓取这一整套动作」。进场 → 转正 → 下探合爪 本来就是一条连贯的路，拆成三个
# 文件没法连起来看，并成一段。摆件是操作员把方块摆到工位上，不是机械臂干的活。
VIDEO_GROUPS = [(("进场 → 方块上方", "挪到方块正上方", "腕子对上方块侧面", "下探到方块"),
                 "进场到下探抓取")]
VIDEO_SKIP = ("摆件",)


def video_groups(chunks: list[tuple[str, list[dict]]]) -> list[tuple[str, list[dict]]]:
    """把按 state.seg 切细的段落并成「看的时候想连着看的那几段」。"""
    out: list[tuple[str, list[dict]]] = []
    for seg, frames in chunks:
        if not frames or seg.startswith(VIDEO_SKIP):
            continue
        name = next((g for keys, g in VIDEO_GROUPS if seg in keys), seg)
        if out and out[-1][0] == name:
            out[-1][1].extend(frames)
        else:
            out.append((name, list(frames)))
    return out


def real_evidence(arm, state, index: int) -> dict:
    """真机上给判官的那份证据。

    仿真那份（`core.round_state.evidence`）读的是方块真值坐标，真机上没有这个查询。
    真机读得到的只有两样，判官 `held` 也就按这两样判：

        grip_deg   轮末现读的 6 号舵机角度（度）—— 嘴里有没有东西
        empty_deg  技能开跑时实测的**合到底**角（度）—— 行程的小头
        open_deg   技能开跑时实测的**张到最大**角（度）—— 行程的大头
        tcp_z      轮末 TCP 的世界高度（米）
        lift_z     超过这个高度才算「离了桌面」（米）= 桌面 z + 摆件躺平半高

    判「夹住」要两头齐：轮末那只爪子可能是**故意张开的**（失败收尾会张爪），
    只跟 `empty_deg` 比会把张开误判成夹住。

    `held` 只给日志看：那是技能自己报的，判官量的是现在的现场，不听人说。

    `empty_deg` 是技能合空爪时现场量的，挂在 `arm` 上（`skills/pick_real.py` 第 0 段）。
    这里只是把它转手给判官 —— 不做任何换算，判官拿到的就是技能当时读到的那个数。
    """
    from servo_bridge import read_gripper_deg            # noqa: PLC0415

    ctrl = getattr(arm, "ctrl", None)
    deg = None if ctrl is None else read_gripper_deg(ctrl)
    specs = geom.object_specs()
    rest_z = float(shapes.from_spec(specs[min(index, len(specs) - 1)]).rest_z)
    return {"target_index": index,
            "grip_deg": deg,
            "empty_deg": getattr(arm, "empty_deg", None),
            "open_deg": getattr(arm, "open_deg", None),
            "tcp_z": float(arm.tcp()[2]),
            "lift_z": geom.real_table_z() + rest_z,
            "held": state.held}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="SO-101 harness 主循环（nexus 仿真）")
    ap.add_argument("--config", type=Path, default=HERE / "configs" / "default.yaml")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--stride", type=int, default=4, help="每几个控制步留一帧画面")
    ap.add_argument("--fps", type=float, default=None,
                    help="视频回放帧率。默认按仿真的实际速度算（50 ÷ stride）。"
                         "写死成 30 会比实速快 2.4 倍，看着不像机械臂真实快慢。")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--live", type=Path, default=None,
                    help="把最新那帧画面覆盖写到这个文件（写 jpg，编码快）。"
                         "**只写文件，不进控制回路** —— 跑真机时人（和我）靠它看"
                         "此刻相机里是什么，视频要等整轮跑完才有。不给就不写。")
    ap.add_argument("--whole", action="store_true",
                    help="整轮只出一个视频（所有段连起来、从头到尾），不按段拆。"
                         "要连着看一整轮就用它。")
    ap.add_argument("--steps", type=str, default=None,
                    help="只走这几步，逗号分隔，比如 park 或 park,pick。"
                         "不给就跑完整的一轮（park,pick,place,home）")
    ap.add_argument("--stop-after", type=str, default=None,
                    help="真机那一路：在 `skills/pick_real.py` 的某一段**走完**就收工，"
                         "按段名给（进场 / 看一眼 / 挪到盒子正上方 / 腕子对上方块侧面 / "
                         "下探 / 合爪 / 抬升），逗号分隔可以给几个。"
                         "只影响 `pick_real` 这一支，不给就一路走到底。")
    ap.add_argument("--yaw-deg", type=str, default=None,
                    help="摆件时把方块按这个朝向摆（度），逗号分隔可以一次跑几个朝向"
                         "做对照，每个值一轮。不给就按 PARK_JITTER_YAW 随机摇。")
    ap.add_argument("--pick-offset", type=str, default=None,
                    help="摆件时把方块挪开工位多少（毫米，写成 dx:dy），逗号分隔可以一次"
                         "跑几个位置做对照，每个值一轮。不给就摆在工位坐标上。")
    ap.add_argument("--place-xy", type=str, default=None,
                    help="落点改由外面给（毫米，底座坐标系，写成 x:y），逗号分隔可以一次"
                         "给几个落点、每个值一轮。这就是真机上到这一步能拿到的全部信息："
                         "一个坐标点，没有标识盘可看。仿真里另外画一个盘出来，"
                         "只为让你一眼看出方块落得准不准 —— 落点算的是这串数，不是那个盘。")
    ap.add_argument("--randomize", action="store_true",
                    help="域随机化：每局的灯光亮度和色温、桌面颜色都换一套。"
                         "同一 seed 抽到的是同一套，能复现。不给就一点不改，"
                         "跟以前逐字节一样。")
    ap.add_argument("--single", action="store_true",
                    help="单方块场景：场上只有 red 一个方块，摆在拾取标识点上，"
                         "朝向随机，标识盘不画（标识点只是「我知道这个坐标」）。"
                         "只在仿真上有意义")
    ap.add_argument("--blind", action="store_true",
                    help="盲测：当场随机抽一件东西摆上桌（正方体 / 长方体 / 实心圆柱 / "
                         "空心圆柱，尺寸也在范围内抽），**形状和尺寸一个字不说、指令只给"
                         "颜色**。真值只留给仿真和判官，策略那一路的形状必须自己从画面里"
                         "看出来（`core.blind`）。同一个 seed 抽到同一件。")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：这一轮的记录已经在 --out 里、而且当时是按同一套参数"
                         "跑的，就认领它、跳过不重跑。**默认关是有意的** —— 改完代码"
                         "想跑同一条命令看效果，开着它反而会被跳过、看着像改动没生效"
                         "（2026-10-13 验 pick.py 那个守卫就正是这么重跑的）。"
                         "跑几十轮的批、断了想接着跑的时候才开。")
    ap.add_argument("--resume-pick", action="store_true",
                    help="断点续跑（**抓取那一段**）：`out/pick_checkpoint.json` 在、"
                         "而且臂还在那个姿态上，就从「合爪前微调」那一眼接着走 —— 不再"
                         "重走 回位、起手合爪、进场、看一眼、纠位、转正、下探。"
                         "断点是 `PICK_CONFIRM=1` 那一路在等点头之前写盘的。"
                         "**跟 `--resume` 不是一回事**：那个是整轮跳过。")
    args = ap.parse_args()

    if args.resume_pick:
        import os                                                   # noqa: PLC0415

        os.environ["PICK_RESUME"] = "1"
        print("⤒ 断点续跑（抓取那一段）开着：先找 out/pick_checkpoint.json，"
              "臂不在断点姿态上就从头走这一趟")

    # 回放速度：不给 --fps 就按仿真的实际速度。默认留了 30，录出来的东西会快一倍多。
    fps = float(args.fps) if args.fps else CONTROL_HZ / args.stride

    cfg = load_config(args.config)
    # **真机那一路的实现按需导入。** `real/arm.py` 一导就拉 mujoco + lerobot + nexus，
    # 仿真那支没必要背这个；反过来真机也用不上 `impls/arm_sim.py` 那些场景件。
    # 判据只有配置里那一行 `arm:`。
    is_real = str(cfg.get("arm", "sim")).lower() == "real"
    # **真机那一套代码也可以跑在仿真臂上**（`arm: sim_real`）：技能、感知、判官全是
    # 真机那份，只有机械臂换个替身。那是[核对「仿真知道的有没有比真机多」]用的 ——
    # 配置里写一行 `real_stack: true` 就把它要的几样装上（真机那几支实现 + 现场常数）。
    # 独立一个开关、不看 `arm:` 那一行，是为了让 `arm: real` 仍然**只**走真机那一支。
    real_stack = is_real or bool(cfg.get("real_stack"))
    if real_stack:
        import impls.arm_sim_real                              # noqa: F401
        import impls.judge_held                                # noqa: F401
        import impls.policy_real                               # noqa: F401
        import real.arm                                        # noqa: F401
        import real.perception                                 # noqa: F401
        if is_real:
            import real.cameras                                # noqa: F401
        import skills.home_real                                # noqa: F401
        import skills.pick_real                                # noqa: F401
        if is_real and args.single:
            raise SystemExit("--single 是仿真上的摆件方式（场上只留一件、把标识盘撤掉），"
                             "真机上摆件是人把盒子放到桌上，这个开关没有对应物。")
        # 真机现场的两个常数：桌面多高、盒子大致摆在哪。**必须在 `make("arm", ...)`
        # 之前设好** —— 感知起来时就读它们（`core/geom.real_table_z` / `real_pick_xy`
        # 读不到会当场抛）。口径跟 `OBJECTS` / `PLACE_TARGET` / `EXTRA_MARKS` 一样。
        rig = cfg.get("real") or {}
        if rig.get("table_z_mm") is None or rig.get("pick_xy_mm") is None:
            raise SystemExit(
                "configs/real.yaml 的 `real:` 那一段缺数：要 `table_z_mm`（桌面在臂基座系"
                "多高，毫米）和 `pick_xy_mm`（盒子大致摆在哪，毫米 [x, y]）。"
                "**这两个数不给就没法把画面反推成世界坐标**，拿猜的去驱动机械臂比不跑更糟。")
        geom.REAL_TABLE_Z = float(rig["table_z_mm"]) / 1000.0
        geom.REAL_PICK_XY = np.array([float(v) for v in rig["pick_xy_mm"]],
                                     dtype=np.float64) / 1000.0
        # 腕相机反推的固定偏移（标定误差的补丁）。**没配就是不扣**（跟上面两个不一样，
        # 那两个少了就没法算）。配了就在这儿打出来 —— 这个数直接影响落点，必须看得见。
        geom.REAL_PERCEP_OFFSET = np.array(
            [float(v) for v in rig.get("percep_offset_mm") or (0.0, 0.0)],
            dtype=np.float64) / 1000.0
        if float(np.any(geom.REAL_PERCEP_OFFSET)):
            print(f"感知反推的坐标扣掉固定偏移 "
                  f"{np.round(geom.REAL_PERCEP_OFFSET * 1000, 1)}mm"
                  f"（`real.percep_offset_mm`）")
        if is_real and args.fps is None:
            print("! 没给 --fps：这套按仿真的 50Hz 算回放速度，真机一步约 0.1 秒，"
                  "录出来的视频会比实速快几倍。想要实速就写 --fps 10。")
    # 盲测开关。**要在建场、建感知、跑策略之前置好** —— 那三处都读它。
    blind.ON = bool(args.blind or cfg.get("blind"))
    # 指定了朝向/偏移就一个值一轮，命令行给了几个值就是几轮
    yaws = ([float(v) for v in args.yaw_deg.split(",") if v.strip()]
            if args.yaw_deg else None)
    offs = ([tuple(float(v) / 1000.0 for v in p.split(":"))
             for p in args.pick_offset.split(",") if p.strip()]
            if args.pick_offset else None)
    places = ([np.array([float(v) / 1000.0 for v in p.split(":")], dtype=float)
               for p in args.place_xy.split(",") if p.strip()]
              if args.place_xy else None)
    n_list = max(len(yaws or []), len(offs or []), len(places or []))
    # 命令行明确给了 --rounds 就听它；没给才按「扫了几组朝向/偏移/落点就是几轮」。
    # 给了 --place-xy 一个坐标就想连跑五轮，靠的就是这一条。
    rounds = args.rounds or n_list or int(cfg.get("rounds", 2))
    seed0 = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    # 时间上限**归策略管**（只有它清楚自己试到第几次了、这一次是什么时候开的），
    # 主循环只负责「到点把这一轮停下来」—— `arm.terminated` 每一步都会被环境重置，
    # 得每步重新立，这件事只有主循环做得到。两条口径见 `policy.deadline()`。
    cam_names = list(cfg.get("cams") or ["sim_wrist"])
    steps = ([s.strip() for s in args.steps.split(",") if s.strip()]
             if args.steps else None)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"机械臂 {available('arm')} / 相机 {available('camera')} / "
          f"感知 {available('perception')} / 策略 {available('policy')} / "
          f"判官 {available('judge')} / 技能 {available('skill')}")

    # 给了落点坐标就在场景里把那个点画出来（EXTRA_MARKS），只为给人看。真机上没有。
    geom.EXTRA_MARKS = list(places or [])
    # **摆件清单：这一局桌上摆的是什么。** 配置里 `objects:` 写了就用它，没写就是老
    # 口径那三块 30mm 方块。**必须在 make("arm","sim") 之前设**：仿真照它建场景、
    # 感知照它建词表、判官照它算躺平多高，三处读的是同一份。
    geom.OBJECTS = list(cfg.get("objects") or [])
    # **盲测：这一局摆什么，当场抽。** 抽出来这一行只落在 `geom.OBJECTS` 里 —— 仿真
    # 照它建场景、判官照它算躺平多高。策略那一路拿到的是抹过的版本（`specs_for_policy`）。
    if blind.ON:
        spec = blind.sample(np.random.default_rng(seed0))
        geom.OBJECTS = [spec]
        blind.set_truth(spec)
    # 不给 --randomize 时一个 kwarg 都不传 —— 默认行为跟以前逐字节一样，
    # 现有实验和判官不看这个开关也不会被它影响。
    extra = {"randomize": True} if args.randomize else {}
    if args.single:
        geom.OBJECTS = geom.object_specs()[:1]      # 单件场景：只留清单里的第一件
        # 单件场景默认什么盘都不画；给了落点坐标就只画那一个盘，不画工位圆环
        arm = make("arm", cfg.get("arm", "sim"), objects=geom.object_specs(),
                   markers=bool(places), ring=False, **extra)
        s0 = geom.object_specs()[0]
        print(f"单件场景：场上只有一件 {s0['color']} {s0['shape']}，"
              + (f"只画出落点那个盘 {np.round(places[0], 3)}" if places
                 else "标识盘不画（标识点只是「我知道这个坐标」）"))
    elif is_real:
        # `RealArm` 不认识 `objects=`（场景是仿真照摆件清单建出来的，真机上桌子就是
        # 桌子），串口和速度档从 `real:` 那一段来。
        arm = make("arm", "real", **{k: v for k, v in (cfg.get("real") or {}).items()
                                     if k in ("port", "speed")})
    else:
        arm = make("arm", cfg.get("arm", "sim"), objects=geom.object_specs(), **extra)
    cam_kw = ({"index": int((cfg.get("real") or {}).get("cam_index", 1))}
              if is_real else {})
    cams = {n: make("camera", n, source=arm, **cam_kw) for n in cam_names}
    judge = make("judge", cfg.get("judge", "truth"))

    # 感知：看着俯视画面、按指令说「要抓的是哪一件、它在哪」。参考图是「空工位」那张
    # —— 桌上什么都没有时拍的，`foreground_mask` 拿它比出「这一帧多出来了什么」。
    # 真机上就是开机拍一张空桌面；仿真这边由机械臂现造一张。
    pname = cfg.get("perception", "nearest")
    # **看哪一路画面归配置说。** `brown_rect` 默认盯 `real_wrist`（它是在真机上写的）；
    # 仿真臂上跑它的时候得写明 `perception_cam: sim_wrist`，不然它会去找一路没有的
    # 相机、当场报「没有 real_wrist 这一路」。别的感知不吃这个 kwarg，就不传。
    percep_kw = ({"cam_name": str(cfg["perception_cam"])}
                 if cfg.get("perception_cam") else {})
    percep = make("perception", pname, cams=cams,
                  ref=arm.ref_frame() if hasattr(arm, "ref_frame") else None,
                  # 盲测：**清单要抹过再给** —— 只有颜色，形状是 unknown。给它真清单，
                  # 感知就按真形状投剪影了，这一支考的东西当场没了。
                  objects=blind.specs_for_policy(geom.object_specs()),
                  **percep_kw)
    print(f"用：机械臂={arm.name}  相机={cam_names}  感知={percep.name}  "
          f"策略={cfg.get('policy')}  判官={judge.name}")
    if blind.ON:
        # 真值还是印出来 —— 但只印给**看日志的人**对着核，策略读不到这行字（它拿到
        # 的是 `specs_for_policy` 抹过的那一份）。不印的话，「看出来的是什么」没法对账。
        print(f"桌上摆的：随机抽的一件，只告诉它颜色 —— {geom.object_specs()[0]['color']}"
              f"    （真值 {blind.TRUTH['label']}，这一行只有人和判官看得见）\n")
    else:
        print("桌上摆的：" + "、".join(
            f"{o['color']} {o['shape']} "
            f"{float(o.get('side_mm', o.get('diameter_mm', 0))):.0f}mm"
            for o in geom.object_specs()) + "\n")

    results = []
    n_skipped = [0]                # 断点续跑跳过了几轮；结尾要交代，不然「跑了 5 轮」
                                   # 会让人以为这次真的跑了 5 局
    t_start = time.time()
    for k in range(rounds):
        # 扫朝向/扫偏移做对照时几轮必须同一个 seed —— 换 seed 的话机械臂自己的初始
        # 姿态也跟着变，量到的差别就分不清是「朝向/偏移」还是「这一局的随机摆位」造成的。
        # 扫朝向/偏移/落点做对照时几轮必须同一个 seed —— 换 seed 的话机械臂自己的初始
        # 姿态也跟着变，量到的差别就分不清是「朝向/偏移」还是「这一局的随机摆位」造成的。
        # 但明确给了 --rounds 是「连跑几轮看稳不稳」，那种正相反，要逐轮换 seed。
        same_seed = bool(yaws or offs or places) and args.rounds is None
        seed = seed0 + (0 if same_seed else k)
        # 这一轮的三个摆场参数先算出来：断点续跑要靠它们认指纹，真要跑时又要拿它们摆场。
        # 一处算、两处用，免得「指纹认的是 A、摆的是 B」。
        r_yaw = None if not yaws else yaws[min(k, len(yaws) - 1)]
        r_off = None if not offs else offs[min(k, len(offs) - 1)]
        r_place = None if not places else places[min(k, len(places) - 1)]
        key = resume_key(seed=seed, yaw=r_yaw, off=r_off, place=r_place, cfg=cfg,
                         steps=steps, single=args.single, randomize=args.randomize)

        # **断点续跑**：跑过的轮次不重跑。两条判据一起看 —— 记录文件在，且指纹对得上。
        # 判在 `arm.reset` **之前**：先过了这一关才开新的一局，不然是白开一局再跳过。
        rec_path = args.out / f"round_{k + 1:02d}.json"
        if args.resume and rec_path.exists():
            old = read_round(rec_path)
            if old is not None and (old.get("meta") or {}).get("resume_key") == key:
                v_old = old.get("verdict") or {}
                done = bool(v_old.get("success"))
                results.append(done)     # 跳过的轮也算进结尾的成功率，不然那个数会虚高
                n_skipped[0] += 1
                print(f"第 {k + 1} 轮：要抓 {old.get('target')}"
                      f"（seed {old.get('seed')}，{old.get('task')}）")
                print(f"  ⏭ 跳过：{rec_path} 在、指纹 {key} 对得上，这轮跑过了"
                      f"（{'成功' if done else '失败'} — {v_old.get('reason') or '没说'}）\n")
                continue
            if old is not None:
                print(f"  ↻ {rec_path} 在，但指纹对不上（记的是 "
                      f"{(old.get('meta') or {}).get('resume_key')}，这次是 {key}）"
                      f"—— 参数或配置跟那轮不一样，重跑")

        info = arm.reset(seed)
        # 每局重拍一张空工位参考图：`--randomize` 会把桌面颜色换掉，上一局那张对不上
        # 这一局的桌子。没开随机化时桌面没变，重拍一张也一样，代价是一次渲染。
        if hasattr(arm, "ref_frame") and hasattr(percep, "set_reference"):
            percep.set_reference(arm.ref_frame())
        skills.sweep.YAW_DEG = r_yaw
        skills.sweep.OFFSET_XY = r_off
        # 这一轮的落点：外面给的坐标（真机口径）或表里的放置标识点
        geom.PLACE_TARGET = r_place
        idx = int((info or {}).get("target_index", 0))
        # 这一轮抓的是哪一件：颜色取自摆件清单，不是写死的三色表。
        color = geom.object_specs()[min(idx, len(geom.object_specs()) - 1)]["color"]
        # 喂给策略的那句话。**盲测下不能用 nexus 生成的那句** —— 它带着形状词
        # （「Pick up the red cube.」），递过去等于把答案一起递过去了。
        # **真机上 nexus 那句也没有**（`RealArm.reset()` 返回空 dict，场上就一件、
        # 摆件的是人），所以就按手上这件东西现拼一句。
        task = (blind.task_sentence(color) if blind.ON
                else (info or {}).get("task") or cfg.get("task")
                or f"pick up the {color} box")

        # 摆场记录：仿真记「三个方块各在哪」，真机没有真值查询，只记「这一轮抓第几件」。
        layout = {"target_index": idx}
        if hasattr(arm, "cube_pos") and hasattr(arm, "n_cubes"):
            layout["spawn"] = [[float(v) for v in arm.cube_pos(i)]
                               for i in range(arm.n_cubes)]
        rnd = Round(index=k + 1, task=task, target=color, seed=seed, layout=layout)
        state = RoundState()
        policy = make("policy", cfg.get("policy", "expert"),
                      arm=arm, cams=cams, state=state, order=steps, percep=percep,
                      judge=judge,
                      max_attempts=cfg.get("max_attempts", 3),
                      attempt_seconds=cfg.get("attempt_seconds", 30),
                      round_seconds=cfg.get("round_seconds", 120),
                      # 盲测那一支只到「抓起来」为止（判官也换成 grasp）
                      goal=cfg.get("goal", "place"))
        policy.reset(info)
        # **`--stop-after 进场`：只走前几段。** 真机那一路的三整趟跑下来十几分钟，问题
        # 出在第一段的话后面全是白耗舵机。这个开关只钉 `skills/pick_real.py` 那一个模块级
        # 变量，别的技能、别的文件一个字不动 —— 不给（默认 None）就是走到底。
        if args.stop_after:
            from skills import pick_real as _pick_real      # noqa: PLC0415
            _pick_real.knobs.STOP_AFTER = args.stop_after
            print(f"⏹ 只走到：{args.stop_after}（走完这一段就收工，不重试）\n")
        # **`--steps park`：仿真上「操作员把盒子摆到工位」这一步得有个人干。** 真机的
        # 策略（`impls/policy_real.py`）里没有 `park` —— 那边摆件就是人把盒子放上去，
        # 所以它写死了「回位 → 抓 → 回位」。要在仿真上按真机那一趟的坐标复现，就得先
        # 让仿真把盒子摆到指定位置（`--pick-offset` / `--yaw-deg` 走的就是 `skills/sweep.py`
        # 那一步）。**这一步不动臂**（只写场景状态），所以挂在策略外面、主循环里，
        # 不动真机那一支的三个技能。
        if steps and "park" in steps and hasattr(arm, "place_cube"):
            make("skill", "park").run(arm, cams, state=state)

        n = [0]
        frames = [0]
        # 按「这一步在干什么」（state.seg）把画面切成一段一段：一个技能内部分了好几个
        # 阶段（进场 / 下探 / 合爪 / 抬升），日志里看得见、视频里也分得开。
        chunks: list[tuple[str, list[dict]]] = []
        cut = [False]          # 这一轮是不是被时间上限掐掉的
        cut_why = [""]         # 掐掉时策略给的说法

        # **到点不能只是打印，得让它真的收工。** 办法是把 `arm.terminated` 立起来：
        # `follow` 和 `hold` 每一步都看它，一看就返回 'term'，各技能顺着往上收，整轮
        # 几步之内停住。为什么这样立得住：`arm_sim.step_action` 是**先**重置 terminated
        # （读环境的 term/trunc，那两个永远是 False），**再**调 on_step —— 所以在这儿
        # 置 True 之后，同一步的 follow 就能看到它。
        def on_step():
            n[0] += 1
            if not cut[0]:
                dl = policy.deadline()      # (到点的墙钟秒, 理由)；None = 没设上限
                if dl is not None and time.time() > dl[0]:
                    cut[0] = True
                    cut_why[0] = dl[1]
                    print(f"    ⏱ 到点了：{dl[1]}，收工（已走 {n[0]} 个控制步）")
            if cut[0]:
                # **每一步都得重新立一遍。** `step_action` 每步开头都会把它重置成
                # 环境的 term/trunc（永远是 False），只在掐掉那一步置 True 的话，
                # 下一步就恢复成 False，整轮接着往下跑 —— 兜底等于没有。
                arm.terminated = True
            seg = state.seg or "—"
            if not chunks or chunks[-1][0] != seg:
                chunks.append((seg, []))
            if n[0] % args.stride:
                return
            frames[0] += 1
            cam = {name: cam_.shot() for name, cam_ in cams.items()}
            # 留证：方块这一刻在哪、两个指垫各顶着几个接触点。光看关节角分不出
            # 「方块被挤着往下走」还是「被腕子的转动带下去」，这两个数才能分。
            # **真机这三样都没有**（没有真值坐标、没有接触传感），extra 就留空，
            # 画面和关节角照记 —— 现场出问题回看视频靠的就是这两样。
            extra = None
            if hasattr(arm, "finger_cube_contacts") and hasattr(arm, "cube_pos"):
                nf, nm, zc = arm.finger_cube_contacts(idx)
                extra = {
                    "cube": [[round(float(v), 5) for v in arm.cube_pos(i)]
                             for i in range(arm.n_cubes)],
                    "touch": [int(nf), int(nm),
                              None if zc != zc else round(float(zc), 5)],
                }
            # 记的是「发出去的那一维指令」。仿真手上的是七维动作；真机没有这个字段，
            # 退而记六个关节角（`RealArm.angles` 和 `SimArm.angles` 同口径，都是度）。
            act = getattr(arm, "last_action", None)
            if act is None:
                act = arm.angles()
            rnd.add_frame(cam, arm.angles(), act, extra=extra)
            chunks[-1][1].append(cam)
            # **实时画面。** 每留一帧就把它覆盖写到 `--live` 那个文件上（相机给的是
            # RGB，`imwrite` 要 BGR，所以这儿转一下）。多路相机就横着拼起来，跟
            # 视频里一样的拼法。写 jpg 不写 png：1080p 的 png 编码要几十毫秒，
            # 而真机一个控制步才 ~90ms，那会实打实地把整轮拖慢。
            if args.live is not None:
                import cv2                                     # noqa: PLC0415

                names = list(cam)
                im = (np.hstack([np.asarray(cam[c]).astype(np.uint8) for c in names])
                      if len(names) > 1
                      else np.asarray(cam[names[0]]).astype(np.uint8))
                cv2.imwrite(str(args.live), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        print(f"第 {rnd.index} 轮：要抓 {color}（seed {seed}，{task}）")
        if geom.PLACE_TARGET is not None:
            print(f"  落点由外面给：{np.round(geom.PLACE_TARGET, 3)}"
                  f"（毫米 {geom.PLACE_TARGET[0] * 1000:.0f}:{geom.PLACE_TARGET[1] * 1000:.0f}）"
                  f"  —— 机械臂只知道这串数，场景里那个盘只是画给你看的")
        print(f"  这一步走：{' → '.join(policy.order)}")
        arm.on_step = on_step
        t0 = time.time()
        try:
            policy.plan({"task": task, "target": color,
                         "arm": arm.angles(), "cam": {n2: c.shot() for n2, c in cams.items()}})
        finally:
            arm.on_step = None

        for name, out in policy.log:
            print(f"    {name:<6} {out}")

        # 判官要的证据：三个方块最后各在哪、爪里还有没有东西。**和策略轮中那一次
        # 用的是同一个函数**（`core.round_state.evidence`），两处不许各写一份。
        rnd.evidence = (real_evidence(arm, state, idx) if real_stack
                        else evidence(arm, state, idx,
                                      geom.place_xy(idx) if places else None))
        rnd.verdict = judge.judge(rnd)
        rnd.meta = {"policy": policy.name, "judge": judge.name,
                    "resume_key": key,      # 断点续跑认的就是这一个（见 resume_key()）
                    "seconds": round(time.time() - t0, 1), "n_frames": frames[0],
                    "timeout": cut[0], "timeout_why": cut_why[0],
                    "attempts": getattr(policy, "attempt", None),
                    "stop_why": getattr(policy, "stop_why", "")}

        v = rnd.verdict
        if cut[0]:
            # 判官只按最后的现场判，它不知道这一轮是被掐掉的。这一行要单独说清楚，
            # 不然「还挂在天上」会被当成抓取本身的问题。
            print(f"  ⏱ 注意：这一轮是被时间上限掐掉的（{cut_why[0]}），不是自己走完的")
        if getattr(policy, "stop_why", ""):
            print(f"  收工：{policy.stop_why}")
        print(f"  判定：{'成功' if v.success else '失败'}（{v.source}）— {v.reason}")
        print(f"  用时 {rnd.meta['seconds']}s，{frames[0]} 帧\n")
        results.append(v.success)

        # 一段一个视频，文件名带上它在干什么，方便一段一段看。
        # `--whole` 时整轮只出一个文件（所有段连起来，从头到尾），不再按段拆。
        rnd.meta["chunks"] = [{"seg": s, "n_frames": len(fs)} for s, fs in chunks]
        # **先记分段再落盘**：离线定位（failure_scan.py）要拿这个把出事的那一帧
        # 翻回「是进场、下探还是搬运那一段」。原来写在落盘后面，于是每份 json 都
        # 少了这一段 —— 记录本身是好的，只是没写进去。
        (args.out / f"round_{rnd.index:02d}.json").write_text(
            __import__("json").dumps(rnd.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
        if not args.no_video:
            # 指定了朝向/偏移/落点就写进文件名，几轮对照放一个目录里也分得清谁是谁
            tag = ""
            if yaws:
                tag += (f"yaw{yaws[min(k, len(yaws) - 1)]:+.0f}"
                        .replace("+", "p").replace("-", "m") + "_")
            if offs:
                ox, oy = offs[min(k, len(offs) - 1)]
                tag += (f"off{ox * 1000:+.0f}_{oy * 1000:+.0f}mm"
                        .replace("+", "p").replace("-", "m") + "_")
            if places:
                px, py = places[min(k, len(places) - 1)]
                tag += (f"at{px * 1000:+.0f}_{py * 1000:+.0f}mm"
                        .replace("+", "p").replace("-", "m") + "_")
            try:
                if args.whole:
                    allf = [f for _, fs in chunks for f in fs]
                    path = args.out / f"{tag}round_{rnd.index:02d}_完整.mp4"
                    write_video(path, allf, cam_names, fps)
                    print(f"  整轮 {len(chunks)} 段 {len(allf)} 帧 -> {path}")
                else:
                    for j, (seg, fs) in enumerate(video_groups(chunks), start=1):
                        if not fs:
                            continue
                        safe = "".join(ch if ch.isalnum() else "_" for ch in seg)[:24]
                        path = args.out / f"{tag}round_{rnd.index:02d}_s{j:02d}_{safe}.mp4"
                        write_video(path, fs, cam_names, fps)
                        print(f"  第 {j} 段「{seg}」{len(fs)} 帧 -> {path}")
            except Exception as exc:                       # 存视频失败不该带走整轮
                print(f"  ✗ 视频没写成：{type(exc).__name__}: {exc}")
            print()

    arm.close()
    wall = time.time() - t_start
    ok = sum(results)
    print("-" * 70)
    print(f"跑了 {len(results)} 轮，总 {wall:.1f}s"
          + (f"（其中 {n_skipped[0]} 轮是跳过没跑的，判定从旧记录里读回来）"
             if n_skipped[0] else ""))
    print(f"成功率 {ok}/{len(results)}"
          + (f"（{ok / len(results) * 100:.0f}%）" if results else ""))
    print(f"结果在 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
