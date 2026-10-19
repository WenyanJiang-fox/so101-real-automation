"""模块 A：靠 6 号舵机（夹爪）的角度判断「夹住了没有」。

原理：夹爪是位置控制舵机。合爪指令发的是「转到 GRIPPER_CLOSED_DEG（=0°）」，
嘴里有东西就转不到 0°，会被物体顶在一个更大的角度上。所以读回来的角度
直接反映嘴里有没有东西 —— 不需要相机、不需要模型、没有延迟。

方向（用户实测行程，见 memory so101-gripper-jaw-travel）：
    0°   = 合到底，爪面间距约 30.9 mm
    180° = 全开，  爪面间距约 55.9 mm
于是：空抓 -> 角度接近 0；夹住方块 -> 角度明显大于 0。

**线上这条路是自适应的：不问「这只盒子对应哪个角度区间」。** 每次开跑前把行程两端
都量一遍 —— 合一次空爪（`measure_empty`）、再张到最大（`measure_open`），后面判
「读数是不是卡在这两头中间」（`classify_ref`）。两头都是本次实测的，`empty_margin_deg`
只是噪声地板，跟盒子无关 —— 所以换盒子、换爪面都不用重标。

`calibrate_grasp.py` + `classify()` 那条绝对阈值的路留着，是离线对账用的：
它给出的是「这只夹爪空抓停在几度、夹住停在几度」，看一眼就知道机械状态对不对。
线上不用它，`thresholds.json` 里那几个 null 也就不会挡住抓取。

命令行：
    python grasp_check.py --watch              实时打印 6 号舵机角度（标定时看）
    python grasp_check.py --check              读一次，按 thresholds.json 给结论
    python grasp_check.py --all                顺便把 6 个舵机都打印一遍
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TH_PATH = HERE / "thresholds.json"

HELD, EMPTY, UNCLEAR, NO_READING = "held", "empty", "unclear", "no-reading"

# 兜底余量（度）。正常从 thresholds.json 的 `gripper.empty_margin_deg` 取。
# 它跟「哪只盒子」无关，是**噪声地板**：空抓重复测得有多散、舵机停位有多重复。
# 盒子换了不用动它，这也是「自适应」能成立的原因。
DEFAULT_EMPTY_MARGIN_DEG = 8.0
DEFAULT_DROP_DELTA_DEG = 10.0


def load_thresholds(path: Path = TH_PATH) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_thresholds(data: dict, path: Path = TH_PATH) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def gripper_cal(cal: dict) -> dict:
    """把 gripper 那一段取出来，没标定就直接抛 —— 不猜阈值。"""
    g = cal.get("gripper") or {}
    missing = [k for k in ("empty_max_deg", "held_min_deg", "drop_delta_deg")
               if g.get(k) is None]
    if missing:
        raise RuntimeError(
            f"thresholds.json 里的 gripper 还没标定（缺 {missing}）。"
            f"先跑：python {HERE / 'calibrate_grasp.py'}"
        )
    return g


def classify(angle_deg, cal: dict) -> str:
    """按**标定好的绝对阈值**给结论。angle_deg 为 None 表示这次没读到。

    这是离线那条路（`calibrate_grasp.py` 标、`grasp_check --check` 看）。线上抓取
    走的是 `classify_ref`，它不需要预先标定。"""
    g = gripper_cal(cal)
    if angle_deg is None:
        return NO_READING
    if angle_deg >= g["held_min_deg"]:
        return HELD
    if angle_deg <= g["empty_max_deg"]:
        return EMPTY
    return UNCLEAR


def empty_margin(cal: dict) -> float:
    """「比空抓基准高多少度才算夹住」。盒子无关的噪声地板，见模块顶部常量。"""
    v = (cal.get("gripper") or {}).get("empty_margin_deg")
    return DEFAULT_EMPTY_MARGIN_DEG if v is None else float(v)


def drop_delta(cal: dict) -> float:
    """抬起过程中角度回落多少度算「掉了」。同样与盒子无关。"""
    v = (cal.get("gripper") or {}).get("drop_delta_deg")
    return DEFAULT_DROP_DELTA_DEG if v is None else float(v)


def measure_empty(controller, samples: int = 5, settle: float = 0.8):
    """合爪到空嘴里，读回那个角度 —— 这一趟的「空抓基准」。

    **这就是自适应的那一步。** 不再问「这只盒子对应哪个角度区间」，而是每次开跑前
    让夹爪自己在空气里合到底一次，量出它这一趟停在哪；后面所有判断都是「比这个数
    高多少」。换盒子、换爪面、舵机老化都不用重标。

    `settle` 是等舵机转到底的时间：合爪指令发下去那一刻角度还在变，立刻读会读到
    半路上的值。
    """
    from servo_bridge import close_gripper, read_gripper_deg   # noqa: PLC0415

    close_gripper(controller, speed=0)
    time.sleep(settle)
    return read_gripper_deg(controller, samples=samples)


def measure_open(controller, samples: int = 5, settle: float = 0.8):
    """张到最大，读回那个角度 —— 行程的**另一头**。

    为什么要有这一头：判「夹住」只写「比空抓基准高」是不够的（见 `classify_ref`），
    一只完全张开的爪也「比基准高」。量出张开那头，判据才能要求读数**卡在两头中间**。
    """
    from servo_bridge import open_gripper, read_gripper_deg   # noqa: PLC0415

    open_gripper(controller, speed=0)
    time.sleep(settle)
    return read_gripper_deg(controller, samples=samples)


def classify_ref(angle_deg, empty_deg, cal: dict,
                 open_deg: float | None = None) -> str:
    """拿**本次实测的两个端点**判，不查标定表。读数为 None 就不报结论。

    夹爪是位置控制的，行程两端都能实测出来：

        合到底（空嘴）  `empty_deg`  ← 小的一头
        张到最大        `open_deg`   ← 大的一头

    嘴里咬着东西时，爪合不到底，**卡在两头中间**。所以判据是「比空抓那头明显大、
    又够不着张开那头」—— 两头都是本次实测的，`empty_margin_deg` 只是噪声地板。

    **上界不能省。** 少了它，一只**完全张开**的爪（168°，比空抓基准高出一百多度）
    会被当成「夹着东西」。2026-10-14 首跑就踩了这个：技能那一轮根本没合过爪，
    判官在轮末读到全开的 168° 报了 `held`。`open_deg` 没给就退回只看下界的老行为。

    三档（`open_deg` 给了的时候）：
        跟空抓那头差不到 `empty_margin_deg`  -> empty
        差过了这条地板，又够不着张开那头      -> held
        贴着张开那头（= 没在合）             -> empty，不是 held

    **判据是「离空抓基准有多远」，不分方向**（用户 2026-10-16：「可是抓住了啊，只要大于
    1° 就算夹住」）。原来只认「比空抓那头**大**」这一边，另一边的差被当成 empty。
    实测那一趟：空抓基准 55.3°、合爪完读到 46.5° —— 差了 8.8°，判了 empty。
    方向为什么反的不在这儿猜（真机符号、或者那一趟空抓基准量的时机），只把判据改成
    看绝对值。
    """
    if angle_deg is None:
        return NO_READING
    if empty_deg is None:
        return UNCLEAR
    m = empty_margin(cal)
    d = abs(float(angle_deg) - float(empty_deg))
    if d <= m * 0.5:
        return EMPTY
    if d < m:
        return UNCLEAR
    if open_deg is not None and angle_deg > open_deg - m:
        return EMPTY
    return HELD


class GraspMonitor:
    """抬起过程中反复读角度，用来判「抓起来之后掉没掉」。

    单个角度只能说明合爪那一刻嘴里有没有东西；这一层补的是后面几秒。
    用法：
        m = GraspMonitor(ctrl, cal)
        ... 合爪、抬起到开始抬升 ...
        m.mark_peak()          # 抬起前调一次，记基准（会自动采一次样）
        for _ in 抬起过程:
            m.poll()           # 每 0.2 秒左右调一次
        if m.dropped(): ...    # True = 比基准掉得超过 drop_delta_deg
    """

    def __init__(self, controller, cal: dict, samples: int = 5,
                 empty_deg: float | None = None, open_deg: float | None = None):
        from servo_bridge import read_gripper_deg  # 延迟导入，纯逻辑测试时不必装串口
        self._read = read_gripper_deg
        self.c = controller
        self.cal = cal
        self.samples = samples
        # 本次实测的行程两端。只影响 `verdict()`；`dropped()` 只看回落差值，跟它们无关。
        self.empty_deg = empty_deg
        self.open_deg = open_deg
        self.peak = None
        self.last = None
        self.min_seen = None
        self.history: list[tuple[float, float]] = []

    def poll(self):
        a = self._read(self.c, samples=self.samples)
        self.last = a
        if a is None:
            return None
        if self.peak is None or a > self.peak:
            self.peak = a
        if self.min_seen is None or a < self.min_seen:
            self.min_seen = a
        self.history.append((time.time(), a))
        return a

    def mark_peak(self):
        """把当前角度设成基准。call 在开始抬升之前。"""
        a = self.poll()
        self.peak = a
        self.min_seen = a
        return a

    def verdict(self):
        """按本次实测的两端给结论。没给 `empty_deg` 就退回标定表那条路。"""
        if self.empty_deg is None:
            return classify(self.last, self.cal)
        return classify_ref(self.last, self.empty_deg, self.cal, self.open_deg)

    def dropped(self):
        """None = 信息不够，判不了。"""
        if self.peak is None or self.last is None:
            return None
        return (self.peak - self.last) >= drop_delta(self.cal)


def _main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="夹爪判据（模块 A）")
    ap.add_argument("--port", default=None, help="串口，默认取 thresholds.json 里记的，再不行 COM12")
    ap.add_argument("--watch", action="store_true", help="实时打印 6 号舵机角度，Ctrl-C 停")
    ap.add_argument("--check", action="store_true", help="读一次并按阈值给结论")
    ap.add_argument("--all", action="store_true", help="顺便打印 6 个舵机")
    ap.add_argument("--hz", type=float, default=5.0, help="--watch 的采样频率")
    args = ap.parse_args()

    from servo_bridge import GRIPPER_ID, open_controller, read_all_deg, read_gripper_deg

    cal = load_thresholds()
    port = args.port or (cal.get("gripper") or {}).get("_port") or "COM12"
    c = open_controller(port=port)
    try:
        if args.watch:
            print(f"实时读 {GRIPPER_ID} 号（夹爪）角度，Ctrl-C 停")
            try:
                while True:
                    a = read_gripper_deg(c)
                    print(f"  {a if a is None else f'{a:6.1f}°'}"
                          + (f"   {classify(a, cal)}" if _is_calibrated(cal) else "   (还没标定)"),
                          flush=True)
                    time.sleep(1.0 / max(args.hz, 0.1))
            except KeyboardInterrupt:
                pass
            return 0

        a = read_gripper_deg(c)
        print(f"{GRIPPER_ID} 号（夹爪）= {a if a is None else f'{a:.1f}°'}")
        if _is_calibrated(cal):
            print(f"结论：{classify(a, cal)}")
        else:
            print("thresholds.json 还没标定，只报读数不报结论。先跑 calibrate_grasp.py")
        if args.all:
            for sid, ang in read_all_deg(c).items():
                print(f"  {sid}: {ang if ang is None else f'{ang:.1f}°'}")
        return 0
    finally:
        c.close()


def _is_calibrated(cal: dict) -> bool:
    g = cal.get("gripper") or {}
    return all(g.get(k) is not None
               for k in ("empty_max_deg", "held_min_deg", "drop_delta_deg"))


if __name__ == "__main__":
    sys.exit(_main())
