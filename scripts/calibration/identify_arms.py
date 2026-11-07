"""认领端口：哪根线插的是主臂、哪根是从臂。**只读，一个寄存器都不写。**

为什么要有这个：Windows 给串口编号跟 USB 转串口模块有没有唯一序列号有关 ——
没有序列号的（便宜的 CH340 常见）会把每个物理口当成新设备，换个 USB 口就换个
COM 号。于是「程序里写死 COM12」这件事会变成「今天开到的可能是另一条臂」。
这个脚本把端口号从判断里拿掉：挨个口读回六个舵机的标定寄存器，跟两份标定文件
比，跟哪份对得上就是哪条臂。

判据用的是标定寄存器，不是舵机 ID —— 两条臂的 ID 都是 1~6，同一个命名空间，
光看 ID 分不出来。标定时 LeRobot 把 `Homing_Offset` / `Min_Position_Limit` /
`Max_Position_Limit` 写进了舵机自己的寄存器（`feetech.write_calibration`），
两条臂这 18 个数各不相同，所以读回来一比就认得出。

读寄存器走的是 `servo_bridge`（也就是 mcp_robot 那个真机驱动），不是 lerobot 的
总线 —— lerobot 的 `FeetechMotorsBus` 要 `scservo_sdk`，这个环境里没装，为了认领
端口去引一个新依赖不划算。

**安全约定（这个文件必须一直守住）**：
  * 只调 `read_register`，不调任何 `write`；全程不使能力矩。
  * 收尾只 `close()` 串口 —— 不碰 lerobot 那套 `disconnect(disable_torque=True)`，
    那一句会往寄存器里写。这个脚本一个字节都不写。
  * 认不出来就说认不出来，不猜、不给兜底值 —— 拿个不信的结论去发运动指令，
    比当场停下糟得多。

标定文件在 `assets/lerobot_calibration/`：
    little_little_pig.json     = 从臂（follower，原本在 robots/so_follower）
    little_little_sister.json  = 主臂（leader，原本在 teleoperators/so_leader）

用法（不给参数就是只列端口、只认领）：
    G:/anaconda/envs/sim2real/python.exe identify_arms.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import servo_bridge  # noqa: E402

HERE = Path(__file__).resolve().parents[2]
CALIB_DIR = HERE / "assets" / "lerobot_calibration"

# 标定文件 -> 这个文件说的是哪条臂。名字按文件来，不按端口来 —— 端口号会变。
ARMS = {
    "little_little_pig.json": "从臂 follower（robots/so_follower）",
    "little_little_sister.json": "主臂 leader（teleoperators/so_leader）",
}

# 六个舵机的规范顺序（全项目一致，见 servo_bridge.SERVO_TO_JOINT）
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper")

# 要比的三个寄存器。`Present_Position` 不参与 —— 那个读数取决于臂现在摆成什么样，
# 跟「这是哪条臂」无关。
REGISTERS = {
    "homing_offset": "Homing_Offset",
    "range_min": "Min_Position_Limit",
    "range_max": "Max_Position_Limit",
}

BAUD = 1_000_000          # 两条臂都是 1Mbps（SO-101 官方口径）

# 认领的门槛：18 个数（6 个舵机 × 3 个寄存器）里对得上一半以上，才敢说「是这条臂」。
MATCH_MIN = 9


def list_ports() -> list:
    """Windows 上现有的串口，连描述和序列号一起列出来。

    有没有 `serial_number` 直接决定「换个 USB 口会不会换号」—— 顺带打出来，
    看一眼就知道要不要靠这个脚本认领。
    """
    from serial.tools import list_ports as lp

    return list(lp.comports())


def is_usb_serial(port) -> bool:
    """这个口是不是 USB 转串口设备。

    两条臂都是靠 USB 转串口模块接进来的（CH340 / CP210x 那种），hwid 里带 "USB"。
    蓝牙那几个「标准串行 (COMx)」不带 —— 而且**打开它们会卡**：实测每个蓝牙口
    `serial.Serial()` 要等十几秒才超时失败，其中 COM6 那种挂在 60 秒以上没回来。
    所以认领之前先按这个筛一遍，别去开蓝牙口。
    """
    return "USB" in (port.hwid or "").upper()


def load_refs() -> dict:
    """两份标定文件读进来。缺哪份就少一条对照，不编。"""
    refs = {}
    for fname, who in ARMS.items():
        path = CALIB_DIR / fname
        if not path.exists():
            print(f"  ! 标定文件不在：{path} —— 这条臂没法对照")
            continue
        refs[who] = json.loads(path.read_text(encoding="utf-8"))
    return refs


def read_arm(port: str) -> dict | None:
    """读回这个口上六个舵机的标定寄存器。**只读。**

    连不上或者一个舵机都没回话，返回 None。个别舵机读不到就把那一个记成 None，
    不因为一个读不到就放弃整条臂 —— 认领阶段要的是「读得到多少算多少」。
    """
    try:
        ctrl = servo_bridge.open_controller(port, baudrate=BAUD)
    except Exception as exc:                 # 端口被占 / 不是这套设备 / 没通电
        print(f"  · 打不开或连不上（{type(exc).__name__}: {exc}）")
        return None

    try:
        raw = servo_bridge.read_calibration_registers(ctrl)
    finally:
        ctrl.close()                         # 只关串口，不写寄存器、不动力矩

    # 驱动是按舵机 ID 1~6 给的，转成关节名。顺序和 SERVO_TO_JOINT 一致。
    out = {name: (raw.get(i + 1) or {}) for i, name in enumerate(MOTORS)}

    if all(v is None for vals in out.values() for v in vals.values()):
        print("  · 这个口上没读到任何舵机（没插臂、臂没通电，或者端口被别的程序占着）")
        return None
    return out


def compare(read_back: dict, ref: dict) -> tuple[list, list]:
    """拿读回来的 18 个数跟一份标定文件比。返回（对上的舵机, 对不上的明细）。"""
    same, diff = [], []
    for name in MOTORS:
        got = read_back.get(name) or {}
        want = ref.get(name) or {}
        bad = {k: (got.get(k), want.get(k)) for k in REGISTERS
               if got.get(k) != want.get(k)}
        (diff.append((name, bad)) if bad else same.append(name))
    return same, diff


def identify(port: str, refs: dict, read_back: dict) -> str:
    """给这个口下一个结论。认不出就明说。"""
    verdicts = []
    for who, ref in refs.items():
        same, diff = compare(read_back, ref)
        # 对上几个寄存器（18 个里）。整块对上的算 3 个，对不上的那块里也有对上几个的。
        n = len(same) * 3 + sum(3 - len(bad) for _, bad in diff)
        verdicts.append((n, who, same, diff))

    verdicts.sort(reverse=True)
    if not verdicts:
        return "没法判（一份标定文件都没读到）"

    n_top, who_top, same_top, diff_top = verdicts[0]
    if n_top < MATCH_MIN:
        return (f"**认不出**（最接近的也只对上一半以下：{who_top} "
                f"{len(same_top)}/6 个舵机）—— 停下来核对，别发指令")
    if len(verdicts) > 1 and verdicts[1][0] >= MATCH_MIN:
        return (f"**认不出**：跟两份标定都对得上一半以上（{who_top} 和 "
                f"{verdicts[1][1]}）—— 两份标定太像，靠这个分不开")

    line = f"{who_top}（6 个舵机里对上 {len(same_top)} 个）"
    if diff_top:
        detail = "；".join(
            f"{name}:" + ",".join(f"{k} 读到 {g} 该是 {w}" for k, (g, w) in bad.items())
            for name, bad in diff_top)
        line += f"  ⚠ 有对不上的，逐个列出：{detail}"
    return line


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 72)
    print("认领端口（只读，不写任何寄存器、不动臂）")
    print("=" * 72)

    ports = list_ports()
    if not ports:
        print("\n一个串口都没有 —— 线没插上，或者驱动没装。")
        return 1

    print(f"\n串口 {len(ports)} 个：")
    for p in ports:
        ser = p.serial_number or "（没有序列号 —— 换 USB 口就会换 COM 号）"
        print(f"  {p.device:<8} {p.description}   {'' if is_usb_serial(p) else '[跳过：不是 USB 串口]'}")
        print(f"           hwid={p.hwid}")
        print(f"           序列号={ser}")

    candidates = [p for p in ports if is_usb_serial(p)]
    if not candidates:
        print("\n里面没有 USB 串口设备 —— 臂的 USB 线没插上（蓝牙口不是臂）。")
        return 1

    refs = load_refs()
    if not refs:
        print("\n一份标定文件都没有，没法认领。")
        return 1

    print("\n" + "-" * 72)
    print("挨个口读回标定寄存器，跟两份标定文件比：")
    print("-" * 72)

    found = []
    for p in candidates:
        print(f"\n---- {p.device} ----")
        read_back = read_arm(p.device)
        if read_back is None:
            continue
        for name in MOTORS:
            v = read_back[name]
            print(f"    {name:<14} 零点 {v['homing_offset']}  "
                  f"行程 {v['range_min']}~{v['range_max']}")
        for who, ref in refs.items():
            same, diff = compare(read_back, ref)
            print(f"    对 {who}：{len(same)}/6 个舵机一模一样"
                  + ("" if not diff else f"，对不上的是 {[n for n, _ in diff]}"))
        who = identify(p.device, refs, read_back)
        print(f"    >> {p.device} = {who}")
        found.append((p.device, who))

    print("\n" + "=" * 72)
    if not found:
        print("一个口都没认出来。检查：臂通电了吗？端口被 mcp_server.py 占了吗？")
    else:
        print("结论：")
        for dev, who in found:
            print(f"  {dev}  ->  {who}")
        print("\n注：认出来的结论只当这一次用 —— 下次插拔前重新跑一遍，别把结论写死。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
