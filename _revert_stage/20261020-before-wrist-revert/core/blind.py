"""盲测：桌上放一件**随机**的东西，只告诉它颜色，形状和尺寸一个字不说。

**为什么要这样考。** 之前每一局形状和尺寸都是清单里写死的：仿真照它建场景、感知
照它建词表、策略照它挑夹法（`core.shapes.from_spec`）。那是「已知答案去抓」，测的
是轨迹和夹法，不是「看不看得出来」。这一支把答案收走：随机抽一件摆在桌上，真值只
留给**建场景的仿真**和**打分的判官**，策略那一路只拿得到一句话 —— 颜色。

**怎么保证收走了。** 三处一起断：
    仿真      照真值建（`_nexus_object` 认 width/wall 那些键）
    判分      照真值算落地高度（`core.geom` 那份清单还是真值）
    策略      只认 `specs_for_policy()` 抹过的那份：形状是 `unknown`、没有尺寸；
              要形状得走 `shape_of()`，它读的是**相机估出来的那一份**（`ESTIMATE`）。
一句话：真值在仿真和判官那一侧，估计值在策略那一侧，两边不共用一份数。

**抽的范围**（2026-10-13 用户定）：
    正方体      边长 20~40mm
    长方体      两条边 20~40mm，另一条边 20~100mm
    实心圆柱    直径 20~40mm，高 = 直径（仿真这支造不出高≠直径的圆柱）
    空心圆柱    同上，壁厚 2~3mm
尺寸一律**毫米**，跟 nexus 和 `core.shapes` 的口径一致。
"""

from __future__ import annotations

from core import shapes

# ---- 抽的范围（毫米）----
SIDE_MIN, SIDE_MAX = 20.0, 40.0      # 正方体的边长；长方体那两条短边
LONG_MIN, LONG_MAX = 20.0, 100.0     # 长方体那条长边
WALL_MIN, WALL_MAX = 2.0, 3.0        # 空心圆柱的壁厚

KINDS = ("cube", "box", "cylinder", "cup")

# **橙色不抽**：机械臂自己就是橙色，`vision.color_mask` 直接报错。黑色也不抽 ——
# 腕部相机看的是深色桌面，黑的东西会跟桌面连成一片，不是「看不清」而是「看错了」。
BLIND_COLORS = ("red", "yellow", "green", "blue", "purple", "white")

# 还没看一眼的时候，形状这一路先按什么算。**不能用真值**，也不能太小：进场、
# 纠位那两步按它定目标和夹取高度，估得太离谱会走到一个荒唐的地方去。
PLACEHOLDER_MM = 30.0
PLACEHOLDER_Z = PLACEHOLDER_MM / 2000.0    # 上面那个数换成「中心离地多高」（米）

ON = False            # 盲测开没开。主循环按命令行/配置置它
TRUTH: dict | None = None      # 这一局的真值（仿真和判官读）
ESTIMATE: dict | None = None   # 策略看到的那一份（相机估出来的）


def sample(rng, colors=BLIND_COLORS) -> dict:
    """抽一件。返回一行清单，键跟 `core.shapes.from_spec` / `_nexus_object` 对得上。"""
    kind = str(rng.choice(KINDS))
    color = str(rng.choice(list(colors)))
    if kind == "cube":
        side = float(rng.uniform(SIDE_MIN, SIDE_MAX))
        return {"color": color, "shape": "cube", "side_mm": side,
                "label": f"正方体 边长{side:.0f}mm"}
    if kind == "box":
        # 长边**躺着**（横在桌面上）。立起来 100mm 高的柱子固然也能夹，但这一版
        # 桌子上的东西一律躺着放（`core/shapes` 开头那句），不另开一种姿势。
        long_ = float(rng.uniform(LONG_MIN, LONG_MAX))
        w = float(rng.uniform(SIDE_MIN, SIDE_MAX))
        h = float(rng.uniform(SIDE_MIN, SIDE_MAX))
        return {"color": color, "shape": "box", "size_mm": [long_, w, h],
                "label": f"长方体 长{long_:.0f}×宽{w:.0f}×高{h:.0f}mm"}
    d = float(rng.uniform(SIDE_MIN, SIDE_MAX))
    if kind == "cup":
        wall = float(rng.uniform(WALL_MIN, WALL_MAX))
        return {"color": color, "shape": "cylinder", "diameter_mm": d, "height_mm": d,
                "cup": True, "wall_mm": wall, "hollow": True,
                "label": f"空心圆柱 直径{d:.0f}mm 壁厚{wall:.1f}mm"}
    return {"color": color, "shape": "cylinder", "diameter_mm": d, "height_mm": d,
            "label": f"实心圆柱 直径{d:.0f}mm"}


def set_truth(spec: dict) -> None:
    """这一局的真值。仿真照它建场景，判官照它算落地高度。"""
    global TRUTH, ESTIMATE
    TRUTH = dict(spec)
    ESTIMATE = None            # 换一局，上一次看出来的那份作废


def specs_for_policy(specs) -> list[dict]:
    """给策略那一侧看的清单：**只留颜色**，形状和尺寸都抹掉。

    感知拿它建词表（所以指令里的颜色词能对上）、`_want_index` 拿它比形状词 —— 抹成
    `unknown` 之后，「白圆柱」这种词在指令里就再也对不上了，只有颜色能对上。这正是
    要的口径：指令只给颜色。
    """
    if not ON:
        return list(specs)
    return [{"color": str(s.get("color", "red")), "shape": "unknown"} for s in specs]


def shape_of(arm, index: int):
    """策略这一侧「这件东西是什么形状」。盲测下**读相机估出来的那一份**。

    还没看一眼的时候（进场那一步要用它定 z），给一个中性的 30mm 方块 —— 那是
    「我什么都不知道」的位置，不是答案。估出来之后 `run` 会重新问一次。
    """
    if not ON:
        get = getattr(arm, "object_shape", None)
        return get(index) if get is not None else shapes.Box(PLACEHOLDER_Z)
    if ESTIMATE is None:
        return shapes.Box(PLACEHOLDER_Z)
    return shapes.from_spec(ESTIMATE)


def note(est: dict | None) -> None:
    """把相机估出来的这一份记下来。下次问形状就是它。"""
    global ESTIMATE
    ESTIMATE = None if est is None else dict(est)


def task_sentence(color: str) -> str:
    """喂给策略的那句话。**只有颜色，没有形状也没有尺寸。**"""
    return f"Pick up the {color} object."
