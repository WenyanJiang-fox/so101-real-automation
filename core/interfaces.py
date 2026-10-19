"""六个接口：机械臂、相机、感知、策略、判官、小模块。

要求只有一条：**主循环不许知道背后是谁**。它只认这六个，看不出背后是仿真还是
真机、是脚本还是大模型。想换一样，只动配置里那一行。

谁实现了哪个，去 impls/ 和 skills/ 里找；名字写在 configs/*.yaml 里。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .types import Round, Verdict


class Arm(ABC):
    """机械臂。只管动，不管看。

    仿真机械臂、真机械臂各写一个，主循环看的是同一个接口。

    **接口定在「爪子尖要去哪」，不是「六个关节多少度」。** 原因是仿真和真机
    收的东西不一样：仿真收的是「末端往哪挪一点」，关节角度它自己内部算；真机
    收的是关节角度。定在关节角度上的话，两边得互相迁就；定在爪子尖上，两边
    各用各的逆解，对外是同一种说法。
    """

    name: str

    # 每真正推进一步回调一次。主循环挂上它来录帧（记录 + 存视频）—— 后端只管在
    # 步进之后叫一下，不关心回调里干什么。挂在这儿而不是让主循环去查后端有没有
    # 这个属性：主循环不该知道背后是谁。
    on_step: Any = None

    @abstractmethod
    def tip(self) -> tuple[np.ndarray, np.ndarray]:
        """现在爪子尖在哪、朝哪个方向。方向给的是四元数 [w,x,y,z]。"""

    @abstractmethod
    def reset(self, seed: int) -> dict[str, Any]:
        """开一局。

        返回这个后端知道的现场信息：仿真里能返回「这一局要抓哪个方块」和任务描述，
        真机里没人知道真值、由人来给，就返回空字典。主循环按有啥用啥。
        """

    @abstractmethod
    def drive(self, to_position: np.ndarray, to_direction: np.ndarray | None = None,
              closing: np.ndarray | None = None, grip: float | None = None) -> None:
        """朝这个目标走**一小步**，然后换一拍。

        「走一小步」而不是「直接走到」：中间要留出每一步看反馈的余地。
        循环里反复调它，直到 tip() 到位。

        - to_position：爪子尖要去的点。
        - to_direction：爪子**指向**哪（工具 z 轴）。默认竖直向下。
        - closing：爪子**开合的方向**要朝哪。平行夹爪要贴在物体的一对侧面上，
          所以抓之前得让开合轴对上面法向；不对齐就是拿指尖去顶棱，夹不住。
        - grip：那一维夹爪指令。可以是常数，也可以是每步现算的函数（守位用）。
        """

    @abstractmethod
    def angles(self) -> np.ndarray:
        """现在六个关节各多少度。记录用，也用来跟真机舵机口径对齐。"""

    @abstractmethod
    def contacts(self) -> int:
        """现在碰到几个东西。真机上没有就返回 0。"""


class Camera(ABC):
    """相机。只管给画面，别的不管。

    仿真渲染、腕上相机、全局相机各写一个，名字不同而已。
    """

    name: str

    @abstractmethod
    def shot(self) -> np.ndarray:
        """要一帧。"""

    @abstractmethod
    def size(self) -> tuple[int, int]:
        """画面的宽高。策略要按这个对齐。"""

    def mount(self) -> dict | None:
        """这台相机在世界里的位姿 + 内参（相机标定 + 手眼标定的结果）。

        要从画面里反过来算出某个东西在世界的哪个方向，就得有这一份。真机上同样
        标得出来；标不出来的相机（比如俯视那种固定的）返回 None 就行。

        约定和 `core/vision.py` 一致：pos 是相机原点，mat 是 cam→world 的 3x3，
        fovy 是垂直视场角（度）。
        """
        return None


class Perceptor(ABC):
    """感知：看着一张画面和一句话，说「这话指的是哪一个、它在哪」。

    和相机分开是有意的 —— **相机只管给画面，认东西是另一件事**。同一个相机可以配
    「按工位坐标取最近的那块」，也可以配「拿一句话去挑」，机械臂和技能一个字不动。

    真机上以后接别的（更好的检测模型、别的相机组合）也实现这一个，名字不同而已。
    """

    name: str

    @abstractmethod
    def locate(self, instruction: str, *, guess=None) -> tuple[np.ndarray | None, dict]:
        """指令说的那个东西在哪。返回 (它中心的世界坐标 或 None, 这一眼的事实字典)。

        **认不出来就返回 None，不给兜底值。** 拿一个自己都不信的位置去带路、去转腕子，
        比当场承认没认出来更糟。事实字典里要带上「为什么没认出来」和这一步看到的东西，
        它会被打进日志。

        `guess`：外面知道的「大概在这一带」（标识点坐标之类）。**只用来定地面那个平面
        和兜底**，不是答案。
        """


class Policy(ABC):
    """策略。看到东西，出一串动作。

    脚本专家、SmolVLA、大模型都实现这个，区别只在肚子里装的是什么。

    **一次问一串，不是一步问一次**：模型本来就一次吐一串，机器人自己慢慢做。
    不想攒串的（比如脚本每步现算），返回一个长度为 1 的序列就行。
    返回空序列 = 这一轮我干完了。
    """

    name: str

    @abstractmethod
    def reset(self, info: dict[str, Any]) -> None:
        """开新一轮前清干净。info 里有这一轮的任务、目标、摆放。"""

    @abstractmethod
    def plan(self, obs: dict[str, Any]) -> list[np.ndarray]:
        """看一眼，出一串动作。obs 的键固定是 cam / arm / task / target。"""

    def deadline(self) -> tuple[float, str] | None:
        """到什么时候必须收工，以及到时候拿什么理由写上。None = 不管这一条。

        返回 (墙钟秒, 理由)。**时间归策略管、掐归主循环掐** —— 因为「一次尝试
        多少秒」只有策略知道（它才清楚自己试到第几次了），而「怎么把整轮停下来」
        只有主循环做得到（`arm.terminated` 每步都会被重置，得每步重新立）。
        不实现这一条的策略就是没有时间上限。
        """
        return None


class Judge(ABC):
    """判官。给一轮的完整记录，说成没成。

    仿真用真值判，真机用界线判，将来还能用视觉模型判 —— 都实现这个。
    """

    name: str

    @abstractmethod
    def judge(self, r: Round) -> Verdict:
        """下一句结论。"""


class Skill(ABC):
    """小模块。一个技能一个文件，只管干一件小事。

    抓一次、放一次、复位、安全检查，各是一个 Skill。

    **策略调它**：专家策略是人写死的调用顺序，大模型策略是模型当场决定顺序，
    两边用的是同一批小模块 —— 这就是「大模型参与控制」这件事的全部：
    换一个策略，机械臂、相机、判官一个字都不用改。
    """

    name: str

    @abstractmethod
    def run(self, arm: Arm, cams: dict[str, Camera], **kw: Any) -> dict[str, Any]:
        """干完这一件，返回它观察到的结果（给策略下一步做决定用）。"""
