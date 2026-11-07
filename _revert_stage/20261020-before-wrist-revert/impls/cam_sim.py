"""仿真相机：把机械臂那份观测里的画面切出来。

**相机不认识机械臂。** 它只拿着一个「能从哪儿要画面」的对象（`source.obs`）和一个
键名。真机上换成从摄像头抓帧，同样实现 `shot()` / `size()`，主循环和策略一个字不改。

两路画面：
    wrist_camera      腕上那个，跟着手走，看得见爪子和方块怎么接触
    overhead_camera   高处俯视，看得见整张工位

**渲染线程亲和**：画面由 `SimArm` 在它自己线程里渲染（`arm.observation()` 已经建好
离屏 renderer），这里只是取数组切片，不新建 renderer —— 跨线程建 GL 上下文在
Windows 上会全黑。
"""

from __future__ import annotations

import numpy as np

from core import vision
from core.interfaces import Camera
from core.registry import register


class _SimCamera(Camera):
    """从 source.obs[key] 取一帧。"""

    key = ""

    def __init__(self, source, key: str | None = None):
        self.source = source
        self.key = key or self.key

    def shot(self) -> np.ndarray:
        frame = self.source.obs[self.key]
        return np.asarray(frame)

    def size(self) -> tuple[int, int]:
        h, w = np.asarray(self.source.obs[self.key]).shape[:2]
        return int(w), int(h)


@register("camera", "sim_wrist")
class SimWristCamera(_SimCamera):
    """腕上相机。"""

    name = "sim_wrist"
    key = "wrist_camera"

    def mount(self) -> dict:
        """腕部相机的世界位姿 + 内参。

        **必须现读，不能缓存**：这个环境的腕部相机每开一局都会重新随机化俯仰角、
        安装位置和视场角，位姿和内参一局一个样。

        这份数据是「相机标定 + 手眼标定」的结果，真机上标出来的是同一份东西；
        和方块自己的位姿是两回事 —— 读它不违反「只认标识点坐标」。
        """
        u = self.source.u
        cid = int(u._wrist_cam_id)
        return {"pos": u.data.cam_xpos[cid].copy(),
                "mat": u.data.cam_xmat[cid].reshape(3, 3).copy(),
                "fovy": float(u.model.cam_fovy[cid])}

    def tip_px(self, arm=None) -> np.ndarray:
        """固定爪尖投在画面里那个像素 —— 仿真这一支的「点一次」。

        真机上这个数是操作员在画面上点出来的，存进 `real_wrist_tip_px.json`：相机和
        固定爪都长在同一个刚体上（nexus 的 `wrist_camera_mount_link="Fixed_Jaw"`），
        相对位置是死的，爪尖**永远是同一个像素**。仿真里不用点：机位和内参模型里就有，
        投一次就得到同一个常数。

        **但每一局 nexus 会重新随机机位、俯仰和视场角**
        （`base_env.py::_randomize_wrist_camera`），所以不像真机那样存一个跨局不变的值 ——
        这里按「机位 + 视场角 + 画面尺寸」缓存，这三样一变就重投一次。

        为什么要它：真机没有接触传感，只能用「爪尖那个像素离红区多少像素」判
        「让开够了没有」（`skills/pick_real.py::_set_clearance`）。仿真这一支要跑同一份
        代码，就得给出同一件事的读数。
        """
        if arm is None:
            raise RuntimeError("仿真腕机算爪尖像素要机械臂 —— 调 fixed_tip_pos() 用。")
        mount = self.mount()
        key = (tuple(np.round(np.asarray(mount["pos"], dtype=np.float64), 9)),
               round(float(mount["fovy"]), 9), self.size())
        if getattr(self, "_tip_px_key", None) == key:
            return self._tip_px_val
        w, h = self.size()
        k = vision.intrinsics(float(mount["fovy"]), int(w), int(h))
        px = vision.project(mount["pos"], mount["mat"], k,
                            np.asarray(arm.fixed_tip_pos(), dtype=np.float64))
        if px is None:
            raise RuntimeError("固定爪尖这一刻投不到画面上（在相机背后？）")
        self._tip_px_key = key
        self._tip_px_val = np.asarray(px, dtype=np.float64)
        return self._tip_px_val


@register("camera", "sim_overhead")
class SimOverheadCamera(_SimCamera):
    """俯视相机。位置固定，看得见整张工作台。"""

    name = "sim_overhead"
    key = "overhead_camera"

    def mount(self) -> dict:
        """俯视相机的世界位姿 + 内参 —— **直接问渲染器要**。

        和腕部那台不一样：这台是按 lookat/distance/elevation/azimuth 摆的自由相机，
        模型里没有它的 `cam_xpos`。但渲染这张图的那个 `Renderer` 手上就有一份现成的
        解算结果（`scene.camera[0]`），那就是像素真正由它算出来的那个机位。

        **不要自己拿 lookat/distance 推三角函数。** 试过了：按文档公式推出来的机位和
        渲染器实际用的差约 3°（整整 20~40 像素）。

        场景相机只给 pos / forward / up，没有现成的 3x3 矩阵，得自己拼：相机看向 -z，
        所以 z 轴 = 视线方向取反；x 轴向右 = y × z。视场角同理，从视锥的 top/near 反解
        （`fovy = 2·atan(top/near)`），这才是 mjr_render 真正用的那个。

        这套读法在 `measure_stereo.py::overhead_mount` 里对着标识点验过（残差 5 像素
        以内）。那边是量测脚本、要边读边打日志，所以留着自己那份；正式这条路走这里。
        """
        u = self.source.u
        rend = u._overhead_obs_renderer
        rend.update_scene(u.data, camera=u._overhead_obs_cam)
        c = rend.scene.camera[0]

        z = -np.asarray(c.forward, dtype=float)
        z = z / np.linalg.norm(z)
        x = np.cross(np.asarray(c.up, dtype=float), z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        near = float(c.frustum_near)
        fovy = float(np.degrees(2.0 * np.arctan(float(c.frustum_top) / near)))
        return {"pos": np.asarray(c.pos, dtype=float).copy(),
                "mat": np.column_stack([x, y, z]),
                "fovy": fovy}
