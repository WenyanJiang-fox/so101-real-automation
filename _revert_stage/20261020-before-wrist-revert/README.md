# SO-101 真机 harness（初代）

## 这个目录是什么

真机上两个判定模块 + 它们的标定脚本：

- **模块 A**：靠 6 号舵机（夹爪）的角度判断「夹住了没有」，以及抬起过程中「掉没掉」。
- **模块 B**：从一张腕部相机图里判断「夹住的是红 / 绿 / 蓝哪个」。

两个模块都是**纯阈值、纯白盒**的：没有神经网络，没有大模型，每一步的中间量都能打印出来手工核对。

**和 `G:\sim2real\so101-smolvla-sim`（仿真 + SmolVLA 训练那套）完全分开，互不引用。** 本目录不读那边的任何文件。

## 现在到哪一步了

- 设计定了，5 个脚本写完，**只做了语法检查**（`py_compile` 通过）。
- **一行都没在真机上跑过，逻辑路径也没执行过。**
- `thresholds.json` 全是 `null`。两个模块在没标定时会**直接抛异常拒绝工作**，这是故意的 —— 不猜阈值。

## 硬件前提

### 已核实（读的是 `G:\sim2real\mcp_robot\sources\servo_control_standalone.py`）

| 项 | 位置 | 内容 |
|---|---|---|
| 构造 | L289 | `ServoControllerStandalone(port_name="COM4", baudrate=1000000, timeout=0.1, debug=False)` |
| 发角度 | L456 | `set_servo_angle(servo_id, angle, speed=0)`，angle 0–180，speed=0 是最高速 |
| 读角度 | L499 | `get_servo_angle(servo_id)` → 0–180 或 `None` |
| 读原始值 | L557 | `read_position(servo_id)` → 0–4095 或 `None` |
| 批量读 | L543 / L668 | `read_positions(ids)` / `sync_read_positions(ids)` |
| 扭力 | L804 / L816 / L828 / L865 | `enable_torque` / `disable_torque` / `sync_enable_torque` / `sync_disable_torque` |
| 回原点 | L902 | `go_home(speed=0)` |
| 地址常量 | L285 | **只读了 `ADDR_PRESENT_POSITION = 56`。负载 / 电流寄存器没读** |

舵机 → 关节映射（全项目一致）：`1=shoulder_pan, 2=shoulder_lift, 3=elbow_flex, 4=wrist_flex, 5=wrist_roll, 6=gripper`。

### 未核实（接手的人必须先确认，别当成前提）

1. **真机相机怎么取帧。** 整个仓库里没找到真机 USB 相机的代码 —— `G:\sim2real\mcp_robot\sources\view_wrist_cam.py` 走的是 MuJoCo 的 `render_observation`，是仿真的。`color_check.py` 默认用 `cv2.VideoCapture(index)`，也可以 `--image` 喂文件。**真机走哪条路要你定**，定完可能要改 `open_camera`。
2. **夹爪角度的方向。** 「0° = 合到底（爪面约 30.9 mm），180° = 全开（约 55.9 mm）」来自用户实测（memory `so101-gripper-jaw-travel`），**本目录没有在真机上验证过**。第一次跑 `python grasp_check.py --watch` 时手动开合一下爪子，确认方向对不对。
3. **抓起来之后方块在不在腕机画面里。** 这决定模块 B 能不能用。腕机装在腕上看前方，方块夹在爪子正下方，很可能落在画面下缘，或者被爪子挡住。**先拍一张「夹住举起来」的图看看。** 如果看不到，就得改在抬到高位的瞬间读，或者调相机角度。
4. 方块尺寸、颜色饱和度、桌面背景色 —— 都影响阈值，全部靠标定步骤解决，不要在代码里猜。

## 文件

| 文件 | 干什么 |
|---|---|
| `G:\sim2real\so101-real-harness\servo_bridge.py` | 把 `mcp_robot` 的舵机驱动引进来，加中位数滤波的读法。**唯一一个知道驱动在哪的文件。** |
| `G:\sim2real\so101-real-harness\grasp_check.py` | 模块 A：`classify()` + `GraspMonitor` + 命令行 |
| `G:\sim2real\so101-real-harness\color_check.py` | 模块 B：HSV 分割 + 通道排序交叉验证 + 命令行 |
| `G:\sim2real\so101-real-harness\calibrate_grasp.py` | 跑真机采两组角度，算阈值写回 json |
| `G:\sim2real\so101-real-harness\calibrate_color.py` | 喂样本图，算 HSV 区间写回 json |
| `G:\sim2real\so101-real-harness\thresholds.json` | 唯一的阈值来源。标定产物写这里 |

## 接口

```python
# 模块 A
from grasp_check import load_thresholds, classify, GraspMonitor
cal = load_thresholds()
classify(angle_deg, cal) -> "held" | "empty" | "unclear" | "no-reading"

m = GraspMonitor(controller, cal)
m.mark_peak()      # 抬起之前调一次，记基准
m.poll()           # 抬起过程中每 0.2 秒左右调一次
m.dropped()        # True / False / None（信息不够）

# 模块 B
from color_check import which_color, color_ratios, channel_order_guess
which_color(frame_bgr, cal)        -> ("red", {"red":0.42,"green":0.03,"blue":0.01})
channel_order_guess(frame_bgr, cal) -> ("red", {"blue":..,"green":..,"red":..})

# 工具
from servo_bridge import open_controller, read_gripper_deg, close_gripper, open_gripper
```

## 怎么标定

用 `G:/anaconda/envs/sim2real/python.exe`。

**第 1 步：夹爪。** 会动真机（发合爪指令），跑之前确认爪里不会夹到线或桌面、串口没被 `mcp_server.py` 占着。

```powershell
cd G:\sim2real\so101-real-harness
G:/anaconda/envs/sim2real/python.exe calibrate_grasp.py --port COM12 --seconds 4
```

它会让你做两次采样：先空抓，再夹住方块举起来。**连续跑 2~3 遍**，看两次算出的阈值差多少。差得多说明夹爪重复性不够，把 `--margin` 调大。

**第 2 步：三个颜色。** 每个颜色采 20 张左右。

```powershell
G:/anaconda/envs/sim2real/python.exe calibrate_color.py --color red  --camera 0 --n 20
G:/anaconda/envs/sim2real/python.exe calibrate_color.py --color green --camera 0 --n 20
G:/anaconda/envs/sim2real/python.exe calibrate_color.py --color blue --camera 0 --n 20
```

或者先存图再标（推荐，可以反复试）：

```powershell
G:/anaconda/envs/sim2real/python.exe color_check.py --camera 0 --n 20 --save G:\sim2real\so101-real-harness\samples\red
G:/anaconda/envs/sim2real/python.exe calibrate_color.py --color red --images "G:\sim2real\so101-real-harness\samples\red\*.jpg"
```

**第 3 步：验证。**

```powershell
G:/anaconda/envs/sim2real/python.exe grasp_check.py --check
G:/anaconda/envs/sim2real/python.exe color_check.py --image G:\sim2real\so101-real-harness\samples\red\<随便一张>.jpg
```

标定期间的两个辅助开关：

- `python grasp_check.py --watch` —— 实时打印 6 号舵机角度。手动开合爪子，看空抓是不是接近 0°。
- `python color_check.py --camera 0 --test-hue` —— 打印前景像素的色相直方图前 6 名，用来手工定区间。

## 已知坑

- **位置舵机在持续发「闭合」指令时会一直往闭合方向挤**，读回来的角度会缓慢变化。所以不要拿单次读数和固定阈值比，用 `GraspMonitor` 的「峰值基准 + 变化量」判据。
- **单次 SCS 读会丢包**。所有读都走 `read_gripper_deg()` 的中位数滤波，别直接调 `get_servo_angle`。
- **红色跨色相 0 点**，所以 `color_check` 和 `calibrate_color` 都做了「色相平移到 -90..89」的处理。标定红色时出来两段区间是正常的。
- **光照色温会把色相整体推偏**。`channel_order_guess()` 不看色相、只看 B/G/R 哪个通道强，换灯不用重标。它和 HSV 结论不一致时，`_report` 会打印「这帧别信」。
- **负载判据现在拿不到**。驱动只读了位置寄存器（L285）。想加「使多大劲」这个物理量，得自己补一个寄存器读 —— 但那是改 `mcp_robot` 的驱动，会影响真机 MCP server，先问用户。
- 模块 B 的分母是**前景像素数**（够彩的像素），不是全图 —— 这样方块离得远近不影响占比。前景为 0 时返回全 0，不返回垃圾值。

## 下一步（按顺序）

1. 确认真机相机取帧走哪条路，改 `open_camera` 或固定用 `--image`。
2. 拍一张「夹住方块举起来」的图，确认方块在腕机画面里。**这一步不通过，模块 B 就白做。**
3. 跑 `calibrate_grasp.py` 2~3 遍，看阈值稳定性。
4. 三个颜色各采 20 张，跑 `calibrate_color.py`。
5. `--check` 和 `--image` 各验证一遍。
6. 以上都过了，才是把它接进 harness 主循环（每轮：夹爪判有 → 颜色判色 → 抬起过程判掉没掉 → 写一行结果）。

## 不要做的事

- **不要把阈值写死在代码里。** 唯一的阈值来源是 `thresholds.json`。
- **不要用 `lerobot-annotate`。** 它走 VLM 做语言/VQA 标注，是黑盒，和本目录的路线相反。
- **不要改 `G:\sim2real\mcp_robot\sources\servo_control_standalone.py`。** 那是真机 MCP server 正在用的驱动。

## 和 LeRobot 官方那套的关系

LeRobot 0.6 自带 `lerobot-rollout`，有 `sentry` / `highlight` / `dagger` 等策略（源码在 `G:\anaconda\envs\sim2real\Lib\site-packages\lerobot\scripts\lerobot_rollout.py`），现成、能用。但它判成功走的是**学习式的 reward classifier**（ResNet-10 二分类，`configuration_classifier.py`，`num_classes: 2`），是黑盒。本目录是它之外的平行方案：判定用阈值，不用模型。

两者不冲突 —— `lerobot-rollout` 负责「让策略跑、录数据」，本目录负责「判这一轮成没成、夹的是什么色」。要接的话，本目录的模块挂到 rollout 的每轮首尾即可。
