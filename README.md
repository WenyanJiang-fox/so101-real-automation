# SO-101 真机抓放：脚本技能 + π0.5 末端策略 + Qwen3-VL 判官

在 SO-101 六轴机械臂上实现真机抓取—放置：脚本负责几何对轴，π0.5 LoRA 接管末段，Qwen3-VL 在合爪前做否决。无人值守采集 6 小时产出 120 条、有效率 68%；加入 VLA 末段后单趟平均 100 s → 43 s，抓取成功率 83%。

## 1. 项目简介

这是一套面向真机的抓取自动化与数据采集系统，仿真仅用于对照验证。系统将一次抓取拆分为五个可替换模块：机械臂、相机、策略、判官、技能；主循环只依赖接口，替换实现仅需修改 `configs/` 中的一行配置。

核心设计原则是「脚本负责确定性环节，模型只接管末段」：

- 脚本技能负责可解释的部分：回位、悬停、转腕对轴、下探、合爪、抬升、放置。
- π0.5 只接最后一段（横向对齐、合爪）；另有提前接管变体，下探也交由模型执行。模型只输出动作，成败判定仍由脚本完成。
- Qwen3-VL 只做合爪前的否决，不指挥机械臂，也不参与轮末成败判定。
- 失败按失败码归因，三档策略按码路由；失败样本回流为 VLM 判官的训练对。

同一套接口也可在 MuJoCo 中运行（`run_sim.py`），用于对照脚本策略与仿真真值判官。

## 2. 核心结果

| 维度 | 结果 |
|---|---|
| 数据采集 | 4 Hz、腕 + 侧双相机；无人值守连续采集 6 小时，产出 120 条，有效率 68% |
| 失败归因 | 7 类失败码：`off_axis` / `too_low` / `too_high` / `no_gap` / `slip` / `descend_fail` / `other` |
| 末端策略 | π0.5 LoRA（仅 adapter，`action_horizon=10`） |
| 端到端 | 脚本对轴下探 + π0.5 接管末段（横向对齐 / 合爪）：单趟平均 100 s → 43 s，抓取成功率 83% |
| 策略路由 | 三档 `oneshot` / `full` / `vla_end`，按失败码自动路由 |
| VLM 判官 | Qwen3-VL LoRA r16/α32；6 类失败场景 300 条成对训练，280 条评测二分类准确率 92.9% |

表中的端到端时间、成功率、采集条数和有效率来自真机实测记录；其余数字可在代码与配置中核对。有效率 = 判定为 `held` 并写入数据集的集数占全部采集集数的比例；未抓取成功的整集丢弃。

## 3. 方法与流水线

### 3.1 五接口

主循环依赖五个模块：臂、相机、策略、判官、技能。感知也单独成接口（`Perceptor`，随相机模块装配），因此 `core/interfaces.py` 实际有 6 个抽象基类。

| 模块 | 职责 | 现有实现 |
|---|---|---|
| 机械臂 `Arm` | 运动：读取 TCP、朝目标走一小步、读取关节角/接触 | `impls/arm_sim.py`、`real/arm.py` |
| 相机 `Camera` / 感知 `Perceptor` | 视觉：返回图像、尺寸、机位/内参；感知：按指令定位目标并输出世界坐标 | `impls/cam_sim.py`、`real/cameras.py`、`real/side_cam.py`；`impls/perception.py`、`real/perception.py` |
| 策略 `Policy` | 决策：一次输出一串动作 | `impls/policy_expert.py`、`impls/policy_real.py` |
| 判官 `Judge` | 轮末判定成败 | `impls/judge_truth.py`、`impls/judge_grasp.py`、`impls/judge_held.py` |
| 技能 `Skill` | 执行单一操作：抓取/放置/回位/安全检查 | `skills/pick.py`、`skills/pick_real/` |

装配由 `core/registry.py` 的「名字 → 类」表和 `configs/*.yaml` 完成；替换实现只需修改配置中的对应行。

### 3.2 一次抓取的执行顺序

```
腕相机 / 侧相机
      │
      ▼
感知：HSV 找目标色块 → 矩形度过滤 → 内参 + 手眼反投影 → 中心 / 朝向 / 尺寸
      │
      ▼
脚本技能：回位 → 悬停 → 转腕对轴 → 竖直下探 → 合爪 → 抬升 → 放置
      │                                   │
      │                                   └─ vla_end 变体：从下探起交给 π0.5
      ▼
判官：6 号舵机角度（held / empty） + 轮末 Judge
```

1. 回初始位：`home_real` 将六轴回到 `real_home.RAW_HOME`，腕相机正对桌面。
2. 采集一帧：`real/perception.py` 用 HSV 定位目标色块，用矩形填充率过滤同色杂物；`core/vision.py` 用相机内参和手眼标定将像素反推到基座系，估计中心、朝向和尺寸。
3. 进场与转腕：`skills/pick_real/motion.py` 运动到悬停点；`align.py` 将合爪轴（工具 x 轴）转到方块侧面法向。
4. 对轴下探：`rulers.py` 同时检测画面缝宽和 3D 离轴；`decide.py` 用上一趟实测的下探过衡量修正本趟终点；`motion.py` 执行竖直下探。`vla_end` 提前接管变体将这一步交给 π0.5。
5. 合爪与夹持判定：`grasp.py` 合爪至压不动；`grasp_check.py` 读取 6 号舵机角度，与本趟现测的空抓基准比较，判定 `held` / `empty`。`vla_end` 档由模型合爪，但判据仍由此处执行。
6. 抬升与放置：抬升到搬运高度；抓取成功后按坐标放到随机点，为下一集更换位置，再回初始位。
7. 安全闸：每一步自动走位前后均经过 `real/safety.py`；越界立即置 `terminated`，该集作废。

### 3.3 三档策略与失败码路由

`configs/strategies.json` 是策略的唯一来源。三档并非三份技能，而是同一份 `pick_real` 的三组参数覆盖：

| 档 | 脚本行为 | 典型使用 |
|---|---|---|
| `oneshot` | 固定落点抓取：跳过合爪前微调，爪尖直接到计算落点 | 默认；`too_low` |
| `full` | 视觉伺服微调：画面雅可比微调 + 中心 −1 mm | `off_axis` / `no_gap` / `slip` / `too_high` / `descend_fail` |
| `vla_end` | 提前接管变体：脚本只到「腕部对上方块侧面」，下探、横向对齐、合爪交给 π0.5 | 需要 VLA 末段时 |

失败码到下一档的映射：

| 失败码 | 典型含义 | 下一档 |
|---|---|---|
| `off_axis` | 合爪轴偏离 | `full` |
| `too_low` | 下探过深 | `oneshot` |
| `too_high` | 下探过浅 | `full` |
| `no_gap` | 爪面没有缝 | `full` |
| `slip` | 未夹持 / 抬升后滑落 | `full`；同一档先按实测再下探一次 |
| `descend_fail` | 下探未执行 | `full` |
| `other` | 其他 | 按阶梯取未试过的一档；都试过就停 |

`no_detect`（未识别到目标）是停止条件，不进入重试。路由代码在 `skills/pick_real/strategy.py::code()` / `next_after()`；面板连续抓取时用它切换档位。

无人值守采集另有 `ppo/policy.py` 的重试账本：`baseline` / `no_measure` / `full` 三组参数，每次只更换一个变量，全部试过后回到实测开口最大的一档。它只接收技能交回的实测数，不直接操作硬件。

### 3.4 π0.5 末端策略

π0.5 只执行末段。观测为腕相机与侧相机两路图像，动作为 6 个关节的目标角，一次输出 10 步的 action chunk（`action_horizon=10`）。训练只更新 LoRA adapter（`gemma_2b_lora` + `gemma_300m_lora`），基座权重冻结。

有两个接入点：

- 策略表中的 `vla_end`（提前接管变体）：脚本完成「识别方块 → 移动到目标上方 → 腕部对上方块侧面」后接入，模型执行下探、横向对齐、合爪。
- 环境变量 `PICK_VLA_END=1`（旧入口）：脚本对轴下探到底之后接入，模型执行横向纠偏、最后一段竖直下探、合爪。

两条路径都由 `_close_and_lift(already_closed=True)` 判定是否夹持成功。客户端在 `_pi05_run.py` 和 `_pi05_link.py`；`skills/pick_real/vla_end.py` 负责将技能接入模型。策略服务运行在外部 GPU 上，本机只做端口转发。

### 3.5 Qwen3-VL 判官

它不是轮末的 `Judge`，而是合爪前的一次否决闸：下探到位、合爪之前，将当前帧和一张下探前参考帧交给 Qwen3-VL，询问「方块是否仍在两爪之间」。同一组图像查询两次，结果一致才生效；仅 `no` 触发「抬回悬停点、重新下探」，`yes` / `unsure` / 超时一律放行。重下之后的采集帧只记录，不再触发动作。

它不修改档位、不输出偏移、不直接控制关节；一集仍然只合爪一次。代码在 `ppo/vlm.py::judge`，接线在 `skills/pick_real/vlm_gate.py`；采集脚本默认启用，`--no-vlm-judge` 可关闭。

### 3.6 无人值守采集与安全闸

`collect_real_dataset.py` 的单集流程为：回位（不录制）→ `pick_real` 执行一趟（录制）→ 判定 `held` 后才写入数据集。未抓取成功的整集丢弃，但失败码、原图、实测数保留在 `out/` 供归因。抓取成功后技能自动将方块摆到随机点，为下一集更换位置。

停止仅通过 `out/STOP` 文件触发，避免强制终止进程导致 parquet 截断。`out/` 每次启动会按天数清理。

安全闸阈值按实测确定：单步单关节 45°、TCP 包络半径 300 mm / z −30~200 mm、2 号肩抬 ≥ −45°、3 号肘两臂夹角 ≥ 75°。触发即停止，该集作废；现场交 VLM 存档，随后回初始位继续下一集。

## 4. 目录结构

```
仓库根目录/
├─ core/                    接口、记录/注册表、几何/视觉/轨迹
│  ├─ interfaces.py         Arm / Camera / Perceptor / Policy / Judge / Skill
│  ├─ types.py              Round / Verdict（一轮唯一记录形状）
│  ├─ registry.py           名字 → 实现
│  ├─ follow.py             笛卡尔跟踪
│  ├─ trajectory.py         toppra 时间最优轨迹
│  ├─ geom.py / shapes.py   工位几何、形状表
│  ├─ vision.py             从腕相机量位置/朝向
│  ├─ params.py             速度/高度/夹持力参数
│  ├─ round_state.py        一轮内技能间共享状态
│  └─ blind.py              仿真盲测
├─ impls/                   接口的仿真/真机实现
│  ├─ arm_sim.py / cam_sim.py
│  ├─ arm_sim_real.py       真机代码跑在仿真臂上
│  ├─ perception.py
│  ├─ policy_expert.py / policy_real.py
│  └─ judge_truth.py / judge_grasp.py / judge_held.py
├─ skills/                  小模块
│  ├─ pick.py / place.py / home.py / sweep.py   仿真技能
│  └─ pick_real/            真机抓取技能：编排、运动、对齐、合爪、失败码、策略
├─ panel/                   tkinter 真机面板：画面、按钮、断点、连续采集
├─ real/                    真机硬件层：串口臂、相机、感知、录制、安全闸、侧相机
├─ ppo/                     重试账本、定位守卫、VLM 诊断/判官
├─ configs/                 装配配置（default / real / sim_real）+ 三档策略表
├─ assets/                  相机标定、标定板
├─ docs/                    接口说明、调试笔记、演示媒体
├─ scripts/
│  ├─ probes/               IK/FK、坐标、夹爪行程等一次性探针
│  ├─ calibration/          相机内参/手眼、颜色、夹爪标定与检查
│  ├─ experiments/          感知与语言条件实验
│  ├─ training/             π0.5 末段与 Qwen3-VL 判官的数据构建、配置、训练
│  ├─ data/                 数据集后处理、失败扫描、视频导出
│  ├─ ops/                  现场运维与单次量测脚本
│  └─ legacy/               早期仿真 harness 原型（已被 run_sim.py 取代）
├─ datasets/                LeRobot 数据集（按环境生成，不随仓库发布）
├─ collect_real_dataset.py  真机无人值守采集入口
├─ run_sim.py               MuJoCo 主循环入口
├─ _pi05_run.py             π0.5 推理入口
├─ _pi05_link.py            π0.5 客户端接线
├─ _vla_end_try.py          脚本 + π0.5 末段的一趟演示
├─ servo_bridge.py          Feetech 舵机驱动桥
├─ real_home.py             真机初始点位
├─ grasp_check.py           6 号舵机夹持判据
├─ thresholds.json          HSV / 夹爪阈值
└─ requirements.txt
```

## 5. 环境依赖

### 软件

开发环境是 Windows 10/11 + Python 3.12。主要依赖：

- `numpy`、`opencv-python`、`scipy`、`toppra`、`mujoco`、`PyYAML`、`Pillow`
- `lerobot==0.6.1`（数据集读写）
- `websockets` + `openpi_client`（π0.5 客户端）
- `pyserial`（Feetech 总线，经外部驱动）
- `so101_nexus`（运动学、归一化和 MuJoCo 抓取环境）
- `tkinter`（Windows Python 自带，面板用）
- `pandas`、`pyav`（只在切数据集/统计时用）
- `paramiko`（只在 SSH 端口转发脚本里用）
- `torch` / `transformers` / `ms-swift`（只在 Qwen3-VL 训练/部署时用）

仓库的 `requirements.txt` 列出主循环所需依赖；`openpi_client`、`so101_nexus`、Qwen3-VL 权重不在 PyPI，按下一节的说明单独准备。

### 外部仓库与模型

以下都不在本仓库内，路径需要按环境配置：

- `SO-ARM100` 的 MuJoCo 模型 `so101_new_calib.xml`：`real/arm.py` 用它做正解。
- Feetech 舵机驱动 `servo_control_standalone.py`：`servo_bridge.py` 把它加进 `sys.path`。
- `so101_nexus`：仿真环境和运动学工具。
- openpi：π0.5 策略服务；仓库只含客户端。
- Qwen3-VL + ms-swift：判官训练和 OpenAI 兼容部署。

### 硬件

- SO-101 从臂，6 个 Feetech SCS 舵机，USB 串口总线。
- 腕部 USB 相机（原生 1920×1080；数据中存 960×540）。
- 可选侧视 USB 相机：只进数据、不参与抓取；双相机数据集需要它。
- π0.5 和 Qwen3-VL 需要外部 GPU；本地脚本和面板不需要 GPU。

## 6. 如何运行

以下命令都在仓库根目录执行；`python` 指装有上述依赖的解释器。

### 6.1 仿真自检

```bash
python run_sim.py --config configs/default.yaml --rounds 2 --seed 0
```

默认配置用 `siglip2` 感知；没有模型权重时，把 `configs/default.yaml` 的 `perception` 改成 `nearest`（不依赖模型）。仿真需要 `so101_nexus` 和 `SO-ARM100` 模型路径可用。

### 6.2 真机面板

```bash
python -m panel
```

面板是一个原生 tkinter 窗口：左侧腕相机画面、右侧按钮和读数。F1 跑一趟，F6 开关连续采集。串口、相机序号、桌面高度、感知偏移都在 `configs/real.yaml` 里，按机器修改。

```bash
python -m panel --collect
```

开窗即开始自动采集。

### 6.3 无人值守采集

```bash
python collect_real_dataset.py --episodes 10 --root datasets/so101_real_grasp_2cam_4hz --repo-id so101_real/pick_grasp_2cam
```

`--episodes 0` 表示不设上限；`--dry-run` 只跑不写盘；`--no-vlm-judge` 关闭合爪前的 Qwen 判官。

### 6.4 π0.5 末段

先启动外部策略服务并做本机端口转发，然后：

```bash
python _pi05_run.py --steps 200
```

动作块按 `--ctl-hz` 连续下发（默认 30 Hz）；`--hz` 只用于打印步频，不改变下发频率。该命令让 π0.5 驱动整段。要运行「脚本对轴 + π0.5 接管末段」的一趟：

```bash
python _vla_end_try.py
```

`configs/strategies.json` 里的 `vla_end` 是提前接管变体的策略条目。

### 6.5 标定

- 内参 + 手眼：`real/calib_views.py` 采视图，`real/mount_check.py` 用 ChArUco 验反投影。
- 颜色 / 夹爪阈值：`scripts/calibration/calibrate_color.py`、`scripts/calibration/calibrate_grasp.py` 写 `thresholds.json`。
- 工位常数：`configs/real.yaml` 的 `table_z_mm`、`pick_xy_mm`、`percep_offset_mm`，换桌面或换相机后需重新测量。

## 7. 数据与训练

### 7.1 数据集格式

采集写 LeRobot v3.0（`lerobot 0.6.1`）：

- `observation.state` / `action`：6 维，关节角（度，0~180 舵机口径），顺序 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`。
- `observation.images.camera1`：腕相机，960×540。
- `observation.images.camera2`：侧相机（配置了侧相机时才有）。
- `task`：每帧一句，来自 `configs/real.yaml` 里的目标颜色。
- `fps=4`：0.2 s 的采样闸口隔一步录一帧。

仅录制抓取阶段：夹爪标定段和摆件段不录制；抓取成功（`held` 且未滑落）的集才写盘。无人值守采集 6 小时产出 120 条，有效率 68%。

### 7.2 π0.5 LoRA

- 切末段：`scripts/training/_build_endseg.py` 从 6 号舵机第一次读到 >15° 的前 2 帧切到集尾；262 集实测中位 28 帧/集。丢弃 ep 0-19：这 20 集的开合范围达到 99–100°，与现行 20° 上限不是同一套标定。
- 训练配置：`pi05=True`、`action_horizon=10`、`paligemma_variant=gemma_2b_lora`、`action_expert_variant=gemma_300m_lora`、20k steps、恒定 lr 1e-5、AdamW 裁剪 1.0、batch 2；只训练 LoRA adapter。
- 配置片段在 `scripts/training/_add_endseg_cfg.py`、`scripts/training/_add_v2_cfg.py`，启动脚本在 `scripts/training/_train_pi05_v2.sh`。这些脚本中的路径为云端环境专用，公开使用前需按环境修改。

### 7.3 Qwen3-VL 判官

- 数据：`scripts/training/_build_vlm_sft.py` 将成败样本构造成成对图像（合爪前 + 当前），标签来自 6 号舵机合爪角（`held` = 方块顶住，`empty` = 合到底）。
- 规模：6 类失败场景，300 条成对样本训练；280 条评测样本上二分类准确率 92.9%。
- 模型：Qwen3-VL-4B-Instruct + LoRA（rank 16、alpha 32）。
- 部署：ms-swift 起 OpenAI 兼容端点，`ppo/vlm.py::judge` 通过 `VLM_JUDGE_BASE_URL` 接入。

### 7.4 复现说明

数据集、模型权重、训练日志都不在本仓库。`datasets/` 和 `out/` 由采集/训练脚本按环境生成；`configs/real.yaml` 里的数据集根目录、串口、相机序号是现场值。

## 8. 演示媒体

以下为 GIF 预览，会直接在 README 中循环播放；每行末尾附对应的完整 mp4。

### 无人值守连续采集

| 预览 | 片段说明 |
|---|---|
| ![15 秒采集](docs/media/gifs/00-unattended-15s.gif) | **单趟完整抓放（15 秒原速）**：从初始位定位方块，转腕对轴、竖直下探、合爪，抬升到目标点后松开。[完整 mp4](docs/media/videos/00-unattended-15s.mp4) |
| ![23 分钟 8 倍速](docs/media/gifs/01-unattended-23min-8x.gif) | **连续多趟（23 分钟压缩至 8 倍速）**：抓取完成后自动将方块摆到下一位置并回零，继续下一趟；未成功的趟也会回零重试。[完整 mp4](docs/media/videos/01-unattended-23min-8x.mp4) |

### 脚本三档

| 预览 | 片段说明 |
|---|---|
| ![固定落点抓取](docs/media/gifs/02-script-oneshot.gif) | **固定落点抓取（`oneshot`）**：合爪前不做微调，爪尖直接到由方块中心计算的落点，竖直下探、合爪、抬升。[完整 mp4](docs/media/videos/02-script-oneshot.mp4) |
| ![单次量测修正](docs/media/gifs/03-script-measure-once.gif) | **单次量测修正**：下探前在腕相机中测量爪面到方块的缝宽，按该帧实测值修正落点后再下探。[完整 mp4](docs/media/videos/03-script-measure-once.mp4) |
| ![视觉伺服微调](docs/media/gifs/04-script-full-finetune.gif) | **视觉伺服微调（`full`）**：合爪前用画面雅可比小幅平移，将方块调整至两爪中心再合爪。[完整 mp4](docs/media/videos/04-script-full-finetune.mp4) |
| ![多次迭代修正](docs/media/gifs/05-script-finetune-multi.gif) | **多次迭代修正（历史演示）**：当时未设次数上限，同一落点连续纠偏 5 次，展示偏差较大时的收敛过程；现行策略的合爪前微调有次数上限（`CLEAR_MAX_MOVES`）。[完整 mp4](docs/media/videos/05-script-finetune-multi.mp4) |

### VLA 末段

| 预览 | 片段说明 |
|---|---|
| ![腕部对齐后交给 π0.5](docs/media/gifs/06-vla-handover.gif) | **腕部对齐后交给 π0.5（提前接管变体）**：脚本先将腕部转到方块侧面的法向，π0.5 从该帧接管下探、横向对齐和合爪。[完整 mp4](docs/media/videos/06-vla-handover.mp4) |
| ![空抓后重试](docs/media/gifs/07-vla-retry.gif) | **空抓后重试**：π0.5 第一次合爪空抓，6 号舵机判定为 `empty` 后抬回悬停点，第二次下探才夹持成功。[完整 mp4](docs/media/videos/07-vla-retry.mp4) |

## 9. 已知边界

- 真机常数强依赖现场：串口、相机序号、桌面高度、感知偏移、初始舵机原始值分别在 `configs/real.yaml`、`real_home.py`、`assets/` 里，换一台机器需重新标定。
- Qwen3-VL 判官只覆盖「合爪前方块是否在两爪之间」；合爪后的滑落不在其判据内。
- π0.5 的策略服务在外部 GPU，仓库只含客户端和末段接线，不能单独启动模型。
- 真机面板单趟默认只合爪一次（`max_attempts: 1`）；重试由无人值守采集的 `ppo/policy.py` 或面板的策略阶梯负责，不原样重放。
- 数据集、权重、日志不随仓库发布。

## 10. 贡献者

本项目由两名开发者共同完成，方案设计、真机实验与工程实现由双方共同推进；代码提交统一使用仓库作者账号。

<!-- 共同开发者：姓名 / GitHub 账号，确认后补充 -->
