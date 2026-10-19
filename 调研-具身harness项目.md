# 具身 / 机器人 harness 调研（2026-10-05）

用途：给自己（后续 session）查。回答一个问题——**别人怎么把「跑 episode → 判定成功 → 落盘 → 评测 checkpoint → 真机部署」这套循环搭起来的**，好对齐我们 `G:\sim2real\so101-real-harness` 缺的四块。

不确认的条目一律标「未确认」，不编数字。

---

## 0. 我们的问题清单（调研就是为这六条服务的）

| # | 问题 | 我们现状 |
|---|---|---|
| 1 | 仿真与真机怎么抽象成同一个接口 | `backend.py` 有 `SimBackend`；`RealBackend.__init__` 直接 `raise NotImplementedError` |
| 2 | episode 主循环 + 停止条件 | `harness.py` 只有「采够条数就停」 |
| 3 | 成功判定（仿真真值 / 真机阈值并存） | `judge.py` 有 `GroundTruthJudge` + `ThresholdJudge` 影子模式设计 |
| 4 | 数据落盘格式 + 流式写入 | 未接。`demo_picklift_round.py` 是攒内存最后写一个 mp4（1452 帧 = 2.68 GB） |
| 5 | checkpoint rollout 怎么组织 | `drivers.py` 的 `PolicyDriver.__init__` 也是 `raise NotImplementedError` |
| 6 | 标定/阈值这类「每台硬件一份的参数」放哪 | `thresholds.json` 全是 `null`，模块拒绝工作（故意的） |

---

## 1. 方法

四路并行调研，每路一个簇：

- A 机器人学习框架（LeRobot / openpi / RoboVerse / ManiSkill3 / RoboCasa / LIBERO / RoboTwin / SIMPLER / robomimic / Isaac Lab）
- B 通用评测 harness 模式（lm-evaluation-harness / HELM / Inspect AI / SWE-bench harness / OpenCompass）
- C 真机部署与评测回路（LeRobot 真机脚本 / SERL / dora-rs / ros2_control hardware_interface / gym-hil）
- D 具身评测基准（LIBERO / SimplerEnv / VLABench / EmbodiedBench / BEHAVIOR-1K / RoboArena / Genie Sim）

star 数经 **GitHub REST API + ungh.cc** 两源实抓交叉核对；抓不到的写「未确认」。
注意：调研环境里 `github.com` 直连被封，star 是靠 API 拿的，不是页面。

---

## 2. 最对口的一个：allenai/vla-evaluation-harness

**URL** https://github.com/allenai/vla-evaluation-harness ｜ 论文 arXiv:2603.13966 ｜ PyPI `vla-eval`
**star** 635（2026-10-05） ｜ **年份** 2026-03（很新） ｜ **许可证** Apache-2.0

这是目前唯一一个「跨基准的通用 VLA 评测 harness」，结构跟我们想做的几乎一一对应：

```
CLI(cli/main.py) → Orchestrator → Benchmark(benchmarks/base.py, 跑在 Docker 内)
                                   → EpisodeRunner → Connection ←WebSocket/msgpack→ ModelServer
```

- **模型侧只实现一个 `predict()`**；模型服务器是独立 `uv` 脚本，跑在另一台机器。
- 两条命令：`vla-eval serve -c configs/model_servers/<model>.yaml` 和 `vla-eval run -c configs/<benchmark>.yaml`。就绪判定 = `GET /health` 返回 200。远程 GPU 用 `--server-url ws://<node>:8000`。
- **基准跑在 Docker 里**——这是解决依赖地狱的关键（我们这边 nexus / lerobot / sim2env 的依赖冲突就是同一类问题）。
- **结果用 SQLite 逐条落盘**（`RecordingStore`），episode 级错误隔离（单条失败不中断整轮）；分片跑各自连同一个 model server，跑完 `vla-eval merge`。
- 模型可在 `EPISODE_START` 拿到 `(sid, eid, eval_id, db_path)`，自己开 `StepRecorder` 写行——用 SQLite 的 `json_patch` 并字段。
- **文档里记的真实坑**（比设计更值钱）：proprio 来源搞错 → 成功率 97.8%→42%；绝对/增量动作模式混乱 → 0%；四元数↔轴角归一化不一致；OpenVLA 未公开的中心裁剪 ≈ 3pp。

**要抄的**：`predict()` 单点接口 + Docker 隔离 + WebSocket/msgpack + SQLite 逐条落盘 + 分片 merge。

---

## 3. 评测基准这一簇（细节）

### 3.1 LIBERO
- https://github.com/Lifelong-Robot-Learning/LIBERO ｜ arXiv:2306.03310 ｜ star **2382** ｜ 2023 ｜ MIT
- 任务集：**BDDL 场景文件**（`bddl_files/`）+ 每任务**固定初始状态数组**（`init_states/*.pruned_init`）。四 suite：Spatial / Object / Goal / Long（= `libero_10`），共 130 任务。
- 策略接入：`algo.policy.get_action(data)`（`libero/lifelong/evaluate.py`）；走 LeRobot 的话是 `lerobot-eval --env.type=libero --env.task=...`。
- 规模/计分：官方脚本 `env_num=20` 并行环境，`success_rate = num_success / env_num`。社区标准协议（OpenVLA/π0）：**每任务 50 初始状态 × 10 任务 = 500/suite，四 suite 共 2000 条，seed 7**。
- 判定：仿真真值 `env.check_success()`（BDDL 谓词）。
- 落盘：`torch.save` 存 `{loss, success_rate}` 到 `{benchmark}_{algo}_{policy}_{seed}...stats`，**逐任务一个文件** → 天然可按任务续跑。
- 一致性：初始状态来自固定数组，`indices = np.arange(env_num) % init_states.shape[0]`；`env.seed(cfg.seed)`；预热 5 步零动作让物理稳定。
- **可抄**：BDDL「初始谓词 + 目标谓词 + 固定初始状态数组 + `check_success()`」三件套。

### 3.2 SimplerEnv / SIMPLER
- https://github.com/simpler-env/SimplerEnv ｜ arXiv:2405.05941 ｜ star **1174** ｜ 2024（CoRL） ｜ MIT
- 任务即 ManiSkill2 环境（`google_robot_pick_coke_can` 等）；入口 `simpler_env/evaluation/main_inference.py` → `maniskill2_evaluator.py`。
- GR00T/OpenVLA 的 fork 走**服务器+客户端**（`run_gr00t_server.py` + `rollout_policy.py`）。
- 规模：Pick Coke Can 是 3 朝向 × 25 网格位 = 75 条；GR00T 复现实测 200 条/任务。
- **两种协议**：VM（Visual Matching，匹配真机纹理/光照/位姿，与真机相关性最好）/ VA（Variant Aggregation，跨变体平均，看鲁棒性）。
- 结论输出：`tools/calc_metrics.py` 算 **MMRV（Mean Maximum Rank Violation，越低越好）** 和 **Pearson r**——用来判断「仿真评测排名 ≈ 真机排名」。
- 一致性：Octo 因扩散头非确定性对 3 个种子平均；Google Robot 对 4 种配色版本平均；控制频率固定（Google 3 Hz，Bridge 5 Hz）。
- **可抄**：MMRV + Pearson 这套**评测管线自检指标**。我们用仿真当代理，这个能证明代理可信。

### 3.3 VLABench
- https://github.com/OpenMOSS/VLABench ｜ arXiv:2412.18194 ｜ star **476** ｜ 2024-12 ｜ MIT
- 任务集版本化靠 **`VLABench/configs/evaluation/tracks/`** 下的 track 配置：`track_1_in_distribution` / `track_2_cross_categroy` / `track_3_common_sense` / `track_4_semantic_instruction` / `track_5_cross_task` / `track_6_unseen_texture`。官方推荐用 track 采样 episode，比直接从环境采样更可控。
- 接入：`scripts/evaluate_policy.py --n-sample 20 --model openvla --eval_track ...`；模型配置是 JSON（`configs/model/openvla_config.json`，需先算 `norm_stat`）。
- 计分：**Progress Score PS = α·(n_correct/N) + (1−α)·(m_done/M)**——长程任务比 0/1 合适。
- **可抄**：「评测集 = 一组命名的 track 配置文件」，跨机器同 episode。

### 3.4 EmbodiedBench
- https://github.com/EmbodiedBench/EmbodiedBench ｜ arXiv:2502.09560 ｜ star **346** ｜ 2025-02（ICML Oral） ｜ 许可证**未确认**（仓库似无 LICENSE）
- 1128 测试任务，4 环境（EB-ALFRED / Habitat / Navigation / Manipulation）；六个能力子集各 50 实例。
- 命令式：`python -m embodiedbench.main env=eb-hab model_name=gpt-4o ...`。**三个独立 conda 环境**（依赖冲突无法合并）。
- 判定：各仿真器的脚本谓词，非人工。结果逐 episode 存 JSON；竞赛用 `copy_json.py` 导出打包提交 EvalAI。
- **可抄**：按**能力维度**拆子集评分（常识 / 空间 / 长程…），比单一总分更能定位短板。

### 3.5 BEHAVIOR-1K / OmniGibson
- https://github.com/StanfordVL/BEHAVIOR-1K ｜ arXiv:2403.09227 ｜ star **1735**（OmniGibson 同为 1735，已归档并入，数值存疑） ｜ 2024（CoRL） ｜ MIT（资产另有条款，未逐条确认）
- 1000 个家庭活动，BDDL 定义初始/目标条件。2025 挑战 = v3.7.2 + Isaac Sim 4.5.0 + B50；2026 评测器 = v3.9.1、100 任务，`python -m omnigibson.eval.eval`。
- 接入：**WebSocket policy server**。健康检查 `/healthz` 后开 WS，用 `WebsocketPolicyServer`（改编自 openpi），观测 flatten、返回 **msgpack 编码的 action 数组**。支持 action-chunk 重放（`--replay-action-chunk-size`）。
- 计分：**q-score = 满足的 BDDL 目标谓词数 / 目标谓词总数**（部分分）；`task_sr` 要求全部满足。每条 JSON 含 `q_score / time / agent_distance`。
- 落盘：**每条 rollout 一个 JSON + 一个 MP4**，最多 500 份打包提交。支持 `--partial_scene_load`、`--testing_on_train_instances`（查过拟合）、`--max_steps`（默认 = 人类时长 ×2）。
- 一致性：实例 0–19 公开 / 20–39 隐藏；wrapper 代码人工审查防作弊（禁瞬移机器人、禁改物体状态）。
- **可抄**：**每条 rollout 一个 JSON + 一个视频、允许部分提交、未跑记 0 分**——「跑一条存一条、断了能续」最干净的实现。

### 3.6 RoboArena（分布式真机评测）
- https://github.com/robo-arena/roboarena ｜ arXiv:2506.18123 ｜ star **115**（评测器 repo 8） ｜ 2025-06（CoRL Oral） ｜ MIT
- **不固定任务**——评测者自由选任务/环境，这是它的核心创新。
- 策略实现 `BasePolicy`（`policy.py`），用 `PolicyServerConfig(image_resolution, needs_wrist_camera, n_external_cameras, action_space=...)` 声明所需观测/动作，再包 `WebsocketPolicyServer`。
- 计分：**双盲 A/B 成对比较**。七个机构、DROID 平台、600+ 对真机 episode、7 个策略。每对给 progress score（0–100）、成对偏好（A/B/tie）、自然语言反馈。
- 判定：**人类看真机视频**（非仿真真值）。
- 汇总：Bradley–Terry MLE / Elo 在线更新 / 部分成功平均；每周评测预算，贡献评测换额度。
- **可抄**：任务没法统一时，「**成对比较 + Bradley-Terry 排名**」能在保留多样性的同时给可信排序。

### 3.7 Genie Sim / AgiBot World
- https://github.com/AgibotTech/genie_sim ｜ arXiv:2601.02078 ｜ star **1411** ｜ 2026-01（Genie Sim 3.0） ｜ 许可证 "other"（未逐条读）
- AgiBot-World 数据集 https://github.com/OpenDriveLab/AgiBot-World ｜ star **3201** ｜ 100 台双臂机器人、100 万+ 轨迹 / 217 任务 / 5 场景（是数据集不是评测）｜ 许可证未确认（论文 CC BY-NC-SA 4.0）
- 评测任务 JSON：`source/geniesim/benchmark/ader/eval_tasks/*.json`（objects / recording_setting / robot / scene / stages），另有 `task_definitions/default_problem.json`。任务生成用 **LLM + ADER**。
- 接入：子类化 `BasePolicy`，实现 `__init__ / reset / act(observations, **kwargs) -> np.ndarray`（观测 = 三路相机 + 关节，返回 `target_joint`）；`run_client {task} --policy_class ... --env_class ...`。竞赛提交走 **Docker 镜像**。
- 判定：仿真真值逐步 success rate + **VLM 自动判读**视频。汇总 `python3 scripts/stat_average.py output/benchmark`；leaderboard 每队取最高分、每日 2 次提交。
- 4 格设计：训练数据仿真/真机 × 评测仿真/真机；报告称仿真-真机评测误差 < 5%。
- **可抄**：「LLM 生成任务配置 + VLM 自动判分」，把造任务和判成功两件重活自动化。

### 3.8 其它新基准（表格）
| 基准 | URL / star | 年份 | 许可证 | 一句话 |
|---|---|---|---|---|
| RoboTwin 2.0 | https://github.com/RoboTwin-Platform/RoboTwin ｜ star **2939** | 2025-06 | MIT | 双臂 50 任务，Easy/Hard 两档，每任务 100 rollouts；有 leaderboard |
| LIBERO-Plus | https://github.com/sylvestf/LIBERO-plus ｜ star **470** | 2025-10 | 未确认 | LIBERO 鲁棒性扩展，7 个扰动维度，每 cell 10 任务 × 50 episode |
| VLA-Arena | https://github.com/PKU-Alignment/VLA-Arena ｜ star **216** | 2025-09 | Apache-2.0 | 11 套件 / 170 任务，难度 L0–L2（只 L0 可微调）+ 语言/视觉扰动；结果存 CSV |
| WorldArena | ICML 2026（repo 未定位） | 2026 | 未确认 | 评测具身世界模型，RoboTwin 2.0 上 50 场景 / 2500 视频 |
| ForesightSafety-VLA | arXiv:2606.27079 | 2026 | 未确认 | VLA 安全诊断基准，13 类 66 场景 |
| IndustrialVLA-Bench | arXiv:2609.25562 | 2026 | 未确认 | 六个开源 VLA 的可追溯多轴评测，3 个 run 级种子 |

新基准共性趋势：① 都建在 LIBERO/RoboTwin 上做扰动扩展；② 强调可复现（多 seed、可追溯）；③ 难度分层 + 只允许最低层微调。

---

## 4. 横向对照（四路汇总，逐条回答第 0 节那六个问题）

### 4.1 问题 1：仿真与真机怎么抽象成同一个接口

四种成熟写法，按耦合度从松到紧：

| 写法 | 代表 | 形状 |
|---|---|---|
| **注册制** | lerobot `EnvConfig` + `@register_subclass` | 每个 benchmark 一个子类，真机也是子类（`gym_manipulator`）；`create_envs()` 统一返回 `VectorEnv` |
| **单一 ABC 撑多后端** | RoboVerse `BaseSimHandler` | `launch/render/set_states/set_dof_targets/simulate`，10 个后端各一目录；Genesis 只是被封装的一个引擎 |
| **复用仿真 env 类 + 真机 agent** | ManiSkill `Sim2RealEnv(sim_env, BaseRealAgent)` | 复用仿真的 `observation_space/action_space`，`_check_observations()` **逐键断言真机观测与仿真 shape/dtype 一致** |
| **适配器接上层框架** | RoboCasa `EnvRobocasa` 实现 robomimic `EnvBase` | 一个适配器换来整套训练/评测 harness |
| （无 backend 层） | openpi | 只有 per-platform `DataTransformFn` 插件 |

### 4.2 问题 2：episode 主循环 + 停止条件

**停止条件三件套**（A 路明确）：① 「采够条数」`n_episodes` / `n_rollouts`；② 「硬上限」`max_episode_steps` / `horizon`（RoboTwin 单独放 `_eval_step_limit.yml` 按任务名查表，缺省 1000）；③ 「成功率早停」——**四路里只有 robomimic `config.experiment.rollout.rate` 确认到该机制**。

**终止条件可组合化**（Isaac Lab）：`TerminationManager` 把 `time_out` 与各种成功/失败 term 并列成一张字典，`reset_terminated` 与 `reset_time_out` 分开两个缓冲。

**「成功」与「结束」必须解耦**（robosuite `ignore_done`、RoboCasa `is_done()` 恒 `False`、Isaac Lab 独立 `time_out`）——真机上尤其如此，因为真机基本拿不到 `done`。

### 4.3 问题 3：成功判定

| 场景 | 做法 | 代表 |
|---|---|---|
| 仿真 | env 暴露真值谓词 | `_check_success()`（robosuite/LIBERO/RoboCasa）、`evaluate()["success"]`（ManiSkill）、`info["is_success"]`（lerobot）、BDDL goal（LIBERO/BEHAVIOR） |
| 仿真（可替换） | 独立 checker 对象，真值/感知并存 | RoboVerse `checker` + `BaseDetector` |
| 真机（几何阈值） | **几何阈值 + 夹爪状态** | RoboTwin `0.08 < dist(obj,target) < 0.2` 且 `obj.x < target.x` 且 `|Δy| < 0.05` 且两爪张开 |
| 真机（学习式） | 相机奖励分类器 | HIL-SERL（空格标正样本，负样本 2–3 倍）、lerobot `rewards/classifier/`（`success_threshold=0.5`）、AutoEval（VLM 三分类，与人工 oracle Pearson≈0.942） |
| 真机（人工） | 人看 | ALOHA/ACT（`get_reward()` 恒 0，实际成功率靠人）、RoboArena（双盲偏好 + Bradley-Terry） |

**关键事实：开源里没有一个成熟的「用舵机电流/力矩判抓取」的实现。** 真机判成功只有「几何阈值」和「图像分类器」两条路。

### 4.4 问题 4：数据落盘格式

**两派**：LeRobot dataset（lerobot / openpi / RoboVerse / ManiSkill / RoboCasa 都可导出）vs HDF5（robomimic / robosuite / LIBERO / RoboTwin / Isaac Lab Mimic）。

**流式写入的最佳范例是 lerobot**：`create() → add_frame() → save_episode() → finalize()`，视频走 `StreamingVideoEncoder` 边跑边编码；v3 格式用**偏移量重建 episode 边界，而不是一集一个文件**。

**「跑一条存一条、文件在就是已完成」**：SWE-bench 每个实例一个 `report.json`，存在即跳过；BEHAVIOR-1K 每条 rollout 一个 JSON + 一个 MP4，支持部分提交（未跑记 0 分）。vla-eval 用 SQLite（`RecordingStore`）做 episode 级隔离 + `merge` 分片。

### 4.5 问题 5：checkpoint rollout 怎么组织

- **接口**：方法级单点——`get_action()`（LIBERO）、`select_action()`（lerobot）、`predict()`（vla-eval）、`infer(obs)->dict`（openpi）、`act()`（Genie Sim）。
- **进程内 → 服务式**是明确趋势：SimplerEnv-GR00T / BEHAVIOR / RoboArena / vla-eval / Genie Sim / openpi 全是 **WebSocket policy server**（openpi 用 msgpack_numpy，ndarray 编成 `{__ndarray__:true, dtype, shape, data}`，入站上限 64 MiB）。
- **推理延迟 > 控制周期**三种解法（C 路）：(a) 动作分块 + 开环执行（openpi `open_loop_horizon=8`；ACT 直接丢掉 chunk 前 `BASE_DELAY=13` 步）；(b) 后台线程 + `ActionQueue` + RTC guidance（lerobot `execution_horizon=10`、`queue_threshold=30`）；(c) 低频率 + 增量动作 + 人机硬切换（HIL-SERL 10 Hz delta）。
- **调度**：RoboTwin `eval_policy_multitask.py` 按 GPU slot 调度（`PolicyEndpoint` + `EvalJob`）。

### 4.6 问题 6：标定/阈值这类「每台硬件一份的参数」放哪

- **只有 lerobot 真机侧有完整的标定文件管理**：`Robot._load_calibration/_save_calibration` → draccus JSON，默认 `calibration_fpath`，配 `lerobot_calibrate` CLI 与 `calibration_gui.py`。
- RoboTwin 把每任务步数上限、embodiment 参数集中在 `env_cfg/*.yml`。
- HELM / OpenCompass 是「每个模型/部署一份配置」（`model_deployments.yaml`、`configs/models/*.py` 里的 `run_cfg{num_gpus, num_procs}`）。
- **其余几乎全是纯仿真，没有硬件标定这个概念。**

### 4.7 附：协议一致性（怎么做才可复现）

固定初始状态数组 + 锁种子（LIBERO `.pruned_init` / SimplerEnv / RoboTwin）｜ 公开/隐藏 instance（BEHAVIOR 0–19 / 20–39）｜ 同名 track 配置文件跨机器同 episode（VLABench）｜ 双盲背靠背（RoboArena）｜ 只允许最低难度层微调（VLA-Arena）｜ 演示 HDF5 里 embed `env_meta` + `env_version`，加载时比对告警（robomimic）。

### 4.8 附：评测管线自检

**SimplerEnv 是唯一一个把「仿真评测的意义」操作化的项目**：内置真机基准表 `REAL_PERF` 与仿真表 `SIMPLER_PERF`，算 `pearson_correlation` 与 `mean_maximum_rank_violation`（MMRV），回答「仿真排名 ≈ 真机排名吗」。

---

## 5. 对 SO-101 real-harness 的结论（四路汇总后的最终版）

### 5.1 六问对照表

| 我们的问题 | 抄谁 | 具体做什么 |
|---|---|---|
| 1 backend 抽象 | **ManiSkill `Sim2RealEnv`** | `backend.py` 已经是最接近的形状；补一个 `_check_observations` 式断言——真机后端接上后，逐键比对观测 shape/dtype 与仿真一致，不一致直接报错 |
| 2 停止条件 | **lerobot 三层 + robomimic 早停** | `harness.py` 现在只有「采够条数」。补硬上限（每 episode 步数）+ 时间上限（真机用）+ 可选成功率滑窗 |
| 3 成功判定 | **RoboTwin 几何阈值 + 夹爪状态** | 这是我们 `judge.py` / `thresholds.json` 的模板：把「成功」写成 `距离阈值 + 夹爪开合 + 颜色` 的布尔式，而不是靠时间猜 |
| 4 落盘 | **lerobot 流式写入** | 治「帧攒内存最后写一个 mp4」。`add_frame → save_episode → finalize`，视频流式编码。或先做最简的 SWE-bench 式：每轮一个 JSON，文件在就跳过 |
| 5 checkpoint rollout | **vla-eval `predict()` / openpi `infer(obs)`** | 补 `drivers.py` 的 `PolicyDriver`：单点接口 + 动作分块 + `reset()` 清队列 |
| 6 标定/阈值 | **lerobot 的 per-robot JSON** | `thresholds.json` 就是我们的 `calibration_fpath`，按 lerobot 的 draccus JSON 组织即可 |

### 5.2 四条跨路线的硬结论

1. **成功判定与 episode 结束必须解耦**（robosuite `ignore_done` / RoboCasa `is_done()` 恒 `False` / Isaac Lab 独立 `time_out`）。真机拿不到 `done`，只能轮询 success——我们 `judge.py` 现在把两者混着看，要拆开。
2. **真机判「夹住 / 颜色」只有两条开源路线**：几何阈值（RoboTwin）或相机分类器（HIL-SERL / AutoEval）。**「从爪舵机角度判夹取」这条路线在开源里没有先例**——我们 README 里模块 A 要自己验，别指望有现成的可抄。
3. **复位（reset）必须单独建**，否则无法无人值守滚下去。AutoEval 用 50–100 条复位示范训一个 reset policy，或脚本化复位。我们 `demo_picklift_round.py` 里的 `park_cubes()`（操作员摆件）就是脚本化复位的雏形。
4. **报告要加部分分列**（BEHAVIOR q-score / VLABench PS / RoboArena progress 0–100）。抓取里的「夹住了但没抬够」「抬起来但掉在半路」用 0/1 全丢。

### 5.3 我们的标识点表在别人那里叫什么

**就是 LIBERO 的 `.pruned_init` / BEHAVIOR 的固定 instance**——「固定初始状态」这一条我们已经在做了，方向对。下一步照 LIBERO 补「目标谓词」，就凑齐了「初始谓词 + 目标谓词 + 固定初始状态 + `check_success()`」三件套。

### 5.4 四条待修，按优先级

1. `harness.py` 的停止条件（现在只有一条，缺硬上限和成功率早停）
2. `drivers.py::PolicyDriver`（raise NotImplementedError）
3. `backend.py::RealBackend`（raise NotImplementedError）
4. `thresholds.json` 标定（全 null，模块拒绝工作——这个设计本身是对的，照 lerobot 的 per-robot JSON 填即可）

---

## 6. 通用评测 harness 的模式（B 路，已回）

研究对象：语言/代码模型的评测 harness。**跟机器人无关，但它们把「任务 × 模型 × 硬件」这套矩阵跑了十年，工程模式最成熟。**
star 与提交日期均为 2026-10-05 实抓（GitHub 页面 + REST API）。

### 6.1 六个项目速览

| 项目 | URL | Stars | 最近提交 | 许可证 |
|---|---|---|---|---|
| EleutherAI/lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | 14,129 | 2026-09-14 | MIT |
| stanford-crfm/helm | https://github.com/stanford-crfm/helm | 2,933 | 2026-06-05 | Apache-2.0 |
| UKGovernmentBEIS/inspect_ai | https://github.com/UKGovernmentBEIS/inspect_ai | 2,935 | **2026-10-05（当天）** | MIT |
| princeton-nlp/SWE-bench | https://github.com/princeton-nlp/SWE-bench | 5,971 | 2026-09-02 | MIT |
| openai/evals | https://github.com/openai/evals | 19,552 | 2026-04-14 | MIT |
| open-compass/opencompass | https://github.com/open-compass/opencompass | 7,492 | 2026-09-28 | Apache-2.0 |
| bigcode-project/bigcode-evaluation-harness | https://github.com/bigcode-project/bigcode-evaluation-harness | 1,062 | 2025-07-22 | Apache-2.0 |

### 6.2 每个项目的一句话要点（挑跟我们相关的）

- **lm-evaluation-harness**：任务定义成**纯数据 YAML**（`task:` + `doc_to_*` + `metric_list` + `output_type`），`output_type` 反查模型该调哪个方法 → 任务与模型彻底解耦。模型接口 `LM` 三个方法：`loglikelihood` / `loglikelihood_rolling` / `generate_until`，每后端一个文件用 `@register_model("hf")` 注册。结果：聚合 `results_<时间戳>.json` + **每样本 `samples_<task>.jsonl`（逐行）**。缓存两层：请求结果 pickle + 模型响应 sqlitedict（SQLite），`--use_cache`。CLI 三个子命令 `run / ls / validate`。
- **HELM**：一次运行冻结成不可变 **`RunSpec`**（数据 `scenario_spec` + 适配 `adapter_spec` + 指标 `metric_specs`）。**每个模型/部署一份 YAML**（`model_deployments.yaml`：`client_spec{class_name, args}`），`AutoClient` 按名字动态实例化。续跑靠 **`--skip-completed-runs`**（5 个输出文件齐了就算完成）。缓存后端可选 SQLite/Mongo/BlackHole。
- **inspect_ai（最现代）**：任务拆成 **dataset + solver + scorer** 三个可组合 Protocol（`solver` = 策略闭环，`scorer` = 判分）。日志 `.eval` 是 **ZIP**，内部 `header.json` + 逐样本 `samples/` 条目，**先写 header 后续 append**。断点续跑最强：`SampleBufferDatabase`（**SQLite WAL**）记已完成样本，`retry_immediate=True` 失败立即重试并复用已完成样本；`incomplete_action` 处理崩溃日志。
- **SWE-bench**：把模型退化成**一份外部文件**（predictions JSONL，每行 `instance_id / model_name_or_path / model_patch`），harness 只做「隔离环境执行 + 判分」。**每个实例一个 `report.json`，文件存在即视为已完成、直接跳过**；运行级 `results.json` 汇总（`resolved_instances` 等 + `schema_version: 2`）。判分三态 `RESOLVED_FULL / PARTIAL / NO`。并发 ThreadPool `--max_workers`。
- **openai/evals**：**事件流 recorder**——每次评估是一条 `Event` 序列（`match / sampling / metrics / error` 等类型），落本地 JSON；有 `--local-run` / `--http-run`（流式 POST）两路。YAML 注册表 `class: 模块:类` + `args:`。
- **OpenCompass**：两段原子任务 **`OpenICLInferTask`（生成 predictions）+ `OpenICLEvalTask`（读 predictions 算分）** 完全分离。`Partitioner` 把整份 Config 拆成 (model × dataset) 子任务，交给 `Runner` 调度。续跑 **`-r/--reuse <时间戳>`**。每个模型一份 `configs/models/*.py`，里面 `run_cfg{num_gpus, num_procs}` 声明资源。
- **bigcode**：**`--generation_only` 只生成 / `--load_generations_path` 只评测**，两个阶段可在不同硬件上分开跑。代码执行在 Docker 镜像里隔离，`--allow_code_execution` 显式开。

### 6.3 六条通用轴（照这个对照我们）

1. **任务注册**：YAML 自动发现（lm-eval）/ `@run_spec_function` + 扫文件（HELM）/ `@task @solver @scorer` 装饰器（inspect_ai）/ YAML 注册表（openai）/ `type=` 的 Python 配置（OpenCompass）/ 纯字典 `TASK_REGISTRY`（bigcode）/ 数据集里每实例 TestSpec、无注册表（SWE-bench）。
2. **模型抽象**：单点方法为主（`loglikelihood` / `make_request` / `generate`）；SWE-bench 极端到只吃文件，openai 只吃 `CompletionFn`。
3. **结果存储**：每样本 jsonl（lm-eval）/ 每 run 一个目录多 json（HELM）/ 单 zip 内 samples（inspect_ai）/ 每实例 report.json + 运行级 results.json（SWE-bench）/ 事件流 JSON（openai）/ predictions/ + results/ + summary（OpenCompass）。
4. **续跑/去重**：**文件存在即跳过**（SWE-bench、HELM）是最简单且最通用的做法；进阶是 SQLite 缓存（lm-eval、HELM）与 SQLite 样本缓冲（inspect_ai）。
5. **CLI**：`run/ls/validate`（lm-eval）、`helm-run/summarize/server`、`inspect`（14 个子命令）、`swebench infer/eval/report/submit`、`oaieval/oaievalset`、`opencompass <config>`。
6. **配置/代码分离 + 按模型定制**：最强的是 **HELM（`model_deployments.yaml`）与 OpenCompass（`configs/models/*.py` 含 `run_cfg` 资源声明）**；最弱/最简的是 `--model_args` 字符串。

### 6.4 对 SO-101 harness 的三条新增结论

- **「跑一条存一条、文件在就是已完成」这个模式（SWE-bench）最简单也最稳**，比 SQLite 方案省事，且天然解决了我们「帧攒内存最后写一个 mp4」的问题。第一条 rollout 就写盘。
- **两段式（`--generation_only` / `--load_generations_path`，OpenCompass 的 Infer/Eval 分离）**：我们「采轨迹」和「判成功」也应该分开、各自可缓存重跑。判据改了不用重采。
- **`run_cfg` 式资源声明（OpenCompass）**：把「这个策略需要几张卡 / 这个场景需要哪个后端」写进配置，而不是写死在代码里——正好对上我们 `backend.py` 想做的「真机/仿真只换一份 deployment 配置」。

### 6.5 B 路的未确认项

- HELM「2026-06-01 进入维护模式」：WebSearch 提及，仓库页面未见 → **未确认**。
- openai/evals「不再接受带自定义代码的 eval」：WebSearch 提及，未逐条核实 → **未确认**。
- SWE-bench `harness/constants/python.py` 正文（jsDelivr 对该文件 404），仅确认文件存在。
- 同日 WebSearch 给出的 star 旧值（如 lm-eval 13.1k–14.1k、HELM 1,691/2,860）与直抓页面不同 → 以直抓为准。

---

## 7. 真机策略部署与评测回路（C 路，已回）

**这一节最贴近我们的真实目标**——我们最终要跑真机，不是仿真基准。
star / 提交日期 2026-10-05 实抓（api.github.com；抓不到限流的用 shields.io 并标注）。

### 7.1 速览表

| 项目 | URL | Star | 最近提交 | 许可证 | 控制频率 | 推理位置 | 自动判成功？ |
|---|---|---|---|---|---|---|---|
| huggingface/lerobot | https://github.com/huggingface/lerobot | 27,954 | 2026-10-05 | Apache-2.0 | 30 Hz（可调） | 同机/同进程（可后台线程） | **否** |
| rail-berkeley/hil-serl | https://github.com/rail-berkeley/hil-serl | 1,539 | 2025-10-27 | Apache-2.0 | 策略 10 Hz | Actor 在机器人机 / Learner 在 GPU 机 | **是**（图像奖励分类器） |
| rail-berkeley/serl | https://github.com/rail-berkeley/serl | 863 | 2025-10-27 | Apache-2.0 | 同上 | 同上 | 已弃用 |
| Physical-Intelligence/openpi | https://github.com/Physical-Intelligence/openpi | 14,094 | 2026-08-24 | Apache-2.0 | 例 DROID 15 Hz | **独立机器** | **否** |
| markfzp/mobile-aloha | https://github.com/markfzp/mobile-aloha | 4,473 | 2024-06-22 | MIT | 50 Hz | 同机同进程 | **否**（奖励恒 0） |
| MarkFzp/act-plus-plus | https://github.com/MarkFzp/act-plus-plus | 3,667 | 2024-05-15 | MIT | 50 Hz | 同机同进程 | **否** |
| dora-rs/dora | https://github.com/dora-rs/dora | 3,993 | 2026-10-05 | Apache-2.0 | 由图决定 | 任意 | — |
| ros-controls/ros2_control | https://github.com/ros-controls/ros2_control | 1,026 | 2026-10-03 | Apache-2.0 | 单主环（例 200 Hz） | 任意 | — |
| huggingface/gym-hil | https://github.com/huggingface/gym-hil | 234 | 2026-09-18 | Apache-2.0 | 50 Hz（物理 500 Hz） | 进程内 | 否（仿真 reward） |
| zhouzypaul/auto_eval | https://github.com/zhouzypaul/auto_eval | 108 | 2026-03-27 | MIT | 未确认 | — | **是**（分类器 + 自动复位） |

### 7.2 逐个要点

- **lerobot（真机）**：`lerobot-record / replay / rollout / teleoperate` 一条 CLI 覆盖采集→回放→部署→DAgger。`--fps` 默认 30，控制循环打印实际 Hz + 各步骤耗时（`Cadence reporting`）。`--strategy.type` 四种：`base`（跑 `--duration` 秒）/ `sentry`（连续录、每 5 集上传，**episode 边界按 `fps×分辨率` 自动切，策略状态跨集保留、机器人不 reset**）/ `highlight`（内存环形缓冲 `ring_buffer_seconds=30`，按键存）/ `dagger`（人机交替，`Space` 暂停、`Tab` 接管，纠正帧标 `intervention=True`）。**没有任何自动成功判定**——结束全靠时长或按键。推理引擎两种：`SyncInferenceEngine`（线程内联）/ **RTC**（后台线程 + `ActionQueue`，`execution_horizon=10`、`queue_threshold=30`）。数据集 LeRobot v3：低维走 Parquet，视频走 MP4，`meta/info.json` 记 `fps: 30`，episode 边界**用偏移量重建而非一集一文件**，收尾必须 `finalize()`。
- **HIL-SERL（唯一内建自动判成功）**：`done = path_length >= max_episode_length or reward or terminate`，`MAX_EPISODE_LENGTH=100`。**reward 来自一个现学的图像奖励分类器**（`record_success_fail.py` 按住空格标正样本，建议负样本 2–3 倍；`train_reward_classifier.py` 存 `classifier_ckpt`）。通信三层：Actor→机器人 Flask HTTP（`:5000`），server 内部→机器人 ROS1 topic，Actor↔Learner **ZMQ（agentlace）** 异步，参数每 1–2 s push 回 actor。动作是 7 维 `[-1,1]` delta（`new_pose = current + delta*ACTION_SCALE`），**低速 10 Hz + 增量**所以人机硬切换不跳变。安全靠 `clip_safety_box()` + `ABS_POSE_LIMIT`。
- **openpi**：`serve_policy.py --env=...` 起在 **:8000**，机器人侧只装轻量 `openpi-client`。**WebSocket + msgpack_numpy** 二进制帧，ndarray 编成 `{__ndarray__:true, dtype, shape, data}`，入站上限 64 MiB、idle 30 s 断。推理整块搬到独立 GPU 机；DROID 例 `DROID_CONTROL_FREQUENCY=15`、`open_loop_horizon=8`（≈0.5 s 查一次，中间开环执行 chunk）、`max_timesteps=600`。无自动判定，靠 Ctrl+C。
- **ALOHA / ACT**：ROS1 + interbotix，`DT=0.02 / FPS=50`。ACT 的 `BASE_DELAY=13`——**真机上从 chunk 里直接丢掉前 13 步**对齐执行延迟。`real_env.get_reward()` 直接 `return 0`，所以 `eval_bc` 算出的真机成功率**恒等于 1，实际靠人看**。录制 HDF5 + JPEG。
- **dora-rs**：Rust 中间件，应用 = **YAML 声明式有向图**，节点是独立进程；>4 KB 走 **Zenoh 零拷贝共享内存**，跨机退化为 pub-sub，用 Apache Arrow 统一内存布局（声称比 ROS2 Python 快 10–17×）。**不是 ALOHA 团队做的**（作者 Haixuan Xavier Tao + Philipp Oppermann），是**被用来优化 ALOHA 遥操作**（50 Hz→500 Hz，双臂延迟 2 ms）。
- **ros2_control**：硬件分 Actuator/Sensor/System，实现 `on_init/on_configure/on_activate` + `read()/write()`，返回 `ERROR` 触发 `on_error()`。`controller_manager` **单主环**，`update_rate` 定周期（例 CRANE-X7 200 Hz）；硬件要跑不同速率就在 `read()/write()` 里**计数分频**。
- **gym-hil**：`control_dt=0.02`（50 Hz）/ `physics_dt=0.002`（500 Hz）/ `n_substeps=10`。**人在环抽象成 wrapper**：`InputsControlWrapper` + `info["is_intervention"]` / `info["teleop_action"]` 统一契约——这个契约可以直接搬到真机。**只有仿真，无真机环境**。
- **AutoEval（最像「harness」的一个）**：三个可学习模块替代人工——(1) **success classifier**：~1000 张 success/failure/invalid 图微调 PaliGemma 三分类，兼作 safety detector；(2) **reset policy**：50–100 条复位示范微调（OpenVLA + LoRA）或脚本化；(3) 安全/故障：电流上限、电机重启、工作空间边界、Slack 告警。任务以 server job 提交到 web UI 排队。**与人类 oracle 的 Pearson ≈ 0.942**；24 h 跑 ~850 episode，仅 3 次人工介入；每 6 h 停机 20 min 防过热。论文 arXiv:2503.24278（CoRL 2025）。

### 7.3 横向结论（可直接引用）

1. **控制频率**：lerobot 30 Hz ｜ ALOHA/ACT 50 ｜ openpi DROID 15 ｜ HIL-SERL 策略 10（底层阻抗环 1 kHz）｜ ros2_control 50–200 ｜ gym-hil 50（物理 500）。
2. **通信**：串口 → ROS1 topic → Flask HTTP → WebSocket+msgpack → ZMQ → 共享内存/Zenoh → ROS2。**越靠后越解耦、越能跨机**。
3. **推理位置**：同进程（lerobot 默认、ALOHA）→ 同机后台线程（lerobot RTC）→ 跨机 actor/learner（HIL-SERL）→ **独立机器 websocket（openpi，最彻底）**。
4. **「推理延迟 > 控制周期」三种解法**：(a) 动作分块 + 开环执行（openpi `open_loop_horizon=8`；ACT 丢前 13 步）；(b) 后台线程 + ActionQueue + RTC guidance（lerobot）；(c) 低频率 + 增量动作 + 人机硬切换（HIL-SERL 10 Hz delta）。
5. **谁决定 episode 结束**：时间 → 按键 → **学习式奖励分类器** → **分类器 + 自动复位**（AutoEval）。**越往右越能无人值守。**
6. **真机怎么判「夹住 / 颜色」**：开源里**只有图像分类器路线**（HIL-SERL / LeRobot 的 reward classifier 微调 CNN / VLM；AutoEval 的 VLM 三分类）。ALOHA/ACT 完全靠人（`get_reward()` 恒 0）。**没看到一个成熟的拿舵机电流/力矩判抓取的实现**——夹爪侧只有位置/速度归一化（ALOHA 的位置归一化；HIL-SERL 的 `GRIPPER_SLEEP=0.6s` + 位置阈值二值合爪）。
7. **安全**：**普遍没有独立硬件 E-stop 代码**。实际手段是安全盒/工作空间裁剪（HIL-SERL `clip_safety_box` + `ABS_POSE_LIMIT`；LeRobot `lerobot-find-joint-limits` + `EEBoundsAndSafety`）、阻抗控制器限参考点、动作 clip + 超时、人工接管/终止键、电机重启 + 电流上限（AutoEval）。
8. **录制格式**：**LeRobot v3（parquet + MP4，fps 30）是唯一事实标准**；ALOHA 用 HDF5 + JPEG（50 Hz）；HIL-SERL 用 replay buffer；AutoEval 用 RLDS/dlimp；openpi 训练前转 LeRobot v2.0。**与仿真对齐靠同一份 state/action 特征定义**（SO-101 就是 6 维 `SERVO_TO_JOINT` 顺序 + fps 30）。

### 7.4 对我们（SO-101 real-harness）最直接的三条

- **控制/推理分离照 lerobot**：同机 `fps=30` + 后台线程 RTC（策略是 flow/VLA 时），或照 openpi 把策略放另一台机走 msgpack-websocket。
- **自动判成功照 HIL-SERL / AutoEval**：训一个相机图像 reward classifier（约 200 正 / 400–1000 负），成功即终止 episode。**这正好补我们「空抓 / 颜色」判定的缺口**——`so101-empty-grasp-root-cause` 已经证明「靠时间猜合爪时刻」走不通。**注意：C 路明确说开源里没有用舵机电流/力矩判抓取的成熟实现**——我们 README 里模块 A 那条「从爪舵机角度判夹取」的路线，在开源里没有先例可抄，要自己验。
- **复位必须单独建**（AutoEval 的 reset policy 或脚本化复位），否则没法无人值守滚下去。

### 7.5 C 路的未确认项

- lerobot 是否有硬件 E-stop / 扭矩上限 → **未确认**。
- hil-serl 的 10 Hz 控制频率来自中文二手解读，官方 README 未写 → 标注为「社区来源确认」。
- gym-hil 是否支持真机 → **未确认**（仓库只有仿真）。
- openpi 成功判定的具体实现 → **未确认**（例子只提示可存 success/duration）。
- AutoEval 的控制频率、fork 数 → **未确认**。
- star 来源：api.github.com 实抓；gym-hil / auto_eval / serl_franka_controllers 用 img.shields.io（API 限流）。

---

## 8. 机器人学习框架（A 路，已回）

核验方式：star 抓 GitHub HTML 的 `repo-stars-counter-star` 字段；最近提交用 blobless 浅克隆后 `git log -1`；源码经 jsDelivr 镜像读（`raw.githubusercontent` 和 WebFetch 在本环境被墙）。

### 8.1 速览表

| 项目 | URL | Stars | 最近提交 | 许可证 | 成功判定核心 | 真机接入 | backend 抽象代表 |
|---|---|---|---|---|---|---|---|
| lerobot | https://github.com/huggingface/lerobot | 27,954 | 2026-10-03 | Apache-2.0 | env `info["is_success"]` / 真机 reward classifier(0.5) | 有（`Robot` + HIL-SERL） | `EnvConfig` 注册制 + `make_env` |
| openpi | https://github.com/Physical-Intelligence/openpi | 14,094 | 2026-08-24 | Apache-2.0 | env `done` | 有（示例，无阈值判定） | 无，靠 per-platform transform |
| RoboVerse | https://github.com/RoboVerseOrg/RoboVerse | 1,865 | 2026-09-08 | Apache-2.0 | **checker 对象** | 有（hybrid / 感知 detector） | **`BaseSimHandler` + 10 后端** |
| ManiSkill3 | https://github.com/mani-skill/ManiSkill | 3,375 | 2026-08-01 | Apache-2.0 | `evaluate()["success"]` | **有（`Sim2RealEnv`）** | `BaseEnv` + `sim_backend` |
| RoboCasa | https://github.com/robocasa/robocasa | 1,775 | 2026-09-24 | MIT | robosuite `_check_success()` | 无 | 继承 robosuite |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | 2,382 | 2025-03-15 | MIT | BDDL `_check_success()` | 无 | robosuite 包装类 |
| RoboTwin 2.0 | https://github.com/RoboTwin-Platform/RoboTwin | 2,939 | 2026-09-24 | MIT | **几何阈值 + 夹爪态** | 无 | 单一 SAPIEN |
| SimplerEnv | https://github.com/simpler-env/SimplerEnv | 1,174 | 2026-09-30 | MIT | env `done` | 无（用真机表对照） | ManiSkill2_real2sim |
| Genesis | https://github.com/Genesis-Embodied-AI/genesis-world | 30,023 | 2026-10-05 | Apache-2.0 | **无**（用户自写） | 无 | 引擎，被上层封装 |
| robomimic | https://github.com/ARISE-Initiative/robomimic | 1,572 | 2026-08-09 | MIT | `is_success()["task"]` | 无 | `EnvBase` + `EnvType` |
| robosuite | https://github.com/ARISE-Initiative/robosuite | 2,642 | 2026-07-11 | MIT | `_check_success()` | 无 | `MujocoEnv` 层级 |
| Isaac Lab | https://github.com/isaac-sim/IsaacLab | 8,282 | 2026-10-05 | BSD-3-Clause | termination term（无独立 success） | 无 | Manager 组合 / `DirectRLEnv` |

### 8.2 逐个要点（只记 harness 相关的）

- **lerobot**：`EnvConfig`（`draccus.ChoiceRegistry` + ABC）是后端基类，每个 benchmark 一个 `@EnvConfig.register_subclass(...)`——`aloha / pusht / libero / metaworld / robocasa / robomme / robotwin / vlabench`，真机是 `HILSerlRobotEnvConfig`（注册名 `gym_manipulator`）。`create_envs()` 统一返回 `{suite: {task_id: VectorEnv}}`。真机不走 env，走 `robots/robot.py::Robot(ABC)`（`connect/calibrate/get_observation/send_action/configure`），rollout 层包成 `ThreadSafeRobot`。**成功判定直接约定 env 必须吐 `info["is_success"]`**。停止条件三层：`max_steps`（gym 注册）+ `n_episodes`（采够条数）+ `cfg.duration`（真机）。**没有成功率滑窗早停**。标定在 `Robot._load_calibration/_save_calibration`，draccus 存 JSON。数据集 v3 写入 `create() → add_frame() → save_episode() → finalize()` + `StreamingVideoEncoder` 流式编码。
- **openpi**：**没有独立 backend 层**。核心是 `Policy.infer(obs: dict) -> dict`，前后各一串可组合的 `DataTransformFn`（`transforms.py`），平台差异靠 transform 插件（`aloha_policy / droid_policy / libero_policy`）。episode 循环在各 `examples/<platform>/main.py`，`replan_steps=5` 做 chunk 重规划，LIBERO 每 suite 独立 `max_steps`（220/280/300/520/400）。成功判定 = env 的 `done`。
- **RoboVerse（backend 抽象最彻底）**：`sim/base.py::BaseSimHandler(ABC)` = `launch/render/close/set_states/get_states/set_dof_targets/simulate/...`；具体后端 `isaacsim/ isaacgym/ mujoco/ mjx/ pybullet/ sapien/ genesis/ blender/ newton/`。另有 `HybridSimHandler`（半真半仿）。**成功判定独立成 `checker` 对象**（`EmptyChecker / DetectedChecker(配 BaseDetector) / JointPosChecker / JointPosShiftChecker`），统一接口 `reset(handler, env_ids)` + `check(handler, states) -> BoolTensor`——**真值 checker 与感知 checker 并存、可替换**。
- **ManiSkill3**：`Sim2RealEnv(sim_env, agent: BaseRealAgent)` 是本簇**最直接的「同一 env 类跑真机」**——强制 `sim_backend == "physx_cpu"`，复用仿真的 `observation_space/action_space`，`_check_observations()` **逐键断言真机观测与仿真观测 shape/dtype 一致**。真机 agent 接口 `reset(qpos) / capture_sensor_data / get_sensor_params / stop`。另有 `sensor_data_preprocessing_function`（真机相机 center-crop + resize 对齐仿真）和 `real_reset_function`。成功判定 `evaluate() -> {"success": BoolTensor, "fail": ...}`。
- **RoboCasa**：不自造后端，直接继承 robosuite。关键在 `robomimic_env_wrapper.py::EnvRobocasa` 实现 robomimic 的 `EnvBase` 协议 → **一个 benchmark 靠一个适配器就复用了整套 robomimic 训练/评测 harness**。`is_done()` 恒 `False`（固定 horizon）。数据三格式并存：robomimic HDF5 / LeRobot / GR00T。
- **LIBERO**：BDDL `.bddl` 定义 goal，`_check_success()` 与 goal 一一对应；`*.pruned_init` 存每任务初始状态。`ControlEnv → OffScreenRenderEnv / SegmentationRenderEnv / DemoRenderEnv` 三层包装。`reset()` 里对 `RandomizationError` 循环重采样。
- **RoboTwin 2.0**：`check_success()` 是**「几何阈值 + 夹爪状态」**，例：`0.08 < dist(obj,target) < 0.2` 且 `obj.x < target.x` 且 `|Δy| < 0.05` 且两爪都张开。**每个任务的评测步数上限单独放 `env_cfg/task_config/_eval_step_limit.yml`**（按任务名查表，缺省 1000）。`eval_policy_multitask.py` 是带 GPU slot 调度的评测调度器。
- **SimplerEnv**：灵魂是 `utils/metrics.py`——内置真机基准表 `REAL_PERF` 与仿真表 `SIMPLER_PERF`，提供 `pearson_correlation`、`mean_maximum_rank_violation`、`print_all_kruskal_results`。三重循环扫初始位姿网格 × 物体位姿。success/failure 编码进视频文件名。
- **Genesis**：**不是 harness 框架，是物理引擎**。`Simulator` 只有 `build/destroy/reset/step/get_state/save_ckpt/load_ckpt`；无 `_check_success`、无 env 注册、无数据集格式，episode 循环全在 example 里自写。它在本簇的关系是「被 RoboVerse 当后端调用」。可迁移的只有 `save_ckpt/load_ckpt` 和可微 `step`。
- **robomimic**：`EnvBase(ABC)` + `EnvType`（ROBOSUITE/GYM/IG_MOMART），`is_success() -> {"task": bool}` 是全簇最通用约定。**演示 HDF5 里 embed `env_meta`（`env_name/type/env_kwargs`）与 `env_version`**，`check_env_version()` 比对数据集与本地环境，不一致告警。`rollout()` 里 `if done or success: break`；`config.experiment.rollout.rate` 是**唯一确认到的「成功率滑窗达标即停」机制**。
- **robosuite**：`_check_success()` 是独立于 `done` 的可覆盖方法（`Lift._check_success()` = `cube_height > table_height + 0.04`），`self.done = (timestep >= horizon) and not self.ignore_done`——**「任务是否成功」与「episode 是否结束」解耦**。真机拿不到 done 只能轮询 success 时，正是要复用这个模式。
- **Isaac Lab**：`ManagerBasedRLEnv` 用**声明式 manager 组合**（Observation/Action/Reward/Termination/Curriculum/Command/Event/Recorder）。**没有单独的 `is_success`——成功就是某个 termination term**（`reset_terminated` 承载，`reset_time_out` 与之并列）。recorder 挂在 reset/step 事件上（`PreStepActionsRecorder` 等）。

### 8.3 A 路的六条横向结论（原文保留）

1. **「仿真+真机同一接口」最成熟的两种写法**：① ManiSkill `Sim2RealEnv(sim_env, BaseRealAgent)`——复用仿真 env 类 + `_check_observations` 一致性断言；② lerobot `EnvConfig` 注册制——真机是 `gym_manipulator` 子类、机器人是 `Robot` 接口，两条入口共享同一 rollout 框架。**可对照我们 `MujocoServoController` / `ServoControllerStandalone` 的 twin-controller 设计。**
2. **成功判定**：仿真侧一律是 env 暴露真值（`is_success()` / `_check_success()` / `evaluate()["success"]` / `info["is_success"]`），且**都把「是否成功」与「episode 是否结束」解耦**（robosuite `ignore_done`、Isaac Lab 独立 `time_out`、RoboCasa `is_done()` 恒 False）。真机侧只有 lerobot 给了机制（`rewards/classifier/` + `success_threshold=0.5`）。**RoboTwin 的 `check_success()`（几何阈值 + 夹爪状态）最贴近我们真机阈值判定的写法。**
3. **停止条件三件套的落点**：采够条数（`n_episodes` / `n_rollouts`）、硬上限（`max_episode_steps` / `horizon`，RoboTwin 单独放 `_eval_step_limit.yml`）、成功率早停（**只有 robomimic `experiment.rollout.rate` 确认到**）。
4. **数据格式两派**：LeRobot dataset（lerobot/openpi/RoboVerse/ManiSkill/RoboCasa 均可导出）vs HDF5（robomimic/robosuite/LIBERO/RoboTwin/Isaac Lab Mimic）。**流式写入最佳范例是 lerobot `add_frame → save_episode → finalize` + `StreamingVideoEncoder`**。
5. **checkpoint 评测组织**：轻量派（robomimic `run_trained_agent`、openpi `examples/*/main.py`）直接脚本 rollout；分布式/服务化派（RoboTwin `eval_policy_multitask.py` 按 GPU slot 调度 + 策略 server、openpi websocket policy server + 多 env worker）。
6. **「每硬件一份参数」**：**只有 lerobot 真机侧有明确的标定文件管理**（`Robot._load_calibration/_save_calibration`，draccus JSON，配 `lerobot_calibrate` CLI 与 `calibration_gui.py`）；RoboTwin 把每任务步数上限、embodiment 参数集中在 `env_cfg/*.yml`。其余几乎全是纯仿真、无硬件标定概念。

### 8.4 A 路的未确认项

- 各仓库最新 release tag 未取到（`/releases/latest` 的 302 在本环境不稳定），统一用默认分支 HEAD 提交日期判断新旧。
- Genesis star 抓 HTML 得 30,023（同日另一快照 29,957，缓存差异）。
- Isaac Lab RL 训练脚本未逐文件核验（Mimic/robomimic 对接已核验）。
- RoboVerse hybrid 模式与 ManiSkill `BaseRealAgent` 的具体硬件驱动实现未展开。
- lerobot `dataset_metadata.py` 的 `CODEBASE_VERSION` 常量未直接读到（由转换脚本与文件头注释判定为 v3）。
- **RoboCasa / robosuite 的 GitHub API `spdx_id` 显示 NOASSERTION，但 LICENSE 文件首行确为 MIT**——以文件为准。

---

## 附：未确认项（不编数字）

- EmbodiedBench / AgiBot-World / LIBERO-plus 的**许可证**：GitHub API 与 ecosyste.ms 均未检出标准 license 文件 → 未确认
- genie_sim 许可证为 "other"（自定义），未逐条读
- OmniGibson 与 BEHAVIOR-1K star 都返回 1735，OmniGibson 已归档并入 BEHAVIOR-1K，数值存疑
- WorldArena / ForesightSafety-VLA / IndustrialVLA-Bench 的官方 repo 与 star 未定位到 → 未确认，仅据论文标题
- **调研环境里 github.com 直连被封，WebFetch 全域同样被封**；star 数经 GitHub REST API + ungh.cc + ecosyste.ms 三源交叉核对，主要仓库双源一致
