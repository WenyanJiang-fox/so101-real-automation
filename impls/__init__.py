"""五个接口的各种实现都放这儿，一种一个文件。

    arm_sim.py            仿真机械臂
    cam_sim.py            仿真相机
    policy_expert.py      专家策略（按顺序拼小模块）
    policy_smolvla.py     模型策略
    policy_llm.py         大模型策略
    judge_truth.py        仿真真值判官
    judge_threshold.py    真机界线判官

每个文件顶上挂一个牌子（register），名字就是配置里写的那个。
阶段 1 开始往这里搬东西，现在还是空的。
"""
