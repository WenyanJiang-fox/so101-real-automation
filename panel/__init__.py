"""真机 SO-101 的面板：**整条臂的运动都从这块窗口上按出来。** 一个原生 Windows 窗口，
左边画面、右边按钮。

**为什么要单独一包。** `skills/pick_real.py` 把「策略」和「交互」焊在一起了 —— 它每走
一步就在那儿等一个人从外面写一个 JSON（`out/clear_go.json`），不写就干等到超时（15
分钟）然后判这一趟不合爪。实跑一轮 `out/run_real_amp.log` 就栽在这儿。时间全花在等
那个 JSON 上。

这一包把几层拆开，一层一个文件：

    panel/rig.py          硬件那一层 —— 串口 + 腕相机 + 感知 + 关节轨迹。跑在自己的线程里。
    panel/measure.py      算那一层   —— 纯函数：从一帧画面量出缝/轴偏、再算出下一步给多少。
    panel/view.py         画那一层   —— 把算出来的东西标在画面上（只画，不算）。
    panel/skill_driver.py 断点那一层 —— 把技能跑起来（闸口全开），在它自己的断点上写
                                      `out/clear_go.json` 当答复。
    panel/ui.py           按那一层   —— tkinter 窗口、按钮、键盘。只读快照、只下命令。
    panel/__main__.py     入口       —— `python -m panel`。

**臂怎么动。** 按「▶ 跑一趟」＝ 把整个 `pick_real` 技能跑起来，闸口全开。技能会在
「微调每一步 / 下探那一截之前 / 判过了先不合 / 横移掉下去要抬回来」这**四处自己停下来**
（`skills/pick_real.py` 里那四个 `_confirm_*`），等面板写 `out/clear_go.json`：
`{"go": true}` 照算的走、`{"move": [dx,dy]}` 直接给位移（毫米）、`{"descend": true}` 微调
收工直接下探、`{"dz": -3}` 改钉住的那一档高度、`{"abort": true}` 这一趟不干了。

**停在断点上的时候，面板的关节、夹爪、TCP 点动照样能用。** 靠的是技能自己那根
`WAIT_TICK` 钩子（`skills/pick_real.py`）：等的时候它每约 0.1 秒被调一次，面板就在那一头
把队列里的按键做掉 —— 串口只有一条，本来就该由等的那一头去动。见 `rig.pump_wait`。


**窗口是 tkinter，不是 cv2。** 原来那版用 `cv2.namedWindow` 自己画按钮，两个硬伤：
按钮是画出来的假按钮，而且 `cv2.putText` **只认 ASCII**（这套 cv2 里连 `freetype` 都没有），
中文画出来是一排问号 —— 整块面板被迫全英文。tkinter 是标准库、用的是系统原生控件，
中文正常、按钮能按。**画面上的字仍然是英文**（那是 `cv2.putText` 画进帧里的，换窗口救
不了它），中文一律走这个窗口的「读数」「日志」两块和终端。

**判据和阈值一个都不在这一包里定。** 缝的带宽、轴偏容差、"下一步该走多少"的算式、
下探终点用哪个常数，全部从 `skills/pick_real.py` 里取现成的那一份（`CLEAR_BAND_*` /
`AXIS_TOL_PX` / `_gap_ok` / `_gap_axis_px` / `_box_center_px` / `_solve_delta` /
`GRASP_Z_MARGIN_REAL` / `NUDGE_STEP_M`）。那边改一个数，这里跟着变 —— 再抄一份的话
两边迟早不是一回事。

跑：

    cd G:\\sim2real\\so101-real-harness
    G:/anaconda/envs/sim2real/python.exe -m panel

**它占着串口和相机。** 和 `run_sim.py` / `view_cam.py` / `mcp_server.py` 不能同时开。
"""
