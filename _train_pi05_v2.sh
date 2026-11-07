#!/bin/bash
# 2026-10-25：数据集换成 285 条的 clean2_4hz，从 so101_v1/159999 的 params 接着练 2 万步。
# 日志 /root/autodl-tmp/train_v2.log。**不加 --overwrite / --resume**：新目录 so101_v2，
# 那两个参数一个会把现有 checkpoint 删掉、一个要求目录里已经有东西。
#
# **这几个 export 一条都不能少**（2026-10-25 第一次启动就栽在这儿）：
# 非交互 SSH 不会 source `~/.bashrc`，`HF_LEROBOT_HOME` 缺了就退回
# `~/.cache/huggingface/lerobot` —— 那里没有数据集，lerobot 转去 hub 上找，
# 而服务器到 huggingface.co 是不通的，直接 Network is unreachable。
set -x
exec >> /root/autodl-tmp/train_v2.log 2>&1

export PATH=/root/miniconda3/bin:$PATH
export HF_LEROBOT_HOME=/root/autodl-tmp/hf_lerobot
export HF_ENDPOINT=https://hf-mirror.com
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_data
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

cd /root/autodl-tmp/openpi || exit 1
date
df -h /root/autodl-tmp | tail -1
/root/miniconda3/envs/pi05/bin/python -c "
import json
p='/root/autodl-tmp/hf_lerobot/so101_real/pick_grasp_2cam_clean2_4hz/meta/info.json'
j=json.load(open(p)); print('DATASET', j.get('total_episodes'), 'eps', j.get('total_frames'), 'frames', j.get('fps'), 'fps')
"
/root/miniconda3/envs/pi05/bin/python scripts/train.py pi05_so101_lora_v2 --exp-name so101_v2
echo "TRAIN EXIT $?"
date
