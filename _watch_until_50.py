# -*- coding: utf-8 -*-
"""盯数据集集数，到 50 就给面板发 WM_CLOSE（走面板自己的退出流程：回初始位置 + finalize）。

一次性的现场小工具，不挂系统定时任务。40 分钟没到 50 就放弃，不碰面板。
"""
import json
import subprocess
import sys
import time

INFO = r"G:\sim2real\so101-real-harness\datasets\so101_real_grasp_2cam_4hz\meta\info.json"
TARGET = 50
DEADLINE_S = 40 * 60


def total() -> int:
    try:
        with open(INFO, encoding="utf-8") as f:
            return int(json.load(f).get("total_episodes") or 0)
    except Exception:
        return -1


def panel_pids() -> list[int]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
         "Where-Object { $_.CommandLine -match 'm panel' } | "
         "Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True).stdout.split()
    return [int(x) for x in out if x.strip().isdigit()]


def main() -> int:
    deadline = time.time() + DEADLINE_S
    last = None
    print("watcher start, total =", total(), flush=True)
    while time.time() < deadline:
        n = total()
        if n != last:
            print(time.strftime("%H:%M:%S"), "total =", n, flush=True)
            last = n
        if n >= TARGET:
            pids = panel_pids()
            print(time.strftime("%H:%M:%S"), "到 50，收工：面板 PID", pids, flush=True)
            for pid in pids:
                # 不带 /F = 发 WM_CLOSE，走面板自己的退出流程（回初始位置 + finalize）
                subprocess.run(["taskkill", "/PID", str(pid)], capture_output=True)
            time.sleep(25)
            for pid in panel_pids():
                print("还在，强杀", pid, flush=True)
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            print("已停。最终 total =", total(), flush=True)
            return 0
        time.sleep(5)
    print("超时40分钟，没停面板。当前 total =", total(), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
