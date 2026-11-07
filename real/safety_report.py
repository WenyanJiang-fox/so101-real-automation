# -*- coding: utf-8 -*-
"""安全闸触发时问一次 VLM，把「原因 + 整改方案」落盘存档。

**只记录，不改任何参数、不改任何代码、不重试这一集。** 采集循环随后清掉
`arm.terminated` / `arm.safety_why`，回初始位置，接着跑下一集。

落盘结构（`out/safety_reports/<时间>_ep<集号>/`）：
    wrist.jpg     触发瞬间的腕相机画面
    side.jpg      侧相机画面（config 里有才存）
    raw.txt       模型原始响应（解析失败时唯一能看出它到底回了什么）
    report.json   触发上下文 + 解析结果 + 原始响应
    report.md     给人看的一页
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

_API = "https://api.deepseek.com/chat/completions"


def _secret(name: str) -> str:
    """环境变量优先；否则读仓库根目录的 secrets.env（该文件不提交）。"""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    p = Path(__file__).resolve().parents[1] / "secrets.env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, val = line.partition("=")
                if k.strip() == name:
                    return val.strip()
    return ""


_KEY = _secret("DEEPSEEK_API_KEY")
MODEL = os.environ.get("PPO_VLM_MODEL") or "deepseek-flash"

_PROMPT_HEAD = """你是 SO-101 机械臂的安全闸诊断器，同时负责给出下一步的限制。\
这一趟自动采集被安全闸拦下了：程序发现某条指令会让机械臂做一次幅度过大的动作\
（或抓取点跑出工作包络），所以没有写舵机、立刻停手。图是触发瞬间的相机画面。

下面 JSON 是现场读数（角度是度，坐标是基座系毫米）。`state.safety_why` 是程序\
自己写的触发原话，`state.joint_deg` 是那一刻的关节角，`state.last_action_deg` 是\
上一条已经发下去的指令。

**你不指挥机械臂。** 不要给目标点、关节角、轨迹或速度 —— 那些由我们自己的策略算。\
你只调下面这几个旋钮；调完我们的三档策略会带着它们继续跑。

"""

_PROMPT_TAIL = """

cause 只能是：no_detect, off_axis, no_gap, too_low, too_high, slip, descend_fail, safety, other
action 只能是：baseline（先量一眼再微调）, no_measure（不量直接下扎）, full（完整微调）
remediation 是一句给人看的整改说明，可以自由写。

**按 `state.trip.joint` 选旋钮，别一律给 1 号那条**（2026-10-22：上一轮 62 次安全闸里\
56 次是 5 号关节，模型每次都给 `joint1_step_deg:30` —— 它管不到 5 号，而且 30 就是\
1 号本来的默认值，等于什么都没改）：
  · `trip.joint` 是 1 → 给 `joint1_step_deg`，只有比 30 更小才有意义；
  · `trip.joint` 是 2~5 → 给 `step_deg`（默认 45 = 不额外限制），报 45~55° 就给 35 或 30；
  · 没有可用的旋钮时 `params` 给 `{}`，别为了填而填。

只输出一个 JSON 对象，不要解释，不要 markdown 代码块：
{"cause":"...","evidence":"...","action":"...","params":{},"remediation":"...","confidence":0.0}

cause 不要写成「可能是」「建议检查」这类空话；evidence 必须引用上面 JSON 里的数字。
confidence 是 0~1。"""


class _Safe(dict):
    """`json.dumps` 时把 numpy / Path 变成原生类型。"""

    @classmethod
    def of(cls, v):
        if isinstance(v, dict):
            return {str(k): cls.of(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [cls.of(x) for x in v]
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, np.ndarray):
            return cls.of(v.tolist())
        if hasattr(v, "tolist"):
            return cls.of(v.tolist())
        return str(v)


def _save_jpg(frame, path: Path) -> bool:
    if frame is None:
        return False
    try:
        img = np.asarray(frame)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        return bool(cv2.imwrite(str(path), img))
    except Exception:                                              # noqa: BLE001
        return False


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _parse(text: str) -> dict | None:
    s = str(text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.split("\n", 1)[-1] if "\n" in s else s
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(s[i:j + 1])
    except Exception:                                              # noqa: BLE001
        return None
    return d if isinstance(d, dict) else None


def _ask(images: list[Path], context: dict, timeout: float = 120.0) -> tuple[dict | None, dict]:
    """调一次 VLM。返回 (解析结果, 原始响应信息)；任何失败都不抛。"""
    from ppo.vlm import VARS_DOC, filter_params                      # noqa: PLC0415

    # **变量说明和校验共用 `ppo/vlm.py` 那一份**，改范围只改一处。
    prompt = _PROMPT_HEAD + VARS_DOC + _PROMPT_TAIL
    content: list[dict] = [{"type": "text", "text": prompt}]
    for p in images:
        content.append({"type": "text", "text": f"这一张是【{p.stem}】。"})
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + _b64(p)}})
    content.append({"type": "text",
                    "text": "现场读数：\n" + json.dumps(context, ensure_ascii=False)})
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1200,
            "thinking": {"type": "disabled"}}
    req = urllib.request.Request(
        _API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + _KEY})
    info: dict = {"model": MODEL, "images": len(images)}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        msg = d["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        info.update({"elapsed_s": round(time.time() - t0, 2),
                     "finish_reason": d["choices"][0].get("finish_reason"),
                     "usage": d.get("usage"), "content": text,
                     "reasoning_content": msg.get("reasoning_content")})
    except Exception as exc:                                       # noqa: BLE001
        info.update({"elapsed_s": round(time.time() - t0, 2),
                     "error": f"{type(exc).__name__}: {exc}"})
        print(f"  [安全记录] VLM 调用失败：{type(exc).__name__} {exc}", flush=True)
        return None, info
    dec = _parse(text)
    if dec is not None:
        # 越界的单个参数丢掉，不整条作废 —— 诊断文字还要留着。
        dec["params"] = filter_params(dec.get("params"))
    return dec, info


def record(arm, wrist, side, *, ep: int, round_no: int, strategy, out: dict,
           root: Path, ask: bool = True) -> tuple[Path | None, dict | None]:
    """把一次安全闸触发存档。返回 `(存档目录, VLM 决策)`；写不进去返回 `(None, None)`。"""
    why = str(getattr(arm, "safety_why", "") or "")
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = Path(root) / f"{ts}_ep{ep:03d}_r{round_no}"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as exc:                                       # noqa: BLE001
        print(f"  [安全记录] 建目录失败：{exc}", flush=True)
        return None, None

    images: list[Path] = []
    if wrist is not None:
        p = d / "wrist.jpg"
        try:
            if _save_jpg(wrist.shot(), p):
                images.append(p)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [安全记录] 腕相机取帧失败：{exc}", flush=True)
    if side is not None:
        p = d / "side.jpg"
        try:
            if _save_jpg(side.shot(), p):
                images.append(p)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [安全记录] 侧相机取帧失败：{exc}", flush=True)

    try:
        ang = np.round(np.asarray(arm.angles(), dtype=np.float64), 2).tolist()
    except Exception:                                              # noqa: BLE001
        ang = None
    try:
        tcp = np.round(np.asarray(arm.tcp(), dtype=np.float64).reshape(-1) * 1000.0, 1).tolist()
    except Exception:                                              # noqa: BLE001
        tcp = None
    try:
        last = (None if getattr(arm, "last_action", None) is None
                else np.round(np.asarray(arm.last_action, dtype=np.float64), 2).tolist())
    except Exception:                                              # noqa: BLE001
        last = None

    # **把「哪个关节、要动多少度」单独拎出来**：`why` 是一句中文，模型以前没抓住
    # 关节号，一律照着提示词里的例子给 `joint1_step_deg`（2026-10-22 实测 43 次）。
    _m = re.search(r"一步要动 ([\d.]+)°（(\d+) 号关节", why)
    trip = ({"joint": int(_m.group(2)), "deg": float(_m.group(1)),
             "limit_deg": 45.0} if _m else None)

    ctx = _Safe.of({
        "episode": int(ep),
        "round": int(round_no),
        "safety_why": why,
        "trip": trip,
        "joint_deg": ang,
        "tcp_mm": tcp,
        "last_action_deg": last,
        "strategy": getattr(strategy, "name", str(strategy)),
        "grasp_out": out,
    })
    # `ask=False`（`--no-vlm-judge`）：存档照写，**不调 API**，也就没有 decision。
    decision, info = _ask(images, ctx) if ask else (None, {"skipped": "VLM 支路关着"})

    try:
        (d / "raw.txt").write_text(str(info.get("content") or info.get("error") or ""),
                                   encoding="utf-8")
        (d / "report.json").write_text(json.dumps(
            {"context": ctx, "decision": decision, "response": info},
            ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            f"# 安全闸记录 {ts}（第 {ep} 集 / 第 {round_no} 轮）", "",
            f"- 触发原因：{why}",
            f"- 关节角（度）：{ang}",
            f"- 抓取点（mm）：{tcp}",
            f"- 上一条指令（度）：{last}",
            f"- 这一集用的一档：{getattr(strategy, 'label', getattr(strategy, 'name', strategy))}",
            f"- 画面：{', '.join(str(p.name) for p in images) or '（没取到）'}", "",
            "## VLM 判断（params 会写进策略旋钮）", "",
        ]
        if decision:
            lines += [f"- 根因：{decision.get('cause', '')}",
                      f"- 依据：{decision.get('evidence', '')}",
                      f"- 整改方案：{decision.get('remediation', '')}",
                      f"- 套用的变量：{json.dumps(decision.get('params') or {}, ensure_ascii=False)}",
                      f"- 置信度：{decision.get('confidence', '')}"]
        else:
            lines.append(f"- 没解析出 JSON（原始响应见 raw.txt）：{info.get('error', '')}")
        (d / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:                                       # noqa: BLE001
        print(f"  [安全记录] 写文件失败：{exc}", flush=True)

    print(f"  [安全记录] 已存档：{d}", flush=True)
    if decision:
        print(f"  [安全记录] VLM：{decision.get('cause', '')} / "
              f"变量：{decision.get('params')}", flush=True)
    return d, decision
