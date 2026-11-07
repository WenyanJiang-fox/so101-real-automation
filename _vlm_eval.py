# -*- coding: utf-8 -*-
"""在 AutoDL 那台上跑 Qwen3-VL，量「方块在不在两爪之间」判得准不准。

    /root/qwen3vl_env/bin/python -u /root/autodl-tmp/_vlm_eval.py <jsonl> [adapter]

`adapter` 给了就套 LoRA（ms-swift 存出来的 checkpoint-XXX 目录）；不给就是基座。
输出逐条结果 + 混淆矩阵。**判据口径只有一个**：解析出 `{"between": "yes"|"no"}`，
解析不出来算「没答」，单独计数 —— 不替它猜。
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/vlm_judge_sft/val_remote.jsonl"
    adapter = sys.argv[2] if len(sys.argv) > 2 else ""
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        from peft import PeftModel                                  # noqa: PLC0415

        model = PeftModel.from_pretrained(model, adapter)
        print(f"[eval] LoRA：{adapter}", flush=True)
    model.eval()

    cm = Counter()
    bad = []
    t0 = time.time()
    for i, r in enumerate(rows):
        want = json.loads(r["messages"][1]["content"])["between"]
        text = r["messages"][0]["content"]
        content = [{"type": "image", "image": p} for p in r["images"]]
        content.append({"type": "text", "text": text.replace("<image>", "")})
        msgs = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        got = proc.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        pred = "?"
        try:
            pred = json.loads(got[got.index("{"):got.rindex("}") + 1])["between"]
            pred = "yes" if str(pred).lower().startswith("y") else "no"
        except Exception:                                           # noqa: BLE001
            pass
        cm[(want, pred)] += 1
        if pred != want:
            bad.append((i, want, pred, got[:60]))
            print(f"  ✗ {i:02d} 该 {want} 判 {pred}：{got[:60]!r}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  … {i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    n = len(rows)
    right = sum(v for (w, p), v in cm.items() if w == p)
    print(f"\n[结果] {path}  样本 {n}  对 {right}  "
          f"准确率 {right / max(n, 1) * 100:.1f}%  用时 {time.time() - t0:.0f}s")
    for w in ("yes", "no"):
        for p in ("yes", "no", "?"):
            if cm[(w, p)]:
                print(f"  真 {w:3s} → 判 {p:3s}: {cm[(w, p)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
