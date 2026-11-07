# -*- coding: utf-8 -*-
"""零样本探针：合爪后那张图，模型能不能看出「方块稳稳夹在两爪面之间」。

    /root/qwen3vl_env/bin/python -u /root/autodl-tmp/_vlm_eval_hold.py <jsonl> [adapter]

`adapter` 不给就是基座。只解析 `{"held": "yes"|"no"}`，解析不出来算「没答」，
单独计数 —— 不替它猜。
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"

PROMPT = (
    "这是机械爪刚合上的一瞬间，从腕部相机往下拍。"
    "方块是不是稳稳地落在两个爪面之间？判断标准：方块整体夹在两片爪面中间、"
    "夹住的高度足够，才算 yes；只在指尖搭了一点、歪在爪缝外面、或者爪里没有方块，"
    "都算 no。只回答一行 JSON：{\"held\": \"yes\" 或 \"no\"}"
)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/vlm_hold_probe/probe_remote.jsonl"
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
        want = r["label"]
        content = [{"type": "image", "image": p} for p in r["images"]]
        content.append({"type": "text", "text": PROMPT})
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
            pred = json.loads(got[got.index("{"):got.rindex("}") + 1])["held"]
            pred = "yes" if str(pred).lower().startswith("y") else "no"
        except Exception:                                           # noqa: BLE001
            pass
        cm[(want, pred)] += 1
        flag = "" if pred == want else "  ✗"
        print(f"[{i + 1}/{len(rows)}] 真 {want:4s} 判 {pred:4s}{flag}  {r.get('name', '')}",
              flush=True)
        if pred != want:
            bad.append({"name": r.get("name"), "want": want, "pred": pred, "got": got[:120]})

    n = len(rows)
    right = sum(v for (w, p), v in cm.items() if w == p)
    print("\n=== 混淆（真 → 判） ===")
    for k in sorted(cm):
        print(f"  {k[0]:4s} → {k[1]:4s}  {cm[k]}")
    print(f"\n准确率 {right}/{n} = {right / max(1, n):.1%}"
          f"   用时 {time.time() - t0:.0f}s")
    miss = [b for b in bad if b["pred"] == "?"]
    if miss:
        print(f"没答 {len(miss)} 条：{json.dumps(miss, ensure_ascii=False)[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
