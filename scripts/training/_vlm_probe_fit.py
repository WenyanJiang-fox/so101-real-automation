# -*- coding: utf-8 -*-
"""同一个提示格式下，比基座和 LoRA 对留出样本的拟合 + 生成。

    python -u _vlm_probe_fit.py <jsonl> [adapter] [n]

输出每条：gold、yes/no 两个候选 token 的 logprob 差、贪心回答。
判据只有两个数字：`dlogp` 的符号对不对、读 LoRA 前后 loss 差多少。
"""
from __future__ import annotations

import json
import sys

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL = "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"


def build(proc, r):
    from PIL import Image

    text = r["messages"][0]["content"].replace("<image>", "")
    # 图用 PIL 真对象传（传路径字符串在 transformers 5.19 下也能工作，2026-10-24
    # 实测黑图对照 dlogp 5.5→0.5）。**但 peft 手工加载 LoRA 那条路不可信**：
    # 同一适配器在 peft 下黑图只差 0.1，在 swift 下 24/28 正确 —— 评测一律用 swift。
    content = [{"type": "image", "image": Image.open(p).convert("RGB")}
               for p in r["images"]]
    content.append({"type": "text", "text": text})
    inp = proc.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt")
    if inp.get("pixel_values") is None:
        raise SystemExit("processor 没吐出 pixel_values —— 图没进去")
    return inp


def main() -> int:
    path = sys.argv[1]
    adapter = sys.argv[2] if len(sys.argv) > 2 else ""
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    rows = [json.loads(x) for x in open(path, encoding="utf-8")][:n]

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"[probe] LoRA：{adapter}", flush=True)
    model.eval()

    # `yes` / `no` 在词表里的下一个 token —— 用第一个字的 token 做比较。
    tok = proc.tokenizer
    id_yes = tok.encode("yes", add_special_tokens=False)[0]
    id_no = tok.encode("no", add_special_tokens=False)[0]
    prefix = tok.encode('{"between": "', add_special_tokens=False)
    print(f"[probe] token yes={id_yes} no={id_no} 前缀 {prefix}", flush=True)

    # 对照：同一提示把两张图换成全黑，看 logits 会不会动。不动 = 视觉没进模型。
    if rows:
        from PIL import Image as _I

        r0 = rows[0]
        _inp = build(proc, r0)
        print(f"[probe] input_ids {tuple(_inp['input_ids'].shape)} "
              f"pixel_values {tuple(_inp['pixel_values'].shape)}", flush=True)
        for tag, imgs in (("原图", None), ("全黑", [_I.new("RGB", (1920, 1080))
                                                   for _ in r0["images"]])):
            rr = dict(r0)
            if imgs is not None:
                rr = {**r0, "images": imgs}
                text = r0["messages"][0]["content"].replace("<image>", "")
                content = [{"type": "image", "image": im} for im in imgs]
                content.append({"type": "text", "text": text})
                inp = proc.apply_chat_template(
                    [{"role": "user", "content": content}],
                    add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to(model.device)
            else:
                inp = _inp.to(model.device)
            ids = torch.cat([inp["input_ids"],
                             torch.tensor([prefix], device=model.device)], dim=1)
            mm = inp.get("mm_token_type_ids")
            if mm is not None:
                mm = torch.cat([mm, torch.zeros((1, len(prefix)), dtype=mm.dtype,
                                                device=mm.device)], dim=1)
            with torch.no_grad():
                lg = model(input_ids=ids, pixel_values=inp.get("pixel_values"),
                           image_grid_thw=inp.get("image_grid_thw"),
                           mm_token_type_ids=mm).logits[0, -1]
            print(f"[probe] {tag}: dlogp={float(lg[id_yes] - lg[id_no]):+.3f} "
                  f"max|logit|={float(lg.abs().max()):.1f}", flush=True)

    hit = 0
    for i, r in enumerate(rows):
        gold = json.loads(r["messages"][1]["content"])["between"]
        inp = build(proc, r).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=16, do_sample=False)
            got = proc.decode(out[0][inp["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            # 把答案前缀接在提示后面，再看模型在该位置给 yes / no 谁的分高。
            ids = torch.cat([inp["input_ids"],
                             torch.tensor([prefix], device=model.device)], dim=1)
            mm = inp.get("mm_token_type_ids")
            if mm is not None:
                mm = torch.cat([mm, torch.zeros(
                    (1, len(prefix)), dtype=mm.dtype, device=mm.device)], dim=1)
            logits = model(input_ids=ids, pixel_values=inp.get("pixel_values"),
                           image_grid_thw=inp.get("image_grid_thw"),
                           mm_token_type_ids=mm).logits[0, -1]
        d = float(logits[id_yes] - logits[id_no])
        pred = "yes" if "yes" in got else ("no" if "no" in got else "?")
        arg = "yes" if d > 0 else "no"
        hit += pred == gold
        print(f"  {i:02d} 真{gold:3s} dlogp(ya-n)= {d:+7.2f} 前缀判{arg:3s}  "
              f"生成判{pred:3s}  {got!r}", flush=True)
    print(f"[probe] 贪心对 {hit}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
