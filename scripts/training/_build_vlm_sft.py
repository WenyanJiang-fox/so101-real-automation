"""把 `out/vlm_gallery/` 里的成败样本转成 ms-swift 的多模态 SFT 数据集。

    G:/anaconda/envs/sim2real/python.exe -u _build_vlm_sft.py

做三件事：

1. 把图**拷出 `out/`** —— 那边的清理会删 3 天前的文件，不拷就等于三天后样本全没了；
2. 按时间分层切成训练 / 留出（约 8:3），held 和 empty 各自按比例；
3. 写 `train.jsonl` / `val.jsonl` / `meta.json`（ms-swift 的多模态格式：
   `messages` 里用 `<image>` 占位，路径放 `images` 列表）。

标签只有一个来源：6 号舵机合爪到不到底 —— `_held` = yes（方块顶住爪子），
`_empty` = no（合到底）。`unknown` 没有可信标签，整条丢掉。
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(r"G:\sim2real\so101-real-harness")
SRC = ROOT / "out" / "vlm_gallery"
DST = ROOT / "datasets" / "vlm_judge_sft"
VAL_EVERY = 4          # 每 4 条挑 1 条进留出

PROMPT = """你看的是一台机械臂合爪之前的腕相机画面。

给你两张图：
  ·【此刻】下探之后、爪子张开、还没合拢的那一刻；
  ·【参考】同一趟下探之前的位置。

只回答一件事：方块此刻是不是落在两个爪子之间、合拢时能夹到它。

只输出一行 JSON，不要别的字：{"between": "yes"} 或 {"between": "no"}
"""


def _rows() -> list[dict]:
    rows: list[dict] = []
    if not SRC.is_dir():
        return rows
    for d in sorted(p for p in SRC.iterdir() if p.is_dir()):
        name = d.name
        if name.endswith("_held"):
            label = "yes"
        elif name.endswith("_empty"):
            label = "no"
        else:
            continue
        now, pre = d / "now.jpg", d / "pre.jpg"
        if not (now.is_file() and pre.is_file()):
            print(f"  [跳过] {name}：缺图")
            continue
        rows.append({"src": d, "name": name, "label": label})
    return rows


def _split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """按 label 分开，各自每 VAL_EVERY 条挑 1 条进留出 —— 保持两类的比例。

    `FIXED_VAL=1` 时沿用**上一份 `val.jsonl` 里那几条**当留出，新样本全进训练：
    换留出集就没法拿前后两轮的数字比。上一份 val 里已经被删的样本跳过。
    """
    if os.environ.get("FIXED_VAL") == "1" and (DST / "val.jsonl").is_file():
        keep = set()
        for line in (DST / "val.jsonl").read_text(encoding="utf-8").splitlines():
            for p in json.loads(line)["images"]:
                keep.add(Path(p).parent.name)
        train = [r for r in rows if r["name"] not in keep]
        val = [r for r in rows if r["name"] in keep]
        print(f"  [固定留出] {len(val)} 条（旧 val 里还在的）；其余 {len(train)} 条进训练")
        return train, val
    train: list[dict] = []
    val: list[dict] = []
    for label in ("yes", "no"):
        part = [r for r in rows if r["label"] == label]
        for i, r in enumerate(part):
            (val if i % VAL_EVERY == 0 else train).append(r)
    return train, val


def _emit(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            dst_dir = DST / "images" / r["name"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            now = dst_dir / "now.jpg"
            pre = dst_dir / "pre.jpg"
            if not now.is_file():
                shutil.copy2(r["src"] / "now.jpg", now)
            if not pre.is_file():
                shutil.copy2(r["src"] / "pre.jpg", pre)
            rec = {
                "messages": [
                    {"role": "user",
                     "content": f"<image><image>{PROMPT}"},
                    {"role": "assistant",
                     "content": json.dumps({"between": r["label"]})},
                ],
                "images": [str(now), str(pre)],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    rows = _rows()
    if not rows:
        print(f"{SRC} 里没有可用样本")
        return 1
    train, val = _split(rows)
    DST.mkdir(parents=True, exist_ok=True)
    _emit(train, DST / "train.jsonl")
    _emit(val, DST / "val.jsonl")
    (DST / "meta.json").write_text(json.dumps({
        "source": str(SRC),
        "total": len(rows),
        "train": len(train),
        "val": len(val),
        "label_yes": sum(1 for r in rows if r["label"] == "yes"),
        "label_no": sum(1 for r in rows if r["label"] == "no"),
        "prompt": PROMPT,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    y_tr = sum(1 for r in train if r["label"] == "yes")
    y_va = sum(1 for r in val if r["label"] == "yes")
    print(f"总 {len(rows)}（yes {sum(1 for r in rows if r['label'] == 'yes')} / "
          f"no {sum(1 for r in rows if r['label'] == 'no')}）")
    print(f"训练 {len(train)}（yes {y_tr} / no {len(train) - y_tr}）→ {DST / 'train.jsonl'}")
    print(f"留出 {len(val)}（yes {y_va} / no {len(val) - y_va}）→ {DST / 'val.jsonl'}")
    print(f"图 {DST / 'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
