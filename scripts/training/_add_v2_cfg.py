# -*- coding: utf-8 -*-
"""在服务器上的 `openpi/src/openpi/training/config.py` 里加一条 `pi05_so101_lora_v2`。

和已有的 `pi05_so101_lora_endseg` 同一套路：**只换数据集和步数**，权重从
`pi05_so101_lora/so101_v1/159999` 接着来，其余超参一字不改。这样 159999 那份
checkpoint 不会被 `max_to_keep=1` 删掉（继续训同一个目录会删）。

    /root/miniconda3/envs/pi05/bin/python -u /root/autodl-tmp/_add_v2_cfg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

F = Path("/root/autodl-tmp/openpi/src/openpi/training/config.py")
ANCHOR = '    TrainConfig(\n        name="pi05_so101_lora_endseg",'

NEW = '''    TrainConfig(
        name="pi05_so101_lora_v2",
        # 2026-10-25：数据集从 202 条涨到 285 条（`clean2_4hz` 补齐了没传上去的那些），
        # 从 `pi05_so101_lora/so101_v1/159999` 接着练 2 万步。**训练步数只加这一段**，
        # 别的超参和 `pi05_so101_lora` 一模一样；LR 恒在 1e-5（peak = decay）。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotSo101DataConfig(
            repo_id="so101_real/pick_grasp_2cam_clean2_4hz",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/autodl-tmp/openpi_runs/checkpoints/pi05_so101_lora/so101_v1/159999/params"
        ),
        batch_size=2,
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200, peak_lr=1e-5, decay_steps=20_000, decay_lr=1e-5
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        assets_base_dir="/root/autodl-tmp/openpi_runs/assets",
        checkpoint_base_dir="/root/autodl-tmp/openpi_runs/checkpoints",
        save_interval=1000,
        keep_period=None,
        wandb_enabled=False,
        num_workers=2,
    ),

'''


def main() -> int:
    src = F.read_text(encoding="utf-8")
    if 'name="pi05_so101_lora_v2"' in src:
        print("已经有 v2 了，跳过")
        return 0
    if ANCHOR not in src:
        print("找不到插入点（endseg 那条）")
        return 1
    bak = F.with_suffix(".py.bak_20261025")
    if not bak.exists():
        bak.write_text(src, encoding="utf-8")
    F.write_text(src.replace(ANCHOR, NEW + ANCHOR, 1), encoding="utf-8")
    print("已插入 pi05_so101_lora_v2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
