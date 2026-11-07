# -*- coding: utf-8 -*-
"""在 AutoDL 那台的 `openpi/src/openpi/training/config.py` 里加一条末端微调的配置。

    python3 /root/autodl-tmp/_add_endseg_cfg.py

插在 `pi05_so101_lora` 那条后面、Aloha 那一段前面。重复跑会直接报 already there。
"""
from pathlib import Path

p = Path("/root/autodl-tmp/openpi/src/openpi/training/config.py")
s = p.read_text(encoding="utf-8")
if "pi05_so101_lora_endseg" in s:
    raise SystemExit("already there")
anchor = s.index('name="pi05_so101_lora",')
tail = s.index("\n    #\n    # Fine-tuning Aloha configs.", anchor)
block = '''
    TrainConfig(
        name="pi05_so101_lora_endseg",
        # 末端那一段的微调（2026-10-23）：从 pi05_so101_lora/so101_v1/159999 接着练，
        # 数据换成「下探到底 → 合爪」的切片（本机 `_build_endseg.py` 出的那份）。
        # 其余超参和 pi05_so101_lora 一模一样，只有训练步数和权重来源不同。
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotSo101DataConfig(
            repo_id="so101_real/pick_grasp_2cam_clean2_end_4hz",
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
        save_interval=2000,
        keep_period=None,
        wandb_enabled=False,
        num_workers=2,
    ),
'''
s = s[:tail] + "\n" + block + s[tail:]
p.write_text(s, encoding="utf-8")
print("ok", len(s))
