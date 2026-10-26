# OPD/OPMD Research Repo

Research codebase for On-Policy Distillation (OPD) of LLMs. Paper: https://arxiv.org/abs/2604.13016

## Project structure

- `OPD/` — Published OPD code (ICML 2026 FoGen). Based on [verl](https://github.com/verl-project/verl) v0.7.0 and [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) v0.9.5.
- `OPMD/` — Experimental extension: applying OPD to mid-training (continual pre-training). Currently placeholder (empty `datasets/`).
- `OPMD_models/` — Local model checkpoints (`DeepSeek-R1-Distill-Qwen-1.5B`, `qwen3-4b-base`, `qwen3-8b-base`)
- `OPMD_idea.md` — Research plan (Chinese)
- `OPMD_logs/`, `OPMD_graph/` — Empty output directories

Upstream verl AGENTS.md at `OPD/verl/verl/AGENTS.md`.

## Setup

Two conda environments:
```bash
# OPD/RL (verl v0.7.0)
conda create -n verl python=3.12
conda activate verl
cd OPD/verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify

# SFT (LlamaFactory v0.9.5)
conda create -n sft python=3.11
cd OPD/LlamaFactory/
pip install -e .
pip install -r requirements/metrics.txt
```

## Key commands

```bash
# OPD training
bash OPD/on_policy_distillation.sh

# RL (GRPO)
bash OPD/grpo.sh

# SFT
llamafactory-cli train OPD/LlamaFactory/examples/train_full/qwen3_base_full_sft.yaml

# Teacher response generation (for SFT data)
python OPD/scripts/infer/vllm_rollout.py --input-parquet <path> --model-path <path>

# Validation
cd OPD/scripts/val/eval && python gen_vllm.py && python grade.py

# Deduplicate DeepMath against DAPO-Math-17K
python OPD/scripts/infer/dedup_deepmath.py
```

## Critical quirks

- **verl v0.7.0 validation bug**: built-in validation under-estimates by 5–7 points. Set `trainer.test_freq=-1` and validate separately with `OPD/scripts/val/`. Fix in verl v0.8.0.
- **Non-thinking models**: must add `+data.apply_chat_template_kwargs.enable_thinking=False` to training script (OPD and RL).
- **Training requires**: `ray stop --force; ray start --head` before launching (already in shell scripts).
- **Hardware**: 8× NVIDIA A800 80GB GPUs, Slurm cluster (partition `TEST1`).
- **Logging**: SwanLab by default (`trainer.logger=['console','swanlab']`).

## OPD key parameters

| Variable | Default | Notes |
|---|---|---|
| `ADV_ESTIMATOR` | `token_reward_direct` | OPD; set to `grpo` for RL |
| `LOG_PROB_TOP_K` | `16` | `0` = sampled-token OPD (fallback) |
| `TOP_K_STRATEGY` | `only_stu` | `only_tch`, `intersection`, `union`, `union-intersection` |
| `REWARD_WEIGHT_MODE` | `student_p` | `teacher_p` or `none` |
| `USE_KL` | `False` | Optional KL penalty |
| `LOSS_AGG_MODE` | `token-mean` | How to aggregate token losses |

## OPMD research context

OPMD applies OPD-style distillation to mid-training (continual pre-training). Uses a proxy model (DeepSeek-R1-Distill-Qwen-1.5B) for anchor token selection, then applies student-generated CoT + teacher evaluation. Teacher can be any model (including weaker ones). Based on ideas from RPT and RLP. See `OPMD_idea.md` for full plan.
