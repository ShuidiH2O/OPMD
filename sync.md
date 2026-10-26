# 存储迁移记录（sync.md）

> 账号：`zhengruize-20260612`
> 源：`/mnt/cpfs/epic-user/zhengruize-20260612`（CPFS 共享存储）
> 目标：`/mnt/epic-oss/epic-user/zhengruize-20260612`（OSS 对象存储，ossfs 挂在 `/mnt/data`）
> 时间：2026-09-21 ~ 2026-09-22

## 1. 背景与目的

集群 CPFS 存储调整，需要把研究数据安全备份到对象存储（OSS）。本文件记录：原始 CPFS 里有什么、各有什么用、用什么方式迁移、迁移了什么、以后如何在新的服务器上恢复使用。

## 2. 原始 CPFS 目录结构与作用

账号根目录 `/mnt/cpfs/epic-user/zhengruize-20260612` 下的主要内容：

| 目录/文件 | 作用 |
|---|---|
| `OPMD/` | 主要研究项目：On-Policy Mid-training Distillation。含源码（`custom/`、`scripts/`、`config/`、`opmd_train.sh`）、训练 checkpoint（`checkpoints/`）、数据集（`datasets/`）、评测结果（`eval_results/`）、日志（`logs/`）。 |
| `NTP/` | 另一个预训练实验项目（Next-Token-Prediction 基线）。含源码脚本、训练 checkpoint（`checkpoints/`）、tokenize 后的数据（`processed_data/`）、评测结果（`eval_results/`）。 |
| `OPD/` | 已发表 OPD 论文代码（ICML 2026 FoGen）。基于 verl v0.7.0 + LlamaFactory v0.9.5。 |
| `OLMo-core/` | AllenAI OLMo 源码，NTP 预训练框架参考。 |
| `OPMD_models/` | 本地下载的基座模型权重（Qwen3 系列 / Llama / DeepSeek 等，都是 HuggingFace 格式）。 |
| `OPMD_graph/` | 绘图输出（pdf/png）。 |
| `miniconda3/` | conda 环境（opmd、opmd_eval_cu12、opmd_train_cu12、opmd_train_qwen35 等）。 |
| `venv/`、`.nvm/`、`.npm/`、`.local/` | 独立 python venv 及 Node 工具链、用户级安装。 |
| `.kimi-code/`、`.codex/`、`.opencode/`、`.continue/` | 各 AI 编程工具的配置与会话数据。 |
| `.cache/`、`.triton/`、`.ray/`、`.conda/`、`.anaconda/`、`.modelscope/`、`.nv/`、`.opmd_tmp/` | 各种缓存、临时目录。 |
| `conda_env_exports/` | 本次导出的 4 个 conda 环境配置文件（env_*.yml）。 |
| `text.txt` | 训练/评测/预处理等命令的历史记录文档。 |
| `paper.md`、`OPMD_idea.md`、`AGENTS.md`、`example.md`、`coding_dataset.md`、`information&score_*.md` | 论文草稿、研究计划、项目说明、笔记等文档。 |
| `sync.md` | 本文件。 |
| 其它散文件 | `miniconda.sh`（miniconda 安装脚本）、`opmd_backup_*.tar.gz`（旧备份）、`kimi-debug-*.zip`、`.bashrc` 等 shell 配置。 |

## 3. 迁移方式

- 用 `tar` **流式打包**（`tar -cf 目标.tar -C 源目录 ...`），**不压缩**（数据基本是已压缩/二进制，压缩收益小、更慢），**不落地**（直接写到 ossfs 挂载点）。
- 分卷：每个 tar 对应一个数据类型或一个 run，体积可控、坏一个不影响其它、便于按需解包。
- **ossfs 单文件上限约 78 GiB（80000 MiB = 10000 分片 × 8 MiB）**，超过会报 `File too large`，所以超大内容按目录/run 拆分成多个 tar（每个 < 78 GiB）。
- 对大体积 checkpoint 默认 **只备份模型权重，排除优化器状态（optim/optim_*）**（optim 只在续训时需要，体积约为权重的 1.5~2 倍）；但「4b-teacher」这个尚未训完、需要断点续传的 run，**完整保留 optim 状态**。
- 用低 IO/CPU 优先级（`nice -n 19 ionice -c3 -n7`）执行，不占用训练用的 GPU 算力。
- 实测 ossfs 写大 tar 约 30~300 MB/s（文件越大、越接近单文件上限时整体平均越快）。

## 4. 本次迁移的内容（11 个 tar）

目标目录 `/mnt/epic-oss/epic-user/zhengruize-20260612/` 下的 tar：

| tar 文件 | 约大小 | 内容 |
|---|---|---|
| `code_and_docs.tar` | 314 MB | `OPD/`、`OLMo-core/`、`OPMD` 源码（custom/scripts/config/opmd_train.sh）、`NTP` 源码脚本、根目录所有 `.md`/`text.txt` 文档、`conda_env_exports/`。 |
| `toolchain.tar` | 2.71 GB | `.nvm/`、`.npm/`、`.local/`、`venv/`。 |
| `datasets.tar` | 42.7 GB | `OPMD/datasets/`：`test_data/`（评测测试集 AIME25/DAPO/MATH-500/Minerva 等）、`megamath_web_pro_1B.jsonl`（原始语料）、anchor 选取 parquet、`entropy_cache/`、`megamath_web_pro_opmd_*_train.parquet` 等。 |
| `ntp_checkpoint.tar` | 22.8 GB | `NTP/checkpoints/ntp_qwen3_1.7b_megamath_8192ctx_lr1e-5_bs64_steps-1_20260905_160044/`，仅 HF 权重（每个 step 的 `model.safetensors` + config/tokenizer + trainer_state 等），排除 DeepSpeed 的 `global_step*/` optim、`rng_state_*.pth`、`scheduler.pt`。 |
| `opmd_checkpoint_qwen3-4b-teacher_full.tar` | 71.5 GB | `OPMD/checkpoints/opmd_MegaMath-Web-Pro-OPMD-0.05_Qwen3-1.7B-Base_qwen3-4b-base_..._2026-09-21_19-52-56/` 的 global_step 100~700 完整 checkpoint（**含 optim，用于断点续传**）。 |
| `opmd_eval_teacher.tar` | 44 GB | `OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_teacher/`（高熵臂评测结果）。 |
| `opmd_eval_teacher_low.tar` | 58 GB | `OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_teacher_low/`（低熵臂评测结果）。 |
| `opmd_eval_teacher_rand.tar` | 44 GB | `OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_teacher_rand/`（随机臂评测结果）。 |
| `opmd_ckpt_high.tar` | ~45 GB | `OPMD/checkpoints/opmd_MegaMath-Web-Pro-OPMD-0.05_...`（高熵臂 run，仅模型权重分片，排 optim）。 |
| `opmd_ckpt_low.tar` | ~64 GB | `OPMD/checkpoints/opmd_MegaMath-Web-Pro-OPMD-low-0.05_...`（低熵臂 run，仅模型权重分片，排 optim）。 |
| `opmd_ckpt_rand.tar` | ~45 GB | `OPMD/checkpoints/opmd_MegaMath-Web-Pro-OPMD-rand-0.05_...`（随机臂 run，仅模型权重分片，排 optim）。 |

## 5. 未迁移 / 舍弃的内容

| 内容 | 原因 |
|---|---|
| `OPMD/checkpoints/` 其余 run、`NTP/checkpoints/` 其余 run | 本轮只按需迁移了指定的 run，其余暂未备份。 |
| `OPMD/eval_results/` 其余目录 | 同上，只迁了 14b teacher 三臂的评测结果。 |
| `OPMD_models/` 模型权重（约 141 GB） | 基座模型可从 ModelScope/HuggingFace 重新下载，暂未备份；需要时可单独补。 |
| `miniconda3/` conda 环境（约 54 GB） | 已用 `conda_env_exports/env_*.yml` 记录包版本，可在新存储重建，不迁整个目录。 |
| 「14b-teacher」各 checkpoint 中的 `optim_world_size_*.pt`、`extra_state` 之外的 optim | 这些 run 只用于评测/加载，不需要续训，故排除优化器状态。 |
| `NTP/processed_data/`（约 73 GB） | megamath 系可由 `megamath_web_pro_1B.jsonl` 重新 tokenize；finemath 系与 fineweb 混合系原始语料已删除、本轮明确放弃。 |
| 各缓存/临时目录（`.cache`、`.triton`、`.ray`、`.conda` 等）、`miniconda.sh`、旧备份 zip/tar.gz | 可重建或不再需要。 |

注：`opmd_checkpoint_qwen3-4b-teacher_full.tar` 是唯一的「完整含 optim」checkpoint，因为该 run（teacher=qwen3-4b-base → student=qwen3-1.7b-base）训练尚未结束（截至备份时保存到 global_step 700），将来需要断点续传。

## 6. 如何在新的服务器上恢复使用

1. **解包**（在目标机器挂载 OSS 后）：
   ```bash
   cd /path/to/restore
   for t in *.tar; do tar -xf /mnt/epic-oss/epic-user/zhengruize-20260612/$t; done
   ```
   注意：每个 tar 里都带目录前缀（代码包里是 `OPD/`、`OLMo-core/` 等；checkpoint 包里是 run 名 / 评测目录名），恢复后把各目录放回 `zhengruize-20260612/` 根下对应位置即可。

2. **重建 conda 环境**：
   ```bash
   conda env create -f conda_env_exports/env_opmd.yml
   conda env create -f conda_env_exports/env_opmd_eval_cu12.yml
   conda env create -f conda_env_exports/env_opmd_train_cu12.yml
   conda env create -f conda_env_exports/env_opmd_train_qwen35.yml
   ```
   另需按 `text.txt` / `AGENTS.md` 里的说明补：CUDA 12 驱动、本地源码安装 verl/LlamaFactory（`OPD/verl`、`OPD/LlamaFactory`）、`pip install math-verify` 等。

3. **恢复模型权重**：
   - NTP checkpoint 已是 HF 格式（`model.safetensors`），可直接被 transformers/vLLM 加载。
   - OPMD（14b-teacher 三臂）是 FSDP 分片，先 merge 再加载：`python OPMD/scripts/merge_checkpoint.py --checkpoint_path <run>/global_step_<N>/actor --output_path <输出HF目录>`。

4. **4b-teacher 断点续传**：
   解包 `opmd_checkpoint_qwen3-4b-teacher_full.tar` 得到完整 run 目录（含每个 `global_step_*/actor` 的 model/optim/extra_state/huggingface + `latest_checkpointed_iteration.txt`），在原 `OPMD/checkpoints/` 位置放回后，用：
   ```bash
   RESUME_CKPT=<该 run 目录> STUDENT_MODEL=qwen3-1.7b-base TEACHER_MODEL=qwen3-4b-base ARM=high \
     bash OPMD/opmd_train.sh
   ```
   即可从最新 step 继续训练。

5. **评测复现**：`OPMD/scripts/eval/` 下的 `eval_qwen3_*.sh` 脚本，配合 `datasets.tar` 解出的 `OPMD/datasets/test_data/` 测试集。

6. **训练复现**：`code_and_docs.tar` 里包含 `opmd_train.sh`、`NTP` 各 submit 脚本；`datasets.tar` 里包含训练所需数据集；模型权重按需从 ModelScope 下载到 `OPMD_models/`。14b-teacher 三臂 checkpoint 不含 optim，仅能用于评测/加载、无法接续训练；只有 `4b-teacher` 保留了 optim 可续训。