# Coding Domain 训练与评测记录

## 一、训练信息（NTP mid-training on StarCoderData）

| 项目 | 值 |
|---|---|
| 模型 | Qwen3-1.7B-Base |
| 数据 | StarCoderData 纯编程语言子集（1B Qwen3 tokens，78 种语言，Python 占 11.9%） |
| 训练目标 | 标准 Next-Token Prediction（causal LM） |
| 数据路径 | `OPMD/datasets/coding_1b/train.jsonl` |
| 最大序列长度 | 8192 |
| 学习率 | 1e-5 |
| Batch size | 64 |
| Epoch | 1 |
| Warmup ratio | 0.03 |
| LR scheduler | cosine |
| Weight decay | 0.01 |
| 精度 | bf16 |
| 总步数 | 1902 |
| 总 FLOPs | 8.43 × 10^18 |
| 训练 loss | 0.7587 |
| 训练时间 | 33164 s（≈ 9.2 h） |
| 硬件 | Slurm 集群（TEST1），多卡 NVIDIA 80GB GPU |

训练 checkpoint 目录：
`NTP/checkpoints/ntp_qwen3_1.7b_StarCoderData_8192ctx_lr1e-5_bs64_steps-1_20260908_171721`
（含 checkpoint-400 / 800 / 1200 / 1600 / 1902）。

## 二、评测信息

评测方式：greedy 解码（temperature=0），pass@1。

| 模型 | HumanEval+ (plus pass@1) | MBPP+ (plus pass@1) | LiveCodeBench (private pass@1) | BigCodeBench (Original pass@1) |
|---|---:|---:|---:|---:|
| base | 33.5% | 54.2% | 1.71% | 16.49% |
| step_400 | 9.1% | 18.5% | 0.95% | 15.70% |
| step_800 | 9.8% | 15.3% | 0.57% | 15.18% |
| step_1200 | 8.5% | 12.7% | 0.66% | 15.35% |
| step_1600 | 4.9% | 14.0% | 0.57% | 15.53% |
| step_1902（最终） | 6.7% | 12.4% | 0.47% | 15.44% |

> SantaCoder-FIM 已放弃：base 0.13%、所有 checkpoint 全为 0.00%，无区分度
> （Qwen3-1.7B 非 FIM 训练模型，NTP/OPD 也均为从左到右的因果预测）。

### Benchmark 版本

- HumanEval+：EvalPlus 官方 release v0.1.10（164 题）
- MBPP+：EvalPlus 官方 release v0.2.0（378 题）
- LiveCodeBench：release_latest（1055 题，2023-05-07 ～ 2025-04-06）
- BigCodeBench：data v0.1.4，`complete` split（1140 题，package `bigcodebench` 0.2.5）

数据目录：`OPMD/datasets/test_data/{HumanEvalPlus, MBPPPlus, LiveCodeBench, BigCodeBench}`。
评测结果：`NTP/eval_results/coding_{base, step_400, step_800, step_1200, step_1600, step_1902}`。

### 结论

NTP 在该 StarCoderData code 语料上继续预训练，显著拉低了简单函数补全类 benchmark：

- HumanEval+：33.5% → 6.7%
- MBPP+：54.2% → 12.4%
- LiveCodeBench：1.71% → 0.47%

但更接近真实代码补全的 BigCodeBench 仅从 16.49% 小幅降到 15.2%～15.7%，基本持平。
说明 NTP 主要破坏了「单个函数从 docstring 填空」的能力，对数据库依赖的完整函数补全影响较小。