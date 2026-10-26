# OPMD（On-Policy Mid-training Distillation）方法说明

本文件记录 OPMD 在数据构造、消融对照与训练流程上的关键设计，重点说明（1）anchor token 的选取方式，（2）为保证消融实验公平性所采用的样本量对齐与下采样策略，以及（3）OPMD 相较于面向后训练（post-training）的 OPD 在算法与训练细节上的主要差异。所有符号沿用 `OPMD_idea.md` 中的设定。

## 1. Anchor Token 选取

OPMD 从开放领域语料中构造 on-policy 蒸馏样本，核心操作是预先选定语义上值得进行隐式推理的 **anchor token**。具体流程如下。

### 1.1 代理模型逐 token 预测熵估计

给定语料文档 `D = (x_1, ..., x_T)`，使用一个冻结的代理模型（proxy model）`π_proxy`（本工作采用 `DeepSeek-R1-Distill-Qwen-1.5B`）对文档做逐 token 条件预测，并计算每个位置 `t` 的预测熵：

```text
H(x_t | x_{<t}) = -Σ_{v∈V} π_proxy(v | x_{<t}) · log π_proxy(v | x_{<t})
```

其中 `V` 为词表。文档长度超过代理模型单次最大处理长度（4096 tokens）时，采用步长减半的滑动窗口（stride = 2048）分块前向，重叠区域取两侧熵值的平均。该步骤只做一次前向，熵值按文档落盘为可复用的缓存。

### 1.2 三臂（高熵 / 低熵 / 随机）Anchor 选取

基于上一步得到的文档级逐 token 熵序列，构造三个消融臂：

- **高熵臂（high）**：每个文档选取预测熵最高的 `k = max(1, ⌊r·T⌋)` 个 token 作为 anchor，`r = 0.05`。工程实现为先以 `r = 0.15` 做一次粗筛，再按每文档熵值排序保留前 `1/3`，从而得到约 5% 的高熵 anchor。
- **低熵臂（low）**：每个文档选取预测熵最低的 `k = max(1, ⌊0.05·T⌋)` 个 token，即 bottom-5%，排序采用稳定排序以保证确定性。
- **随机臂（rand）**：每个文档从全部 `T` 个 token 位置上**无放回均匀随机**抽取 `k = max(1, ⌊0.05·T⌋)` 个 token，不依赖熵值。随机种子设置为全局 `seed = 42`，并按文档索引 `d` 使用 `seed_d = 42 + d`，保证逐文档可复现。

由 anchor 构造训练 prompt 时，仅保留 anchor 之前的上下文 `x_{<t} = (x_1, ..., x_{t-1})` 作为 prompt，**不包含 anchor token 自身**。每个样本的 prompt 长度最多为 `max_prompt_length = 1024` tokens。在 anchor 之前的上下文不足或解码后文本过短（< 20 字符）时丢弃该样本；文档总长不足 `max_prompt_length` 时同样予以丢弃。

### 1.3 冗余 anchor 去重

由于相邻 anchor 的上下文窗口高度重叠，会引入几乎重复的训练信号。为降低冗余，构造样本前先对同一文档内的 anchor 按 512-token（`max_prompt_length // 2`）的非重叠窗口做去重：将 anchor 位置按窗口编号分桶，**每个窗口仅保留熵值最高的一个 anchor**。该窗口大小与提示长度保持协调，在保留信息密度的同时控制计算预算。

## 2. 消融实验的样本量对齐（下采样）

为保证三个臂（high / low / rand）可以直接对比，三臂在**训练样本数量上严格一致**，以避免不同迭代步数或不同训练 token 量引入混淆因素。

由于 high、low、rand 三臂进入锚点去重阶段前均由每文档 5% 的比例采样，理论上每文档 anchor 数相等；但由于 anchor 的熵空间分布不同，经 512-token 窗口去重后各臂最终产出的训练样本量存在差异。高熵 anchor 更集中，去重后保留样本更少。以 MegaMath-Web-Pro 1B-token 子集为例，三臂去重后的中间样本量分别为 high 约 0.44M、low 约 1.16M、rand 约 1.22M。

为保证公平、一致的比较，采用如下下采样策略：

- 以三臂中样本量最少的 **高熵臂（high）** 为基准数量 `N = 435,049`；
- 对 low 与 rand 臂，从各自去重后的全量样本中做**无放回随机下采样**，统一截断至 `N = 435,049` 个样本；
- 采样覆盖全部文档（最大 `doc_index` 与原集合一致），即按样本级均匀抽样，而非仅截取前部文档。

最终，high、low、rand 三臂均以 `435,049` 个训练样本进入训练，`trainer.total_epochs = 1`，保证三者训练步数与优化预算一致。

## 3. 与传统后训练 OPD 的算法与训练细节差异

OPMD 将 OPD 从面向指令/推理任务的后训练（post-training）场景迁移到 **mid-training（continual pre-training）** 场景。除训练目标由开放式语料的 soft distillation 取代任务奖励外，以下几点构成主要差异。

### 3.1 数据与监督信号来源

- **传统 OPD（post-training）**：使用问答式或可验证推理数据（如数学问题），prompt 通常是问题本身，student 生成完整推理链与最终答案，teacher 对整条生成序列做 token 级概率评估；部分实现可额外引入格式奖励或结果校验器。
- **OPMD（mid-training）**：使用开放式专业语料（如 MegaMath-Web-Pro 抽取的约 1B-token 子集）。不提供显式答案或可验证结果；监督信号完全来自冻结教师对 student 自生轨迹的 **token 级 soft probability**，无 outcome reward / verifier。

### 3.2 Anchor 机制与 implicit reasoning 的引入

- 传统后训练 OPD 通常对整个 prompt 或问题直接生成，不显式区分“需要推理的位置”。
- OPMD 通过代理模型预测熵显式标注高信息 anchor token，并只在 anchor 之前的上下文上触发 student 的生成。这相当于在中训练数据流中人为构造“需要隐式推理”的位置，与 RPT/RLP 中“先思考再预测下一个 token”的思路一致。低熵与随机臂是该 anchor 策略的消融对照。

### 3.3 Rollout 采样规模

- 传统后训练 OPD/RL 通常对同一 prompt 并行采样多条轨迹（如 `n = 4/8/16/32`）以获得组内相对优势或更好的探索。
- OPMD 对每个 anchor **仅采样 1 条 continuation**（`N_RESPONSES = 1`）。其优势估计不依赖组内比较，而是使用 `token_reward_direct` 将逐 token 蒸馏奖励直接作为优势，因此单条轨迹即可稳定更新。

### 3.4 蒸馏奖励的近似与加权

- OPMD 采用 Reverse KL 的逐 token 形式：
  `r_i = log π_ψ(y_i | s_i) - log π_θ^(old)(y_i | s_i)`
- 为降低对全词表概率的计算开销，采用 **top-K 近似**：`LOG_PROB_TOP_K = 16`，`TOP_K_STRATEGY = only_stu`，即仅使用 student 当前分布中最高的 16 个候选 token 计算教师概率；`REWARD_WEIGHT_MODE = teacher_p` 表示将这 16 个候选上的教师概率做归一化后作为加权权重。奖励最终为
  `rm_score = -(log π_θ - log π_ψ) · w_teacher`，与最小化 Reverse KL 的方向一致。
- `ADV_ESTIMATOR = token_reward_direct`，3D（batch × seq × K）的 token 级奖励直接映射为 PPO advantage；`loss_agg_mode = token-mean`。

### 3.5 教师与学生的配置

- 传统后训练 OPD 的教师通常是同任务的指令模型或更强的 SFT/RL 模型；student 多为某个 SFT 初始化模型。
- OPMD 的教师与学生均为**同系列更强/更弱的 base 模型**（如 `qwen3-14b-base → qwen3-1.7b-base`）。为做 scaling-law 验证，扩展到 `qwen3-30b-a3b-base → qwen3-8b-base`。教师全程冻结，只用于前向计算概率，不参与梯度更新，也不参与 rollout 生成。

### 3.6 优化目标与正则

- student 更新等价于最小化其自身探索轨迹分布上的 Reverse KL；此外叠加**与参考模型的 KL 正则**（`USE_KL = True`，`kl_loss_type = low_var_kl`，`kl_loss_coef = 0.005`），以约束策略不偏离自身初始化分布，防止 mid-training 阶段的灾难性偏离。
- 学习率 `1e-6`，线性 warmup 比例 `0.03`；优化器、precision 与工程细节与 verl 的 PPO 配置一致（`model_dtype = bf16`，FSDP，gradient checkpointing，activation offload）。

### 3.7 训练规模与评估

- 数据规模约 1B-token 语料，anchor 采样比例 5%，经去重与下采样后每臂 `435,049` 个训练样本。
- `max_prompt_length = 1024`、`max_response_length = 512`；student rollout `temperature = 1.0`、`top_p = 0.95`；教师评分温度 `teacher_temperature = 1.0`。
- `trainer.total_epochs = 1`，`save_freq = 100`，`test_freq = -1`（训练阶段关闭内建验证，验证在训练完成后单独离线执行），`MINI_BATCH_SIZE = 64`，每步训练 batch 为 `64 × N_GPUS`。
- 与传统后训练 OPD 不同，OPMD 不依赖格式奖励、无验证器信号，也不使用 GRPO 的组内归一化优势；其唯一训练信号为教师 soft probability 得到的 token 级 Reverse KL 蒸馏奖励。

## 4. 关键超参速查

| 项 | 取值 |
|---|---|
| anchor 比例 `anchor_ratio` | 0.05（每文档 5%） |
| anchor 分支 | high / low / rand |
| 随机臂种子 | `seed = 42`，逐文档 `42 + doc_idx` |
| prompt 长度 | 1024 tokens |
| response 长度 | 512 tokens |
| anchor 去重窗口 | 512 tokens |
| 最终每臂样本数 | 435,049 |
| student rollout 采样数 `n` | 1 |
| 采样温度 | 1.0 |
| top-K 蒸馏 | `K = 16`，`only_stu` |
| 奖励加权 | `teacher_p` |
| 优势估计 | `token_reward_direct` |
| KL 正则 | 开启，`low_var_kl`，`coef = 0.005` |
| loss 聚合 | `token-mean` |
| 学习率 | `1e-6` |
| warmup 比例 | `0.03` |
| 精度 | `bfloat16` |
| 训练 epoch | 1 |