# 一、背景

**OPD（On-policy distillation）** 是一种在线的蒸馏方法，核心思想是：

1. 使用当前策略模型（student）采样生成轨迹

2. 使用教师模型（teacher）对学生轨迹的每个token进行评分（即获取冻结的教师模型在该轨迹上的 token 级概率分布）

3. 通过前向 KL（Forward KL）优化学生模型（student），最小化学生和教师预测分布之间的差异，强制使学生在自身探索的轨迹上向教师的思维模式和输出概率对齐。



**Mid-training** 是 Pre-training 的延续，作为 Continual Pre-training，是基于极大似然估计的 Next-Token Prediction，引入数据分布偏移，注入极低熵、高信息密度、强垂直的语料（如海量纯代码、医学文献、超长文本）。其本质是驱动模型在既有的通用参数空间上，向特定的专业高维子空间进行二次深度拟合。



本研究尝试将 OPD 的蒸馏方式应用在 Mid-training 中，以突破硬标签限制、消除领域适应中的“暴露偏差” 、同时增加模型回答的多样性。



# 二、之前的工作

## Reinforcement Pre-Training （https://arxiv.org/html/2506.08007v1）

论文提出了在Pre-Training中融入 Reinforcement 的训练方式 RPT。RPT 在处理千亿级别无标注语料时的具体工作流可以分为以下五个核心步骤：

### 1. 数据清洗与高熵 Token 筛选 (Data Filtering)

在传统的预训练中，模型会按顺序预测语料中的每一个词。但并非所有词都需要“深度思考”（比如“的”、“了”或者固定的标点）。为了节省算力，RPT 首先进行过滤：

* **代理模型评估：** 引入一个轻量级的代理模型（如 1.5B 的小模型），让它在真实语料库上跑一遍预测。

* **计算预测熵：** 记录小模型对每个 Token 的预测概率分布。如果某个 Token 的预测熵很高（意味着小模型觉得很难，无法依靠简单的统计规律猜出），这个 Token 就会被标记为**需要推理的锚点（Anchor Token）**。RPT 的算力主要集中在这些锚点上。

### 2. 状态初始化 (State Initialization)

当训练推进到某个被标记的锚点时，强化学习循环正式启动。

* **当前状态（State）：** 模型当前接收到的所有上下文序列，记为 $$x_{<t}$$。

* **目标（Ground Truth）：** 语料库中真实的下一个词，记为 $$x_t$$。

### 3. 隐式推理与多路径探索 (Next-Token Reasoning & Exploration)

这是 RPT 区别于传统预测（NTP）最核心的一步。模型不再是直接输出下一个词，而是被要求先进行“思考”。

* **生成思考轨迹（Action）：** 基于当前策略 $$\pi_\theta$$，模型针对同一个上下文 $$x_{<t}$$，并行地生成 $$G$$ 条不同的思考路径（Thinking Trajectories）。

* **输出组合：** 每一条路径都包含一段隐式的推理链（$$c_t^i$$）和最终预测的 Token（$$y_t^i$$）。即动作输出为 $$o_t^i = (c_t^i, y_t^i)$$，其中 $$i \in \{1, 2, ..., G\}$$。

### 4. 奖励验证 (Intrinsic Verifiable Reward)

系统需要对模型生成的这 $$G$$ 条路径进行打分。由于是预训练阶段，无法依赖人工标注或复杂的奖励模型，RPT 采用极简的客观匹配规则：

* **0/1 硬匹配：** 提取第 $$i$$ 条路径的最终预测 Token $$y_t^i$$。

* 如果 $$y_t^i$$ 与语料中真实的 $$x_t$$ **完全一致**，则该路径获得奖励 $$R = 1$$。

* 如果不一致，哪怕是同义词，也严格判定为 $$R = 0$$。

### 5. 策略更新与梯度下降 (Policy Update)

拿到奖励信号后，模型通过标准的 On-Policy 强化学习算法（如 PPO 的变体或 GRPO）进行参数更新：

* **正向激励：** 对于获得 $$R=1$$ 的路径，算法会增加生成该“思考轨迹（$$c_t^i$$）”和最终预测的概率。这就相当于模型自己发现了一条通往正确答案的逻辑回路，并强化了这段逻辑。

* **KL 散度约束：** 为了防止模型在强化学习过程中“跑偏”（比如破坏语言模型的连贯性或遗忘基础知识），更新时会引入一个 KL 散度惩罚项，确保更新后的策略模型不会偏离初始参考模型太远。

**流程总结：**

输入上下文 $$\rightarrow$$ 遇到高难度 Token $$\rightarrow$$ 强制模型发散思维生成 $$G$$ 种推理+预测 $$\rightarrow$$ 拿预测结果跟书本原文死磕对比 $$\rightarrow$$ 答对的逻辑保留并强化，答错的抛弃。

## **RLP: Reinforcement as a Pretraining Objective（https://research.nvidia.com/labs/adlr/RLP/）**

NVIDIA 提出的 **RLP (Reinforcement Learning Pretraining)** 的核心流程，是一套旨在让模型“在预测下一个词之前先思考”的完全自监督闭环管线。由于它摒弃了外部验证器（Verifier-Free），整个流程直接在标准的预训练文本流上运行。

### 1. 状态初始化与“无思考”基线评估 (No-Think Baseline)

RLP 不需要像 RPT 那样刻意去寻找高熵的锚点，它直接在连续的预训练语料上进行。

1. **给定上下文与目标：** 在预训练文本流中，给定当前上下文 $$x_{<t}$$，并将真实的下一个词设为目标 Token $$x_t$$。

2. **计算基准概率：** 系统引入一个**指数移动平均 (EMA) 基线模型**（带有参数 $$\phi$$，它更新得很慢，代表了模型过去的平均水平）。该 EMA 模型在不经过任何额外思考的情况下，直接基于 $$x_{<t}$$ 预测目标词 $$x_t$$，记录下此时的对数概率 $$\log p_\phi(x_t | x_{<t})$$。这就是“No-think”（不思考）的基准线。

### 2. 思考轨迹采样 (CoT Action Rollout)

模型将生成内部逻辑视为一种“探索性动作”。

1. **多路径采样：** 当前的策略模型（参数为 $$\theta$$）接收上下文 $$x_{<t}$$，并并行地采样生成 $$G$$ 条不同的隐式思考轨迹（Chain-of-Thought），记为 $$c^{(1)}, c^{(2)}, ..., c^{(G)}$$。

2. **文本拼接：** 每一条生成的思考轨迹 $$c^{(i)}$$ 都会被无缝拼接到原始上下文的末尾，形成增强后的新上下文：$$[x_{<t}, c^{(i)}]$$。

### 3. 基于思考的预测评估 (Conditioned Prediction)

此时，模型需要证明它刚才生成的思考轨迹是否真的有用。

1. **再次预测目标词：** 策略模型 $$\theta$$ 基于拼接了思考轨迹的新上下文 $$[x_{<t}, c^{(i)}]$$，再次预测那个真实的下一个词 $$x_t$$。

2. **记录新概率：** 记录下加入思考后，模型对目标词预测的对数概率 $$\log p_\theta(x_t | x_{<t}, c^{(i)})$$。

### 4. 信息增益奖励计算 (Information-Gain Reward)

这是 RLP 区别于所有其他方法的核心：**用概率差值作为奖励，而不是用 0/1 硬匹配。**

1. 对于生成的第 $$i$$ 条思考轨迹，其获得的绝对奖励 $$r(c^{(i)})$$ 定义为：

2. $$r(c^{(i)}) = \log p_\theta(x_t | x_{<t}, c^{(i)}) - \log p_\phi(x_t | x_{<t})$$

3. **原理解释：** 它的物理含义是“信息增益”。如果加上这段思考后，预测真实词的概率比“不思考”时更高（即差值为正），说明这段思考是有价值的；反之，如果思考把模型带偏了，导致预测真实词的概率降低，就会得到负奖励。这种机制完美地避开了同义词惩罚。

### 5. 组内优势计算与参数更新 (Update & Replicator Dynamic)

为了保证梯度下降的稳定，RLP 采用了基于组相对优势（Group-relative advantages）的更新机制。

1. **优势归一化：** 计算出 $$G$$ 条路径奖励的平均值 $$\bar{r}$$，然后得出每条路径的相对优势 $$A^{(i)} \propto (r(c^{(i)}) - \bar{r})$$。

2. **策略梯度更新：** 将优势 $$A^{(i)}$$ 代入策略梯度公式，更新策略模型 $$\theta$$ 的权重。哪怕所有的绝对奖励都是负的，这种组内比较也能挑出“最不坏”的那条思考轨迹，保证模型在期望上单调递增（Monotonic Improvement）。

3. **滑动更新基线：** 策略模型 $$\theta$$ 更新完毕后，基线模型 $$\phi$$ 会以极小的步长（EMA）向新的 $$\theta$$ 滑动，从而防止模型自己钻奖励机制的空子（Reward Inflation）。



# 三、我们的思路

**本质是要验证OPD在mid-training上的可行性，不追求复杂的技术路线创新。**

## 核心符号定义

在设计公式前，我们先统一符号体系：

* $$x_{<t}$$：语义锚点（Anchor Token）之前的原始上下文。

* $$x_t$$：代理模型在文本中挑选的用于预测的高信息熵的目标词（锚点词）。

* $$c = [y_1, y_2, \dots, y_K]$$：当前策略模型 $$\theta$$ 生成的包含 $$K$$ 个 token 的隐式思考轨迹。

* $$\psi$$：教师模型参数（Teacher Model）。

* $$\theta$$：当前正在训练的策略模型参数（Student/Policy Model）。



我的思路：

**核心流程：针对专业领域的dataset与初步pre-training后的大模型（如qwen3-1.7B-base）进行OPMD（on-policy mid-training），teacher model 使用能力更强的模型。**



数据集选取：从MegaMath-Web-Pro中选取 1B token 作为mid - training数据

## 1. **使用 小型代理模型 进行锚点标记**

使用 Deepseek-R1-Distill-Qwen-1.5B 作为一个小型代理模型进行 Anchor token 标记。

## 2. **采样：从 Anchor 之前的上下文开始生成**

模型仅接收 anchor token 之前的上下文 `x_{<t}` 作为 prompt（不包含 `x_t`）。Student 自主从 `x_{<t}` 生成后续文本，理想情况下第一个生成 token 即为 anchor token `x_t`。

**路径采样：** 当前策略模型（参数为 $$\theta$$）接收上下文 $$x_{<t}$$，采样生成后续文本。

完整 rollout 序列：

$$(x_{<t}, y_1, y_2, \dots, y_K)$$

其中 $$y_1$$ 应为模型预测的 anchor token，$$y_2, \dots, y_K$$ 为后续 continuation。K 为 response_length（默认 512 tokens）。

## 3. **核心奖励函数 (Reward Function) 设计：Reverse KL Distillation**

由于 rollout 数据来自 Student 自身的探索分布 $$y \sim \pi_\theta$$，我们优化 **Reverse KL**：

$$L_{\text{OPD}} = D_{KL}(P_\theta \parallel P_\psi) = \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_i \left( \log \pi_\theta(y_i \mid x_{<t}, y_{<i}) - \log \pi_\psi(y_i \mid x_{<t}, y_{<i}) \right) \right]$$

Token-level 蒸馏奖励：

$$r_i = \log \pi_\psi(y_i \mid x_{<t}, y_{<i}) - \log \pi_\theta^{(\text{old})}(y_i \mid x_{<t}, y_{<i})$$

> 如果 Teacher 认为该 token 比 Student 更可能 → $$r_i > 0$$（增强）；反之 $$r_i < 0$$（抑制）。

在实际实现中（参考 OPD 项目），通过 top-K 采样（K=16）在 Student 的最高概率候选 token 上近似计算 reward，使用 `token_reward_direct` 优势估计器直接将 token-level reward 传入 PPO loss 进行梯度下降。

优化目标：

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_i r_i \right] \quad \text{（等价于最小化 Reverse KL）}$$

# 四、实验设计

## Phase 1: 确立 Baseline (标准 NTP)

这是第一个对照组。我们将使用相同的MegaMath-Web-Pro提取的 1B token 的 datasets 在 student model（qwen3-4b-base）上进行mid-training。

## Phase 2: 引入 OPD

本阶段将标准的 Next-Token Prediction 替换为 On-policy Distillation (OPD)，验证 OPD 是否能够作为一种 Mid-training Objective，用于专业领域 Continual Pre-training。

整个实验保持 **训练数据、训练 Token 数、学习率、Batch Size、Optimizer、Student Model** 与 Phase 1 完全一致，仅改变训练目标。

主代码仿照 OPD 项目代码（https://github.com/thunlp/OPD.git）

### Anchor Token 选择

首先利用代理模型

> DeepSeek-R1-Distill-Qwen-1.5B

扫描 MegaMath-Web-Pro。

对于每个 token $$x_t$$ 计算预测熵

$$H(xt)=−∑_{i}p_i \cdot logpi$$

仅保留高熵 Token 作为 Anchor。

这样做的目的在于：

* 避免大量简单 token（标点、格式词）的训练浪费；

* 将 OPD 的计算预算集中在真正需要推理的区域；

* 与 RPT 保持一致，使不同方法具有可比性。

***

### 3.1 Student On-policy Rollout

对于每一个 Anchor token `x_t`，Student Model $$\pi_\theta$$ 仅读取 anchor 之前的上下文 `x_{<t}` 作为 prompt（不包含 `x_t` 本身）。

Student 从 `x_{<t}` 出发，根据自身当前策略自主生成后续文本。理想情况下，生成的第一个 token 应为 anchor token `x_t`，随后继续生成后续 continuation：

$$c = (y_1, y_2, \cdots, y_K)$$

采用 stochastic sampling 而非 greedy decoding：

* Temperature = 1
* Top-p = 0.95

生成过程：

$$y_i \sim \pi_\theta(y_i \mid x_{<t}, y_{<i})$$

因此整个 rollout trajectory：

$$\tau = (x_{<t}, y_1, y_2, \cdots, y_K)$$

完全来自 Student policy：$$\tau \sim \pi_\theta$$

Teacher 不参与生成过程。

Student 同时保留自身生成时每个 token 的概率：

$$\log \pi_\theta(y_i \mid x_{<t}, y_{<i})$$

**设计要点**：`x_t` 不在 prompt 中，而是 Student 需要自主推理出的第一个生成 token。这与 RPT 的核心思想一致——模型在高熵锚点处被强制推理，而非直接看到答案。

***

### 3.2 Teacher Evaluation

冻结 Teacher $$\pi_\psi$$。

Teacher 接收 Student 已经生成的完整序列：

$$(x_{<t}, y_1, y_2, \cdots, y_K)$$

Teacher 不重新采样，仅执行一次 forward pass，对 Student 已产生的每一个 token 计算概率：

$$\log \pi_\psi(y_i \mid x_{<t}, y_{<i})$$

因此对于每一个生成的 token `y_i`（包括锚点 token `y_1`），获得：

* **Student probability**: $$P_\theta(y_i \mid x_{<t}, y_{<i})$$
* **Teacher probability**: $$P_\psi(y_i \mid x_{<t}, y_{<i})$$

Teacher 全程冻结（$$\nabla_\psi L = 0$$）。

Teacher 不重新采样，仅执行一次 forward pass，对 Student 已产生的每一个 token 计算概率：

$$\log \pi_\psi(y_i \mid x_{<t}, x_t, y_{<i})$$

因此对于每一个 token `y_i`，获得：

* **Student probability**: $$P_\theta(y_i \mid s_i)$$，其中 $$s_i = (x_{<t}, x_t, y_{<i})$$
* **Teacher probability**: $$P_\psi(y_i \mid s_i)$$

Teacher 全程冻结（$$\nabla_\psi L = 0$$）。

***

### 3.3 Reverse KL Distillation (Token-level)

由于 rollout 数据 $$y \sim P_\theta$$ 来自 Student 自身的探索分布，优化目标对应 **Reverse KL**：

$$L_{\text{OPD}} = D_{KL}(P_\theta \parallel P_\psi)$$

展开：

$$L_{\text{OPD}} = \sum_y P_\theta(y) \log \frac{P_\theta(y)}{P_\psi(y)}$$

对于 autoregressive language model，token-level 分解为：

$$L_{\text{OPD}} = \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_i \left( \log \pi_\theta(y_i \mid s_i) - \log \pi_\psi(y_i \mid s_i) \right) \right]$$

其中 $$s_i = (x_{<t}, x_t, y_{<i})$$。

**Token-level Distillation Reward：**

定义每个 token 的即时奖励为：

$$r_i = \log \pi_\psi(y_i \mid s_i) - \log \pi_\theta^{(\text{old})}(y_i \mid s_i)$$

* 如果 Teacher 比 Student 更倾向该 token：$$r_i > 0$$，Student 获得增强信号
* 如果 Student 生成了 Teacher 低概率 token：$$r_i < 0$$，该 token 被抑制

实际上，verl 的 PPO trainer 中通过 `token_reward_direct` 优势估计器直接将 $$r_i$$ 作为 advantage 信号，PPO loss 的梯度下降等价于最小化 Reverse KL。

OPD 项目中进一步通过 top-K 采样（K=16, `only_stu`）在 Student 的最高概率候选 token 上近似计算该 reward，以降低计算开销：

$$r_i^{(k)} = \left( \log \pi_\psi(k \mid s_i) - \log \pi_\theta^{(\text{old})}(k \mid s_i) \right) \cdot \frac{\pi_\psi(k \mid s_i)}{\sum_{j \in \text{top-K}} \pi_\psi(j \mid s_i)}$$

**核心直觉：**

> Reverse KL（mode-seeking）迫使 Student 在自己的探索轨迹上，通过 Teacher 提供的 token-level preference signal，不断修正自身 policy。不同于 Forward KL 的平均化行为（mode-covering），Reverse KL 鼓励 Student 聚焦于教师高置信度的 generation 模式，同时保持从自身分布采样的 on-policy 特性。

***

### 3.4 参数更新

Student 参数更新目标（等价于最小化 Reverse KL）：

$$\theta \leftarrow \theta - \eta \nabla_\theta L_{\text{OPD}}$$

Teacher 全程冻结：

$$\nabla_\psi L = 0$$

虽然优化形式使用梯度下降，但它不再是传统的 supervised learning——监督信号由 Teacher 的 soft distribution 提供，而非 hard label。这本质上是一种 **token-level preference-based policy optimization**。

***

