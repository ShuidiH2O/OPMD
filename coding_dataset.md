# Coding Dataset Reprocessing / Coding 数据集重新处理

## 1. Goal / 目标

Our previous **StarCoderData** coding corpus did not perform well. Preliminary inspection suggests it contains too much non-pure-code content, such as web pages, forums, Markdown, documentation, JSON/YAML, etc.

我们之前使用 **StarCoderData** 构建的 coding corpus 效果不好。初步检查发现其中包含较多非纯代码内容，例如网页、论坛、Markdown、文档、JSON/YAML 等。

For the next experiment, **do not continue heavily modifying StarCoderData**. Instead, investigate **The Stack v2**, especially its `train-smol` subset, as the new primary data source.

下一轮实验**不要继续在 StarCoderData 上做复杂筛选**。请优先调查 **The Stack v2**，尤其是 `train-smol` 子集，作为新的主要数据源。

---

## 2. Dataset / 数据集

Please verify:

请先确认：

* The exact languages included in `the-stack-v2-train-smol`
* `train-smol` 的具体语言列表
* Whether the actual source code can be directly obtained
* 是否可以直接获得实际 source code
* Dataset version/revision and licensing
* 数据集版本、revision 和 license
* Whether it is practical to construct at least 1B tokens

如果 `train-smol` 可以方便地获得实际代码，并且规模足够，优先使用它。

---

## 3. Filtering / 数据过滤

Keep the corpus **multilingual and code-centric**. Do **NOT** make it Python-only or Python-heavy just because HumanEval+/MBPP+ are mainly Python.

保持**多语言、以代码为中心**，不要因为 HumanEval+/MBPP+ 主要是 Python 就把训练集改成 Python-only 或 Python-heavy。

### Keep / 保留

* Python
* C / C++
* Java
* JavaScript / TypeScript
* Go
* Rust
* Ruby / PHP
* Shell
* SQL
* other genuine programming languages

### Remove or strongly restrict / 删除或强烈限制

* Markdown
* standalone documentation
* forum/web text
* JSON/YAML-only files
* generated/minified files
* obvious non-source-code content

Comments and docstrings **should be kept**.

代码中的 comments 和 docstrings **应该保留**，因为它们是真实 source code 的一部分。

---

## 4. Build Dataset / 构建数据集

Construct approximately:

```text
1B Qwen3 tokens
```

using the tokenizer:

```text
Qwen/Qwen3-1.7B-Base
```

Use a reproducible shuffle/sampling seed.

不要简单取数据集前面的 1B tokens；需要 shuffle 后进行采样，并记录 seed。

Do not create a Python-dominated corpus unless the language distribution of the source dataset is already naturally dominated by Python.

除非原始数据本身就是 Python 占绝大多数，否则不要人为构造 Python-heavy corpus。

---

## 5. Quick Audit / 简单数据检查

Before building the final 1B-token dataset, sample a small amount of data and report:

在生成完整 1B-token corpus 前，先抽样检查并报告：

```text
language distribution / 语言分布
approximate code purity / 大致代码纯度
Markdown ratio / Markdown 比例
JSON/YAML ratio / JSON/YAML 比例
documentation/web content / 文档和网页内容
```

Also compare briefly with the previous StarCoderData corpus.

同时简单和之前的 StarCoderData 做对比。

We do **not** need a complicated data-quality analysis. The main purpose is simply to verify that the new dataset is substantially more code-centric.

不需要复杂的数据质量分析，主要确认新数据集确实比 StarCoderData 更偏向真实 source code。

---

## 6. Evaluation / 评测

Keep the existing coding benchmarks:

```text
HumanEval+
MBPP+
LiveCodeBench
```

Also investigate adding:

```text
BigCodeBench
FIM / code completion
```

The purpose of FIM/code completion is to evaluate code modeling ability that is closer to the pretraining distribution.

FIM/code completion 的目的是评估与 coding pretraining distribution 更接近的代码建模能力。

Do not invent a new FIM benchmark unless necessary; first check whether there is an established, reproducible benchmark.

不要自己随意构造 FIM test set，优先寻找已有的、可复现的 benchmark。

---

## 7. Pilot / 小规模验证

Before running the full 1B-token experiment, run a small pilot of roughly:

```text
50M–100M tokens
```

Check whether training behavior is reasonable.

先跑约 **50M–100M tokens** 的小规模实验，确认：

* training loss
* validation loss
* throughput
* coding benchmark performance

Then, if everything looks reasonable, construct/run the full 1B-token experiment.

确认没有明显问题后，再进行完整的 1B-token NTP vs OPD experiment。

---

## 8. Final Output / 最终输出

Save the processed dataset under the existing project structure:

```text
/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/
```

Please report only:

1. Dataset source and exact version
2. Languages included
3. Filtering rules
4. Final token count
5. Language distribution
6. Comparison with the previous StarCoderData corpus
7. Exact path of the processed dataset
8. Any important issues or limitations

**Do not start the full 1B-token training until the dataset construction and the 50M–100M token pilot look reasonable.**
